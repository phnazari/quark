"""Train a GPT-style transformer for language modeling.

Usage:
    python train.py
    python train.py model=delta_net
    python train.py training.steps_budget=10000
    python train.py training.optim=nadamw training.scheduler=wsd training.cooldown_steps=100
    torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import time
from collections import defaultdict

import hydra
from checkpoint_utils import maybe_load_checkpoint, save_checkpoint
from data import get_dataloaders
from engine import TorchEngine
from models import DeltaNet, DeltaNetWrapperConfig, Transformer, TransformerConfig
from omegaconf import DictConfig, OmegaConf
from torch_utils import destroy_ddp, pytorch_setup
from tqdm import tqdm
from utils import get_variant_name, log, maybe_make_dir, print_master


def build_model(cfg: DictConfig):
    """Build a model from config, dispatching on model_type."""
    vocab_size = cfg.data.vocab_size
    model_type = cfg.model.model_type

    if model_type == "delta_net":
        config = DeltaNetWrapperConfig(
            vocab_size=vocab_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            expand_k=cfg.model.expand_k,
            expand_v=cfg.model.expand_v,
            use_gate=cfg.model.use_gate,
            use_beta=cfg.model.use_beta,
            use_short_conv=cfg.model.use_short_conv,
            conv_size=cfg.model.conv_size,
        )
        return DeltaNet(config), config

    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        head_dim=cfg.model.head_dim,
        block_size=cfg.data.seq_len,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
    )
    return Transformer(config), config


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):  # noqa: C901
    """Main training function."""
    # Setup
    local_rank, world_size, device, master_process = pytorch_setup(cfg)

    if master_process:
        maybe_make_dir(cfg)

    # Checkpoint
    ckpt = maybe_load_checkpoint(cfg)

    # Data
    trainloader, validloader = get_dataloaders(cfg)

    # Model
    model, config = build_model(cfg)

    if master_process:
        variant_name = get_variant_name(cfg)
        num_params = sum(p.numel() for p in model.parameters())
        total_tokens = (
            cfg.training.steps_budget
            * cfg.training.grad_accumulation_steps
            * cfg.training.micro_batch_size
            * cfg.data.seq_len
            * world_size
        )
        print("=" * 80)
        print(f"Training {variant_name}")
        print(f"  Device: {device}, dtype: {cfg.system.dtype}")
        print(f"  Hidden: {config.hidden_size}, Layers: {config.num_layers}")
        if hasattr(config, "head_dim"):
            print(f"  Heads: {config.num_heads}, Head dim: {config.head_dim}")
        else:
            print(f"  Heads: {config.num_heads}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Steps budget: {cfg.training.steps_budget}")
        print(f"  Total training tokens: {total_tokens:,}")
        print("=" * 80)

    # Engine
    engine = TorchEngine(model, cfg, device, local_rank, ckpt)

    # W&B
    if cfg.logging.wandb_log and master_process:
        import uuid

        import wandb

        variant_name = get_variant_name(cfg)
        num_params = sum(p.numel() for p in model.parameters())
        total_tokens = (
            cfg.training.steps_budget
            * cfg.training.grad_accumulation_steps
            * cfg.training.micro_batch_size
            * cfg.data.seq_len
            * world_size
        )
        run_id = uuid.uuid4().hex[:6]
        wandb.init(
            project=cfg.logging.wandb_project,
            name=f"{variant_name}-L{cfg.model.num_layers}-D{cfg.model.hidden_size}-lr{cfg.training.lr}-{run_id}",
            config={
                **OmegaConf.to_container(cfg, resolve=True),
                "num_params": num_params,
                "total_tokens": total_tokens,
            },
        )
        wandb.run.log_code(
            root=".",
            include_fn=lambda p, r: (
                p.endswith((".py", ".yaml"))
                and any(
                    os.path.relpath(p, r).startswith(d) for d in ("src/", "configs/", "train.py")
                )
            ),
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    # Training loop
    steps_budget = cfg.training.steps_budget
    micro_step_budget = steps_budget * cfg.training.grad_accumulation_steps
    step_start = cfg.checkpoint.resume_step if cfg.checkpoint.resume else 0
    micro_step_start = step_start * cfg.training.grad_accumulation_steps

    print_master(
        f"=== Start Training from step {step_start}/{steps_budget}, "
        f"micro_step {micro_step_start}/{micro_step_budget} ==="
    )

    metrics = defaultdict(list)
    train_loss_array = []
    t_log_start = time.time()
    micro_step_prev = micro_step_start
    pbar = None
    if master_process:
        pbar = tqdm(
            total=steps_budget,
            initial=step_start,
            desc="Training steps",
            dynamic_ncols=True,
        )

    for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
        step = micro_step // cfg.training.grad_accumulation_steps
        is_step = micro_step % cfg.training.grad_accumulation_steps == 0
        if step > steps_budget and is_step:
            break

        # Train
        train_loss, grad_norm = engine.step(micro_batch)
        train_loss_array.append(train_loss)

        # Progress bar (one tick per optimizer step)
        if master_process and is_step and pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=float(train_loss))

        # Eval
        valid_loss = None
        if cfg.data.eval and validloader and step % cfg.training.eval_every_steps == 0 and is_step:
            print_master("Evaluating on validation set")
            valid_loss = engine.eval(validloader)

        # Log
        if master_process and step % cfg.training.log_every_steps == 0 and is_step:
            # Throughput
            t_log_end = time.time()
            dt = t_log_end - t_log_start
            micro_steps_done = micro_step - micro_step_prev
            throughput = (
                micro_steps_done
                * cfg.training.micro_batch_size
                * cfg.data.seq_len
                * world_size
                / dt
            )
            step_time = dt / (micro_steps_done / cfg.training.grad_accumulation_steps)

            log(
                cfg,
                metrics,
                micro_step,
                train_loss,
                train_loss_array,
                valid_loss,
                engine.optimizer,
                world_size,
                grad_norm,
                throughput=throughput,
                step_time=step_time,
            )
            train_loss_array = []
            t_log_start = time.time()
            micro_step_prev = micro_step

        # Checkpoint
        if (
            master_process
            and cfg.checkpoint.save_intermediate_checkpoints
            and cfg.checkpoint.save_every_steps
            and step % cfg.checkpoint.save_every_steps == 0
            and is_step
        ):
            save_checkpoint(step, model, engine, cfg, metrics)

    # End of training
    if master_process and pbar is not None:
        pbar.close()

    print_master("=== Training Completed! ===")
    if master_process and cfg.checkpoint.save_last_checkpoint:
        save_checkpoint(step, model, engine, cfg, metrics)

    if cfg.logging.wandb_log and master_process:
        import wandb

        wandb.finish()

    destroy_ddp()


if __name__ == "__main__":
    main()
