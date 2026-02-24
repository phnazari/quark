"""Train a GPT-style transformer for language modeling.

Usage:
    python train.py
    python train.py training.steps_budget=10000
    python train.py training.optim=nadamw training.scheduler=wsd training.cooldown_steps=100
    torchrun --standalone --nproc_per_node=4 train.py
"""

import os
from collections import defaultdict

import hydra
from checkpoint_utils import maybe_load_checkpoint, save_checkpoint
from data import get_dataloaders
from engine import TorchEngine
from models import DeltaNet, DeltaNetWrapperConfig, Transformer, TransformerConfig
from omegaconf import DictConfig
from torch_utils import destroy_ddp, pytorch_setup
from utils import flatten_config, log, maybe_make_dir, print_master


def build_model(cfg: DictConfig, flat_cfg):
    """Build a model from config, dispatching on model_type."""
    vocab_size = flat_cfg.vocab_size if hasattr(flat_cfg, "vocab_size") else 50304
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
        block_size=flat_cfg.seq_len,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
    )
    return Transformer(config), config


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):  # noqa: C901
    """Main training function."""
    flat_cfg = flatten_config(cfg)

    local_rank, world_size, device, master_process = pytorch_setup(flat_cfg)

    if master_process:
        maybe_make_dir(flat_cfg)

    ckpt = maybe_load_checkpoint(flat_cfg)

    trainloader, validloader = get_dataloaders(flat_cfg)

    model, config = build_model(cfg, flat_cfg)

    if master_process:
        num_params = sum(p.numel() for p in model.parameters())
        print("=" * 80)
        print(f"Training {cfg.model.model_type}")
        print(f"  Device: {device}, dtype: {cfg.system.dtype}")
        print(f"  Hidden: {config.hidden_size}, Layers: {config.num_layers}")
        if hasattr(config, "head_dim"):
            print(f"  Heads: {config.num_heads}, Head dim: {config.head_dim}")
        else:
            print(f"  Heads: {config.num_heads}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Steps budget: {flat_cfg.steps_budget}")
        print("=" * 80)

    engine = TorchEngine(model, flat_cfg, device, local_rank, ckpt)

    if cfg.logging.wandb_log and master_process:
        import uuid

        import wandb
        from omegaconf import OmegaConf

        num_params = sum(p.numel() for p in model.parameters())
        run_id = uuid.uuid4().hex[:6]
        wandb.init(
            project=cfg.logging.wandb_project,
            name=f"transformer-L{cfg.model.num_layers}-D{cfg.model.hidden_size}-lr{cfg.training.lr}-{run_id}",
            config={**OmegaConf.to_container(cfg, resolve=True), "num_params": num_params},
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

    steps_budget = flat_cfg.steps_budget
    micro_step_budget = steps_budget * flat_cfg.grad_accumulation_steps
    step_start = flat_cfg.resume_step if flat_cfg.resume else 0
    micro_step_start = step_start * flat_cfg.grad_accumulation_steps

    print_master(
        f"=== Start Training from step {step_start}/{steps_budget}, "
        f"micro_step {micro_step_start}/{micro_step_budget} ==="
    )

    metrics = defaultdict(list)
    train_loss_array = []

    for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
        step = micro_step // flat_cfg.grad_accumulation_steps
        is_step = micro_step % flat_cfg.grad_accumulation_steps == 0
        if step > steps_budget and is_step:
            break

        train_loss, grad_norm = engine.step(micro_batch)
        train_loss_array.append(train_loss)

        valid_loss = None
        if flat_cfg.eval and validloader and step % flat_cfg.eval_every_steps == 0 and is_step:
            print_master("Evaluating on validation set")
            valid_loss = engine.eval(validloader)

        if master_process and step % flat_cfg.log_every_steps == 0 and is_step:
            log(
                flat_cfg,
                metrics,
                micro_step,
                train_loss,
                train_loss_array,
                valid_loss,
                engine.optimizer,
                world_size,
                grad_norm,
            )
            train_loss_array = []

        if (
            master_process
            and flat_cfg.save_intermediate_checkpoints
            and flat_cfg.save_every_steps
            and step % flat_cfg.save_every_steps == 0
            and is_step
        ):
            save_checkpoint(step, model, engine, flat_cfg, metrics)

    print_master("=== Training Completed! ===")
    if master_process and flat_cfg.save_last_checkpoint:
        save_checkpoint(step, model, engine, flat_cfg, metrics)

    if cfg.logging.wandb_log and master_process:
        import wandb

        wandb.finish()

    destroy_ddp()


if __name__ == "__main__":
    main()
