"""TorchEngine: encapsulates model, optimizer, scheduler, scaler, and training step.

Adapted from plainLM's engine. Key difference: our model returns (logits, loss)
directly from forward(), so we don't use a separate CrossEntropyLoss criterion.
"""

from contextlib import nullcontext

import torch
from optim import initialize_optimizer, initialize_scheduler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from utils import get_param_groups


class TorchEngine(torch.nn.Module):
    """Wraps model, optimizer, scheduler, scaler into a training step."""

    def __init__(self, model, cfg, device, local_rank, ckpt):
        super().__init__()

        self.micro_steps = 0
        self.accumulated_samples = 0

        self.seq_len = cfg.data.seq_len
        self.accumulation_steps = cfg.training.grad_accumulation_steps
        self.grad_clip = cfg.training.grad_clip
        self.dtype = cfg.system.dtype

        self.device = device

        # Load model state dict if resuming
        if cfg.checkpoint.resume:
            model.load_state_dict(ckpt["state_dict"])
            self.micro_steps = ckpt["step"] * cfg.training.grad_accumulation_steps

        # Move model to device and wrap in DDP
        self.model = model.to(device)
        if torch.distributed.is_initialized():
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=False)

        # Compile
        if cfg.system.compile_model:
            print("Compiling the model...")
            self.model = torch.compile(self.model)

        # AMP
        device_type = "cuda" if "cuda" in device else "cpu"
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            self.dtype
        ]
        self.ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        # Grad scaler (no-op if not fp16)
        self.scaler = torch.amp.GradScaler(enabled=(self.dtype == "float16"))

        # Optimizer and scheduler
        param_groups = get_param_groups(model, cfg.training.weight_decay)
        self.optimizer = initialize_optimizer(param_groups, cfg)
        self.scheduler = initialize_scheduler(self.optimizer, cfg)

        if cfg.checkpoint.resume:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler"):
                self.scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler"):
                self.scaler.load_state_dict(ckpt["scaler"])

    def step(self, batch):  # noqa: C901
        """Run one micro-step: forward, backward, and optionally optimizer step."""
        self.model.train()

        self.micro_steps += 1
        self.accumulated_samples += 1

        inputs, targets = _move_to_device(batch, self.seq_len, self.device)

        # Sync gradients only at the last accumulation step
        if torch.distributed.is_initialized():
            self.model.require_backward_grad_sync = (
                self.accumulated_samples == self.accumulation_steps
            )

        # Forward pass — our model returns (logits, loss)
        with self.ctx:
            _, loss = self.model(inputs, targets)
            loss = loss / self.accumulation_steps

        # Detach for logging (scale up to undo division)
        loss_val = loss.detach() * self.accumulation_steps
        if torch.isnan(loss_val):
            raise ValueError("Train loss is nan")

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Step after accumulation
        grad_norm = None
        if self.accumulated_samples == self.accumulation_steps:
            self.accumulated_samples = 0

            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler:
                self.scheduler.step()

        return loss_val, grad_norm

    @torch.no_grad()
    def eval(self, dataloader):
        """Evaluate model on a dataloader, return average loss."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            inputs, targets = _move_to_device(batch, self.seq_len, self.device)
            with self.ctx:
                _, loss = self.model(inputs, targets)

            if torch.isnan(loss) or loss is None:
                raise ValueError("Validation loss is nan")

            total_loss += loss.item()
            num_batches += 1

        # Reduce across processes
        if dist.is_initialized():
            total_loss_tensor = torch.tensor([total_loss], device=self.device)
            num_batches_tensor = torch.tensor([num_batches], device=self.device, dtype=torch.int)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item()
            num_batches = num_batches_tensor.item()

        return total_loss / num_batches


def _move_to_device(batch, seq_len, device):
    """Slice batch into inputs and targets, move to device."""
    inputs = batch["input_ids"][:, :seq_len].contiguous()
    targets = batch["input_ids"][:, 1 : (seq_len + 1)].contiguous()

    if "cuda" in device:
        inputs = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs, targets = inputs.to(device), targets.to(device)

    return inputs, targets
