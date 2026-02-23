"""Custom implementation of signSGD and signum."""

import torch
from torch.optim import Optimizer


class signSGD(Optimizer):  # noqa: N801
    """SignSGD optimizer with momentum."""

    def __init__(self, params, lr, momentum=0.0, dampening=0.0, weight_decay=0.1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= dampening <= 1.0:
            raise ValueError(f"Invalid dampening: {dampening}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Perform a single optimization step."""
        for group in self.param_groups:
            alpha = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]

                # Weight decay
                p.mul_(1 - alpha * weight_decay)

                # Momentum initialization
                if "m" not in param_state:
                    param_state["m"] = p.grad.detach().clone()

                # Decay momentum
                m = param_state["m"]
                m.mul_(momentum).add_(p.grad, alpha=(1.0 - dampening))

                # Update parameters
                p.add_(torch.sign(m), alpha=-alpha)
