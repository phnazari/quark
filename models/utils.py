"""Shared model utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """Forward pass."""
        return F.normalize(x, dim=-1, eps=self.eps) * self.scale * (self.gamma + 1)


def init_gpt_weights(model, std=0.02, residual_std_factor=0.5):
    """Initialize GPT model weights (GPT-2 style, residual projections scaled down)."""

    def _init(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    model.apply(_init)

    for name, param in model.named_parameters():
        if "c_proj.weight" in name or "proj.weight" in name:
            param.data.normal_(mean=0.0, std=std * residual_std_factor)
