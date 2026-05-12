"""Wrapper around fla's GLA (Gated Linear Attention) for use in quark's training pipeline."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from fla.models.gla import GLAConfig, GLAForCausalLM


@dataclass
class GLAWrapperConfig:
    """Quark-style config for GLA. Maps to fla's GLAConfig."""

    vocab_size: int = 50304
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    expand_k: float = 0.5
    expand_v: float = 1.0
    use_short_conv: bool = False
    conv_size: int = 4
    use_output_gate: bool = True


class GLA(torch.nn.Module):
    """Thin wrapper around GLAForCausalLM with quark's forward interface."""

    def __init__(self, config: GLAWrapperConfig):
        super().__init__()
        self.config = config
        fla_config = GLAConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_heads=config.num_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            use_output_gate=config.use_output_gate,
            # Disable fla's fused losses — we compute loss ourselves
            fuse_cross_entropy=False,
            fuse_linear_cross_entropy=False,
        )
        self.model = GLAForCausalLM(fla_config)

    def forward(self, idx, targets=None):
        """Run forward pass with quark's (idx, targets) -> (logits, loss) interface.

        Args:
            idx: Token indices of shape ``(batch, seq_len)``.
            targets: Pre-shifted target indices of shape ``(batch, seq_len)``.
                Positions with value ``-1`` are ignored in the loss.

        Returns:
            Tuple of ``(logits, loss)`` where loss is ``None`` when targets is not provided.
        """
        hidden_states = self.model.model(
            input_ids=idx, use_cache=False, return_dict=True
        ).last_hidden_state
        logits = self.model.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss
