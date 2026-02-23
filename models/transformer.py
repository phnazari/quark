"""Standard pre-norm GPT transformer."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .utils import RMSNorm, init_gpt_weights


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with separate Q/K/V projections.

    The attention dimension is ``num_heads * head_dim`` and may differ from
    ``hidden_size`` (the projection handles the mapping).
    """

    def __init__(self, hidden_size, num_heads, head_dim, dropout=0.0, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply causal self-attention.

        Args:
            x: ``(batch, seq_len, hidden_size)``

        Returns:
            Output of shape ``(batch, seq_len, hidden_size)``.
        """
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.out_proj(y)


class MLP(nn.Module):
    """SwiGLU feed-forward network (hidden dim = 8/3 * hidden_size)."""

    def __init__(self, hidden_size, bias=False):
        super().__init__()
        hidden_dim = int(8 / 3 * hidden_size)

        self.gate_proj = nn.Linear(hidden_size, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_size, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, hidden_size, bias=bias)

    def forward(self, x):
        """Apply SwiGLU transformation."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass
class TransformerConfig:
    """Configuration for the GPT transformer."""

    vocab_size: int = 50304
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attention + SwiGLU MLP with residuals."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = CausalSelfAttention(
            config.hidden_size, config.num_heads, config.head_dim, config.dropout, config.bias
        )
        self.mlp = MLP(config.hidden_size, config.bias)
        self.ln_1 = RMSNorm(config.hidden_size)
        self.ln_2 = RMSNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP with residual connections."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Standard pre-norm GPT model for language modeling."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        init_gpt_weights(self)

    def forward(self, idx, targets=None):
        """Run forward pass, optionally computing cross-entropy loss.

        Args:
            idx: Token indices of shape ``(batch, seq_len)``.
            targets: Optional target indices. Positions with value ``-1`` are ignored.

        Returns:
            Tuple of ``(logits, loss)`` where loss is ``None`` when targets is not provided.
        """
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively generate new tokens.

        Args:
            idx: Conditioning token indices of shape ``(batch, seq_len)``.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: If set, only sample from the top-k most likely tokens.

        Returns:
            Token indices of shape ``(batch, seq_len + max_new_tokens)``.
        """
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
