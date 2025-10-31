import math

import flax.linen as nn
import jax.numpy as jnp


class MultiHeadSelfAttention(nn.Module):
    """Vanilla multi-head self-attention with optional dropout."""

    dimension: int
    num_heads: int
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        """
        Compute multi-head self-attention.

        Args:
            x: Input tensor of shape (B, N, D).
            mask: Optional attention mask with broadcastable shape.
            deterministic: Disable dropout when True.

        Returns:
            Tensor of shape (B, N, D) after attention.
        """
        b, n, d = x.shape
        h = self.num_heads
        head_dim = d // h
        if d % h != 0:
            raise ValueError("dimension must be divisible by num_heads")

        # Project once to obtain queries, keys, and values (saves parameters).
        qkv = nn.Dense(features=d * 3, use_bias=False, name="qkv")(x)
        qkv = qkv.reshape(b, n, 3, h, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / math.sqrt(head_dim)
        # Attention logits computed in float32 for stability.
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q * scale, k)
        if mask is not None:
            attn_logits = jnp.where(mask == 0, -1e9, attn_logits)
        attn = nn.softmax(attn_logits, axis=-1)
        attn = nn.Dropout(rate=self.attn_dropout)(attn, deterministic=deterministic)

        values = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        values = jnp.transpose(values, (0, 2, 1, 3)).reshape(b, n, d)
        values = nn.Dense(features=d, use_bias=False, name="proj")(values)
        values = nn.Dropout(rate=self.proj_dropout)(values, deterministic=deterministic)
        return values
