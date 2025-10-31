import flax.linen as nn
import jax.numpy as jnp

from .attention import MultiHeadSelfAttention
from .mlp import MLP


class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale and shift are predicted from a conditioning vector."""

    dimension: int

    @nn.compact
    def __call__(self, x, cond):
        shift_scale = nn.Dense(self.dimension * 2, name="modulation")(
            nn.silu(cond)
        )
        shift, scale = jnp.split(shift_scale, 2, axis=-1)
        normed = nn.LayerNorm(epsilon=1e-6, name="norm")(x)
        normed = normed * (1 + scale[:, None, :]) + shift[:, None, :]
        return normed


class AdaLayerNormZero(nn.Module):
    """Adaptive LayerNorm with zero-initialised modulation for stable decoding."""

    dimension: int

    @nn.compact
    def __call__(self, x, cond):
        shift_scale = nn.Dense(
            self.dimension * 2,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="modulation",
        )(nn.silu(cond))
        shift, scale = jnp.split(shift_scale, 2, axis=-1)
        normed = nn.LayerNorm(epsilon=1e-6, name="norm")(x)
        normed = normed * (1 + scale[:, None, :]) + shift[:, None, :]
        return normed


class DiTBlock(nn.Module):
    """Transformer block with adaptive LayerNorm modulation used in DiT."""

    dimension: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, cond, mask=None, deterministic=True):
        # Dense produces the six modulation components expected by DiT blocks.
        modulation = nn.Dense(self.dimension * 6, name="modulation")(
            nn.silu(cond)
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation, 6, axis=-1
        )

        h = nn.LayerNorm(epsilon=1e-6, name="norm1")(x)
        h = h * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        h = MultiHeadSelfAttention(
            dimension=self.dimension,
            num_heads=self.num_heads,
            attn_dropout=self.attn_dropout,
            proj_dropout=self.dropout,
            name="attention",
        )(h, mask=mask, deterministic=deterministic)
        # tanh keeps residual gates in [-1, 1] so blocks start near identity.
        gate_msa = jnp.tanh(gate_msa)[:, None, :]
        x = x + gate_msa * h

        h = nn.LayerNorm(epsilon=1e-6, name="norm2")(x)
        h = h * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        h = MLP(
            hidden_dimension=int(self.dimension * self.mlp_ratio),
            output_dimension=self.dimension,
            dropout=self.dropout,
            name="mlp",
        )(h, deterministic=deterministic)
        gate_mlp = jnp.tanh(gate_mlp)[:, None, :]
        x = x + gate_mlp * h
        return x


class DiTEncoder(nn.Module):
    depth: int
    dimension: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, cond, mask=None, deterministic=True):
        for i in range(self.depth):
            # Conditioning vector is shared across all blocks as in the paper.
            x = DiTBlock(
                dimension=self.dimension,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                name=f"blocks_{i}",
            )(x, cond, mask=mask, deterministic=deterministic)
        return x
