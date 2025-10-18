import flax.linen as nn
import jax.numpy as jnp
from .mlp import MLP
from .attention import MultiHeadSelfAttenton

class TransformerBlock(nn.Module):
    dimension: int
    num_heads: int
    mlp_ratio: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        # 1. LayerNorm
        residual = x
        x = nn.LayerNorm()(x)
        # 2. Attention
        attention_output, _ = MultiHeadSelfAttenton(self.dimension, self.num_heads, self.dropout)(x, deterministic=deterministic)
        # 3. Residual Connection
        x = attention_output + residual

        #4. LayerNorm + MLP
        residual = x
        x = nn.LayerNorm()(x)
        x = MLP(self.dimension * self.mlp_ratio, self.dimension, self.dropout)(x, deterministic=deterministic)
        # 5. Residual Connection
        x = x + residual

        return x

class DiTEncoder(nn.Module):
    depth: int
    dimension: int
    num_heads: int
    mlp_ratio: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        for _ in range(self.depth):
            x = TransformerBlock(self.dimension, self.num_heads, self.mlp_ratio, self.dropout)(x, deterministic=deterministic)
        return x
