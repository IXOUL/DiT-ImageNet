import jax.numpy as jnp
import math
from flax import linen as nn

class TimeEmbedding(nn.Module):
    dimension: int

    @nn.compact
    def __call__(self, t):
        half = self.dimension // 2
        freqs = jnp.exp(-math.log(10000) * jnp.arange(half) / half)
        angles = t[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        emb = nn.Dense(self.dimension * 4)(emb)
        emb = nn.silu(emb)
        return nn.Dense(self.dimension)(emb)
