import math

import jax.numpy as jnp
from flax import linen as nn


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a small MLP."""

    dimension: int

    @nn.compact
    def __call__(self, timesteps):
        timesteps = timesteps.astype(jnp.float32)
        half = self.dimension // 2
        if half == 0:
            raise ValueError("dimension must be at least 2 for sinusoidal embeddings")

        exponent = -math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / (half - 1 + 1e-6)
        freqs = jnp.exp(exponent)
        angles = timesteps[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        emb = nn.Dense(self.dimension * 4, name="fc1")(emb)
        emb = nn.silu(emb)
        emb = nn.Dense(self.dimension, name="fc2")(emb)
        return emb
