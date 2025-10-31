import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """Two-layer feed-forward network used inside the Transformer blocks."""

    hidden_dimension: int
    output_dimension: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(self.hidden_dimension, name="fc1")(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dropout(rate=self.dropout, name="drop1")(x, deterministic=deterministic)
        x = nn.Dense(self.output_dimension, name="fc2")(x)
        x = nn.Dropout(rate=self.dropout, name="drop2")(x, deterministic=deterministic)
        return x
