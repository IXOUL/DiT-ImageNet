import jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
    hidden_dimension: int
    output_dimension: int
    dropout: float

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(self.hidden_dimension)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dimension)(x)
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        return x