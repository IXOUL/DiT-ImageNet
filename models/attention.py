import math
import jax.numpy as jnp
import flax.linen as nn

class MultiHeadSelfAttenton(nn.Module):
    dimension: int
    num_heads: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        """
            x: (B, N, D)
            returns: out (B, N, D), attention (B, H, N, N)
        """
        
        # x.shape = (batchsize, number of tokens, dimension)
        # every token is a D-dimension vector, including the information of an image patch
        B, N, D = x.shape
        H = self.num_heads
        head_dimension = D // H

        assert D % H == 0, "dimension must be divisible by number of heads"

        # QKV projection
        qkv = nn.Dense(D * 3)(x)  # qkv.shape = (B, N, 3 * D)
        qkv = qkv.reshape(B, N, 3, H, head_dimension)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # qkv.shape = (3, B, H, N, head_dimension)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        # Apply dropout to attention output
        values = nn.Dropout(rate=self.dropout)(values, deterministic=deterministic)

        # Concatenate heads and project back to D
        values = values.transpose(0, 2, 1, 3).reshape(B, N, D)
        output = nn.Dense(D)(values)

        return output, attention

    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        # q, k, v shape = (B, H, N, head_dim)
        d_k = q.shape[-1]

        # 1. compute attention logits = Q @ K^T / sqrt(d_k)
        attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)

        # 2. apply mask if provided (mask 0 = ignore)
        if mask is not None:
            attn_logits = jnp.where(mask == 0, -9e15, attn_logits)

        # 3. softmax to get normalized attention weights
        attention = nn.softmax(attn_logits, axis=-1)

        # 4. weighted sum of values
        values = jnp.matmul(attention, v)

        return values, attention
