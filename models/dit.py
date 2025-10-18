import flax.linen as nn
import jax.numpy as jnp
from .patch_embed import PatchEmbed
from .transformer import DiTEncoder
from utils.time_embedding import TimeEmbedding

class PatchDecoder(nn.Module):
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x, patches_hw):
        """
        Reconstruct image from patch tokens

            x: (B, N, D)
            patches_hw: (patches_h, patches_w)
            returns: (B, H, W, out_channels)
        """
        B, N, D = x.shape
        patches_h, patches_w = patches_hw
        x = nn.Dense(self.patch_size**2 * self.out_channels)(x)
        x = x.reshape(B, patches_h, patches_w, self.patch_size, self.patch_size, self.out_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, patches_h * self.patch_size, patches_w * self.patch_size, self.out_channels)
        return x


class DiT(nn.Module):
    image_size: int
    patch_size: int
    dimension: int
    depth: int
    num_heads: int
    mlp_ratio: int = 4
    channels: int = 3
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, timesteps, mask=None, deterministic=True):
        """
            x: (B, H, W, C)
            timesteps: (B, ) diffusion timesteps
            returns: predicted noise / image of shape (B, H, W, C)
        """

        B, H, W, C = x.shape

        # 1. Patch Embedding, from pixels to patches
        x_tokens, patches_hw = PatchEmbed(self.patch_size, self.dimension)(x)
        
        # 2. Time Embedding
        t_emb = TimeEmbedding(self.dimension)(timesteps)  # (B, D)
        x_tokens = x_tokens + t_emb[:, None, :]  # broadcast to (B, N, D)

        # 3. Encoder
        x_tokens = DiTEncoder(self.depth, self.dimension, self.num_heads, self.mlp_ratio, self.dropout)(x_tokens, mask=mask, deterministic=deterministic)

        # 4. LayerNorm + projecting patches back to pixels
        x_tokens = nn.LayerNorm()(x_tokens)
        x_output = PatchDecoder(self.patch_size, self.channels)(x_tokens, patches_hw)

        return x_output