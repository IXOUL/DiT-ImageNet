import jax.numpy as jnp
import flax.linen as nn

class PatchEmbed(nn.Module):
    patch_size: int
    dimension: int

    @nn.compact
    def __call__(self, x):
        # x: shape (B, H, W, C)
        B, H, W, C = x.shape
        patches_h = H // self.patch_size
        patches_w = W // self.patch_size
        total_num_patches = patches_h * patches_w
        
        # 1. reshape the images into patch grids
        # -> (B, patches_h, patch, patches_w, patch, C)
        x = x.reshape(B, patches_h, self.patch_size, patches_w, self.patch_size, C)

        # 2. transpose to bring patch dimensions together
        # -> (B, patches_h, patches_w, patch, patch, C)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))

        # 3. flatten patch pixels -> (B, total_num_patches, patch_size*patch_size*C)
        x = x.reshape(B, total_num_patches, self.patch_size * self.patch_size * C)

        # 4. linear projection to "dimension" for Transformer
        x = nn.Dense(self.dimension)(x)

        # 5. learned positional embedding (1, N, D)
        pos = self.param("pos", nn.initializers.normal(0.02), (1, total_num_patches, self.dimension))
        x = x + pos

        return x, (patches_h, patches_w)
