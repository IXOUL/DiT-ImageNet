import flax.linen as nn
import jax.numpy as jnp


class PatchEmbed(nn.Module):
    """Project images to a sequence of patch tokens with learned position encodings."""

    image_size: int
    patch_size: int
    dimension: int
    in_channels: int = 3

    def setup(self):
        patches_per_dim = self.image_size // self.patch_size
        self.num_patches = patches_per_dim * patches_per_dim

    @nn.compact
    def __call__(self, x):
        proj = nn.Conv(
            features=self.dimension,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
            name="projection",
        )(x)

        b, h, w, d = proj.shape
        if h * w != self.num_patches:
            raise ValueError(
                f"Unexpected token count {h * w}; expected {self.num_patches} based on image size"
            )
        # Flatten the 2-D grid of patches into a 1-D token sequence.
        tokens = proj.reshape(b, h * w, d)
        pos = self.param(
            "positional_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_patches, self.dimension),
        )
        tokens = tokens + pos
        return tokens, (h, w)
