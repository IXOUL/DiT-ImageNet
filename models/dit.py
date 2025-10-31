import flax.linen as nn
import jax
import jax.numpy as jnp

from .patch_embed import PatchEmbed
from .transformer import AdaLayerNormZero, DiTEncoder
from utils.time_embedding import TimeEmbedding


class LabelEmbedding(nn.Module):
    """Class embedding with optional classifier-free guidance dropout."""

    num_classes: int
    dimension: int
    dropout_prob: float = 0.0

    @nn.compact
    def __call__(self, labels, deterministic=True):
        if labels is None:
            raise ValueError("labels must be provided when using label conditioning")

        embed = nn.Embed(
            num_embeddings=self.num_classes + 1,
            features=self.dimension,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="embedding",
        )
        if deterministic or self.dropout_prob <= 0.0:
            conditioned_labels = labels
        else:
            keep_prob = 1.0 - self.dropout_prob
            rng = self.make_rng("dropout")
            # Replace a subset of labels with the null class for classifier-free guidance.
            keep_mask = jax.random.bernoulli(rng, keep_prob, labels.shape)
            null_label = jnp.full_like(labels, self.num_classes)
            conditioned_labels = jnp.where(keep_mask, labels, null_label)
        return embed(conditioned_labels)


class FinalLayer(nn.Module):
    """Final adaptive LayerNorm and projection back to image space."""

    dimension: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x, cond, patches_hw):
        x = AdaLayerNormZero(self.dimension, name="ada_ln")(x, cond)
        x = nn.Dense(
            features=self.patch_size * self.patch_size * self.out_channels,
            name="proj",
        )(x)
        b, n, _ = x.shape
        patches_h, patches_w = patches_hw
        x = x.reshape(
            b,
            patches_h,
            patches_w,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = x.reshape(
            b,
            patches_h * self.patch_size,
            patches_w * self.patch_size,
            self.out_channels,
        )
        return x


class DiT(nn.Module):
    """Diffusion Transformer backbone following DiT-S architecture."""

    image_size: int
    patch_size: int
    dimension: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    channels: int = 3
    dropout: float = 0.0
    attn_dropout: float = 0.0
    num_classes: int | None = None
    class_dropout_prob: float = 0.0

    @nn.compact
    def __call__(self, x, timesteps, class_labels=None, mask=None, deterministic=True):
        if x.shape[1] != self.image_size or x.shape[2] != self.image_size:
            raise ValueError(
                f"Expected square inputs of spatial size {self.image_size}, "
                f"got H={x.shape[1]}, W={x.shape[2]}"
            )

        tokens, patches_hw = PatchEmbed(
            image_size=self.image_size,
            patch_size=self.patch_size,
            dimension=self.dimension,
            in_channels=self.channels,
        )(x)

        t_emb = TimeEmbedding(self.dimension, name="time_embedding")(timesteps)
        if self.num_classes is not None and class_labels is not None:
            y_emb = LabelEmbedding(
                num_classes=self.num_classes,
                dimension=self.dimension,
                dropout_prob=self.class_dropout_prob,
                name="label_embedding",
            )(class_labels, deterministic=deterministic)
        else:
            y_emb = jnp.zeros_like(t_emb)

        # Shared conditioning vector injected into every Transformer block.
        cond = t_emb + y_emb

        tokens = DiTEncoder(
            depth=self.depth,
            dimension=self.dimension,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            name="transformer",
        )(tokens, cond, mask=mask, deterministic=deterministic)

        output = FinalLayer(
            dimension=self.dimension,
            patch_size=self.patch_size,
            out_channels=self.channels,
            name="final_layer",
        )(tokens, cond, patches_hw)
        return output
