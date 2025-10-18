# main.py
import jax
import jax.numpy as jnp
from models.dit import DiT

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = DiT(image_size=64, patch_size=8, dimension=256, depth=4, num_heads=8)
    dummy_x = jax.random.normal(key, (2, 64, 64, 3))
    dummy_t = jnp.array([10, 20])
    params = model.init(key, dummy_x, dummy_t)
    out = model.apply(params, dummy_x, dummy_t)
    print(out.shape)  # (2, 64, 64, 3)
