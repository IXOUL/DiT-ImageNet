from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
from flax import struct


@dataclass
class DiffusionConfig:
    """Hyperparameters controlling the forward diffusion process."""

    timesteps: int = 1000
    beta_schedule: str = "cosine"  # {"linear", "cosine"}
    beta_start: float = 1e-4
    beta_end: float = 0.02
    prediction_type: str = "eps"  # {"eps", "v", "x0"}


@struct.dataclass
class DiffusionSchedule:
    """Pre-computed diffusion coefficients stored as JAX arrays."""

    betas: jnp.ndarray
    alphas: jnp.ndarray
    alphas_cumprod: jnp.ndarray
    alphas_cumprod_prev: jnp.ndarray
    sqrt_alphas_cumprod: jnp.ndarray
    sqrt_one_minus_alphas_cumprod: jnp.ndarray
    posterior_variance: jnp.ndarray


def _linear_beta_schedule(
    timesteps: int, beta_start: float, beta_end: float
) -> jnp.ndarray:
    if timesteps <= 1:
        raise ValueError("timesteps must be > 1 for a valid diffusion schedule")
    return jnp.linspace(beta_start, beta_end, timesteps, dtype=jnp.float32)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> jnp.ndarray:
    if timesteps <= 1:
        raise ValueError("timesteps must be > 1 for a valid diffusion schedule")
    steps = jnp.arange(timesteps + 1, dtype=jnp.float32)
    alphas_cumprod = jnp.cos(((steps / timesteps + s) / (1 + s)) * jnp.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1.0 - alphas
    return jnp.clip(betas, a_min=1e-5, a_max=0.999)


def make_diffusion_schedule(config: DiffusionConfig) -> DiffusionSchedule:
    """Pre-compute diffusion constants from config."""

    if config.beta_schedule == "linear":
        betas = _linear_beta_schedule(
            config.timesteps, config.beta_start, config.beta_end
        )
    elif config.beta_schedule == "cosine":
        betas = _cosine_beta_schedule(config.timesteps)
    else:
        raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")

    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = jnp.concatenate(
        [jnp.array([1.0], dtype=jnp.float32), alphas_cumprod[:-1]], axis=0
    )
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    posterior_variance = (
        betas
        * (1.0 - alphas_cumprod_prev)
        / jnp.clip(1.0 - alphas_cumprod, a_min=1e-5)
    )

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        posterior_variance=posterior_variance,
    )


def extract(schedule: jnp.ndarray, timesteps: jnp.ndarray, broadcast_shape: Tuple[int, ...]) -> jnp.ndarray:
    """Gather per-sample coefficients and reshape for broadcasting."""

    out = jnp.take(schedule, timesteps, axis=0)
    reshape_dims = (timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1)
    return out.reshape(reshape_dims)


def q_sample(
    schedule: DiffusionSchedule,
    x_start: jnp.ndarray,
    timesteps: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
    """Sample from q(x_t | x_0) by mixing x_0 with Gaussian noise."""

    sqrt_alphas_cumprod_t = extract(
        schedule.sqrt_alphas_cumprod, timesteps, x_start.shape
    )
    sqrt_one_minus_alphas_cumprod_t = extract(
        schedule.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def compute_v_target(
    schedule: DiffusionSchedule,
    x_start: jnp.ndarray,
    noise: jnp.ndarray,
    timesteps: jnp.ndarray,
) -> jnp.ndarray:
    """Target for v-prediction parameterisation."""

    sqrt_alphas_cumprod_t = extract(
        schedule.sqrt_alphas_cumprod, timesteps, x_start.shape
    )
    sqrt_one_minus_alphas_cumprod_t = extract(
        schedule.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
    )
    return (
        sqrt_alphas_cumprod_t * noise
        - sqrt_one_minus_alphas_cumprod_t * x_start
    )
