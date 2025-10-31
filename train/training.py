from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints

from models.dit import DiT
from train.diffusion import (
    DiffusionConfig,
    DiffusionSchedule,
    compute_v_target,
    make_diffusion_schedule,
    q_sample,
)
from train.state import TrainState


ArrayDict = Dict[str, jnp.ndarray]
BatchIterator = Iterator[ArrayDict]


@dataclass
class TrainingConfig:
    """Hyperparameters for optimiser and outer training loop."""

    seed: int = 0
    total_steps: int = 250_000
    learning_rate: float = 1e-4
    warmup_steps: int = 5_000
    lr_end_value: float = 1e-6
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip_norm: float = 1.0
    log_every: int = 100
    checkpoint_every: int = 5_000
    workdir: str = "./checkpoints"
    use_ema: bool = True
    ema_decay: float = 0.9999
    resume: bool = True


def _create_learning_rate_schedule(config: TrainingConfig):
    warmup_steps = min(config.warmup_steps, config.total_steps)
    decay_steps = max(config.total_steps, warmup_steps + 1)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=config.lr_end_value,
    )


def _create_optimizer(config: TrainingConfig):
    lr_schedule = _create_learning_rate_schedule(config)
    transforms = []
    if config.grad_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(config.grad_clip_norm))
    transforms.append(
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=config.weight_decay,
        )
    )
    return optax.chain(*transforms), lr_schedule


def _prepare_batch(batch: Mapping[str, Any]) -> ArrayDict:
    images = jnp.asarray(batch["images"], dtype=jnp.float32)
    data: ArrayDict = {"images": images}
    if "labels" in batch and batch["labels"] is not None:
        data["labels"] = jnp.asarray(batch["labels"], dtype=jnp.int32)
    return data


def _diffusion_loss(
    apply_fn: Callable[..., jnp.ndarray],
    params: Any,
    batch: ArrayDict,
    rng: jax.Array,
    dropout_rng: jax.Array,
    schedule: DiffusionSchedule,
    diffusion_config: DiffusionConfig,
) -> Dict[str, jnp.ndarray]:
    rng, timestep_rng, noise_rng = jax.random.split(rng, 3)
    images = batch["images"]
    labels = batch.get("labels", None)

    timesteps = jax.random.randint(
        timestep_rng,
        (images.shape[0],),
        minval=0,
        maxval=diffusion_config.timesteps,
    )
    noise = jax.random.normal(noise_rng, images.shape)
    noisy = q_sample(schedule, images, timesteps, noise)

    model_out = apply_fn(
        {"params": params},
        noisy,
        timesteps,
        class_labels=labels,
        deterministic=False,
        rngs={"dropout": dropout_rng},
    )

    if diffusion_config.prediction_type == "eps":
        target = noise
    elif diffusion_config.prediction_type == "v":
        target = compute_v_target(schedule, images, noise, timesteps)
    elif diffusion_config.prediction_type == "x0":
        target = images
    else:
        raise ValueError(
            f"Unsupported prediction_type: {diffusion_config.prediction_type}"
        )

    mse = jnp.mean(jnp.square(model_out - target))
    return {
        "loss": mse,
        "mse": mse,
        "t_mean": jnp.mean(timesteps.astype(jnp.float32)),
    }


def _train_step(
    state: TrainState,
    batch: ArrayDict,
    rng: jax.Array,
    schedule: DiffusionSchedule,
    diffusion_config: DiffusionConfig,
):
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        metrics = _diffusion_loss(
            state.apply_fn,
            params,
            batch,
            rng,
            dropout_rng,
            schedule,
            diffusion_config,
        )
        return metrics["loss"], metrics

    grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
    grad_norm = optax.global_norm(grads)
    new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    metrics = {**metrics, "grad_norm": grad_norm}
    return new_state, metrics


def train(
    model_config: Mapping[str, Any],
    training_config: TrainingConfig,
    diffusion_config: DiffusionConfig,
    train_iterator: BatchIterator,
    eval_fn: Optional[Callable[[TrainState], Mapping[str, float]]] = None,
) -> TrainState:
    """Train the DiT model using the provided data iterator."""

    if training_config.total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    workdir = os.path.abspath(training_config.workdir)
    os.makedirs(workdir, exist_ok=True)

    optimizer, lr_schedule = _create_optimizer(training_config)
    schedule = make_diffusion_schedule(diffusion_config)

    rng = jax.random.PRNGKey(training_config.seed)
    rng, params_rng = jax.random.split(rng)
    params_rng, dropout_init = jax.random.split(params_rng)

    first_batch = _prepare_batch(next(train_iterator))
    init_timesteps = jnp.zeros(first_batch["images"].shape[0], dtype=jnp.int32)
    model = DiT(**model_config)

    variables = model.init(
        {"params": params_rng, "dropout": dropout_init},
        first_batch["images"],
        init_timesteps,
        class_labels=first_batch.get("labels", None),
        deterministic=False,
    )

    dropout_init, state_dropout = jax.random.split(dropout_init)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        ema_decay=training_config.ema_decay,
        dropout_rng=state_dropout,
        use_ema=training_config.use_ema,
    )

    if training_config.resume:
        restored_state = checkpoints.restore_checkpoint(
            workdir, target=state
        )
        state = restored_state

    start_step = int(state.step)

    if start_step > 0:
        print(f"Resuming training from step {start_step}")

    if start_step == 0:
        train_iterator = itertools.chain([first_batch], train_iterator)
    train_step_fn = jax.jit(
        lambda st, batch, step_rng: _train_step(
            st, batch, step_rng, schedule, diffusion_config
        )
    )

    for step in range(start_step, training_config.total_steps):
        rng, step_rng = jax.random.split(rng)
        batch = _prepare_batch(next(train_iterator))
        state, metrics = train_step_fn(state, batch, step_rng)

        if (step + 1) % training_config.log_every == 0:
            metrics_host = jax.device_get(metrics)
            metrics_host = {
                k: float(v) if hasattr(v, "__array__") else float(v)
                for k, v in metrics_host.items()
            }
            lr_value = float(lr_schedule(step + 1))
            print(
                f"step={step + 1} loss={metrics_host['loss']:.4f} "
                f"grad_norm={metrics_host['grad_norm']:.4f} "
                f"lr={lr_value:.6f} "
                f"timestep={metrics_host['t_mean']:.1f}"
            )

        if (
            training_config.checkpoint_every
            and (step + 1) % training_config.checkpoint_every == 0
        ):
            checkpoints.save_checkpoint(
                workdir,
                target=state,
                step=step + 1,
                overwrite=True,
            )

        if eval_fn and (step + 1) % training_config.checkpoint_every == 0:
            eval_metrics = eval_fn(state)
            print(
                "Evaluation: "
                + ", ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
            )

    checkpoints.save_checkpoint(
        workdir,
        target=state,
        step=training_config.total_steps,
        overwrite=True,
    )

    return state
