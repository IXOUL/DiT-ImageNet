from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state


@struct.dataclass
class TrainState(train_state.TrainState):
    """Extends Flax TrainState with EMA tracking and dropout RNG."""

    ema_params: Any
    ema_decay: float
    dropout_rng: jax.Array
    use_ema: bool = struct.field(pytree_node=False, default=True)

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        tx,
        ema_decay: float,
        dropout_rng: jax.Array,
        use_ema: bool = True,
    ):
        opt_state = tx.init(params)
        return cls(
            step=jnp.array(0, dtype=jnp.int32),
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            ema_params=params,
            ema_decay=ema_decay,
            dropout_rng=dropout_rng,
            use_ema=use_ema,
        )

    def apply_gradients(self, *, grads, dropout_rng: jax.Array):
        state = super().apply_gradients(grads=grads)
        if self.use_ema:
            ema_decay = jax.lax.stop_gradient(
                jnp.array(self.ema_decay, dtype=jnp.float32)
            )
            ema_params = jax.tree_util.tree_map(
                lambda ema, p: ema * ema_decay + (1.0 - ema_decay) * p,
                self.ema_params,
                state.params,
            )
        else:
            ema_params = self.ema_params
        return self.replace(
            step=state.step,
            params=state.params,
            opt_state=state.opt_state,
            ema_params=ema_params,
            dropout_rng=dropout_rng,
        )
