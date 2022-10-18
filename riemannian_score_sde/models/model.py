from dataclasses import dataclass
from typing import Any, NamedTuple, Callable, Sequence
import math

import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp

from score_sde.models import MLP

@dataclass
class ConcatEigenfunctionEmbed(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)

    def __call__(self, x, t):
        t = jnp.array(t)
        if len(t.shape) == 0:
            t = t * jnp.ones(x.shape[:-1])

        if len(t.shape) == len(x.shape) - 1:
            t = jnp.expand_dims(t, axis=-1)

        return self._layer(jnp.concatenate([x, t], axis=-1))
