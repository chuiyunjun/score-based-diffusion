from typing import Callable

import equinox as eqx
import jax.numpy as jnp

from .langevin_utils import langevin_std
from .sde import AbstractSongSDE


class SongWrapper(eqx.Module):
    model: eqx.Module
    sde: AbstractSongSDE

    def __call__(self, t, y):
        _t = t + 0.1
        var = self.sde.var(_t)
        std = jnp.sqrt(var)
        return self.model(t, y) / std


class LangevinWrapper(eqx.Module):
    model: eqx.Module
    friction: float
    mass: float
    initial_velocity: float
    int_beta: Callable[[float], float]

    def __call__(self, t, y, v):
        _, _, Σvvt, _, _, Lvvt, _ = langevin_std(self.friction, self.mass, self.initial_velocity, self.int_beta, t)
        return -v / Σvvt + self.model(t, y, v) / Lvvt
