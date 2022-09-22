import abc
import math
from typing import Callable, Optional, Type

import equinox as eqx
import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr

from .custom_types import Array, Shape
from .langevin_utils import langevin_mean, langevin_std


def _normal_logp(y):
    return -0.5 * (y.size * math.log(2 * math.pi) + jnp.sum(y**2))


class AbstractSDE(eqx.Module):
    @abc.abstractmethod
    def single_dsm_loss_fn(
        self,
        model: Callable[..., Array[Shape]],
        data: Array[Shape],
        t: float,
        key: jr.PRNGKey
    ) -> float:
        pass

    @abc.abstractmethod
    def single_ism_loss_fn(
        self,
        model: Callable[[float, Array[Shape]], Array[Shape]],
        data: Array[Shape],
        t: float,
        divergence_fn: Callable[[Callable[[float, Array[Shape]], Array[Shape]],
                                 Array[Shape], int, jr.PRNGKey],
                                tuple[Array[Shape], float]],
        divergence_num: int,
        key: jr.PRNGKey
    ) -> float:
        pass

    @abc.abstractmethod
    def single_cnf_loss_fn(
        self,
        model: Callable[..., Array[Shape]],
        data: Array[Shape],
        dt0: float,
        t1: float,
        divergence_fn: Callable[[Callable[[float, Array[Shape]], Array[Shape]],
                                 Array[Shape], int, jr.PRNGKey],
                                tuple[Array[Shape], float]],
        divergence_num: int,
        key: jr.PRNGKey,
    ) -> float:
        pass

    @abc.abstractmethod
    def single_sample_fn(
        self,
        model: Callable[..., Array[Shape]],
        data_shape: Shape,
        dt0: float,
        t1: float,
        key: jr.PRNGKey,
    ) -> Array[Shape]:
        pass


class AbstractSongSDE(eqx.Module):
    weight: Callable[[float], float]

    @abc.abstractmethod
    def mean(self, t: float, data: Array[Shape]) -> Array[Shape]:
        pass

    @abc.abstractmethod
    def var(self, t: float) -> float:
        pass

    @abc.abstractmethod
    def vector_field(
        self,
        t: float,
        y: Array[Shape],
        model: Callable[[float, Array[Shape]], Array[Shape]],
    ) -> Array[Shape]:
        pass

    def single_dsm_loss_fn(
        self,
        model: Callable[[float, Array[Shape]], Array[Shape]],
        data: Array[Shape],
        t: float,
        key: jr.PRNGKey
    ) -> float:
        t = jnp.maximum(t, 1e-4)
        mean = self.mean(t, data)
        var = self.var(t)
        std = jnp.sqrt(var)
        noise = jr.normal(key, data.shape)
        yt = mean + std * noise
        pred = model(t, yt)
        return self.weight(t) * jnp.mean((pred + noise / std) ** 2)

    def single_ism_loss_fn(
        self,
        model: Callable[[float, Array[Shape]], Array[Shape]],
        data: Array[Shape],
        t: float,
        divergence_fn: Callable[[Callable[[float, Array[Shape]], Array[Shape]],
                                 Array[Shape], int, jr.PRNGKey],
                                tuple[Array[Shape], float]],
        divergence_num: int,
        key: jr.PRNGKey
    ) -> float:
        t = jnp.maximum(t, 1e-4)
        mean = self.mean(t, data)
        var = self.var(t)
        std = jnp.sqrt(var)
        noise = jr.normal(key, data.shape)
        yt = mean + std * noise
        fn = lambda _y: model(t, _y)
        f, div_f = divergence_fn(fn, yt, divergence_num, key=key)
        loss = jnp.sum(f**2) / 2 + div_f
        return self.weight(t) * loss

    def single_cnf_loss_fn(
        self,
        model: Callable[[float, Array[Shape]], Array[Shape]],
        data: Array[Shape],
        dt0: float,
        t1: float,
        divergence_fn: Callable[[Callable[[float, Array[Shape]], Array[Shape]],
                                 Array[Shape], int, jr.PRNGKey],
                                tuple[Array[Shape], float]],
        divergence_num: int,
        key: jr.PRNGKey,
    ) -> float:
        def vf(t, y, args):  # `self.vector_field` passed in via `args`.
            vector_field, args = args
            fn = lambda _y: vector_field(t, _y, args)
            return divergence_fn(fn, y, divergence_num, key)
        term = dfx.ODETerm(vf)
        args = (self.vector_field, model)
        solver = dfx.Heun()
        t0 = 0
        sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, (data, 0.0), args=args, adjoint=dfx.BacksolveAdjoint())
        (y1,), (delta_logp,) = sol.ys
        return -delta_logp - _normal_logp(y1)

    def single_sample_fn(
        self,
        model: Callable[[float, Array[Shape]], Array[Shape]],
        data_shape: Shape,
        dt0: float,
        t1: float,
        key: jr.PRNGKey,
    ) -> Array[Shape]:
        term = dfx.ODETerm(self.vector_field)
        solver = dfx.Tsit5()
        t0 = 0
        y1 = jr.normal(key, data_shape)
        # reverse time, solve from t1 to t0
        sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1, args=model, adjoint=dfx.NoAdjoint())
        (y0,) = sol.ys
        return y0


class VESDE(AbstractSongSDE):
    sigma: Callable[[float], float]

    def mean(self, t: float, data: Array[Shape]) -> Array[Shape]:
        return data

    def var(self, t: float) -> float:
        return self.sigma(t)**2 - self.sigma(0)**2

    def vector_field(
        self,
        t: float,
        y: Array[Shape],
        model: Callable[[float, Array[Shape]], Array[Shape]],
    ) -> Array[Shape]:
        _, dσ2dt = jax.jvp(lambda _t: self.sigma(_t)**2, (t,), (jnp.ones_like(t),))
        return -0.5 * dσ2dt * model(t, y)


class VPSDE(AbstractSongSDE):
    int_beta: Callable[[float], float]

    def mean(self, t: float, data: Array[Shape]) -> Array[Shape]:
        return data * jnp.exp(-0.5 * self.int_beta(t))

    def var(self, t: float) -> float:
        return 1 - jnp.exp(-self.int_beta(t))

    def vector_field(
        self,
        t: float,
        y: Array[Shape],
        model: Callable[[float, Array[Shape]], Array[Shape]],
    ) -> Array[Shape]:
        _, beta = jax.jvp(self.int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * (y + model(t, y))


class CriticalLangevinSDE(AbstractSDE):
    friction: float
    mass: float
    initial_velocity: float
    int_beta: Callable[[float], float]
    weight: Callable[[float], float]

    def __init__(self, friction, mass, initial_velocity, int_beta, weight):
        if friction ** 2 != 4 * mass:
            raise ValueError("Not critically damped.")

        self.friction = friction
        self.mass = mass
        self.initial_velocity = initial_velocity
        self.int_beta = int_beta
        self.weight = weight

    # Technically H(ybrid)S(core)M(atching), not D(enoising)S(core)M(atching).
    def single_dsm_loss_fn(
        self,
        model: Callable[[float, Array[Shape], Array[Shape]], Array[Shape]],
        data: Array[Shape],
        t: float,
        key: jr.PRNGKey
    ) -> float:
        μyt, μvt = langevin_mean(self.friction, self.int_beta, t, data)
        _, _, _, Lyyt, Lyvt, Lvvt, _ = langevin_std(self.friction, self.mass, self.initial_velocity, self.int_beta, t)
        ynoise, vnoise = jr.normal(key, (2,) + data.shape)
        yt = μyt + Lyyt * ynoise
        vt = μvt + Lyvt * ynoise + Lvvt * vnoise
        pred = model(t, yt, vt)
        return self.weight(t) * jnp.mean((pred + vnoise / Lvvt) ** 2)

    def single_ism_loss_fn(
            self,
            model: Callable[[float, Array[Shape]], Array[Shape]],
            data: Array[Shape],
            t: float,
            divergence_fn: Callable[
                [Callable[[float, Array[Shape]], Array[Shape]],
                 Array[Shape], int, jr.PRNGKey],
                tuple[Array[Shape], float]],
            divergence_num: int,
            key: jr.PRNGKey
    ) -> float:
        raise NotImplementedError

    def single_cnf_loss_fn(
        self,
        model: Callable[..., Array[Shape]],
        data: Array[Shape],
        dt0: float,
        t1: float,
        divergence_fn: Callable[[Callable[[float, Array[Shape]], Array[Shape]],
                                 Array[Shape], int, jr.PRNGKey],
                                tuple[Array[Shape], float]],
        divergence_num: int,
        key: jr.PRNGKey,
    ) -> float:
        raise NotImplementedError

    def single_sample_fn(
        self,
        model: Callable[[float, Array[Shape], Array[Shape]], Array[Shape]],
        data_shape: Shape,
        dt0: float,
        t1: float,
        key: jr.PRNGKey,
    ) -> Array[Shape]:

        def vector_field(t, y__v, args):
            y, v = y__v
            dy = v / self.mass
            dv_drift = -y - v * (self.friction / self.mass)
            dv_diffusion = -self.friction * model(t, y, v)
            _, beta = jax.jvp(self.int_beta, (t,), (jnp.ones_like(t),))
            return beta * dy, beta * (dv_drift + dv_diffusion)

        ykey, vkey = jr.split(key)
        term = dfx.ODETerm(vector_field)
        solver = dfx.Tsit5()
        t0 = 0
        y1 = jr.normal(ykey, data_shape)
        v1 = jnp.sqrt(self.mass) * jr.normal(vkey, data_shape)
        # reverse time, solve from t1 to t0
        sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, (y1, v1), adjoint=dfx.NoAdjoint())
        (y0,), _ = sol.ys
        return y0


class GenericVPSDE(AbstractSDE):

    int_beta: Callable[[float], float]
    weight: Callable[[float], float]

    def __init__(self, int_beta, weight):
        self.int_beta = int_beta
        self.weight = weight

    def mean(self, t: float, data: Array[Shape]) -> Array[Shape]:
        return data * jnp.exp(-0.5 * self.int_beta(t))

    def var(self, t: float) -> float:
        return 1 - jnp.exp(-self.int_beta(t))

    def single_dsm_loss_fn(
            self,
            model: Callable[[float, Array[Shape]], Array[Shape]],
            data: Array[Shape],
            t: float,
            key: jr.PRNGKey
    ) -> float:
        raise ValueError("Implicit score matching is required for non-affine "
                         "diffusion.")

    def single_ism_loss_fn(
        self,
        model: Callable[[float, Array[Shape]], Array[Shape]],
        data: Array[Shape],
        t: float,
        divergence_fn: Callable[[Callable[[float, Array[Shape]], Array[Shape]],
                                 Array[Shape], int, jr.PRNGKey],
                                tuple[Array[Shape], float]],
        divergence_num: int,
        key: jr.PRNGKey
    ) -> float:
        t1 = jnp.maximum(t, 1e-4)
        bkey, dkey = jr.split(key)
        _, beta = jax.jvp(self.int_beta, (t,), (jnp.ones_like(t),))

        def drift(t, y, args):
            return -0.5 * beta * y

        def diffusion(t, y, args):
            return jnp.broadcast_to(beta ** (1/2), data.shape)

        brownian_motion = dfx.VirtualBrownianTree(0, t1, tol=1e-3,
                                                  shape=data.shape, key=bkey)
        terms = dfx.MultiTerm(dfx.ODETerm(drift),
                              dfx.WeaklyDiagonalControlTerm(diffusion,
                                                            brownian_motion))
        solver = dfx.Euler()
        (yt,) = dfx.diffeqsolve(terms, solver, 0, t1, dt0=0.1, y0=data).ys

        fn = lambda _y: model(t, _y)
        f, div_f = divergence_fn(fn, yt, divergence_num, key=dkey)
        loss = jnp.sum(f**2) / 2 + div_f
        return self.weight(t1) * loss

    def single_cnf_loss_fn(
        self,
        model: Callable[..., Array[Shape]],
        data: Array[Shape],
        dt0: float,
        t1: float,
        divergence_fn: Callable[[Callable[[float, Array[Shape]], Array[Shape]],
                                 Array[Shape], int, jr.PRNGKey],
                                tuple[Array[Shape], float]],
        divergence_num: int,
        key: jr.PRNGKey,
    ) -> float:
        raise NotImplementedError

    def vector_field(
            self,
            t: float,
            y: Array[Shape],
            model: Callable[[float, Array[Shape]], Array[Shape]],
    ) -> Array[Shape]:
        _, beta = jax.jvp(self.int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * (y + model(t, y))

