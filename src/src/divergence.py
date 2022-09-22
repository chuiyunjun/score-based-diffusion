import functools as ft
import math
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from .custom_types import Array, Shape, Num


def divergence_exact(fn: Callable[[Array[Shape]], Array[Shape]],
                     y: Array[Shape], num: int, key: jax.random.PRNGKey) -> \
                                                    tuple[Array[Shape], float]:
    del num, key
    y_shape = y.shape
    y_flat = y.reshape(-1)
    fn_flat = lambda _y_flat: fn(_y_flat.reshape(y_shape)).reshape(-1)
    jvp = lambda x: ft.partial(jax.jvp, fn_flat, (y_flat,))((x,))
    eye = jnp.eye(y.size)
    f, jac = jax.vmap(jvp, out_axes=(None, 0))(eye)
    div_f = jnp.trace(jac)
    return f, div_f


def divergence_hutchinson(fn: Callable[[Array[Shape]], Array[Shape]],
                          y: Array[Shape], num: int,
                          key: "jax.random.PRNGKey") -> \
                                                    tuple[Array[Shape], float]:
    eps = jr.normal(key, (num,) + y.shape)
    jvp = lambda x: ft.partial(jax.jvp, fn, (y,))((x,))
    f, jac_eps = jax.vmap(jvp, out_axes=(None, 0))(eps)
    div_f = jnp.sum(jnp.mean(eps * jac_eps, axis=0))
    return f, div_f


def divergence_hutch_plusplus(fn: Callable[[Array[Shape]], Array[Shape]],
                              y: Array[Shape], num: int,
                              key: "jax.random.PRNGKey") -> \
                                                    tuple[Array[Shape], float]:
    if num % 3 != 0:
        raise ValueError("`num` must be a multiple of 3 when using Hutch++")
    frac = num // 3

    data_size = math.prod(y.shape)
    s = jr.rademacher(key, (data_size, frac), dtype=y.dtype)
    g = jr.rademacher(key, (data_size, frac), dtype=y.dtype)
    y_shape = y.shape
    y_flat = y.reshape(-1)
    fn_flat = lambda _y_flat: fn(_y_flat.reshape(y_shape)).reshape(-1)
    jvp = lambda x: ft.partial(jax.jvp, fn_flat, (y_flat,))((x,))
    jac_fn = jax.vmap(jvp, in_axes=-1, out_axes=(None, -1))
    f, jac_s = jac_fn(s)
    q, _ = jnp.linalg.qr(jac_s)
    k = g - q @ q.T @ g
    _, jac_q = jac_fn(q)
    _, jac_k = jac_fn(k)
    term1 = jnp.sum(q * jac_q)
    term2 = jnp.sum(k * jac_k)
    div_f = term1 + 3 * term2 / num
    return f, div_f
