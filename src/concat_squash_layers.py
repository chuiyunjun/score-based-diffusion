from typing import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


def _unsqueeze_left(x, ndim):
    index = tuple(jnp.newaxis for _ in range(ndim - len(x.shape)))
    index = (...,) + index
    return x[index]
        


class ConcatSquash(eqx.Module):
    hyper_gate: eqx.nn.Linear
    hyper_bias: jnp.ndarray

    def __init__(self, out_size, *, key):
        key1, key2 = jr.split(key, 2)
        self.hyper_gate = eqx.nn.Linear(1, out_size, key=key1)
        self.hyper_bias = jr.uniform(key2, (out_size,), minval=-1, maxval=1)

    def __call__(self, t, x):
        t = t[jnp.newaxis]
        hyper_gate = jnn.sigmoid(self.hyper_gate(t))
        hyper_bias = t * self.hyper_bias
        hyper_gate = _unsqueeze_left(hyper_gate, x.ndim)
        hyper_bias = _unsqueeze_left(hyper_bias, x.ndim)
        return x * hyper_gate + hyper_bias


class ConcatSquashLinear(eqx.Module):
    linear: eqx.nn.Linear
    concat_squash: ConcatSquash

    def __init__(self, *args, key, **kwargs):
        lkey, cskey = jr.split(key, 2)
        self.linear = eqx.nn.Linear(*args, **kwargs, key=lkey)
        self.concat_squash = ConcatSquash(self.linear.out_features, key=cskey)

    def __call__(self, t, x):
        return self.concat_squash(t, self.linear(x))
        

class ConcatSquashConv2d(eqx.Module):
    conv: eqx.nn.Conv2d
    concat_squash: ConcatSquash

    def __init__(self, *args, key, **kwargs):
        ckey, cskey = jr.split(key, 2)
        self.conv = eqx.nn.Conv2d(*args, **kwargs, key=ckey)
        self.concat_squash = ConcatSquash(self.conv.out_channels, key=cskey)

    def __call__(self, t, x):
        return self.concat_squash(t, self.conv(x))


class ConcatSquashConvTranspose2d(eqx.Module):
    conv_transpose: eqx.nn.ConvTranspose2d
    concat_squash: ConcatSquash

    def __init__(self, *args, key, **kwargs):
        ckey, cskey = jr.split(key, 2)
        self.conv_transpose = eqx.nn.ConvTranspose2d(*args, **kwargs, key=ckey)
        self.concat_squash = ConcatSquash(self.conv_transpose.out_channels, key=cskey)

    def __call__(self, t, x):
        return self.concat_squash(t, self.conv_transpose(x))


class ConcatSquashMLP(eqx.Module):
    layers: list[ConcatSquashLinear]
    activation: Callable
    final_activation: Callable
    in_size: int = eqx.static_field()
    out_size: int = eqx.static_field()
    width_size: int = eqx.static_field()
    depth: int = eqx.static_field()

    def __init__(self, in_size, out_size, width_size, depth, activation=jnn.relu, final_activation=lambda x: x, *, key, **kwargs):
        super().__init__(**kwargs)
        keys = jr.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(ConcatSquashLinear(in_size, out_size, key=keys[0]))
        else:
            layers.append(ConcatSquashLinear(in_size, width_size, key=keys[0]))
            for i in range(depth - 1):
                layers.append(ConcatSquashLinear(width_size, width_size, key=keys[i + 1]))
            layers.append(ConcatSquashLinear(width_size, out_size, key=keys[-1]))
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, t, x):
        for layer in self.layers[:-1]:
            x = layer(t, x)
            x = self.activation(x)
        x = self.layers[-1](t, x)
        x = self.final_activation(x)
        return x
