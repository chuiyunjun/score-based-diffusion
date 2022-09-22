import math

import einops
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from .concat_squash_layers import ConcatSquashConv2d, ConcatSquashConvTranspose2d
from .concat_squash_layers import ConcatSquashMLP as _ConcatSquashMLP
from .unet import SinusoidalPosEmb, UNet


class MLP(eqx.Module):
    mlp: eqx.nn.MLP
    data_shape: tuple[int] = eqx.static_field()
    t1: float = eqx.static_field()

    def __init__(self, data_shape, width_size, depth, t1, langevin, *, key):
        data_size = math.prod(data_shape)
        if langevin:
            in_size = 2 * data_size + 1
        else:
            in_size = data_size + 1
        self.mlp = eqx.nn.MLP(in_size, data_size, width_size, depth, key=key)
        self.data_shape = data_shape
        self.t1 = t1

    def __call__(self, t, y, v=None):
        t = 2 * (1 - (1 - t / self.t1) ** 4) - 1
        y = y.reshape(-1)
        if v is None:
            in_ = jnp.append(y, t)
        else:
            v = v.reshape(-1)
            in_ = jnp.concatenate([t[None], y, v])
        out = self.mlp(in_)
        return out.reshape(self.data_shape)


class ConcatSquashMLP(eqx.Module):
    mlp: _ConcatSquashMLP
    data_shape: tuple[int] = eqx.static_field()
    t1: float = eqx.static_field()

    def __init__(self, data_shape, width_size, depth, t1, langevin, *, key):
        data_size = math.prod(data_shape)
        if langevin:
            in_size = 2 * data_size + 1
        else:
            in_size = data_size + 1
        self.mlp = _ConcatSquashMLP(in_size, data_size, width_size, depth, activation=jnn.tanh, key=key)
        self.data_shape = data_shape
        self.t1 = t1

    def __call__(self, t, y, v=None):
        t = 2 * (1 - (1 - t / self.t1) ** 4) - 1
        y = y.reshape(-1)
        if v is None:
            in_ = jnp.append(y, t)
        else:
            v = v.reshape(-1)
            in_ = jnp.concatenate([t[None], y, v])
        out = self.mlp(t, in_)
        return out.reshape(self.data_shape)


class _MixerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    patch_mixer: eqx.nn.MLP
    norm2: eqx.nn.LayerNorm
    hidden_mixer: eqx.nn.MLP

    def __init__(self, num_patches, hidden_size, mix_patch_size, mix_hidden_size, *, key):
        tkey, ckey = jr.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.patch_mixer = eqx.nn.MLP(num_patches, num_patches, mix_patch_size, depth=1, key=tkey)
        self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))
        self.hidden_mixer = eqx.nn.MLP(hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey)

    def __call__(self, y):
        y = y + jax.vmap(self.patch_mixer)(self.norm1(y))
        y = einops.rearrange(y, 'c p -> p c')
        y = y + jax.vmap(self.hidden_mixer)(self.norm2(y))
        y = einops.rearrange(y, 'p c -> c p')
        return y


class Mixer2d(eqx.Module):
    t1: float = eqx.static_field()
    conv_in: eqx.nn.Conv2d
    blocks: list[_MixerBlock]
    norm: eqx.nn.LayerNorm
    conv_out: eqx.nn.ConvTranspose2d
    
    def __init__(self, data_shape, patch_size, hidden_size, mix_patch_size, mix_hidden_size, num_blocks, t1, langevin, *, key):
        input_size, height, width = data_shape
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)
        inkey, *bkeys, outkey = jr.split(key, 2 + num_blocks)

        if langevin:
            in_size = 2 * input_size + 1
        else:
            in_size = input_size + 1

        self.t1 = t1
        self.conv_in = eqx.nn.Conv2d(in_size, hidden_size, patch_size, stride=patch_size, key=inkey)
        self.blocks = [_MixerBlock(num_patches, hidden_size, mix_patch_size, mix_hidden_size, key=bkey) for bkey in bkeys]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.conv_out = eqx.nn.ConvTranspose2d(hidden_size, input_size, patch_size, stride=patch_size, key=outkey)

    def __call__(self, t, y, v=None):
        t = 2 * (1 - (1 - t / self.t1) ** 4) - 1
        _, height, width = y.shape
        t_repeat = einops.repeat(t, "-> 1 h w", h=height, w=width)
        if v is None:
            y = jnp.concatenate([t_repeat, y])
        else:
            y = jnp.concatenate([t_repeat, y, v])
        y = self.conv_in(y)
        _, patch_height, patch_width = y.shape
        y = einops.rearrange(y, 'c h w -> c (h w)')
        for block in self.blocks:
            y = block(y)
        y = self.norm(y)
        y = einops.rearrange(y, 'c (h w) -> c h w', h=patch_height, w=patch_width)
        return self.conv_out(y)


class _ConcatSquashMixerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    patch_mixer: _ConcatSquashMLP
    norm2: eqx.nn.LayerNorm
    hidden_mixer: _ConcatSquashMLP

    def __init__(self, num_patches, hidden_size, mix_patch_size, mix_hidden_size, *, key):
        tkey, ckey = jr.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.patch_mixer = _ConcatSquashMLP(num_patches, num_patches, mix_patch_size, depth=1, key=tkey)
        self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))
        self.hidden_mixer = _ConcatSquashMLP(hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey)

    def __call__(self, t, y):
        y = y + jax.vmap(lambda x: self.patch_mixer(t, x))(self.norm1(y))
        y = einops.rearrange(y, 'c p -> p c')
        y = y + jax.vmap(lambda x: self.hidden_mixer(t, x))(self.norm2(y))
        y = einops.rearrange(y, 'p c -> c p')
        return y


class ConcatSquashMixer2d(eqx.Module):
    t1: float = eqx.static_field()
    conv_in: ConcatSquashConv2d
    blocks: list[_ConcatSquashMixerBlock]
    norm: eqx.nn.LayerNorm
    conv_out: ConcatSquashConvTranspose2d
    
    def __init__(self, data_shape, patch_size, hidden_size, mix_patch_size, mix_hidden_size, num_blocks, t1, langevin, *, key):
        input_size, height, width = data_shape
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)
        inkey, *bkeys, outkey = jr.split(key, 2 + num_blocks)

        if langevin:
            in_size = 2 * input_size + 1
        else:
            in_size = input_size + 1

        self.t1 = t1
        self.conv_in = ConcatSquashConv2d(in_size, hidden_size, patch_size, stride=patch_size, key=inkey)
        self.blocks = [_ConcatSquashMixerBlock(num_patches, hidden_size, mix_patch_size, mix_hidden_size, key=bkey) for bkey in bkeys]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.conv_out = ConcatSquashConvTranspose2d(hidden_size, input_size, patch_size, stride=patch_size, key=outkey)

    def __call__(self, t, y, v=None):
        t = 2 * (1 - (1 - t / self.t1) ** 4) - 1
        _, height, width = y.shape
        t_repeat = einops.repeat(t, "-> 1 h w", h=height, w=width)
        if v is None:
            y = jnp.concatenate([t_repeat, y])
        else:
            y = jnp.concatenate([t_repeat, y, v])
        y = self.conv_in(t, y)
        _, patch_height, patch_width = y.shape
        y = einops.rearrange(y, 'c h w -> c (h w)')
        for block in self.blocks:
            y = block(t, y)
        y = self.norm(y)
        y = einops.rearrange(y, 'c (h w) -> c h w', h=patch_height, w=patch_width)
        return self.conv_out(t, y)
