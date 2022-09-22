import math

from einops import rearrange
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def _mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


class SinusoidalPosEmb(eqx.Module):
    emb: float

    def __init__(self, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x):
        emb = x * self.emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class LinearTimeSelfAttention(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    heads: int
    to_qkv: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d

    def __init__(self, dim, key, heads=4, dim_head=32,):
        keys = jax.random.split(key, 2)
        self.layer_norm = eqx.nn.LayerNorm(None, elementwise_affine=False)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = eqx.nn.Conv2d(dim, hidden_dim * 3, 1, key=keys[0])
        self.to_out = eqx.nn.Conv2d(hidden_dim, dim, 1, key=keys[1])

    def __call__(self, x):
        c, h, w = x.shape
        x = self.layer_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, '(qkv heads c) h w -> qkv heads c (h w)', heads=self.heads, qkv=3)
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum('hdn,hen->hde', k, v)
        out = jnp.einsum('hde,hdn->hen', context, q)
        out = rearrange(out, 'heads c (h w) -> (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResnetBlock(eqx.Module):
    dim: int
    dim_out: int
    dropout_rate: float
    time_emb_dim: int
    mlp_layers: list[eqx.nn.Linear]
    block1_layers: list[eqx.nn.LayerNorm | eqx.nn.Conv2d]
    block2_layers: list[eqx.nn.LayerNorm | eqx.nn.Conv2d]
    res_conv: eqx.nn.conv.Conv2d

    def __init__(self, dim, dim_out, *, time_emb_dim, key, dropout_rate=0.):
        keys = jax.random.split(key, 3)
        self.dim = dim
        self.dim_out = dim_out
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim

        self.mlp_layers = [_mish,
                           eqx.nn.Linear(time_emb_dim, dim_out, key=keys[0])]

        # Norm -> non-linearity -> conv format follows
        # https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L55
        self.block1_layers = [eqx.nn.LayerNorm(None, elementwise_affine=False),
                              _mish,
                              eqx.nn.Conv2d(dim, dim_out, 3, padding=1, key=keys[1])]
        self.block2_layers = [eqx.nn.LayerNorm(None, elementwise_affine=False),
                              _mish,
                              eqx.nn.Conv2d(dim_out, dim_out, 3, padding=1, key=keys[2])]
        self.res_conv = eqx.nn.Conv2d(dim, dim_out, 1, key=keys[5])

    def __call__(self, x, t):
        h = x
        for layer in self.block1_layers:
            h = layer(h)
        for layer in self.mlp_layers:
            t = layer(t)
        h = h + t[..., None, None]
        for layer in self.block2_layers:
            h = layer(h)
        return h + self.res_conv(x)


class Residual(eqx.Module):
    fn: LinearTimeSelfAttention

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# Unet implementation based on TorchSDEs example of a Unet in PyTorch
# https://github.com/google-research/torchsde/blob/master/examples/unet.py
# Currently LayerNorm used instead of GroupNorm
class UNet(eqx.Module):
    t1: float = eqx.static_field()
    time_pos_emb: SinusoidalPosEmb
    mlp: eqx.nn.MLP
    first_conv: eqx.nn.Conv2d
    down_res_blocks: list[list[list[ResnetBlock]]]
    down_attn_blocks: list[list[list[Residual]]]
    down_spatial_blocks: list[eqx.nn.Conv2d | None]
    mid_block1: ResnetBlock
    mid_attn: Residual
    mid_block2: ResnetBlock
    ups_res_blocks: list[list[list[ResnetBlock]]]
    ups_attn_blocks: list[list[list[Residual]]]
    ups_spatial_blocks: list[None | eqx.nn.ConvTranspose2d]
    final_conv_layers: list[eqx.nn.LayerNorm | eqx.nn.Conv2d]

    def __init__(self, data_shape: tuple[int], dim_mults: list[int],
                 hidden_size: int, heads: int, dim_head: int,
                 dropout_rate: float, num_res_blocks: int,
                 attn_resolutions: list[int], t1: float, langevin: bool, *,
                 key):

        keys = jax.random.split(key, 9)
        del key

        in_channels, in_height, in_width = data_shape

        if langevin:
            in_channels = 2 * in_channels

        dims = [hidden_size] + [hidden_size * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.t1 = t1
        self.time_pos_emb = SinusoidalPosEmb(hidden_size)
        self.mlp = eqx.nn.MLP(hidden_size, hidden_size, 4 * hidden_size, 1, activation=_mish, key=keys[1])

        self.first_conv = eqx.nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1, key=keys[2])

        h, w = in_height, in_width
        self.down_res_blocks = []
        self.down_attn_blocks = []
        self.down_spatial_blocks = []
        num_keys = len(in_out) * (num_res_blocks + len(attn_resolutions) * num_res_blocks)
        keys_resblock = jr.split(keys[3], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(in_out):
            res_blocks = [
                [ResnetBlock(
                    dim=dim_in,
                    dim_out=dim_out,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    key=keys_resblock[i]
                )]
            ]
            i += 1
            for _ in range(num_res_blocks - 1):
                res_blocks.append([
                ResnetBlock(
                    dim=dim_out,
                    dim_out=dim_out,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    key=keys_resblock[i]
                )])
                i += 1
            self.down_res_blocks.append(res_blocks)

            attn_blocks = []
            if h in attn_resolutions and w in attn_resolutions:
                for _ in range(num_res_blocks):
                    attn_blocks.append(
                        [Residual(LinearTimeSelfAttention(dim_out, heads=heads, dim_head=dim_head, key=keys_resblock[i]))]
                    )
                    i += 1
            self.down_attn_blocks.append(attn_blocks)

            if ind < (len(in_out) - 1):
                spatial_blocks = eqx.nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2, padding=1, key=keys_resblock[i])
                h, w = h // 2, w // 2
            else:
                spatial_blocks = None
            self.down_spatial_blocks.append(spatial_blocks)

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            dim=mid_dim,
            dim_out=mid_dim,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            key=keys[4]
        )

        self.mid_attn = Residual(LinearTimeSelfAttention(mid_dim, heads=heads, dim_head=dim_head, key=keys[5]))
        self.mid_block2 = ResnetBlock(
            dim=mid_dim,
            dim_out=mid_dim,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            key=keys[6]
        )

        self.ups_res_blocks = []
        self.ups_attn_blocks = []
        self.ups_spatial_blocks = []
        num_keys = len(in_out) * (
                    num_res_blocks + len(attn_resolutions) * num_res_blocks)
        keys_resblock = jr.split(keys[7], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            res_blocks = []
            for _ in range(num_res_blocks):
                res_blocks.append([
                    ResnetBlock(
                        dim=dim_out * 2,
                        dim_out=dim_out,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        key=keys_resblock[i]
                    )
                ])
                i += 1
            res_blocks.append([
                ResnetBlock(
                    dim=dim_out + dim_in,
                    dim_out=dim_in,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    key=keys_resblock[i]
                )
            ])
            i += 1
            self.ups_res_blocks.append(res_blocks)

            attn_blocks = []
            if h in attn_resolutions and w in attn_resolutions:
                for _ in range(num_res_blocks):
                    attn_blocks.append(
                        [Residual(LinearTimeSelfAttention(dim_out, heads=heads, dim_head=dim_head, key=keys_resblock[i]))]
                    )
                    i += 1
                attn_blocks.append(
                    [Residual(LinearTimeSelfAttention(dim_in, heads=heads, dim_head=dim_head, key=keys_resblock[i]))]
                )
                i += 1
            self.ups_attn_blocks.append(attn_blocks)

            if ind < (len(in_out) - 1):
                spatial_blocks = eqx.nn.ConvTranspose2d(dim_in, dim_in, kernel_size=4, stride=2, padding=1, key=keys_resblock[i])
                h, w = h * 2, w * 2
            else:
                spatial_blocks = None
            self.ups_spatial_blocks.append(spatial_blocks)

        self.final_conv_layers = [
            eqx.nn.LayerNorm(None, elementwise_affine=False),
            _mish,
            eqx.nn.Conv2d(hidden_size, in_channels, 1, key=keys[9])
        ]

    def __call__(self, t, y, v=None):
        # Normalise t between -1 and 1
        t = 2 * (1 - (1 - t / self.t1) ** 4) - 1
        t = self.time_pos_emb(t)
        t = self.mlp(t)
        if v is None:
            in_ = y
        else:
            in_ = jnp.concatenate([y, v])
        hs = [self.first_conv(in_)]
        for (i, res_blocks, attn_blocks, spatial_block) in \
                zip(range(len(self.down_res_blocks)), self.down_res_blocks, self.down_attn_blocks,
                    self.down_spatial_blocks, strict=True):
            if len(attn_blocks) > 0:
                for res_block, attn_block in zip(res_blocks, attn_blocks):
                    for layer in res_block:
                        h = layer(hs[-1], t)
                    for layer in attn_block:
                        h = layer(h)
                    hs.append(h)
            else:
                for res_block in res_blocks:
                    for layer in res_block:
                        h = layer(hs[-1], t)
                    hs.append(h)
            if spatial_block is not None:
                hs.append(spatial_block(hs[-1]))

        h = hs[-1]
        h = self.mid_block1(h, t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t)

        for (i, res_blocks, attn_blocks, spatial_block) in \
                zip(range(len(self.ups_res_blocks)), self.ups_res_blocks, self.ups_attn_blocks,
                    self.ups_spatial_blocks, strict=True):
            if len(attn_blocks) > 0:
                for res_block, attn_block in zip(res_blocks, attn_blocks):
                    for layer in res_block:
                        h = layer(jnp.concatenate((h, hs.pop()), axis=0), t)
                    for layer in attn_block:
                        h = layer(h)
            else:
                for res_block in res_blocks:
                    for layer in res_block:
                        h = layer(jnp.concatenate((h, hs.pop()), axis=0), t)
            if spatial_block is not None:
                h = spatial_block(h)

        for layer in self.final_conv_layers:
            h = layer(h)
        return h
