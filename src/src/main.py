import functools as ft
import math
import os
import pathlib
import time
from typing import Any, Callable, Optional, Sequence, Type, TypedDict

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
import torch
from torch_fidelity import calculate_metrics
import wandb

from .custom_types import Array, Batch, Shape
from .datasets import cifar10, mnist, toy
from .langevin_utils import langevin_std
from .nn import MLP, ConcatSquashMixer2d, Mixer2d, SinusoidalPosEmb, UNet
from .sde import VESDE, VPSDE, AbstractSDE, CriticalLangevinSDE, GenericVPSDE
from .divergence import (
    divergence_exact,
    divergence_hutchinson,
    divergence_hutch_plusplus
)
from .utils import deserialise, maybe_restart_wandb, serialise, CifarSamplesDataset
from .wrappers import LangevinWrapper, SongWrapper


def batch_loss_fn(
        model: Callable[[float, Array[Shape]], Array[Shape]],
        sde: AbstractSDE,
        data: Array[Batch, Shape],
        dt0: float,
        t1: float,
        key: jr.PRNGKey,
        mode: str,
        train_kwargs: dict[str, Any],
) -> float:
    batch_size = data.shape[0]
    if train_kwargs is not None:
        match train_kwargs["divergence_name"]:
            case "exact":
                divergence_fn = divergence_exact
            case "hutchinson":
                divergence_fn = divergence_hutchinson
            case "hutch++":
                divergence_fn = divergence_hutch_plusplus
            case _:
                divergence_fn = None
    if mode == "cnf":
        key = jr.split(key, batch_size)
        loss_fn = ft.partial(
            sde.single_cnf_loss_fn, model=model, dt0=train_kwargs["dt0"],
            t1=t1,
            divergence_fn=divergence_fn, trace_num=train_kwargs["trace_num"]
        )
        loss_fn = jax.vmap(loss_fn)
        return jnp.mean(loss_fn(data=data, key=key))
    else:
        tkey, losskey = jr.split(key)
        losskey = jr.split(losskey, batch_size)
        # Low-discrepancy sampling over t to reduce variance
        t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
        t = t + (t1 / batch_size) * jnp.arange(batch_size)
        if mode == "dsm":
            loss_fn = ft.partial(sde.single_dsm_loss_fn, model=model)
        elif mode == 'ism':
            loss_fn = ft.partial(
                sde.single_ism_loss_fn, model=model,
                divergence_fn=divergence_fn
            )
        else:
            raise ValueError(f"Invalid {mode=}.")
        loss_fn = jax.vmap(loss_fn)
        return jnp.mean(loss_fn(data=data, t=t, key=losskey))


@eqx.filter_jit
def make_step(
        model: Callable[[float, Array[Shape]], Array[Shape]],
        sde: AbstractSDE,
        data: Array[Batch, Shape],
        dt0: float,
        t1: float,
        key: jr.PRNGKey,
        opt_state,
        opt_update,
        mode: str,
        train_kwargs: dict[str, Any],
):
    key, subkey = jr.split(key)
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    value, grads = loss_fn(model, sde, data, dt0, t1, subkey, mode,
                           train_kwargs)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return value, model, key, opt_state


def generate_samples(sample_key, sample_fn, model, sample_size, dataset):
    sample_keys = jr.split(sample_key, sample_size)
    sample = sample_fn(model, sample_keys)
    sample = dataset.mean + dataset.std * sample
    sample = jnp.clip(sample, dataset.min, dataset.max)

    return sample


def plot_samples(sample_key, sample_fn, model, sample_size, dataset,
                 dataset_name, outdir, use_wandb, caption):
    sample = generate_samples(sample_key, sample_fn, model, sample_size,
                              dataset)
    match dataset_name:
        case "toy":
            plt.scatter(sample[:, 0], sample[:, 1])
        case "mnist":
            n = int(math.sqrt(sample_size))
            sample = einops.rearrange(sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)",
                                      n1=n, n2=n)
            plt.imshow(sample, cmap="Greys")
            plt.axis("off")
            plt.tight_layout()
        case "cifar10":
            n = int(math.sqrt(sample_size))
            sample = einops.rearrange(sample,
                                      "(n1 n2) c h w -> (n1 h) (n2 w) c", n1=n,
                                      n2=n, c=3)
            plt.imshow(sample)
            plt.axis("off")
            plt.tight_layout()
        case _:
            raise ValueError
    if use_wandb:
        wandb.log({"samples": wandb.Image(plt, caption=caption)})
    else:
        plt.savefig(outdir / 'samples.png')
        print(f"Output saved to {outdir}")
    plt.close()


class _TrainingRegime(TypedDict):
    mode: str
    num_steps: int
    optim_name: str
    lr: float
    lr_scale: float
    batch_size: int
    train_kwargs: dict[str, Any]


def main(
        # SDE hyperparameters
        sde_name: str,
        t1: float,
        weight_func: str,
        # Neural network hyperparameters
        model_name: str,
        model_kwargs: dict,
        use_wrapper: bool,
        # Optimisation hyperparameters
        dataset_name: str,
        print_every: int,
        plot_every: int,
        save_every: int,
        calc_metrics_every: int,
        training_regimes: Sequence[_TrainingRegime],
        # Sampling hyperparameters
        dt0: float,
        sample_size: int,
        metrics_sample_size: int,
        # Misc
        seed: int,
        restart: bool,
        wandb_config: dict,
        parallel: bool,
        # Not specified in configs
        outdir: pathlib.Path,
        config_dict: dict

):
    key = jr.PRNGKey(seed)
    model_key, train_key, loader_key, sample_key = jr.split(key, 4)
    match dataset_name:
        case "cifar10":
            dataset_fn = cifar10
        case "mnist":
            dataset_fn = mnist
        case "toy":
            dataset_fn = toy
        case _:
            raise ValueError(f"Invalid dataset {dataset_name}")
    dataset = dataset_fn(loader_key)

    match model_name:
        case "mlp":
            model_cls = MLP
        case "mixer2d":
            model_cls = Mixer2d
        case "concatsquashmixer2d":
            model_cls = ConcatSquashMixer2d
        case "unet":
            model_cls = UNet
        case _:
            raise ValueError(f"Invalid model {model_name}")
    model = model_cls(**model_kwargs, data_shape=dataset.data_shape, t1=t1,
                      langevin=sde_name == "langevin", key=model_key)

    # TODO: train these quantities, e.g. for variance minimisation?
    match sde_name:
        case "langevin":
            friction = 1.0
            mass = 0.25
            initial_velocity = 0.04
            int_beta = lambda t: 4.0 * t

            def weight(t):
                _, _, _, _, _, _, Lvvt2 = langevin_std(friction, mass,
                                                       initial_velocity,
                                                       int_beta, t)
                return Lvvt2

            sde = CriticalLangevinSDE(friction, mass, initial_velocity,
                                      int_beta, weight)
            if use_wrapper:
                model = LangevinWrapper(model, friction, mass,
                                        initial_velocity, int_beta)
        case "ve":
            # So Yang uses `sigma` defined by:
            # ```
            # σ_min = 0.01
            # σ_max = 50
            # sigma = lambda t: σ_min * (σ_max / σ_min) ** (t / t1)
            # ```
            # But this produces bad results for me on toy datasets?
            # Instead I'm using this, so that the SDE is just dx(t) = dw(t) i.e.
            # pure Brownian motion.
            # (With +1e-6 just to avoid numerical issues when taking a jvp of sigma**2.)
            sigma = lambda t: jnp.sqrt(t / t1 + 1e-6)
            scope = locals() | globals()
            weight = eval('lambda t: ' + weight_func, scope)
            sde = VESDE(weight, sigma)
            if use_wrapper:
                model = SongWrapper(model, sde)
        case "vp":
            int_beta = lambda t: t
            scope = locals() | globals()
            weight = eval('lambda t: ' + weight_func, scope)
            sde = VPSDE(weight, int_beta)
            if use_wrapper:
                model = SongWrapper(model, sde)
        case "generic":
            int_beta = lambda t: t
            scope = locals() | globals()
            weight = eval('lambda t: ' + weight_func, scope)
            sde = GenericVPSDE(int_beta, weight)
            if use_wrapper:
                model = SongWrapper(model, sde)

        case _:
            raise ValueError

    training_index_file = outdir / "training_index.checkpoint.npz"
    checkpoint_file = outdir / "model.checkpoint.npz"
    if checkpoint_file.exists():
        assert training_index_file.exists()
        if restart:
            init_tr = jnp.array(0)
            init_step = jnp.array(0)
            opt_state = None
            reloaded = False
            print(f'restarting {checkpoint_file}...')
        else:
            (init_tr,) = deserialise(training_index_file, jnp.array(0))
            _tr = training_regimes[init_tr.item()]
            _opt = optax.chain(optax.adabelief(_tr["lr"]),
                               optax.scale_by_schedule(
                                   optax.exponential_decay(1, _tr["num_steps"],
                                                           _tr["lr_scale"])))
            _opt_state = _opt.init(eqx.filter(model, eqx.is_inexact_array))
            init_step, model, opt_state = deserialise(checkpoint_file,
                                                      jnp.array(0), model,
                                                      _opt_state)
            del _tr, _opt, _opt_state
            reloaded = True
            print(f'resuming {checkpoint_file} from step {init_step}...')
    else:
        assert not training_index_file.exists()
        init_tr = jnp.array(0)
        init_step = jnp.array(0)
        opt_state = None
        reloaded = False
        print(f'no checkpoint found in {checkpoint_file}, starting new run...')

    use_wandb = config_dict['wandb_config']['use_wandb']
    if use_wandb:
        run_id = maybe_restart_wandb(reloaded, outdir, config_dict,
                                     wb_dir=outdir)
    else:
        run_id = ""

    @eqx.filter_jit
    @ft.partial(jax.vmap, in_axes=(None, 0))
    def sample_fn(model, sample_keys):
        return sde.single_sample_fn(model, dataset.data_shape, dt0, t1,
                                    sample_keys)

    for i_tr in range(init_tr, len(training_regimes)):
        tr = training_regimes[i_tr]
        i_tr = jnp.array(i_tr)

        total_value = 0
        total_size = 0

        # Scales down to `lr_scale` x initial learning rate over the course of training.
        opt = optax.chain(
            getattr(optax, tr["optim_name"])(tr["lr"]),
            optax.scale_by_schedule(
                optax.exponential_decay(1, tr["num_steps"], tr["lr_scale"])
            )
        )
        if opt_state is None:
            opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

        for step, data in zip(jnp.arange(init_step, tr["num_steps"]),
                              dataset.train_dataloader.loop(tr["batch_size"])):
            value, model, train_key, opt_state = make_step(model, sde, data,
                                                           dt0, t1, train_key,
                                                           opt_state,
                                                           opt.update,
                                                           tr["mode"],
                                                           tr["train_kwargs"])
            total_value += value.item()
            total_size += 1
            if step % print_every == 0 or step == tr["num_steps"] - 1:
                print(f"Step={step} Loss={total_value / total_size}")
                if use_wandb:
                    scalars = {'loss': total_value / total_size}
                    wandb.log(scalars, step=step)
                total_value = total_size = 0
            if step % plot_every == 0 or step == tr["num_steps"] - 1:
                plot_samples(sample_key, sample_fn, model, sample_size,
                             dataset, dataset_name, outdir, use_wandb, run_id)
            if step % save_every == 0 or step == tr["num_steps"] - 1:
                serialise(training_index_file, i_tr)
                serialise(checkpoint_file, step + 1, model, opt_state)
            match dataset_name:
                case "cifar10":
                    # step+1 used, as metrics take a long time to calculate,
                    # and wish to check code is working before calculating them
                    if (step + 1) % calc_metrics_every == 0 or \
                            step == tr["num_steps"] - 1:
                        key, sample_key = jr.split(key)
                        samples = []
                        # UNet requires too much memory to generate 50000
                        # samples at once, split the generation into separate
                        # batches
                        for i in range(50000//metrics_sample_size):
                            print(f'Generating image '
                                  f'{(i + 1) * metrics_sample_size} '
                                  f'for metrics calculation')
                            key, sample_key = jr.split(key)
                            new_samples = \
                                generate_samples(sample_key, sample_fn,
                                                 model, metrics_sample_size,
                                                 dataset)
                            samples.append(new_samples)

                        samples = jnp.concatenate(samples, axis=0)
                        samples = torch.as_tensor(samples.to_py())
                        samples = CifarSamplesDataset(samples)
                        metrics_dict = calculate_metrics(
                            input1=samples,
                            input2="cifar10-train",
                            cuda=True, isc=True, fid=True, kid=True,
                            verbose=False)
                        is_ = metrics_dict['inception_score_mean']
                        fid = metrics_dict['frechet_inception_distance']
                        kid = metrics_dict['kernel_inception_distance_mean']

                        print(f'Inception score: {is_}')
                        print(f'Frechet inception distance: {fid}')
                        print(f'Kernel inception distance: {kid}')
                        with open(outdir / "IS.txt", "a") as f:
                            f.write(str(is_) + '\n')
                        with open(outdir / "FID.txt", "a") as f:
                            f.write(str(fid) + '\n')
                        with open(outdir / "FID.txt", "a") as f:
                            f.write(str(kid) + '\n')
                        if use_wandb:
                            wandb.log(metrics_dict)

    plot_samples(sample_key, sample_fn, model, sample_size, dataset,
                 dataset_name, outdir, use_wandb, run_id)
    breakpoint()  # Do any manual investigation/debugging if we wish.
