import functools as ft
import os
from copy import deepcopy

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
import optax
import wandb
from omegaconf import OmegaConf


def flatten(data, ref=None):
    if ref is None:
        ref = data
    data = jax.tree_leaves(data)
    ref = jax.tree_leaves(ref)
    flat_data = tuple(d for d, r in zip(data, ref) if eqx.is_array(r))
    return flat_data


def serialise(checkpoint_dir, *state):
    data = flatten(state)
    data = tuple(d.to_py() for d in data)
    jnp.savez(checkpoint_dir, *data)


def deserialise(checkpoint_dir, *state):
    _flatten = ft.partial(flatten, ref=state) 
    with np.load(checkpoint_dir, allow_pickle=False) as data:
        flat_state = flatten(state)
        flat_data = tuple(jnp.array(d) for d in data.values())
        assert len(flat_state) == len(flat_data)
        assert all(s.shape == d.shape for s, d in zip(flat_state, flat_data))
        assert all(s.dtype == d.dtype for s, d in zip(flat_state, flat_data))
        return eqx.tree_at(_flatten, state, flat_data)


def maybe_restart_wandb(resuming, outdir, config_dict, wb_dir, resume='must'):
    """Detect the run id if it exists and resume, otherwise write the run id to file. 
        Returns the (maybe) updated state and saves out wandb run settings.
    """
    # if the run_id was previously saved, resume from there
    wandb_dir = outdir / 'wandb.npy'
    run_id = ''
    if resuming and wandb_dir.exists():
        assert resuming, f'Something went wrong, there was no wandb to resume!'
        wandb_dict = np.load(wandb_dir, allow_pickle=True)[()]
        run_id = wandb_dict['run_id']
        wandb.init(id=run_id, project=wandb_dict['project_name'], name=wandb_dict['exp_name'], config=config_dict, entity=wandb_dict['entity'], dir=wb_dir, resume=resume)
        print(f"successfully resumed run: {run_id}")
    else:
        print( f'requested resume {resuming} / there was no wandb to resume...')
        # if the run_id doesn't exist, then create a new run, and write the run id to the config
        wandb_dict = config_dict['wandb_config']
        new_run = wandb.init(project=wandb_dict['project_name'], name=wandb_dict['exp_name'], config=config_dict, entity=wandb_dict['entity'], dir=wb_dir)
        run_id = new_run.id
        wandb_dict['run_id'] = run_id
        np.save(wandb_dir, wandb_dict, allow_pickle=True)
        print(f'initialized new wandb run_id {run_id}, saved config to {wandb_dir}')

    return run_id


def _test_run(load):
    key = jr.PRNGKey(0)
    model = eqx.nn.MLP(2, 2, 2, 2, key=key)
    optim = optax.adam(3e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    ckpt_dir = 'test_ckpt.npy'
    
    if load:
        init_step = jnp.array(0)
        step, model, opt_state = deserialise(init_step, model, opt_state, ckpt_dir=ckpt_dir)
        print(f'loaded model, opt_state, step {step}')

    serialise(jnp.array(10), model, opt_state, ckpt_dir=ckpt_dir)


class CifarSamplesDataset(torch.utils.data.Dataset):

    def __init__(self, samples):
        samples *= 255
        self.samples = samples.type(torch.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    _test_run(False)
    _test_run(True)
