# Score-based diffusions

## Usage

### Examples

Specify SDE, dataset, model, and training regime:
```
python entry_point.py sde=vp dataset=mnist model=concatsquashmixer2d training=dsm_cnf
```

Override individual parameters:
```
python entry_point.py sde=vp dataset=mnist model=concatsquashmixer2d training=dsm use_wrapper=False print_every=100
```

Override Hydra run parameters:
```
python entry_point.py sde=vp dataset=mnist model=concatsquashmixer2d hydra.run.dir=<checkpoint_path>
```

As normal, set the environment variable `CUDA_VISIBLE_DEVICES` to constrain the GPUs used, e.g. `CUDA_VISIBLE_DEVICES=0 python entry_point.py ....` to use just the first GPU.

### Arguments

To run an experiment do:
```
CUDA_VISIBLE_DEVICES=<id> python entry_point.py sde=<sde> dataset=<dataset> model=<model> training=<training> <any other kwargs>
```
where `<id>` is the ID of the GPU to use, or an empty string if using the CPU;

where `<sde>` can be any config file in [configs/sde](./configs/sde), e.g.:
- `ve`
- `vp`
- `langevin`

where `<dataset>` can be any config file in [configs/dataset](./configs/dataset), e.g.:
- `toy`
- `mnist`
- `cifar10`

where `<model>` can be any config file in [configs/model](./configs/model), e.g.:
- `mlp`
- `mixer2d`
- `concatsquashmixer2d`

where `<training>` can be any config file in [configs/training](./configs/training), e.g.:
- `dsm`  (train via denoising score matching)
- `dsm_cnf`  (train via denoising score matching, then train as a continuous normalising flow)

where `<any other kwargs>` are those corresponding to [src/main.py::main()](./src/main.py), and can be used to override the default value of any other kwarg. For example `num_steps=2_000_000`. In addition it may contain:
- `restart=true` to overwrite a previously saved run instead of resuming from it.
- `parallel=true` to paralellise over all available default devices. (Using `jax.pmap`.)

### Notes

- A run can be interrupted and resumed; the model training is saved periodically.
  - You can pass `restart=true` as a command line argument to overwrite the previously saved run instead of resuming from it.
- Each run should be deterministic. If you want to do multiple runs for the same settings then you will need to explicitly pass `seed=<some integer>`.
  - Caveat: if a run is interrupted and resumed then this will not be true.
- Outputs are saved in `/outputs/<description of config>/`. The precise input parameters used to generate it are found in `/outputs/<description of config>/.hydra/`.
- The hyperparameters for each dataset/model combination and each dataset/training combination must be specified explicitly; not every combination necessarily exists by default. These are specified in [configs/dataset_model](./configs/dataset_model) and [configs/dataset_training](./configs/dataset_training).

## Dependencies

- At least Python 3.10;
- [JAX](https://github.com/google/jax) for autodifferentiation;
- [Equinox](https://github.com/patrick-kidger/equinox) for neural networks, model building etc;
- [Diffrax](https://github.com/patrick-kidger/diffrax) for differential equations;
- [Optax](https://github.com/deepmind/optax) for optimisers;
- [PyTorch (torch, torchvision)](https://github.com/pytorch/pytorch/) for datasets and dataloaders;
- [Torch-fidelity](https://github.com/toshas/torch-fidelity) for metric calculation;
- [Einops](https://github.com/arogozhnikov/einops/) for tensor rearrangement operations;
- [Hydra](https://github.com/facebookresearch/hydra/) for command line orchestration;
- Optionally, [Weights and Biases](https://github.com/wandb/client) for recording results.

```
conda create -n myenv python=3.10
conda activate myenv
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install jax==0.3.4 equinox==0.3.2 diffrax==0.0.5 optax==0.1.1 einops==0.4.1 hydra-core==1.1.1 wandb==0.12.12 torch-fidelity==0.3.0
```

Install jaxlib (note that the command you need to run may differ depending on your version of CUDA/cuDNN, so this is for reference):
```
pip install --upgrade jaxlib==0.3.2+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

