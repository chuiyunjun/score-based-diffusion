import abc
import dataclasses
import pathlib

import jax.numpy as jnp
import jax.random as jr
import torch
import torchvision


_data_dir = pathlib.Path(__file__).resolve().parent / ".." / "data"


class _DropLabel(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return feature

    def __len__(self):
        return len(self.dataset)


class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dataset, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class _TorchDataLoader(_AbstractDataLoader):
    def __init__(self, dataset, *, key):
        self.dataset = dataset
        min = torch.iinfo(torch.int32).min
        max = torch.iinfo(torch.int32).max
        self.seed = jr.randint(key, (), min, max).item()

    def loop(self, batch_size):
        generator = torch.Generator().manual_seed(self.seed)
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=6, shuffle=True,
            drop_last=True, generator=generator
        )
        while True:
            for tensor in dataloader:
                yield jnp.asarray(tensor)


class _InMemoryDataLoader(_AbstractDataLoader):
    def __init__(self, array, *, key):
        self.array = array
        self.key = key

    def loop(self, batch_size):
        dataset_size = self.array.shape[0]
        if batch_size > dataset_size:
            raise ValueError("Batch size larger than dataset size")

        key = self.key
        indices = jnp.arange(dataset_size)
        while True:
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                yield self.array[batch_perm]
                start = end
                end = start + batch_size


@dataclasses.dataclass
class Dataset:
    train_dataloader: _TorchDataLoader
    test_dataloader: _TorchDataLoader
    data_shape: tuple[int, ...]
    mean: jnp.ndarray
    std: jnp.ndarray
    max: jnp.ndarray
    min: jnp.ndarray


def toy(key):
    key0, key1, trainkey, testkey = jr.split(key, 4)
    data0 = jnp.array([2.0, 3.0]) + 0.2 * jr.normal(key0, (1024, 2))
    data1 = jnp.array([1.0, 4.0]) + 0.2 * jr.normal(key1, (1024, 2))
    train_data = test_data = jnp.concatenate([data0, data1])

    mean = jnp.mean(train_data, axis=0)
    std = jnp.std(train_data, axis=0)
    max = 5
    min = 0
    max = jnp.inf
    min = -jnp.inf
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    data_shape = train_data.shape[1:]

    train_dataloader = _InMemoryDataLoader(train_data, key=trainkey)
    test_dataloader = _InMemoryDataLoader(test_data, key=testkey)
    return Dataset(train_dataloader=train_dataloader,
                   test_dataloader=test_dataloader,
                   data_shape=data_shape,
                   mean=mean,
                   std=std,
                   max=max,
                   min=min)


def mnist(key):
    trainkey, testkey = jr.split(key)
    data_shape = (1, 28, 28)
    mean = 0.1307
    std = 0.3081
    max = 1
    min = 0

    train_dataset = torchvision.datasets.MNIST(_data_dir / "mnist", train=True,
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(_data_dir / "mnist", train=False,
                                              download=True)

    # MNIST is small enough that the whole dataset can be placed in memory, so
    # we can actually use a faster method of data loading.

    # (We do need to handle normalisation ourselves though.)
    train_data = jnp.asarray(train_dataset.data[:, None]) / 255
    test_data = jnp.asarray(test_dataset.data[:, None]) / 255
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    train_dataloader = _InMemoryDataLoader(train_data, key=trainkey)
    test_dataloader = _InMemoryDataLoader(test_data, key=testkey)
    return Dataset(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                   data_shape=data_shape, mean=mean, std=std, max=max, min=min)


def cifar10(key):
    trainkey, testkey = jr.split(key)
    data_shape = (3, 32, 32)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    max = 1
    min = 0

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    train_dataset = torchvision.datasets.CIFAR10(_data_dir / "cifar10", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(_data_dir / "cifar10", train=False, download=True, transform=transform)

    train_dataset = _DropLabel(train_dataset)
    test_dataset = _DropLabel(test_dataset)
    train_dataloader = _TorchDataLoader(train_dataset, key=trainkey)
    test_dataloader = _TorchDataLoader(test_dataset, key=testkey)
    return Dataset(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                   data_shape=data_shape, mean=jnp.array(mean)[:, None, None], std=jnp.array(std)[:, None, None],
                   max=max, min=min)
