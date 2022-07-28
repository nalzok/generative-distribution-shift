from typing import Type
from contextlib import redirect_stdout

import jax
import jax.numpy as jnp
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
import click

from models.embedder import Embedder, embedders


@click.command()
@click.option('--embedder_name', type=str, required=True)
@click.option('--embedder_dim', type=int, required=True)
@click.option('--embedder_lr', type=float, required=True)
@click.option('--embedder_epochs', type=int, required=True)
def cli(embedder_name: str, embedder_dim: int, embedder_lr: float, embedder_epochs: int):
    embedder_ckpt_dir = 'mnist/ckpts/embedder'
    log_dir = 'mnist/logs/embedder'
    
    batch_size = 256
    specimen, train_loader, valid_loader, test_loader = load_dataset(batch_size)

    key = jax.random.PRNGKey(42)
    Model: Type[Embedder] = embedders[embedder_name]
    embedder = Model(key, specimen, embedder_dim, embedder_lr, embedder_epochs)

    with open(f'{log_dir}/{embedder.identifier}.txt', 'w') as log:
        with redirect_stdout(log):

            embedder.fit(embedder_ckpt_dir, train_loader, valid_loader)
            del embedder

            embedder_restored = Model(key, specimen, embedder_dim, embedder_lr, embedder_epochs)
            embedder_restored.load(embedder_ckpt_dir)

            test_loss = embedder_restored.evaluate(test_loader)
            print(f'End: test loss {test_loss}')


def load_dataset(batch_size):
    root = '/home/qys/torchvision/datasets'

    specimen = jnp.empty((28, 28, 1))
    transform = T.Compose([
        T.ToTensor(),
        lambda X: torch.permute(X, (1, 2, 0)),
    ])

    train_dataset = MNIST(root, train=True, download=False, transform=transform)
    train_dataset, valid_dataset = random_split(train_dataset, (50000, 10000),
            generator=torch.Generator().manual_seed(42))
    test_dataset = MNIST(root, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    return specimen, train_loader, valid_loader, test_loader


if __name__ == '__main__':
    cli()
