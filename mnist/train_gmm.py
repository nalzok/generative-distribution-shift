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
from models.gmm import GMM


@click.command()
@click.option('--embedder_name', type=str, required=True)
@click.option('--embedder_dim', type=int, required=True)
@click.option('--embedder_lr', type=float, required=True)
@click.option('--embedder_epochs', type=int, required=True)
@click.option('--gmm_init', type=str, required=True)
@click.option('--gmm_k', type=int, required=True)
@click.option('--gmm_r', type=int, required=True)
@click.option('--gmm_lr', type=float, required=True)
@click.option('--gmm_dis', type=float, required=True)
@click.option('--gmm_un', type=float, required=True)
@click.option('--gmm_epochs', type=int, required=True)
def cli(embedder_name, embedder_dim, embedder_lr, embedder_epochs,
        gmm_init, gmm_k, gmm_r, gmm_lr, gmm_dis, gmm_un, gmm_epochs):
    embedder_ckpt_dir = 'mnist/ckpts/embedder'
    gmm_ckpt_dir = 'mnist/ckpts/gmm'
    log_dir = 'mnist/logs/gmm'

    batch_size = 256
    specimen, train_loader, valid_loader, test_loader = load_dataset(batch_size)

    key = jax.random.PRNGKey(42)
    Model: Type[Embedder] = embedders[embedder_name]
    embedder = Model(key, specimen, embedder_dim, embedder_lr, embedder_epochs)
    embedder.load(embedder_ckpt_dir)

    C, K, D, R = 10, gmm_k, embedder_dim, gmm_r
    gmm = GMM(C, K, D, R, gmm_init, gmm_lr, gmm_dis, gmm_un, embedder, gmm_epochs)

    with open(f'{log_dir}/{gmm.identifier}.txt', 'w') as log:
        with redirect_stdout(log):
            if gmm_r > embedder_dim:
                print(f'gmm_r = {gmm_r} > {embedder_dim} = embedder_dim')
                return

            gmm.fit(gmm_ckpt_dir, train_loader, valid_loader)
            del gmm

            gmm_restored = GMM(C, K, D, R, gmm_init, gmm_lr, gmm_dis, gmm_un, embedder, gmm_epochs)
            gmm_restored.load(gmm_ckpt_dir)

            test_acc = gmm_restored.evaluate(test_loader)
            print(f'End: test accuracy {test_acc}')


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
