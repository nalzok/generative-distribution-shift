import jax
import jax.numpy as jnp
import numpy as np
from torchvision.datasets import MNIST
from torch import Generator
from torch.utils.data import random_split, DataLoader
import click

from models.autoencoder import AutoEncoderModel


@click.command()
@click.option('--embedding_dim', type=int, required=True)
@click.option('--lr', type=float, default=1e-3)
@click.option('--epochs', type=int, default=32)
def train_ae(embedding_dim, lr, epochs):
    root = '/mnt/disks/persist/torchvision/datasets'
    ckpt_dir = '/mnt/disks/persist/generative-distribution-shift/mnist/ckpts'

    train_dataset = MNIST(root, train=True, download=True, transform=np.array)
    train_dataset, valid_dataset = random_split(train_dataset, (50000, 10000),
            generator=Generator().manual_seed(42))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size)

    key = jax.random.PRNGKey(42)
    specimen = jnp.empty((28, 28, 1))
    ae = AutoEncoderModel(key, embedding_dim, lr, specimen)

    ae.fit(ckpt_dir, epochs, train_loader, valid_loader)


if __name__ == '__main__':
    train_ae()
