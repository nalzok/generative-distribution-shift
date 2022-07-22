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
@click.option('--ae_lr', type=float, required=True)
@click.option('--epochs', type=int, required=True)
@click.option('--ae_ckpt_dir', type=click.Path(), required=True)
def train_ae(embedding_dim, ae_lr, epochs, ae_ckpt_dir):
    root = '/home/qys/torchvision/datasets'
    train_dataset = MNIST(root, train=True, download=False, transform=np.array)
    train_dataset, valid_dataset = random_split(train_dataset, (50000, 10000),
            generator=Generator().manual_seed(42))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size)

    key = jax.random.PRNGKey(42)
    specimen = jnp.empty((28, 28, 1))
    ae = AutoEncoderModel(key, embedding_dim, ae_lr, specimen)

    ae.fit(ae_ckpt_dir, epochs, train_loader, valid_loader)


if __name__ == '__main__':
    train_ae()
