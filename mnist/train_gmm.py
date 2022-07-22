import jax
import jax.numpy as jnp
import numpy as np
from torchvision.datasets import MNIST

from torch import Generator
from torch.utils.data import random_split, DataLoader, Subset
import click

from models.autoencoder import AutoEncoderModel
from models.gmm import GMM


@click.command()
@click.option('--embedding_dim', type=int, required=True)
@click.option('--ae_lr', type=float, required=True)
@click.option('--ae_ckpt_dir', type=click.Path(), required=True)
@click.option('--unlabeled_factor', type=int, required=True)
@click.option('--init_scheme', type=str, required=True)
@click.option('--k', type=int, required=True)
@click.option('--r', type=int, required=True)
@click.option('--gmm_lr', type=float, required=True)
@click.option('--lambda_', type=float, required=True)
@click.option('--kappa', type=float, required=True)
@click.option('--epochs', type=int, required=True)
@click.option('--gmm_ckpt_dir', type=click.Path(), required=True)
def train(embedding_dim, ae_lr, ae_ckpt_dir, unlabeled_factor, init_scheme, k, r, gmm_lr, lambda_, kappa, epochs, gmm_ckpt_dir):
    if r > embedding_dim:
        print(f'r = {r} > {embedding_dim} = embedding_dim')
        return

    key = jax.random.PRNGKey(42)
    specimen = jnp.empty((28, 28, 1))
    ae = AutoEncoderModel(key, embedding_dim, ae_lr, specimen, ae_ckpt_dir)

    def transform(X):
        return np.asarray(ae.embed(np.array(X).reshape(specimen.shape))).flatten()

    root = '/home/qys/torchvision/datasets'
    train_dataset = MNIST(root, train=True, download=False, transform=transform)
    train_dataset, valid_dataset = random_split(train_dataset, (50000, 10000),
            generator=Generator().manual_seed(42))

    rng = np.random.default_rng(42)
    mask = rng.uniform(size=len(train_dataset)) < 1 / unlabeled_factor
    labeled_dataset = Subset(train_dataset, np.flatnonzero(mask))
    unlabeled_dataset = Subset(train_dataset, np.flatnonzero(~mask))

    C, K, D, R = 10, k, embedding_dim, r
    metadata = f'dim{embedding_dim}_aelr{ae_lr}_ufactor{unlabeled_factor}'
    gmm = GMM(C, K, D, R, gmm_lr, lambda_, kappa, metadata)

    batch_size = 64
    labeled_loader = DataLoader(labeled_dataset, batch_size)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size * unlabeled_factor)
    valid_loader = DataLoader(valid_dataset, batch_size)

    gmm.fit(init_scheme, gmm_ckpt_dir, epochs, labeled_loader, unlabeled_loader, valid_loader)
    del gmm

    # Test loading weights from checkpoint
    gmm_loaded = GMM(C, K, D, R, gmm_lr, lambda_, kappa, metadata, gmm_ckpt_dir)
    test_dataset = MNIST(root, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size)

    correct_cases, total_cases = gmm_loaded.evaluate(test_loader)
    print(f'GMM: end: test accuracy {100*correct_cases/total_cases:.2f}%')


if __name__ == '__main__':
    train()
