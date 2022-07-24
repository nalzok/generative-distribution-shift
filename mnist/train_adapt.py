import jax
import jax.numpy as jnp
import numpy as np
from torchvision.datasets import MNIST

from torch import Generator
from torch.utils.data import random_split, DataLoader
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
@click.option('--gmm_ckpt_dir', type=click.Path(), required=True)
@click.option('--adapt_lr', type=float, required=True)
@click.option('--epochs', type=int, required=True)
@click.option('--adapt_ckpt_dir', type=click.Path(), required=True)
def train(embedding_dim, ae_lr, ae_ckpt_dir, unlabeled_factor,
        init_scheme, k, r, gmm_lr, lambda_, kappa, gmm_ckpt_dir,
        adapt_lr, epochs, adapt_ckpt_dir):
    if r > embedding_dim:
        print(f'r = {r} > {embedding_dim} = embedding_dim')
        return

    key = jax.random.PRNGKey(42)
    specimen = jnp.empty((28, 28, 1))
    ae = AutoEncoderModel(key, embedding_dim, ae_lr, specimen, ae_ckpt_dir)

    def transform(X):
        X = X.rotate(90)
        return np.asarray(ae.embed(np.array(X).reshape(specimen.shape))).flatten()

    root = '/home/qys/torchvision/datasets'
    train_dataset = MNIST(root, train=True, download=False, transform=transform)
    unlabeled_dataset, valid_dataset = random_split(train_dataset, (50000, 10000),
            generator=Generator().manual_seed(42))

    C, K, D, R = 10, k, embedding_dim, r
    metadata = f'dim{embedding_dim}_aelr{ae_lr}_ufactor{unlabeled_factor}'
    gmm = GMM(C, K, D, R, init_scheme, gmm_lr, lambda_, kappa, metadata, gmm_ckpt_dir, adapt_lr)

    batch_size = 64
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size)

    gmm.adapt(adapt_ckpt_dir, epochs, unlabeled_loader, valid_loader)
    del gmm

    # Test loading weights from checkpoint
    gmm_restored = GMM(C, K, D, R, init_scheme, gmm_lr, lambda_, kappa, metadata, adapt_ckpt_dir)
    test_dataset = MNIST(root, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size)

    correct_cases, total_cases = gmm_restored.evaluate(test_loader)
    print(f'GMM: end: test accuracy {100*correct_cases/total_cases:.2f}%')


if __name__ == '__main__':
    train()
