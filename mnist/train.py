import jax
import jax.numpy as jnp
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset

from models.autoencoder import AutoEncoderModel
from models.gmm import GMM


if __name__ == '__main__':
    ckpt_dir = 'mnist/ckpts'

    # AutoEncoder
    embedding_dim = 16
    ae_lr = 1e-3
    specimen = jnp.empty((28, 28, 1))

    key = jax.random.PRNGKey(42)
    key_init, key = jax.random.split(key)
    ae = AutoEncoderModel(key_init, embedding_dim, ae_lr, specimen, ckpt_dir)

    def transform(X):
        return np.asarray(ae(np.array(X).reshape(specimen.shape))).flatten()

    root = '/mnt/disks/persist/torchvision/datasets'
    train_dataset = MNIST(root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root, train=False, download=True, transform=transform)

    unlabeled_factor = 16
    rng = np.random.default_rng(42)
    mask = rng.uniform(size=len(train_dataset)) < 1 / unlabeled_factor
    labeled_dataset = Subset(train_dataset, np.flatnonzero(mask))
    unlabeled_dataset = Subset(train_dataset, np.flatnonzero(~mask))

    # GMM
    C, K, D, R = 10, 2, embedding_dim, 2
    gmm_lr = 5e-4
    lambda_, kappa = 0.5, 0.5

    gmm = GMM(C, K, D, R, gmm_lr, lambda_, kappa)
    # FIXME: load weights from checkpoint

    epochs = 28
    report_every = 3
    batch_size = 64

    labeled_loader = DataLoader(labeled_dataset, batch_size)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size * unlabeled_factor)
    valid_loader = DataLoader(test_dataset, batch_size)

    gmm.fit(epochs, report_every, labeled_loader, unlabeled_loader, valid_loader)
