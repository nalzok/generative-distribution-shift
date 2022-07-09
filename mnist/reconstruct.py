from itertools import islice

import jax
import jax.numpy as jnp
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.autoencoder import AutoEncoderModel


if __name__ == '__main__':
    epochs = 28
    report_every = 3
    batch_size = 64
    lr = 1e-3
    embedding_dim = 16

    root = '/mnt/disks/persist/torchvision/datasets'
    train_dataset = MNIST(root, train=True, download=True, transform=np.array)
    test_dataset = MNIST(root, train=False, download=True, transform=np.array)


    key = jax.random.PRNGKey(42)

    specimen = jnp.empty((28, 28, 1))
    key_init, key = jax.random.split(key)

    ckpt_dir = 'mnist/ckpts'
    ae = AutoEncoderModel(key_init, embedding_dim, lr, specimen, ckpt_dir)
    # ae = AutoEncoderModel(key_init, embedding_dim, lr, specimen)
    # ae.fit(epochs, batch_size, report_every, train_dataset, ckpt_dir)

    fig, axes = plt.subplots(10, 2, constrained_layout=True, figsize=plt.figaspect(11))
    test_loader = DataLoader(test_dataset, batch_size)
    for i, (X, _) in enumerate(islice(test_loader, 10)):
        image = jnp.array(X).reshape((-1, *specimen.shape))/255.
        _, reconstructed = ae(image)
        ax_orig, ax_reco = axes[i, 0], axes[i, 1]
        ax_orig.imshow(image[0], cmap='gist_gray')
        ax_reco.imshow(reconstructed[0], cmap='gist_gray')
        for j in range(2):
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)

    plt.suptitle('Original (left) vs Reconstructed (right)')
    plt.savefig('mnist/plots/reconstructed.png', dpi=200)
