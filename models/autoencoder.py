"""
Jax/Flax translation of 

    https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
"""

from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
import flax.linen as nn
from flax.training import train_state, checkpoints
from torch.utils.data import DataLoader


class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]


class Encoder(nn.Module):
    embedding_dim: int

    @nn.compact
    def __call__(self, X, training):
        X = nn.Conv(8, (3, 3), strides=2, padding=1)(X)
        X = jax.nn.relu(X)
        X = nn.Conv(16, (3, 3), strides=2, padding=1)(X)
        X = nn.BatchNorm(use_running_average=not training)(X)
        X = jax.nn.relu(X)
        X = nn.Conv(32, (3, 3), strides=2, padding=0)(X)
        X = jax.nn.relu(X)

        X = X.reshape((-1, np.prod(X.shape[-3:])))

        X = nn.Dense(128)(X)
        X = jax.nn.relu(X)
        X = nn.Dense(self.embedding_dim)(X)

        return X


class Decoder(nn.Module):
    output_dim: Tuple[int, int, int]

    @nn.compact
    def __call__(self, X, training):
        H, W, _ = self.output_dim
        H, W = H - 25, W - 25   # subtract 25 due to convolution

        X = nn.Dense(128)(X)
        X = jax.nn.relu(X)
        X = nn.Dense(H * W * 32)(X)
        X = jax.nn.relu(X)

        X = X.reshape((-1, H, W, 32))

        # NOTE: I am not sure if this is an word-to-word translation from PyTorch.
        # See https://flax.readthedocs.io/en/latest/howtos/convert_pytorch_to_flax.html#transposed-convolutions
        # However, the shape of activations and parameters match between PyTorch and Flax.
        X = nn.ConvTranspose(16, (3, 3), strides=(2, 2), padding=2)(X)
        X = nn.BatchNorm(use_running_average=not training)(X)
        X = jax.nn.relu(X)
        X = nn.ConvTranspose(8, (3, 3), strides=(2, 2), padding=((1, 2), (1, 2)))(X)
        X = nn.BatchNorm(use_running_average=not training)(X)
        X = jax.nn.relu(X)
        X = nn.ConvTranspose(1, (3, 3), strides=(2, 2), padding=((1, 2), (1, 2)))(X)
        X = jax.nn.sigmoid(X)

        return X


class AutoEncoder(nn.Module):
    embedding_dim: int
    output_dim: Tuple[int, int, int]

    def setup(self):
        self.encoder = Encoder(self.embedding_dim)
        self.decoder = Decoder(self.output_dim)

    def __call__(self, X, training):
        embedding = self.encoder(X, training)
        X = self.decoder(embedding, training)
        return embedding, X


def create_train_state(key, embedding_dim, learning_rate, specimen):
    ae = AutoEncoder(embedding_dim, specimen.shape)
    variables = ae.init(key, specimen, True)
    tx = optax.adam(learning_rate)
    state = TrainState.create(
            apply_fn=ae.apply,
            params=variables['params'],
            tx=tx,
            batch_stats=variables['batch_stats'],
    )

    return state


@jax.jit
def train_step(state, image):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        (_, reconstructed), new_model_state = state.apply_fn(
            variables, image, True, mutable=['batch_stats']
        )
        loss = jnp.sum((reconstructed - image)**2)
        return loss.sum(), new_model_state

    (loss, new_model_state), grads = loss_fn(state.params)

    state = state.apply_gradients(
            grads=grads,
            batch_stats=new_model_state['batch_stats']
    )

    return state, loss


@jax.jit
def test_step(state, image):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    embedding, reconstructed = state.apply_fn(variables, image, False)

    return embedding, reconstructed


class AutoEncoderModel:
    def __init__(self, key, embedding_dim, lr, specimen, ckpt_dir_in = None):
        self.state = create_train_state(key, embedding_dim, lr, specimen)
        self.input_dim = specimen.shape
        if ckpt_dir_in is not None:
            self.state = checkpoints.restore_checkpoint(ckpt_dir_in, self.state)


    def fit(self, epochs, batch_size, report_every, train_dataset, ckpt_dir_out = None):
        train_loader = DataLoader(train_dataset, batch_size)
        for epoch in range(epochs):
            train_loss = 0 
            for X, _ in train_loader:
                image = jnp.array(X).reshape((-1, *self.input_dim))/255.
                self.state, loss = train_step(self.state, image)
                train_loss += loss

            if epoch % report_every == 0:
                print(f'Epoch {epoch}: train loss {train_loss}')

            if ckpt_dir_out is not None:
                checkpoints.save_checkpoint(ckpt_dir_out, self.state, epoch, overwrite=True)


    def __call__(self, X):
        embedding, reconstructed = test_step(self.state, X)

        return embedding, reconstructed
