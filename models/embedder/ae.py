"""
Jax/Flax translation of 

    https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
"""

from typing import Tuple, Any
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
import flax.linen as nn
from flax.training import train_state, checkpoints
from torch.utils.data import DataLoader

from . import Embedder


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
        H, W = 3, 3     # TODO: don't hardcode dimension

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

    def encode(self, X, training):
        return self.encoder(X, training)

    def decode(self, embedding, training):
        return self.decoder(embedding, training)

    def __call__(self, X, training):
        embedding = self.encode(X, training)
        X = self.decode(embedding, training)
        return X


def create_train_state(key, specimen, dim, lr):
    ae = AutoEncoder(dim, specimen.shape)
    variables = ae.init(key, specimen, True)
    tx = optax.adam(lr)
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
        reconstructed, new_model_state = state.apply_fn(
            variables, image, True, mutable=['batch_stats']
        )
        loss = jnp.sum((reconstructed - image)**2, axis=(-3, -2, -1))
        return jnp.sum(loss), new_model_state

    (loss, new_model_state), grads = loss_fn(state.params)

    state = state.apply_gradients(
            grads=grads,
            batch_stats=new_model_state['batch_stats']
    )

    return state, loss


@jax.jit
def valid_step(state, image):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    reconstructed = state.apply_fn(variables, image, False)
    loss = jnp.sum((reconstructed - image)**2, axis=(-3, -2, -1))
    return jnp.sum(loss)


class AutoEncoderModel(Embedder):
    def __init__(self, key: Any, specimen: jnp.ndarray, dim: int, lr: float, epochs: int) -> None:
        self.dim = dim
        self.lr = lr
        self.epochs = epochs

        self.state = create_train_state(key, specimen, dim, lr)


    @property
    def identifier(self) -> str:
        return f'EMBEDDER_ae_dim{self.dim}_lr{self.lr}_epc{self.epochs}'


    def load(self, ckpt_dir: str) -> None:
        prefix = f'{self.identifier}_'
        restored = checkpoints.restore_checkpoint(ckpt_dir, self.state, prefix=prefix)
        if restored is self.state:
            raise ValueError(f'Cannot find checkpoint {prefix}X')
        self.state = restored


    def fit(self, ckpt_dir: str, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        best_valid_loss = float('inf')
        for epoch in range(self.epochs):
            train_loss = 0 
            for X, _ in train_loader:
                image = jnp.array(X)
                self.state, loss = train_step(self.state, image)
                train_loss += loss

            valid_loss = self.evaluate(valid_loader)
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                checkpoints.save_checkpoint(ckpt_dir, self.state, -valid_loss, prefix=f'{self.identifier}_')

            print(f'Epoch {epoch + 1}: train loss {train_loss}, valid loss {valid_loss}')


    def evaluate(self, loader: DataLoader) -> float:
        loss = 0
        for X, _ in loader:
            image = jnp.array(X)
            loss += valid_step(self.state, image)

        return loss


    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        variables = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        embedding = self.state.apply_fn(variables, X, False, method=AutoEncoder.encode)

        return embedding
