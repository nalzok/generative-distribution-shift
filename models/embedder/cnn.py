from typing import Any
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


class CNN(nn.Module):
    embedding_dim: int

    def setup(self):
        self.feature_extractor = Encoder(self.embedding_dim)
        self.last = nn.Dense(10)

    def encode(self, X, training):
        return self.feature_extractor(X, training)

    def __call__(self, X, training):
        embedding = self.encode(X, training)
        X = self.last(embedding)
        return X


def create_train_state(key, specimen, dim, lr):
    cnn = CNN(dim)
    variables = cnn.init(key, specimen, True)
    tx = optax.adam(lr)
    state = TrainState.create(
            apply_fn=cnn.apply,
            params=variables['params'],
            tx=tx,
            batch_stats=variables['batch_stats'],
    )

    return state


@jax.jit
def train_step(state, image, label):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logit, new_model_state = state.apply_fn(
            variables, image, True, mutable=['batch_stats']
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logit, label)
        return jnp.sum(loss), new_model_state

    (loss, new_model_state), grads = loss_fn(state.params)

    state = state.apply_gradients(
            grads=grads,
            batch_stats=new_model_state['batch_stats']
    )

    return state, loss


@jax.jit
def valid_step(state, image, label):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logit = state.apply_fn(variables, image, False)
    prediction = jnp.argmax(logit, axis=-1)
    correct_cases = jnp.sum(prediction == label)
    return correct_cases


class CNNModel(Embedder):
    def __init__(self, key: Any, specimen: jnp.ndarray, dim: int, lr: float, epochs: int) -> None:
        self.dim = dim
        self.lr = lr
        self.epochs = epochs

        self.state = create_train_state(key, specimen, dim, lr)


    @property
    def identifier(self) -> str:
        return f'EMBEDDER_cnn_dim{self.dim}_lr{self.lr}_epc{self.epochs}'


    def load(self, ckpt_dir: str) -> None:
        prefix = f'{self.identifier}_'
        restored = checkpoints.restore_checkpoint(ckpt_dir, self.state, prefix=prefix)
        if restored is self.state:
            raise ValueError(f'Cannot find checkpoint {prefix}X')
        self.state = restored


    def fit(self, ckpt_dir: str, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        best_valid_acc = 0
        for epoch in range(self.epochs):
            train_loss = 0 
            for X, y in train_loader:
                image = jnp.array(X)
                label = jnp.array(y)
                self.state, loss = train_step(self.state, image, label)
                train_loss += loss

            valid_acc = self.evaluate(valid_loader)
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                checkpoints.save_checkpoint(ckpt_dir, self.state, valid_acc, prefix=f'{self.identifier}_')

            print(f'Epoch {epoch + 1}: train loss {train_loss}, valid accuracy {valid_acc}')


    def evaluate(self, loader: DataLoader) -> float:
        correct_cases = 0
        total_cases = 0
        for X, y in loader:
            image = jnp.array(X)
            label = jnp.array(y)
            correct_cases += valid_step(self.state, image, label)
            total_cases += X.shape[0]

        return correct_cases/total_cases


    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        variables = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        embedding = self.state.apply_fn(variables, X, False, method=CNN.encode)

        return embedding
