from typing import Any
from numbers import Number
from abc import ABC, abstractmethod, abstractproperty

import jax.numpy as jnp
from torch.utils.data import DataLoader


class Embedder(ABC):
    @abstractmethod
    def __init__(self, key: Any, specimen: jnp.ndarray, dim: int, lr: float, epochs: int) -> None:
        ...

    @abstractproperty
    def identifier(self) -> str:
        ...

    @abstractmethod
    def load(self, ckpt_dir: str) -> None:
        ...

    @abstractmethod
    def fit(self, ckpt_dir: str, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        ...

    @abstractmethod
    def evaluate(self, loader: DataLoader) -> Number:
        ...

    @abstractmethod
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        ...


from models.embedder.ae import AutoEncoderModel
from models.embedder.cnn import CNNModel


embedders = {
    'ae': AutoEncoderModel,
    'cnn': CNNModel,
}
