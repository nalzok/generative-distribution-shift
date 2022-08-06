from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import CIFAR10

from . import EmbeddingConfig


def load_embeddings(split: str, severity: Optional[int], config: EmbeddingConfig) -> Dataset:
    tensor_path = Path('data/embeddings') / split / f'{config}.pt'

    if split in {'train', 'test'}:
        if severity is not None:
            raise ValueError(f'There are no corruptions in {split}')
        embeddings = torch.load(tensor_path)

        root = 'data/pixels/torchvision'
        cifar10 = CIFAR10(root, train=split=='train', download=False)

        label_list = []
        for _, label in cifar10:
            label_list.append(label)
        labels = torch.tensor(label_list)

    else:
        if severity is None:
            raise ValueError(f'Must specify a severity level for {split}')
        embeddings = torch.load(tensor_path)
        num_images = embeddings.shape[0] // 5
        embeddings = embeddings[(severity - 1) * num_images:severity * num_images]
        labels = torch.as_tensor(np.load('data/pixels/CIFAR-10-C/labels.npy'))

    ds = TensorDataset(embeddings, labels)
    return ds
