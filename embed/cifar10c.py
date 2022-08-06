from typing import Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, TensorDataset

from . import EmbeddingConfig


def load_embeddings(split: str, severity: Optional[int], config: EmbeddingConfig) -> Dataset:
    tensor_path = Path('data/embeddings') / split / f'{config}.pt'

    if split in {'train', 'test'}:
        if severity is not None:
            raise ValueError(f'There are no corruptions in {split}')
        embeddings = torch.load(tensor_path)

    else:
        if severity is None:
            raise ValueError(f'Must specify a severity level for {split}')
        embeddings = torch.load(tensor_path)
        num_images = embeddings.shape[0] // 5
        embeddings = embeddings[(severity - 1) * num_images:severity * num_images]

    ds = TensorDataset(embeddings)
    return ds
