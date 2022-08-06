from typing import cast
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from transformers.models.vit.feature_extraction_vit import ViTFeatureExtractor
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel
from transformers import logging
import click

from . import EmbeddingConfig


corrputions = {
    'brightness',
    'contrast',
    'defocus_blur',
    'elastic_transform',
    'fog',
    'frost',
    'gaussian_blur',
    'gaussian_noise',
    'glass_blur',
    'impulse_noise',
    'jpeg_compression',
    'motion_blur',
    'pixelate',
    'saturate',
    'shot_noise',
    'snow',
    'spatter',
    'speckle_noise',
    'zoom_blur',
}


@click.command()
@click.option('--split', type=str, required=True)
@click.option('--model', type=str, required=True)
@click.option('--global_pool', type=bool, required=True)
@click.option('--mask_ratio', type=float, required=True)
def embed_transformer(split: str, model: str, global_pool: bool, mask_ratio: float):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    config = EmbeddingConfig(model, global_pool, mask_ratio)
    output_path = Path('data/embeddings') / split / f'{config}.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if split in {'train', 'test'}:
        root = 'data/pixels/torchvision'
        dataset = CIFAR10(root, train=split=='train', download=True, transform=T.ToTensor())
    elif split in corrputions:
        images = torch.as_tensor(np.load(f'data/pixels/CIFAR-10-C/{split}.npy'))
        labels = torch.as_tensor(np.load('data/pixels/CIFAR-10-C/labels.npy'))
        dataset = TensorDataset(images, labels)
    else:
        raise ValueError(f'Unknown split {split}')

    # Resizing & Normalization
    feature_extractor = ViTFeatureExtractor.from_pretrained(model)

    # Calculating embedding
    verbosity = logging.get_verbosity()
    logging.set_verbosity_error()
    transformer = cast(ViTMAEModel, ViTMAEModel.from_pretrained(model, mask_ratio=mask_ratio))
    logging.set_verbosity(verbosity)
    transformer.eval()

    loader = DataLoader(dataset, batch_size=1)

    outcomes = []
    with torch.no_grad():
        for images, _ in loader:
            for image in images:
                encoded_inputs = feature_extractor(image, return_tensors='pt')
                pixel_values = encoded_inputs['pixel_values']
                outputs = transformer(pixel_values)
                x = outputs.last_hidden_state

                # https://github.com/facebookresearch/mae/issues/70
                # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_vit.py#L46-L51
                norm = torch.nn.LayerNorm(x.shape[-1])
                if global_pool:
                    x = x[:, 1:, :].mean(axis=-2)
                    outcome = norm(x)
                else:
                    x = norm(x)
                    outcome = x[:, 0, :]

                outcomes.append(outcome)

    combined = torch.cat(outcomes)
    torch.save(combined, output_path)
    print(f'Saved embeddings to {output_path}')


if __name__ == '__main__':
    embed_transformer()
