from typing import cast
from pathlib import Path
from json import dumps
from urllib.parse import quote

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from transformers.models.vit.feature_extraction_vit import ViTFeatureExtractor
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel
from transformers import logging
import click


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
@click.option('--dataset_name', type=str, required=True)
@click.option('--model_name', type=str, required=True)
@click.option('--global_pool', type=bool, required=True)
@click.option('--mask_ratio', type=float, required=True)
def embed_transformer(dataset_name: str,model_name: str, **config):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    config_string = quote(dumps(config, sort_keys=True, separators=(',', ':')), safe='')
    output_path = Path('data/embeddings') / dataset_name / model_name / f'{config_string}.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global_pool = config['global_pool']
    mask_ratio = config['mask_ratio']

    root = 'data/pixels/torchvision'
    if dataset_name == 'train':
        dataset = CIFAR10(root, train=True, download=True, transform=T.ToTensor())
    elif dataset_name == 'vanilla':
        dataset = CIFAR10(root, train=False, download=True, transform=T.ToTensor())
    elif dataset_name in corrputions:
        images = torch.as_tensor(np.load(f'data/pixels/CIFAR-10-C/{dataset_name}.npy'))
        labels = torch.as_tensor(np.load('data/pixels/CIFAR-10-C/labels.npy'))
        dataset = TensorDataset(images, labels)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    # Resizing & Normalization
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    # Calculating embedding
    verbosity = logging.get_verbosity()
    logging.set_verbosity_error()
    model = cast(ViTMAEModel, ViTMAEModel.from_pretrained(model_name, mask_ratio=mask_ratio))
    logging.set_verbosity(verbosity)
    model.eval()

    loader = DataLoader(dataset, batch_size=256)

    outcomes = []
    with torch.no_grad():
        for images, _ in loader:
            for image in images:
                encoded_inputs = feature_extractor(image, return_tensors='pt')
                pixel_values = encoded_inputs['pixel_values']
                outputs = model(pixel_values)
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


if __name__ == '__main__':
    embed_transformer()
