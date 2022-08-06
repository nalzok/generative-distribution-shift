from contextlib import redirect_stdout

import torch
from torch.utils.data import random_split, DataLoader
import click

from embed import EmbeddingConfig
from embed.cifar10c import load_embeddings
from models.gmm import GMM


@click.command()
@click.option('--embedding_model', type=str, required=True)
@click.option('--embedding_global_pool', type=bool, required=True)
@click.option('--embedding_mask_ratio', type=float, required=True)
@click.option('--gmm_init', type=str, required=True)
@click.option('--gmm_k', type=int, required=True)
@click.option('--gmm_r', type=int, required=True)
@click.option('--gmm_lr', type=float, required=True)
@click.option('--gmm_dis', type=float, required=True)
@click.option('--gmm_un', type=float, required=True)
@click.option('--gmm_epochs', type=int, required=True)
def cli(embedding_model: str, embedding_global_pool: bool, embedding_mask_ratio: float,
        gmm_init, gmm_k, gmm_r, gmm_lr, gmm_dis, gmm_un, gmm_epochs):
    gmm_ckpt_dir = 'ckpts/gmm'
    log_dir = 'logs/gmm'

    embedding_config = EmbeddingConfig(embedding_model, embedding_global_pool, embedding_mask_ratio)
    batch_size = 256
    embedding_dim, train_loader, valid_loader, test_loader = load_dataset(embedding_config, batch_size)

    C, K, D, R = 10, gmm_k, embedding_dim, gmm_r
    gmm = GMM(C, K, D, R, gmm_init, gmm_lr, gmm_dis, gmm_un, embedding_config, gmm_epochs)

    with open(f'{log_dir}/{gmm.identifier}.txt', 'w') as log:
        with redirect_stdout(log):
            if gmm_r > embedding_dim:
                print(f'gmm_r = {gmm_r} > {embedding_dim} = embedding_dim')
                return

            gmm.fit(gmm_ckpt_dir, train_loader, valid_loader)
            del gmm

            gmm_restored = GMM(C, K, D, R, gmm_init, gmm_lr, gmm_dis, gmm_un, embedding_config, gmm_epochs)
            gmm_restored.load(gmm_ckpt_dir)

            test_acc = gmm_restored.evaluate(test_loader)
            print(f'End: test accuracy {test_acc}')


def load_dataset(embedding_config: EmbeddingConfig, batch_size: int):
    train_dataset = load_embeddings('train', None, embedding_config)
    train_dataset, valid_dataset = random_split(train_dataset, (45000, 5000),
            generator=torch.Generator().manual_seed(42))
    test_dataset = load_embeddings('test', None, embedding_config)

    train_loader = DataLoader(train_dataset, batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    embedding, _ = train_dataset[0]
    return embedding.shape[0], train_loader, valid_loader, test_loader


if __name__ == '__main__':
    cli()
