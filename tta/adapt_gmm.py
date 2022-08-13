from contextlib import redirect_stdout

from torch.utils.data import DataLoader
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
@click.option('--adapt_corruption', type=str, required=True)
@click.option('--adapt_severity', type=int, required=True)
@click.option('--adapt_lr', type=float, required=True)
@click.option('--adapt_epochs', type=int, required=True)
def cli(embedding_model, embedding_global_pool, embedding_mask_ratio,
        gmm_init, gmm_k, gmm_r, gmm_lr, gmm_dis, gmm_un, gmm_epochs,
        adapt_corruption, adapt_severity, adapt_lr, adapt_epochs):
    gmm_ckpt_dir = 'ckpts/gmm'
    adapt_ckpt_dir = 'ckpts/adapt'
    log_dir = 'logs/adapt'

    embedding_config = EmbeddingConfig(embedding_model, embedding_global_pool, embedding_mask_ratio)
    batch_size = 256
    embedding_dim, shift_loader = load_dataset(embedding_config, adapt_corruption, adapt_severity, batch_size)

    C, K, D, R = 10, gmm_k, embedding_dim, gmm_r
    gmm_dummy = GMM(C, K, D, R, gmm_init, gmm_lr, gmm_dis, gmm_un, embedding_config, gmm_epochs)
    gmm_dummy.mark_adapt(adapt_corruption, adapt_severity, adapt_lr, adapt_epochs)

    with open(f'{log_dir}/{gmm_dummy.identifier}.txt', 'w') as log:
        with redirect_stdout(log):
            if gmm_r > embedding_dim:
                print(f'gmm_r = {gmm_r} > {embedding_dim} = embedding_dim')
                return

            gmm_restored = GMM(C, K, D, R, gmm_init, gmm_lr, gmm_dis, gmm_un, embedding_config, gmm_epochs)
            gmm_restored.load(gmm_ckpt_dir)

            baseline_acc = gmm_restored.evaluate(shift_loader)
            print(f'Begin: test accuracy {baseline_acc}')

            gmm_restored.mark_adapt(adapt_corruption, adapt_severity, adapt_lr, adapt_epochs)
            gmm_restored.adapt(adapt_ckpt_dir, baseline_acc, shift_loader, shift_loader)
            del gmm_restored

            gmm_adapted = GMM(C, K, D, R, gmm_init, gmm_lr, gmm_dis, gmm_un, embedding_config, gmm_epochs)
            gmm_adapted.mark_adapt(adapt_corruption, adapt_severity, adapt_lr, adapt_epochs)
            gmm_adapted.load(adapt_ckpt_dir)

            adapted_acc = gmm_adapted.evaluate(shift_loader)
            print(f'End: test accuracy {adapted_acc}')


def load_dataset(embedding_config: EmbeddingConfig, corruption: str, severity: int, batch_size: int):
    shift_dataset = load_embeddings(corruption, severity, embedding_config)
    shift_loader = DataLoader(shift_dataset, batch_size)

    embedding, _ = shift_dataset[0]
    return embedding.shape[0], shift_loader


if __name__ == '__main__':
    cli()
