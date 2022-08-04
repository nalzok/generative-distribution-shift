from pathlib import Path
import re

import numpy as np
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import click


@click.command()
@click.option('--log_path', type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
def cli(log_path: Path):
    if log_path.match('mnist/logs/adapt/ADAPTED_*.txt'):
        fig, axes = plt.subplots(3, 1)
        plot_adapt(log_path, axes[0])
        plot_gmm(log_path, axes[1])
        plot_embedder(log_path, axes[2])
    elif log_path.match('mnist/logs/gmm/GMM_*.txt'):
        fig, axes = plt.subplots(2, 1)
        plot_gmm(log_path, axes[0])
        plot_embedder(log_path, axes[1])
    elif log_path.match('mnist/logs/embedder/EMBEDDER_*.txt'):
        fig, ax = plt.subplots()
        plot_embedder(log_path, ax)
    else:
        raise ValueError

    fig.tight_layout()
    fig.set_size_inches(16, 9)
    fig.savefig(log_path.with_suffix('.png'), dpi=200)
    plt.close()


def plot_adapt(log_path: Path, ax1: axes.Axes):
    num = r'[+-]?(?:[0-9]*[.])?[0-9]+'
    begin_pattern = re.compile(fr'^Begin: test accuracy ({num})\n$')
    epoch_pattern = re.compile(fr'^Epoch ([0-9]+): train loss ({num}), test accuracy ({num}), delta ({num})\n$')

    with open(log_path) as log:
        line_iter = iter(log)
        first_line = next(line_iter)
        match = begin_pattern.fullmatch(first_line)
        assert match is not None
        epochs, losses, accuracies = [0], [np.nan], [float(match.group(1))]

        for line in line_iter:
            match = epoch_pattern.fullmatch(line)
            if match is not None:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                accuracies.append(float(match.group(3)))

    epochs = np.array(epochs)
    losses = np.array(losses)
    accuracies = np.array(accuracies)

    ax2 = ax1.twinx()
    lns1 = ax1.plot(epochs, losses, 'C1', label='TTA loss')
    lns2 = ax2.plot(epochs, accuracies, 'C2', label='Test accuracy')

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='C1')
    ax2.set_ylabel('Accuracy', color='C2')
    ax1.set_title('Adaption')


def plot_gmm(log_path: Path, ax1: axes.Axes):
    gmm_pattern = re.compile(r'(GMM_.*\.txt)$')
    if log_path.name.startswith('ADAPTED_'):
        match = gmm_pattern.search(log_path.name)
        assert match is not None
        log_path = log_path.parent.parent / 'gmm' / match.group(1)

    num = r'[+-]?(?:[0-9]*[.])?[0-9]+'
    epoch_pattern = re.compile(fr'^Epoch ([0-9]+): train loss ({num}) = joint {num} \+ dis {num} \* {num} \+ un {num} \* {num}, valid accuracy ({num})\n$')
    end_pattern = re.compile(fr'^End: test accuracy ({num})\n$')

    with open(log_path) as log:
        epochs, losses, accuracies = [], [], []
        test_acc = None
        for line in log:
            match = epoch_pattern.fullmatch(line)
            if match is not None:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                accuracies.append(float(match.group(3)))
            else:
                match = end_pattern.fullmatch(line)
                assert match is not None
                test_acc = float(match.group(1))

    epochs = np.array(epochs)
    losses = np.array(losses)
    accuracies = np.array(accuracies)

    ax2 = ax1.twinx()
    lns1 = ax1.plot(epochs, losses, 'C1', label='Train loss')
    lns2 = ax2.plot(epochs, accuracies, 'C2', label='Validation accuracy')
    lns3 = [ax2.axhline(test_acc, color='C2', linestyle=':', label='Test accuracy')]

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='C1')
    ax2.set_ylabel('Accuracy', color='C2')
    ax1.set_title('GMM')


def plot_embedder(log_path: Path, ax1: axes.Axes):
    embedder_pattern = re.compile(r'(EMBEDDER_.*\.txt)$')
    if log_path.name.startswith('ADAPTED_') or log_path.name.startswith('GMM_'):
        match = embedder_pattern.search(log_path.name)
        assert match is not None
        log_path = log_path.parent.parent / 'embedder' / match.group(1)

    supervised = log_path.name.startswith('EMBEDDER_cnn_')

    num = r'[+-]?(?:[0-9]*[.])?[0-9]+'
    epoch_pattern = re.compile(fr'^Epoch ([0-9]+): train loss ({num}), valid loss ({num})\n$')
    end_pattern = re.compile(fr'^End: test loss ({num})\n$')

    with open(log_path) as log:
        epochs, train, valid = [], [], []
        test_loss = None
        for line in log:
            match = epoch_pattern.fullmatch(line)
            if match is not None:
                epochs.append(int(match.group(1)))
                train.append(float(match.group(2)))
                valid.append(float(match.group(3)))
            else:
                match = end_pattern.fullmatch(line)
                assert match is not None
                test_loss = float(match.group(1))

    epochs = np.array(epochs)
    train = np.array(train)
    valid = np.array(valid)

    ax2 = ax1.twinx()
    lns1 = ax1.plot(epochs, train, 'C1', label='Train loss')
    lns2 = ax2.plot(epochs, valid, 'C2', label='Validation accuracy' if supervised else 'Validation loss')
    lns3 = [ax2.axhline(test_loss, color='C2', linestyle=':', label='Test accuracy' if supervised else 'Test loss')]

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='C1')
    ax2.set_ylabel('Accuracy' if supervised else 'Loss', color='C2')
    ax1.set_title('Embedder')


if __name__ == '__main__':
    cli()
