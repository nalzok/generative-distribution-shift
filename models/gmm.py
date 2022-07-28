from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from flax.training import checkpoints
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

from .embedder import Embedder


def init_gmm(C, K, D, R):
    """
    Accept the same input as https://github.com/wroth8/hybrid-gmm/blob/master/models/GMMClassifier.py

    C: Number of classes
    K: Number of clusters per class
    D: Number of input features
    R: Number of dimensions of the low-rank approximation of the DPLR matrix structure.
    """
    pi_logit = jnp.zeros((C,))
    alpha_logit = jnp.zeros((C, K))
    mu = jnp.empty((C, K, D))
    Psi_softplus = jnp.empty((C, K, D))
    S = jnp.ones((C, K, D, R))

    params = {
        'pi_logit': pi_logit,
        'alpha_logit': alpha_logit,
        'mu': mu,
        'Psi_softplus': Psi_softplus,
        'S': S,
    }
    return params


def llk_class(params, X):
    alpha = jax.nn.softmax(params['alpha_logit'])
    mu = params['mu']
    Psi = jax.nn.softplus(params['Psi_softplus'])
    S = params['S']

    _, _, D, _ = S.shape
    Psi = jax.vmap(jax.vmap(jnp.diag))(Psi)
    Sigma = 1e-3 * jax.numpy.eye(D) + Psi + jax.vmap(jax.vmap(jnp.matmul))(S, S.swapaxes(-1, -2))

    cluster_llk = jsp.stats.multivariate_normal.logpdf(X[:, np.newaxis, np.newaxis, :], mu, Sigma)
    class_llk = jsp.special.logsumexp(cluster_llk, b=alpha, axis=-1)
    return class_llk


def llk_hybrid(params, X, y):
    """
    Joint log-likelihood when lambda_ = 0
    Conditional log-likelihood when lambda_ = 1
    """
    pi = jax.nn.softmax(params['pi_logit'])
    class_llk = llk_class(params, X)

    joint_llk = jnp.log(pi[y]) + class_llk[jnp.arange(y.size), y]
    marginal_llk = jsp.special.logsumexp(class_llk, b=pi, axis=-1)
    cond_llk = joint_llk - marginal_llk

    return joint_llk, cond_llk


def llk_unlabeled(params, unlabeled):
    pi = jax.nn.softmax(params['pi_logit'])
    class_llk = llk_class(params, unlabeled)
    llk = jsp.special.logsumexp(class_llk, b=pi, axis=-1)

    return llk


@partial(jax.value_and_grad, has_aux=True)
def objective_hybrid(params, dis, un, embedding, label, unlabeled_embedding):
    joint_llk, cond_llk = llk_hybrid(params, embedding, label)
    if un > 0:
        unlabeled_llk = llk_unlabeled(params, unlabeled_embedding)
    else:
        unlabeled_llk = jnp.zeros(unlabeled_embedding.shape[0])

    joint_llk_batch = jnp.sum(joint_llk) / embedding.shape[1]
    cond_llk_batch = jnp.sum(cond_llk)
    unlabeled_llk_batch = jnp.sum(unlabeled_llk) / unlabeled_embedding.shape[0] * embedding.shape[0]
    llk = joint_llk_batch + dis * cond_llk_batch + un * unlabeled_llk_batch

    return -llk, (joint_llk_batch, cond_llk_batch, unlabeled_llk_batch)


@partial(jax.jit, static_argnames=('tx', 'dis', 'un'))
def train_step(params, opt_state, tx, dis, un, embedding, label, unlabeled_embedding):
    (llk_val, llk_breakdown), grads = objective_hybrid(params, dis, un, embedding, label, unlabeled_embedding)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return llk_val, params, opt_state, llk_breakdown


@partial(jax.value_and_grad)
def objective_unlabeled(params, unlabeled_embedding):
    unlabeled_llk = llk_unlabeled(params, unlabeled_embedding)
    unlabeled_llk_batch = jnp.sum(unlabeled_llk)

    return -unlabeled_llk_batch


@partial(jax.jit, static_argnames=('tx',))
def adapt_step(params, opt_state, tx, unlabeled_embedding):
    llk_val, grads = objective_unlabeled(params, unlabeled_embedding)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return llk_val, params, opt_state


@jax.jit
def valid_step(params, X, y):
    prior = jax.nn.softmax(params['pi_logit'])
    class_llk = llk_class(params, X)

    log_posterior = jnp.log(prior) + class_llk
    log_posterior -= jsp.special.logsumexp(class_llk, b=prior, axis=-1, keepdims=True)  # normalize

    predictions = jnp.argmax(log_posterior, axis=-1)
    correct_cases = jnp.sum(predictions == y)
    return correct_cases


class GMM:
    def __init__(self, C: int, K: int, D: int, R: int,
            init: str, lr: float, dis: float, un: float,
            embedder: Embedder, epochs: int):
        self.C = C
        self.K = K
        self.D = D
        self.R = R
        self.init = init
        self.lr = lr
        self.dis = dis
        self.un = un
        self.embedder = embedder
        self.epochs = epochs
        self.adapted = ''

        self.params = init_gmm(C, K, D, R)

        self.tx = optax.adam(learning_rate=lr)
        self.opt_state = self.tx.init(self.params)


    @property
    def identifier(self) -> str:
        gmm_identifier = f'GMM_{self.init}_K{self.K}_R{self.R}_lr{self.lr}_dis{self.dis}_un{self.un}_epc{self.epochs}_{self.embedder.identifier}'
        return f'{self.adapted}{gmm_identifier}'


    def load(self, ckpt_dir: str) -> None:
        prefix = f'{self.identifier}_'
        restored = checkpoints.restore_checkpoint(ckpt_dir, self.params, prefix=prefix)
        if restored is self.params:
            raise ValueError(f'Cannot find checkpoint {prefix}X')
        self.params = restored


    def fit(self, ckpt_dir: str, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        self._init_params(train_loader)

        best_valid_acc = 0
        for epoch in range(self.epochs):
            llk_val = 0
            joint_llk_val = 0
            cond_llk_val = 0
            unlabeled_llk_val = 0
            for X, y in train_loader:
                image = jnp.array(X)
                embedding = self.embedder(image)
                label = jnp.array(y)
                unlabeled_embedding = embedding   # FIXME: PLACEHOLDER
                llk_val_batch, self.params, self.opt_state, llk_breakdown = train_step(self.params, self.opt_state, self.tx,
                        self.dis, self.un, embedding, label, unlabeled_embedding)
                llk_val += llk_val_batch

                joint_llk_batch, cond_llk_batch, unlabeled_llk_batch = llk_breakdown
                joint_llk_val += joint_llk_batch
                cond_llk_val += cond_llk_batch
                unlabeled_llk_val += unlabeled_llk_batch

            valid_acc = self.evaluate(valid_loader)
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                checkpoints.save_checkpoint(ckpt_dir, self.params, valid_acc,
                        prefix=f'{self.identifier}_')

            print(f'Epoch {epoch + 1}: train loss {llk_val:.2f} = gen {-joint_llk_val:.2f} ' \
                    f'+ dis {self.dis} * {-cond_llk_val:.2f} + un {self.un} * {-unlabeled_llk_val:.2f}, ' \
                    f'valid accuracy {valid_acc}')


    def mark_adapt(self, deg: float, lr: float, epochs: int) -> None:
        self.adapt_deg = deg
        self.adapt_lr = lr
        self.adapt_epochs = epochs

        self.tx = optax.adam(learning_rate=lr)
        self.opt_state = self.tx.init(self.params)
        self.adapted = f'ADAPTED_deg{deg}_lr{lr}_epc{epochs}_'


    def adapt(self, ckpt_dir: str, unlabeled_loader: DataLoader, cheat_loader: Optional[DataLoader] = None) -> None:
        for epoch in range(self.adapt_epochs):
            llk_val = 0
            for X, _ in unlabeled_loader:
                unlabeled_image = jnp.array(X)
                unlabeled_embedding = self.embedder(unlabeled_image)
                llk_val_batch, self.params, self.opt_state = adapt_step(self.params, self.opt_state, self.tx, unlabeled_embedding)
                llk_val += llk_val_batch

            checkpoints.save_checkpoint(ckpt_dir, self.params, epoch, prefix=f'{self.identifier}_')

            cheat_acc = self.evaluate(cheat_loader) if cheat_loader is not None else float('nan')

            print(f'Epoch {epoch + 1}: train loss {llk_val:.2f}, ' \
                  f'cheat accuracy {cheat_acc}')


    def evaluate(self, loader: DataLoader) -> float:
        correct_cases = 0
        total_cases = 0
        for X, y in loader:
            image = jnp.array(X)
            embedding = self.embedder(image)
            label = jnp.array(y)
            correct_cases += valid_step(self.params, embedding, label)
            total_cases += embedding.shape[0]

        return correct_cases/total_cases


    def _init_params(self, labeled_loader: DataLoader) -> None:
        # Initialize cluster means with K-means++
        gm = [GaussianMixture(self.K, max_iter=0, init_params='k-means++', random_state=42)
                for _ in range(self.C)]

        # Initialize cluster covariance with (pooled) empirical covariance matrix
        class_sum = np.zeros((self.C, self.D))
        class_outer_sum = np.zeros((self.C, self.D, self.D))
        class_count = np.zeros(self.C)
        for X, y in labeled_loader:
            image = jnp.array(X)
            embedding = self.embedder(image)
            label = jnp.array(y)
            for cls in range(self.C):
                mask = label == cls
                samples = np.sum(mask)
                cls_batch = embedding[mask, :]

                if samples >= self.K:
                    gm[cls].fit(cls_batch)

                class_sum[cls, :] += np.sum(cls_batch, axis=0)
                class_outer_sum[cls] += np.sum(cls_batch[:, :, np.newaxis] * cls_batch[:, np.newaxis, :], axis=0)
                class_count[cls] += samples

        # Cov(X) = E(X^T @ X) - E(X)^T @ E(X)
        class_mean = class_sum / class_count[:, np.newaxis]
        Sigma = class_outer_sum / class_count[:, np.newaxis, np.newaxis] - class_mean[:, :, np.newaxis] * class_mean[:, np.newaxis, :]

        rng = np.random.default_rng(42)
        mu = rng.normal(size=(self.C, self.K, self.D))
        for cls in range(self.C):
            if hasattr(gm[cls], 'means_'):
                mu[cls, :, :] = gm[cls].means_

        w, v = np.linalg.eigh(Sigma)
        w = np.maximum(w, 0)                # avoid numerical issue
        sigval = np.sqrt(w[:, -self.R:])    # numpy sorts eigenvalues in ascending order
        S = sigval[:, np.newaxis, :] * v[:, :, -self.R:]
        residual = Sigma - jax.vmap(jnp.matmul)(S, S.swapaxes(-1, -2))
        S = np.repeat(S[:, np.newaxis, :, :], self.K, axis=1)

        if self.init == 'zero':
            Psi_softplus = np.zeros((self.C, self.D))
        elif self.init == 'roth':
            Psi_softplus = residual.diagonal(axis1=-2, axis2=-1)
        elif self.init == 'full':
            Psi_softplus = Sigma.diagonal(axis1=-2, axis2=-1)
        elif self.init == 'half':
            Psi_softplus = Sigma.diagonal(axis1=-2, axis2=-1) / 2
        else:
            raise ValueError(f'Unknown Initialization scheme {self.init}')

        Psi_softplus = np.repeat(Psi_softplus[:, np.newaxis, :], self.K, axis=1)

        self.params['mu'] = jnp.array(mu)
        self.params['Psi_softplus'] = jnp.array(Psi_softplus)
        self.params['S'] = jnp.array(S)
