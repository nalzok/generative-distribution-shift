from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from flax.training import checkpoints
import torch
from torch.utils.data import DataLoader

from sklearn.mixture import GaussianMixture
from tqdm import tqdm


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
def objective_hybrid(params, X, y, lambda_, unlabeled, kappa):
    joint_llk, cond_llk = llk_hybrid(params, X, y)
    unlabeled_llk = llk_unlabeled(params, unlabeled)

    joint_llk_batch = jnp.sum(joint_llk) / X.shape[1]
    cond_llk_batch = jnp.sum(cond_llk)
    unlabeled_llk_batch = jnp.sum(unlabeled_llk) / unlabeled.shape[0] * X.shape[0]
    llk = lambda_ * (kappa * joint_llk_batch + (1 - kappa) * unlabeled_llk_batch) + (1 - lambda_) * cond_llk_batch

    return -llk, (joint_llk_batch, cond_llk_batch, unlabeled_llk_batch)


@partial(jax.jit, static_argnums=(2,))
def train_step(params, opt_state, tx, X, y, lambda_, unlabeled, kappa):
    (llk_val, llk_breakdown), grads = objective_hybrid(params, X, y, lambda_, unlabeled, kappa)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return llk_val, params, opt_state, llk_breakdown


@partial(jax.value_and_grad)
def objective_unlabeled(params, unlabeled):
    unlabeled_llk = llk_unlabeled(params, unlabeled)
    unlabeled_llk_batch = jnp.sum(unlabeled_llk)

    return -unlabeled_llk_batch


@partial(jax.jit, static_argnums=(2,))
def adapt_step(params, opt_state, tx, unlabeled):
    llk_val, grads = objective_unlabeled(params, unlabeled)
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
    def __init__(self, C: int, K: int, D: int, R: int, init_scheme: str, lr: float, lambda_: float, kappa: float,
            metadata: str, ckpt_dir_in: Optional[str] = None, adapt_lr: Optional[float] = None):
        self.C = C
        self.K = K
        self.D = D
        self.R = R
        self.init_scheme = init_scheme
        self.lambda_ = lambda_
        self.kappa = kappa
        self.load_prefix = f'gmm_{metadata}_{init_scheme}_K{K}_R{R}_gmmlr{lr}_lambda{lambda_}_kappa{kappa}_'

        self.params = init_gmm(C, K, D, R)
        if ckpt_dir_in is not None:
            restored = checkpoints.restore_checkpoint(ckpt_dir_in, self.params, prefix=self.load_prefix)
            if restored is self.params:
                raise ValueError(f'Cannot find checkpoint {self.load_prefix}X')
            self.params = restored

        self.store_prefix = self.load_prefix
        if adapt_lr is None:
            self.tx = optax.adam(learning_rate=lr)
        else:
            self.store_prefix = f'{self.load_prefix}alr{adapt_lr}_'
            self.tx = optax.adam(learning_rate=adapt_lr)
        self.opt_state = self.tx.init(self.params)


    def fit(self, ckpt_dir_out: str, epochs: int, labeled_loader: DataLoader, unlabeled_loader: DataLoader,
            valid_loader: Optional[DataLoader] = None):
        if valid_loader is None:
            valid_loader = labeled_loader

        self._init_params(labeled_loader)

        for epoch in range(epochs):
            llk_val = 0
            joint_llk_val = 0
            cond_llk_val = 0
            unlabeled_llk_val = 0
            pbar = tqdm(zip(labeled_loader, unlabeled_loader))
            for (X_batch, y_batch), (unlabeled_batch, _) in pbar:
                X_batch = jnp.array(X_batch)
                y_batch = jnp.array(y_batch)
                unlabeled_batch = jnp.array(unlabeled_batch)
                llk_val_batch, self.params, self.opt_state, llk_breakdown = train_step(self.params, self.opt_state, self.tx,
                        X_batch, y_batch, self.lambda_, unlabeled_batch, self.kappa)
                llk_val += llk_val_batch

                joint_llk_batch, cond_llk_batch, unlabeled_llk_batch = llk_breakdown
                joint_llk_val += joint_llk_batch
                cond_llk_val += cond_llk_batch
                unlabeled_llk_val += unlabeled_llk_batch

                pbar.set_description(f"{llk_val_batch.item()=:.2f}")

            correct_cases, total_cases = self.evaluate(valid_loader)

            joint_llk_term = self.lambda_ * self.kappa * -joint_llk_val
            cond_llk_term = (1 - self.lambda_) * -cond_llk_val
            unlabeled_llk_term = self.lambda_ * (1 - self.kappa) * -unlabeled_llk_val
            print(f'GMM: epoch {epoch + 1}: train loss {llk_val:.2f} ' \
                    f'= joint {joint_llk_term:.2f} + unlabeled {unlabeled_llk_term:.2f} + cond {cond_llk_term:.2f}, ' \
                  f'valid accuracy {100*correct_cases/total_cases:.2f}%')

            checkpoints.save_checkpoint(ckpt_dir_out, self.params, correct_cases/total_cases,
                    prefix=self.store_prefix, overwrite=True)


    def adapt(self, ckpt_dir_out: str, epochs: int, unlabeled_loader: DataLoader, valid_loader: Optional[DataLoader] = None):
        for epoch in range(epochs):
            llk_val = 0
            pbar = tqdm(unlabeled_loader)
            for unlabeled_batch, _ in pbar:
                unlabeled_batch = jnp.array(unlabeled_batch)
                llk_val_batch, self.params, self.opt_state = adapt_step(self.params, self.opt_state, self.tx, unlabeled_batch)
                llk_val += llk_val_batch

                pbar.set_description(f"{llk_val_batch.item()=:.2f}")

            if valid_loader is not None:
                correct_cases, total_cases = self.evaluate(valid_loader)
            else:
                correct_cases, total_cases = float('nan'), float('nan')

            print(f'GMM (adapted): epoch {epoch + 1}: train loss {llk_val:.2f}, ' \
                  f'valid accuracy {100*correct_cases/total_cases:.2f}%')

            checkpoints.save_checkpoint(ckpt_dir_out, self.params, correct_cases/total_cases,
                    prefix=self.store_prefix, overwrite=True)


    def evaluate(self, valid_loader: DataLoader):
        correct_cases = 0
        total_cases = 0
        for X_batch, y_batch in valid_loader:
            X_batch = jnp.array(X_batch)
            y_batch = jnp.array(y_batch)
            correct_cases += valid_step(self.params, X_batch, y_batch)
            total_cases += X_batch.shape[0]

        return correct_cases, total_cases


    def _init_params(self, labeled_loader: DataLoader):
        # Initialize cluster means with K-means++
        gm = [GaussianMixture(self.K, max_iter=0, init_params='k-means++', random_state=42)
                for _ in range(self.C)]

        # Initialize cluster covariance with (pooled) empirical covariance matrix
        class_sum = np.zeros((self.C, self.D))
        class_outer_sum = np.zeros((self.C, self.D, self.D))
        class_count = np.zeros(self.C)
        for X_batch, y_batch in labeled_loader:
            for cls in range(self.C):
                mask = y_batch == cls
                samples = torch.sum(mask)
                cls_batch = X_batch[mask, :]

                if samples >= self.K:
                    gm[cls].fit(cls_batch)

                class_sum[cls, :] += np.sum(cls_batch, axis=0)
                class_outer_sum[cls] += np.sum(cls_batch[:, :, np.newaxis] * cls_batch[:, np.newaxis, :], axis=0)
                class_count[cls] += samples

        # Cov(X) = E(X^T @ X) - E(X)^T @ E(X)
        class_mean = class_sum / class_count[:, np.newaxis]
        Sigma = class_outer_sum / class_count[:, np.newaxis] - class_mean[:, :, np.newaxis] * class_mean[:, np.newaxis, :]

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

        if self.init_scheme == 'zero':
            Psi_softplus = np.zeros((self.C, self.D))
        elif self.init_scheme == 'roth':
            Psi_softplus = residual.diagonal(axis1=-2, axis2=-1)
        elif self.init_scheme == 'full':
            Psi_softplus = Sigma.diagonal(axis1=-2, axis2=-1)
        elif self.init_scheme == 'half':
            Psi_softplus = Sigma.diagonal(axis1=-2, axis2=-1) / 2
        else:
            raise ValueError(f'Unknown Initialization scheme {self.init_scheme}')

        Psi_softplus = np.repeat(Psi_softplus[:, np.newaxis, :], self.K, axis=1)

        self.params['mu'] = jnp.array(mu)
        self.params['Psi_softplus'] = jnp.array(Psi_softplus)
        self.params['S'] = jnp.array(S)
