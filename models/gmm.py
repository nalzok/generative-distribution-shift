from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax


def init_params(key, C, K, D, R):
    """
    Accept the same input as https://github.com/wroth8/hybrid-gmm/blob/master/models/GMMClassifier.py

    C: Number of classes
    K: List of length C containing the number of components per class
    D: Number of input features
    R: Number of dimensions of the low-rank approximation of the DPLR matrix structure.
    """
    pi_logit = jnp.zeros((C,))
    alpha_logit = jnp.zeros((C, K))
    mu = 20 * jax.random.normal(key, shape=(C, K, D))
    Psi_softplus = jnp.ones((C, K, D))
    S = jnp.zeros((C, K, D, R))

    params = {
        'pi_logit': pi_logit,
        'alpha_logit': alpha_logit,
        'mu': mu,
        'Psi_softplus': Psi_softplus,
        'S': S,
    }
    return params


@jax.jit
def lk_class(params, X):
    alpha = jax.nn.softmax(params['alpha_logit'])
    mu = params['mu']
    Psi = jax.nn.softplus(params['Psi_softplus'])
    S = params['S']

    _, _, D, _ = S.shape
    Psi = jax.vmap(jax.vmap(jnp.diag))(Psi)
    Sigma = jax.numpy.eye(D) + Psi + jax.vmap(jax.vmap(jnp.matmul))(S, S.swapaxes(-1, -2))

    likelihood_cluster = jsp.stats.multivariate_normal.pdf(X[:, np.newaxis, np.newaxis, :], mu, Sigma)
    likelihood_class = jnp.sum(alpha * likelihood_cluster, axis=-1)

    return likelihood_class


def llk_hybrid(params, X, y):
    """
    Joint log-likelihood when lambda_ = 0
    Conditional log-likelihood when lambda_ = 1
    """
    pi = jax.nn.softmax(params['pi_logit'])
    likelihood_class = lk_class(params, X)

    joint_llk = jnp.log(pi[y] * likelihood_class[jnp.arange(y.size), y])
    marginal_llk = jnp.log(jnp.sum(pi * likelihood_class, axis=-1))
    cond_llk = joint_llk - marginal_llk

    return joint_llk, cond_llk


def llk_unlabelled(params, unlabelled):
    pi = jax.nn.softmax(params['pi_logit'])
    alpha = jax.nn.softmax(params['alpha_logit'])
    mu = params['mu']
    Psi = jax.nn.softplus(params['Psi_softplus'])
    S = params['S']

    _, _, D, _ = S.shape
    Psi = jax.vmap(jax.vmap(jnp.diag))(Psi)
    Sigma = jax.numpy.eye(D) + Psi + jax.vmap(jax.vmap(jnp.matmul))(S, S.swapaxes(-1, -2))

    likelihood_cluster_unlabelled = jsp.stats.multivariate_normal.pdf(unlabelled[:, np.newaxis, np.newaxis, :], mu, Sigma)
    likelihood_class_unlabelled = jnp.sum(alpha * likelihood_cluster_unlabelled, axis=-1)
    llk = jnp.log(jnp.sum(pi * likelihood_class_unlabelled, axis=-1))

    return llk


@jax.value_and_grad
def objective_hybrid(params, X, y, lambda_, unlabelled, kappa):
    joint_llk, cond_llk = llk_hybrid(params, X, y)
    unlabelled_llk = llk_unlabelled(params, unlabelled)
    llk = lambda_ * (kappa * jnp.sum(joint_llk) + (1 - kappa) * jnp.sum(unlabelled_llk)) + (1 - lambda_) * jnp.sum(cond_llk)

    return -llk


@partial(jax.jit, static_argnums=2)
def train_step(params, opt_state, tx, X, y, lambda_, unlabelled, kappa):
    llk_val, grads = objective_hybrid(params, X, y, lambda_, unlabelled, kappa)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    params['pi_logit'] = params['pi_logit'] - jnp.mean(params['pi_logit'], axis=-1, keepdims=True)
    params['alpha_logit'] = params['alpha_logit'] - jnp.mean(params['alpha_logit'], axis=-1, keepdims=True)

    return llk_val, params, opt_state


@jax.jit
def test_step(params, X, y):
    prior = jax.nn.softmax(params['pi_logit'])
    likelihood_class = lk_class(params, X)

    normalizer = jnp.max(likelihood_class, axis=-1)
    posterior = prior * (likelihood_class / normalizer)
    posterior /= jnp.sum(posterior, axis=-1)

    predictions = jnp.argmax(posterior, axis=-1)
    correct_cases = jnp.sum(predictions == y)
    return correct_cases


class GMM:
    def __init__(self, key, C, K, D, R, mu, lr, lambda_, kappa):
        self.params = init_params(key, C, K, D, R)
        self.params['mu'] = mu  # should be calculated with K-means++

        self.tx = optax.adam(learning_rate=lr)
        self.opt_state = self.tx.init(self.params)

        self.lambda_ = lambda_
        self.kappa = kappa


    def fit(self, epochs, report_every, batches_per_epoch, X, y, unlabelled, X_valid, y_valid):
        labelled_batch_size = (X.shape[0] + batches_per_epoch - 1) // batches_per_epoch
        unlabelled_batch_size = (unlabelled.shape[0] + batches_per_epoch - 1) // batches_per_epoch

        for i in range(epochs):
            labelled_iter = range(0, X.shape[0], labelled_batch_size)
            unlabelled_iter = range(0, unlabelled.shape[0], unlabelled_batch_size)

            llk_val = 0
            for labelled_batch_id, unlabelled_batch_id in zip(labelled_iter, unlabelled_iter):
                labelled_slice = slice(labelled_batch_id, labelled_batch_id+labelled_batch_size)
                unlabelled_slice = slice(unlabelled_batch_id, unlabelled_batch_id+unlabelled_batch_size)
                X_batch, y_batch, unlabelled_batch = X[labelled_slice], y[labelled_slice], unlabelled[unlabelled_slice]
                llk_val_batch, self.params, self.opt_state = train_step(self.params, self.opt_state, self.tx, X_batch, y_batch, self.lambda_, unlabelled_batch, self.kappa)
                llk_val += llk_val_batch

            if i % report_every == 0:
                correct_cases = self.evaluate(X_valid, y_valid)
                print(f'Iteration {i}: train loss {llk_val}, validation accuracy {100*correct_cases/X_valid.shape[0]:.2f}%')


    def evaluate(self, X_valid, y_valid):
        correct_cases = test_step(self.params, X_valid, y_valid)
        return correct_cases
