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


@jax.jit
def llk_hybrid(params, X, y, lambda_):
    """
    Joint log-likelihood when lambda_ = 0
    Conditional log-likelihood when lambda_ = 1
    """
    pi = jax.nn.softmax(params['pi_logit'])
    likelihood_class = lk_class(params, X)
    llk = jnp.log(pi[y] * likelihood_class[jnp.arange(y.size), y]) - lambda_ * jnp.log(jnp.sum(pi * likelihood_class, axis=-1))

    return llk


def llk_unlabelled(params, unlabelled, kappa):
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
    llk = kappa * jnp.log(jnp.sum(pi * likelihood_class_unlabelled, axis=-1))

    return llk


@jax.value_and_grad
def objective_hybrid(params, X, y, lambda_, unlabelled, kappa):
    llk = llk_hybrid(params, X, y, lambda_) + llk_unlabelled(params, unlabelled, kappa)
    return -jnp.sum(llk)


@partial(jax.jit, static_argnums=2)
def train_step(params, opt_state, tx, X, y, lambda_, unlabelled, kappa):
    llk_val, grads = objective_hybrid(params, X, y, lambda_, unlabelled, kappa)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return llk_val, params, opt_state


@jax.jit
def test_step(params, X, y):
    likelihood_class = lk_class(params, X)
    predictions = jnp.argmax(likelihood_class, axis=-1)
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


    def fit(self, sample_size, batch_size, report_every, X, y, unlabelled, X_valid, y_valid):
        for i in range(10001):
            llk_val = 0
            for batch_id in range(0, sample_size, batch_size):
                slice_batch = slice(batch_id, batch_id+batch_size)
                X_batch, y_batch, unlabelled_batch = X[slice_batch], y[slice_batch], unlabelled[slice_batch]
                llk_val_batch, self.params, self.opt_state = train_step(self.params, self.opt_state, self.tx, X_batch, y_batch, self.lambda_, unlabelled_batch, self.kappa)
                llk_val += llk_val_batch

            if i % report_every == 0:
                correct_cases = self.evaluate(X_valid, y_valid)
                print(f'Iteration {i}: train loss {llk_val}, validation accuracy {100*correct_cases/sample_size:.2f}%')


    def evaluate(self, X_valid, y_valid):
        correct_cases = test_step(self.params, X_valid, y_valid)
        return correct_cases
