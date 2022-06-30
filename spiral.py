import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax


def get_spiral(key, num_samples, noise_level):
    """
    Adapted from

        https://github.com/tensorflow/playground/blob/02469bd3751764b20486015d4202b792af5362a6/src/dataset.ts#L136-L154
    """
    n = num_samples // 2
    neg_key, pos_key = jax.random.split(key, 2)

    def gen_spiral(key, delta_t):
        i = jnp.arange(n)
        r = i / n * 5
        t = 1.75 * i / n * (2 * np.pi) + delta_t
        X = r[:, np.newaxis] * jnp.column_stack([jnp.sin(t), jnp.cos(t)])
        noise = noise_level * jax.random.uniform(key, (n, 2), minval=-1, maxval=1)
        return X + noise

    X = jnp.concatenate([gen_spiral(neg_key, 0), gen_spiral(pos_key, np.pi)])
    y = jnp.concatenate([np.repeat(0, n), np.repeat(1, n)])

    return X, y


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
def llk_hybrid(params, X, y, regularizer):
    """
    Joint log-likelihood when regularizer = 0
    Conditional log-likelihood when regularizer = 1
    """
    pi = jax.nn.softmax(params['pi_logit'])
    alpha = jax.nn.softmax(params['alpha_logit'])
    mu = params['mu']
    Psi = jax.nn.softplus(params['Psi_softplus'])
    S = params['S']

    _, _, D, _ = S.shape
    Psi = jax.vmap(jax.vmap(jnp.diag))(Psi)
    Sigma = jax.numpy.eye(D) + Psi + jax.vmap(jax.vmap(jnp.matmul))(S, S.swapaxes(-1, -2))

    mvn = jax.vmap(jax.vmap(jsp.stats.multivariate_normal.pdf, (None, 0, 0)), (None, 0, 0))
    likelihood_cluster = mvn(X, mu, Sigma).transpose((2, 0, 1))
    likelihood_class = jnp.sum(alpha * likelihood_cluster, axis=-1)
    llk = jnp.log(pi[y] * likelihood_class[jnp.arange(y.size), y]) - regularizer * jnp.log(jnp.sum(pi * likelihood_class, axis=-1))

    return llk


@jax.value_and_grad
def objective_hybrid(params, X, y, regularizer):
    llk = llk_hybrid(params, X, y, regularizer)
    return -jnp.sum(llk)


@jax.jit
def train_step(params, opt_state, X, y, regularizer):
    llk_val, grads = objective_hybrid(params, X, y, regularizer)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return llk_val, params, opt_state


config = {
    'generative': 0,
    'hybrid': 0.5,
    'discriminative': 1,
}

if __name__ == '__main__':
    C, K, D, R = 2, 7, 2, 2
    key = jax.random.PRNGKey(42)

    data_key, key = jax.random.split(key)
    X, y = get_spiral(data_key, 500, 0.5)

    gm = GaussianMixture(K, max_iter=0, random_state=42)
    mu = np.empty((C, K, D))
    for cls in range(C):
        gm.fit(X[y == cls, :])
        mu[cls, :] = gm.means_
    mu = jnp.array(mu)

    for reg_name, regularizer in config.items():
        print(f'=== {reg_name} ===')

        params_key, key = jax.random.split(key)
        params = init_params(params_key, C, K, D, R)

        params_key, key = jax.random.split(key)
        params['mu'] = mu

        xmin, xmax = X[:, 0].min()-1, X[:, 0].max()+1
        ymin, ymax = X[:, 1].min()-1, X[:, 1].max()+1
        xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
        xnew = np.c_[xx.ravel(), yy.ravel()]

        tx = optax.adam(learning_rate=5e-4)
        opt_state = tx.init(params)

        report_every = 500
        for i in range(10001):
            llk_val, params, opt_state = train_step(params, opt_state, X, y, regularizer)
            if i % report_every == 0:
                print(f'Iteration {i}: {llk_val}')

                fig = plt.figure(figsize=(12, 8))
                axes = [
                    fig.add_subplot(2, 2, 2),
                    fig.add_subplot(2, 2, 4),
                    fig.add_subplot(1, 2, 1),
                ]
                fig.set_tight_layout(True)
                plt.set_cmap('binary_r')
                
                centroids = params['mu']
                alpha = jax.nn.softmax(params['alpha_logit'])
                llk = np.empty((C, xnew.shape[0]))
                for cls in range(C):
                    llk[cls, :] = llk_hybrid(params, xnew, cls * jnp.ones(xnew.shape[:1], dtype=int), regularizer)
                    axes[cls].pcolormesh(xx, yy, llk[cls, :].reshape(xx.shape))
                    axes[cls].scatter(X[y == cls, 0], X[y == cls, 1], marker='.')
                    axes[cls].scatter(centroids[cls, :, 0], centroids[cls, :, 1], marker='*', s=15*K*alpha[cls, :])

                plt.set_cmap('coolwarm')
                llk_diff = (llk[1, :] - llk[0, :]).reshape(xx.shape)
                magnitude = min(np.max(llk_diff), -np.min(llk_diff))
                axes[2].pcolormesh(xx, yy, llk_diff, vmin=-magnitude, vmax=magnitude)
                axes[2].scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')

                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                fig.suptitle(reg_name)

                fig.savefig(f'plots/spiral_{reg_name}_{i//report_every}.png', dpi=100, bbox_inches='tight')
                plt.close()

