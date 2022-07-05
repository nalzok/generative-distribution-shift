from sklearn.mixture import GaussianMixture

import jax
import jax.numpy as jnp
import numpy as np

from .datagen import get_spiral
from models.gmm import GMM
from models.tent import Tent


if __name__ == '__main__':
    C, K, D, R = 2, 7, 2, 2
    sample_size = 500
    batch_size = 64
    report_every = 500

    key = jax.random.PRNGKey(42)
    labelled_key, unlabelled_key, valid_key, key = jax.random.split(key, 4)
    X, y = get_spiral(labelled_key, sample_size, 0.5)
    unlabelled, _ = get_spiral(unlabelled_key, sample_size, 0.5)
    X_valid, y_valid = get_spiral(valid_key, sample_size, 0.5)


    # print(f'=== MLP + Tent ===')
    #
    # params_key, key = jax.random.split(key)
    # print('X.shape', X.shape)
    # tent = Tent(params_key, 2, 5e-4, 1e-5, jnp.empty(X.shape[1:]))
    # tent.fit(sample_size, batch_size, report_every, X, y, unlabelled, X_valid, y_valid)


    print(f'=== GMM ===')

    gmm_config = {
        'generative': (0, 0),
        'generative_ssl': (0, 1),
        'hybrid': (0.5, 0),
        'hybrid_ssl': (0.5, 1),
        'discriminative': (1, 0),
        'discriminative_ssl': (1, 1),
    }

    gm = GaussianMixture(K, max_iter=0, random_state=42)
    mu = np.empty((C, K, D))
    for cls in range(C):
        gm.fit(X[y == cls, :])
        mu[cls, :] = gm.means_
    mu = jnp.array(mu)

    for reg_name, (lambda_, kappa) in gmm_config.items():
        print(f'--- {reg_name} ---')
        
        params_key, key = jax.random.split(key)
        gmm_model = GMM(params_key, C, K, D, R, mu, 5e-4, lambda_, kappa)
        gmm_model.fit(sample_size, batch_size, report_every, X, y, unlabelled, X_valid, y_valid)
