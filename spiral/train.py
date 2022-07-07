from sklearn.mixture import GaussianMixture

import jax
import jax.numpy as jnp
import numpy as np

from .datagen import get_spiral
from models.gmm import GMM
from models.tent import Tent


if __name__ == '__main__':
    epochs = 10001
    report_every = 500
    batches_per_epoch = 64
    labelled_total, unlabelled_total, valid_total = 64, 4096, 128

    key = jax.random.PRNGKey(42)
    labelled_key, unlabelled_key, valid_key, key = jax.random.split(key, 4)
    X, y = get_spiral(labelled_key, labelled_total, 0.5)
    unlabelled, _ = get_spiral(unlabelled_key, unlabelled_total, 0.5)
    X_valid, y_valid = get_spiral(valid_key, valid_total, 0.5)


    # print(f'=== MLP + Tent ===')
    #
    # params_key, key = jax.random.split(key)
    # print('X.shape', X.shape)
    # tent = Tent(params_key, 2, 5e-4, 1e-5, jnp.empty(X.shape[1:]))
    # tent.fit(sample_total, batch_total, report_every, X, y, unlabelled, X_valid, y_valid)


    print(f'=== GMM ===')

    C, K, D, R = 2, 7, 2, 2
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
        gmm_model.fit(epochs, report_every, batches_per_epoch, X, y, unlabelled, X_valid, y_valid)
