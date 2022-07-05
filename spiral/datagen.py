import jax
import jax.numpy as jnp
import numpy as np


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


