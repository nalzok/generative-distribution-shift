from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax

import flax
from flax.training import train_state
from flax.struct import field

from .resnet import ResNet18


class TrainState(train_state.TrainState):
    tent_step: int
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]
    tent_tx: optax.GradientTransformation = field(pytree_node=False)
    tent_opt_state: optax.OptState

    def tent_apply_gradients(self, *, tent_grads, **kwargs):
        updates, new_tent_opt_state = self.tent_tx.update(
            tent_grads, self.tent_opt_state, self.batch_stats)
        new_batch_state = optax.apply_updates(self.batch_stats, updates)
        return self.replace(
            tent_step=self.tent_step + 1,
            batch_state=new_batch_state,
            tent_opt_state=new_tent_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, batch_stats, tent_tx, **kwargs):
        state = super(TrainState, cls).create(apply_fn=apply_fn, params=params, tx=tx)

        tent_opt_state = tent_tx.init(batch_stats)
        return cls(
            tent_step=0,
            batch_stats=batch_stats,
            tent_tx=tent_tx,
            tent_opt_state=tent_opt_state,
            **state.asdict(),
            **kwargs,
        )


def create_train_state(key, num_classes, lr, tent_lr, specimen):
    net = ResNet18(num_classes=num_classes)
    variables = net.init(key, specimen)
    tx = optax.adam(lr)
    tent_tx = optax.adam(tent_lr)
    state = TrainState.create(
            apply_fn=net.apply,
            params=variables['params'],
            tx=tx,
            batch_stats=variables['batch_stats'],
            tent_tx=tent_tx
    )

    return state


@jax.jit
def train_step(state, image, label):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables, image, train=True, mutable=['batch_stats']
        )
        loss = optax.softmax_cross_entropy(logits, label)
        hit = jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(label, axis=-1))
        return loss.sum(), new_model_state, hit

    (loss, new_model_state, hit), grads = loss_fn(state.params)
    accuracy = hit/label.shape[0]

    state = state.apply_gradients(
            grads=grads,
            batch_stats=new_model_state['batch_stats']
    )

    return state, loss, accuracy


@jax.jit
def test_step(state, image, label):
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits = state.apply_fn(
            variables, image, train=False, mutable=False
        )
        loss = optax.softmax_cross_entropy(logits, label)
        hit = jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(label, axis=-1))
        return loss.sum(), hit

    @jax.grad
    def entropy_fn(batch_stats):
        variables = {'batch_stats': batch_stats, 'params': state.params}
        logits = state.apply_fn(variables, image)
        entropy = jnp.sum(jsp.special.entr(logits), axis=-1)
        return entropy

    loss, hit = loss_fn(state.params)
    accuracy = hit/label.shape[0]

    tent_grads = entropy_fn(state.params)
    state = state.tent_apply_gradients(tent_grads=tent_grads)

    return state, loss, accuracy


class MLP:
    def __init__(self, key, num_classes, lr, tent_lr, specimen):
        self.state = create_train_state(key, num_classes, lr, tent_lr, specimen)


    def fit(self, sample_size, batch_size, report_every, X, y, unlabeled, X_valid, y_valid):
        for i in range(10001):
            llk_val = 0
            for batch_id in range(0, sample_size, batch_size):
                slice_batch = slice(batch_id, batch_id+batch_size)
                X_batch, y_batch, unlabeled_batch = X[slice_batch], y[slice_batch], unlabeled[slice_batch]
                llk_val_batch, self.params, self.opt_state = train_step(self.params, X_batch, y_batch)
                llk_val += llk_val_batch

            if i % report_every == 0:
                correct_cases = self.evaluate(X_valid, y_valid)
                print(f'Iteration {i}: train loss {llk_val}, validation accuracy {correct_cases/sample_size}')


    def evaluate(self, X_valid, y_valid):
        correct_cases = test_step(self.params, X_valid, y_valid)
        return correct_cases
