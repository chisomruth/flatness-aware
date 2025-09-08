import jax
import jax.numpy as jnp
from jax import grad, random
from typing import Dict


def cross_entropy_loss(params: Dict, model_fn, x: jnp.ndarray, y: jnp.ndarray):
    logits = model_fn(params, x)
    y_onehot = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(y_onehot * jax.nn.log_softmax(logits), axis=-1))


def hutchinson_trace_estimator(params: Dict, loss_fn, x: jnp.ndarray, y: jnp.ndarray,
                               num_samples: int, key: jax.random.PRNGKey):

    grad_fn = grad(loss_fn, argnums=0)  # grad wrt params

    # Compute initial gradient to know size
    g = grad_fn(params, x, y)
    g_flat, unflatten_fn = jax.flatten_util.ravel_pytree(g)

    def hvp_fn(v: jnp.ndarray) -> jnp.ndarray:
        """Hessian-vector product"""
        def g_dot_v(p):
            g_params = grad_fn(p, x, y)
            g_flat, _ = jax.flatten_util.ravel_pytree(g_params)
            return jnp.dot(g_flat, v)

        hvp = grad(g_dot_v)(params)
        hvp_flat, _ = jax.flatten_util.ravel_pytree(hvp)
        return hvp_flat

    # Sample random vectors and compute HVPs
    keys = random.split(key, num_samples)
    total_hvp_norm_sq = 0.0

    for i in range(num_samples):
        v = random.normal(keys[i], g_flat.shape)
        hvp = hvp_fn(v)
        total_hvp_norm_sq += jnp.sum(hvp ** 2)

    return total_hvp_norm_sq / num_samples


def ca_loss_fn(params: Dict, model_fn, x: jnp.ndarray, y: jnp.ndarray,
               lambda_reg: float, num_hutchinson_samples: int, key: jax.random.PRNGKey):
    task_loss = cross_entropy_loss(params, model_fn, x, y)

    if lambda_reg == 0.0:
        return task_loss

    def task_loss_fn(p, xb, yb):
        return cross_entropy_loss(p, model_fn, xb, yb)

    curvature_penalty = hutchinson_trace_estimator(
        params, task_loss_fn, x, y, num_hutchinson_samples, key
    )

    # Normalize by batch size
    batch_size = x.shape[0]
    curvature_penalty = curvature_penalty / batch_size

    total_loss = task_loss + lambda_reg * curvature_penalty
    return total_loss
