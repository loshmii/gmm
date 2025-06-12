import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnums=(1, 2))
def logsumexp(arr: jax.Array, axis: int = -1, keepdims: bool = False) -> jax.Array:
    max_arr = jnp.max(arr, axis=axis, keepdims=True)
    inter_arr = jnp.exp(arr - max_arr)
    inter_arr = jnp.sum(inter_arr, axis=axis, keepdims=True)
    inter_arr = jnp.log(inter_arr)
    inter_arr = jnp.add(inter_arr, max_arr)
    if not keepdims:
        inter_arr = jnp.squeeze(inter_arr, axis=axis)
    return inter_arr


@jax.jit
def log_mvn_numerics(x: jax.Array, mean: jax.Array, cov: jax.Array) -> jax.Array:
    jitter = 1e-6
    features = x.shape[-1]
    cov = cov + jnp.eye(features) * jitter
    L = jnp.linalg.cholesky(cov)
    diag = jnp.diagonal(L, axis1=-2, axis2=-1)
    logdet = 2 * jnp.sum(jnp.log(diag), axis=-1)
    diff = x - mean
    rht = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    quad = jnp.sum(rht * rht, axis=-1)
    return -0.5 * (quad + logdet + features * jnp.log(2 * jnp.pi))


def log_mvn_from_cholesky(X: jax.Array, mean: jax.Array, L: jax.Array) -> jax.Array:
    dim = X.shape[-1]
    logdet = 2 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    diff = X[..., None, :] - mean
    broadcasted_L = jnp.broadcast_to(L, diff.shape[:-1] + (L.shape[-2], L.shape[-1]))
    rht = jax.scipy.linalg.solve_triangular(broadcasted_L, diff, lower=True)
    mahalanobis = jnp.sum(rht * rht, axis=-1)
    return -0.5 * (mahalanobis + logdet + dim * jnp.log(2 * jnp.pi))
