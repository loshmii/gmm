from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from matplotlib import pyplot as plt

from gmm.core import (
    logsumexp,
    log_mvn_numerics,
    log_mvn_from_cholesky,
)
from gmm.utils import RNGManager


def _init_params(X: jax.Array, K: int, rng_key: jax.Array):
    return _init_params_jitted(X, K, rng_key)


@partial(jax.jit, static_argnames=("K",))
def _init_params_jitted(X: jax.Array, K: int, rng_key: jax.Array):
    N, D = X.shape
    pi = jnp.ones(K) / K
    means = X[jax.random.choice(rng_key, N, shape=(K,), replace=False)]
    convs = jnp.broadcast_to(jnp.eye(D)[None, :, :], (K, D, D))
    return pi, means, convs


@jax.jit
def _e_step(
    X: jax.Array,
    pi: jax.Array,
    means: jax.Array,
    convs: jax.Array,
):
    log_probs = jax.vmap(
        lambda x: jax.vmap(log_mvn_numerics, in_axes=(None, 0, 0))(x, means, convs)
    )(X)
    log_pi = jnp.expand_dims(jnp.log(pi + 1e-6), axis=0)
    resp = log_probs + log_pi
    resp -= logsumexp(resp, axis=-1, keepdims=True)
    return jnp.exp(resp)


@jax.jit
def _m_step(X: jax.Array, responsibilities: jax.Array):
    responsibilities = jnp.clip(responsibilities, min=1e-10, max=1.0 - 1e-10)
    responsibilities = responsibilities / jnp.sum(
        responsibilities, axis=-1, keepdims=True
    )

    pi = jnp.sum(responsibilities, axis=0) / responsibilities.shape[0]
    pi = jnp.clip(pi, min=1e-10, max=1.0 - 1e-10)
    pi = pi / jnp.sum(pi)

    means = jnp.einsum("nk, nd -> kd", responsibilities, X)
    means /= jnp.sum(responsibilities, axis=0, keepdims=False)[:, None]

    diff = X[:, None, :] - means[None, :, :]
    covs = jnp.einsum("nk, nkd, nkf -> kdf", responsibilities, diff, diff)
    covs /= jnp.sum(responsibilities, axis=0)[:, None, None]

    traces = jnp.trace(covs, axis1=-2, axis2=-1)
    jitter = traces[:, None, None] * 1e-6
    covs = covs + jitter
    return pi, means, covs


@jax.jit
def _negative_loglikelihood(
    X: jax.Array, means: jax.Array, covs: jax.Array, pi: jax.Array
) -> jax.Array:
    jittered_cov = covs + 1e-6 * jnp.eye(covs.shape[-1])
    L = jnp.linalg.cholesky(jittered_cov)
    log_probs = log_mvn_from_cholesky(X, means, L)
    pi = jnp.clip(pi, min=1e-6, max=1.0 - 1e-6)
    pi /= jnp.sum(pi)
    log_pi = jnp.log(pi)
    loss = jnp.sum(logsumexp(log_probs + log_pi, axis=-1))
    return -loss


class GaussianMixture:
    """Fast EM-based Gassuain Mixture Model implementation in JAX."""

    pi: jax.Array | None
    means: jax.Array | None
    covs: jax.Array | None
    loss_trace: jax.Array | None

    def __init__(
        self,
        n_components: int,
        *,
        tol: float = 1e-6,
        max_iter: int = 200,
        rng_key: int | jax.Array = 0,
        covariance_regulazier: float = 1e-6,
        assume_full_cov: bool = True,
    ) -> None:
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.covariance_regulazier = covariance_regulazier
        self.assume_full_cov = assume_full_cov
        self.pi = None
        self.means = None
        self.covs = None
        self.loss_trace = None
        self._rng = (
            RNGManager(int(rng_key)) if isinstance(rng_key, int) else RNGManager(0)
        )
        if not isinstance(rng_key, int):
            self._rng.key = rng_key
        self._X = None

    def fit(self, X: jax.Array) -> "GaussianMixture":
        """Fit the model to data ``X`` using the EM algorithm."""
        X = jnp.asarray(X, dtype=jnp.float64)
        self._X = X
        self.pi, self.means, self.covs = _init_params(
            X, self.n_components, self._rng.new_key()
        )
        loss = float(_negative_loglikelihood(X, self.means, self.covs, self.pi))
        losses = [loss]
        for _ in range(self.max_iter):
            resp = _e_step(X, self.pi, self.means, self.covs)
            self.pi, self.means, self.covs = _m_step(X, resp)
            new_loss = float(_negative_loglikelihood(X, self.means, self.covs, self.pi))
            losses.append(new_loss)
            if abs(new_loss - loss) < self.tol:
                break
            loss = new_loss
        self.loss_trace = jnp.array(losses)
        return self

    def predict(self, X: jax.Array) -> jax.Array:
        """Return clusted assignment for ``X``"""
        return jnp.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: jax.Array) -> jax.Array:
        """Return component probabilities for ``X``"""
        X = jnp.asarray(X, dtype=jnp.float64)
        resp = _e_step(X, self.pi, self.means, self.covs)
        return resp

    def score_samples(self, X: jax.Array) -> jax.Array:
        """Log-likelihood of each sampler under the model"""

        X = jnp.asarray(X, dtype=jnp.float64)

        jittered_cov = self.covs + 1e-6 * jnp.eye(self.covs.shape[-1])
        L = jnp.linalg.cholesky(jittered_cov)

        log_probs = log_mvn_from_cholesky(X, self.means, L)

        pi = jnp.clip(self.pi, 1e-16, 1.0 - 1e-6)
        pi = pi / jnp.sum(pi)
        log_pi = jnp.log(pi)
        return logsumexp(log_probs + log_pi, axis=-1)

    def plot(self, iter_losses: bool = True, cluster_scatter: bool = False):
        """Helper to visualize training progress"""
        if iter_losses and self.loss_trace is not None:
            plt.figure()
            plt.plot(self.loss_trace)
            plt.xlabel("Iteration")
            plt.ylabel("Negative Log-Likelihood")
            plt.title("EM Convergence")
            plt.show()
        if cluster_scatter and self.means is not None and self.covs is not None:
            if self._X is None or self._X.shape[1] != 2:
                raise ValueError("Cluster scatter plot requires 2D data.")
            labels = self.predict(self._X)
            plt.figure()
            plt.scatter(self._X[:, 0], self._X[:, 1], c=labels, s=10, alpha=0.6)
            plt.scatter(self.means[:, 0], self.means[:, 1], c="red", marker="x")
            plt.title("GMM Clustering")
            plt.show()
