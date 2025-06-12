import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import jax
import jax.numpy as jnp
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture as SKGMM

from gmm.gmm import GaussianMixture


def generate_blob(key, means):
    n = 200
    assign = jax.random.randint(key, (n,), 0, len(means))
    return jax.random.normal(key, (n, means.shape[1])) + means[assign]


def test_shapes():
    key = jax.random.key(0)
    X = generate_blob(key, jnp.array([[0.0, 0.0], [2.0, 2.0]]))
    model = GaussianMixture(n_components=2, rng_key=key).fit(X)
    assert model.pi.shape == (2,)
    assert model.means.shape == (2, 2)
    assert model.covs.shape == (2, 2, 2)


def test_log_likelihood_monotone():
    key = jax.random.key(1)
    X = generate_blob(key, jnp.array([[0.0, 0.0], [3.0, 3.0]]))
    gm = GaussianMixture(n_components=2, rng_key=key).fit(X)
    diffs = jnp.diff(gm.loss_trace)
    assert jnp.all(diffs <= 0), "Loss should be non-increasing"


def test_em_convergences():
    key = jax.random.key(2)
    true_means = jnp.array([[0.0, 0.0], [4.0, 4.0]])
    X = generate_blob(key, true_means)
    gm = GaussianMixture(n_components=2, rng_key=key).fit(X)
    pred = jnp.sort(gm.means, axis=0)
    truth = jnp.sort(true_means, axis=0)
    assert jnp.allclose(pred, truth, atol=0.2), "Means should converge to true means"


def test_predict_matches_responsibility_argmax():
    key = jax.random.key(3)
    X = generate_blob(key, jnp.array([[0.0, 0.0], [1.0, 1.0]]))
    gm = GaussianMixture(n_components=2, rng_key=key).fit(X)
    pred = gm.predict(X)
    resp = gm.predict_proba(X)
    assert jnp.all(
        pred == jnp.argmax(resp, axis=1)
    ), "Predictions should match argmax of responsibilities"


def test_vs_sklearn():
    X, _ = make_classification(
        n_samples=100,
        n_features=3,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    X = jnp.array(X, dtype=jnp.float64)
    gm = GaussianMixture(n_components=3, max_iter=200, rng_key=0).fit(X)
    skl = SKGMM(n_components=3, random_state=0, max_iter=200, init_params="random").fit(
        X
    )
    from gmm.gmm import _negative_loglikelihood

    skl_nll = -skl.score(X) * X.shape[0]
    our_nll = float(
        _negative_loglikelihood(
            X,
            jnp.array(skl.means_, dtype=jnp.float64),
            jnp.array(skl.covariances_, dtype=jnp.float64),
            jnp.array(skl.weights_, dtype=jnp.float64),
        )
    )

    assert (
        abs(skl_nll - our_nll) < 1e-2
    ), "Negative log-likelihoods should match between sklearn and our implementation"
