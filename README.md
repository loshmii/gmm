# JAX Gaussian Mixture Model

Fast JAX implementation of the Expectation-Maximisation algorithm for Gaussian Mixture Models (GMMs). This library leverages JAX JIT compilation, 64-bit precision and Cholesky-based covariance calculations for numerical-stability. 

## Installation

```bash
pip install -r requirements.txt
```

For GPU acceleration install `jax[cuda]` following the [JAX](https://github.com/google/jax#installation) instructions.

## Quick start

```python
import jax
import jax.numpy as jnp
from gmm import GaussianMixture

key = jax.random.key(0)
true_means = jnp.array([[0., 0.], [4., 4.]])
X = jax.random.normal(key, (300, 2)) + true_means[jax.random.randint(key, (300,), 0, 2)]
model = GaussianMixture(2, rng_key=key).fit(X)
model.plot(iter_losses=True, cluster_scatter=True)
```

```python
from sklearn.datasets import make_classification
X, _ = make_classification(n_samples=1000, n_features=4, n_classes=3, random_state=0)
X = jnp.array(X)
model = GaussianMixture(3).fit(X)
print("NLL:", model.loss_trace[-1])
```

## API Reference

### `GaussianMixture`

`fit(X)` → Fit model to data.

`predict(X)` → Cluster assignments.

`predict_proba(X)` → Responsibilities.

`score_samples(X)` → Sample log-likelihoods.

`plot(iter_losses=True, cluster_scatter=False)` → Visualise training.

## DEMO

The demo is located in the [Jupyter Notebook](./gmm_jax_em_demo.ipynb)