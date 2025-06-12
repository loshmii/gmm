import jax
import jax.numpy as jnp
from gmm import GaussianMixture

key = jax.random.key(0)
true_means = jnp.array([[0.0, 0.0], [4.0, 4.0]])
X = jax.random.normal(key, (300, 2)) + true_means[jax.random.randint(key, (300,), 0, 2)]
model = GaussianMixture(2, rng_key=key).fit(X)
model.plot(iter_losses=True, cluster_scatter=True)