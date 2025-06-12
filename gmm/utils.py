import jax


class RNGManager:

    def __init__(self, seed: int = 0) -> None:
        self.key = jax.random.key(seed)

    def new_key(self) -> jax.Array:
        self.key, new = jax.random.split(self.key)
        return new

    def new_keys(self, n: int) -> jax.Array:
        keys = jax.random.split(self.key, n + 1)
        self.key = keys[0]
        return keys[1:]
