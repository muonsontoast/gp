"""
Author: Shaun Preston (John Adams Institute, University of Oxford, Diamond Light Source, Ada Lovelace Centre)

Kernel functional definitions
"""
import jax.numpy as jnp
from jax import jit

@jit
def SE(x0, x1, h):
    h, v = jnp.asarray(h[0]), x0 - x1
    return h[0] * jnp.exp(-.5 * v.T @ jnp.diag(1 / h[1:] ** 2) @ v)

@jit
def RQ(x0, x1, h):
    h, v = jnp.asarray(h[0]), x0 - x1
    hlen = int((h.shape[0] - 1) / 2) + 1
    return h[0] * jnp.prod((1 + v ** 2 / (2 * h[1:hlen] * h[hlen:])) ** -h[hlen:])

@jit
def Per(x0, x1, h):
    h, v = jnp.array(h[0]), x0 - x1
    dim = int((h.shape[0] - 1) / 2)
    return h[0] * jnp.exp(-2 * jnp.sum((jnp.sin(jnp.pi * abs(v) / h[dim + 1:]) / h[1:dim + 1]) ** 2))

@jit
def M12(x0, x1, h):
    h, v = jnp.array(h[0]), x0 - x1
    return jnp.exp(-jnp.sum(abs(v) / h))

@jit
def M32(x0, x1, h):
    h, v = jnp.array(h[0]), x0 - x1
    norm = abs(v)
    prod = jnp.prod(1 + jnp.sqrt(3) * norm / h)
    return prod * jnp.exp(-jnp.sqrt(3) * jnp.sum(norm / h))

@jit
def M52(x0, x1, h):
    h, v = jnp.array(h[0]), x0 - x1
    norm = abs(v)
    prod = jnp.prod(1 + jnp.sqrt(5) * norm / h + 5 * v ** 2 / (3 * h ** 2))
    return prod * jnp.exp(-jnp.sqrt(5) * jnp.sum(norm / h))

@jit
def Linear(x0, x1, h):
    h = jnp.array(h[0])
    dim = int(h.shape[0] / 3)
    return (h[dim:2 * dim] ** 2 + (x0 - h[:dim]).T @ jnp.diag(h[2 * dim:] ** 2) @ (x1 - h[:dim]))