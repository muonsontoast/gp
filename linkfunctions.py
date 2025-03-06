"""
Author: Shaun Preston (John Adams Institute, University of Oxford, Diamond Light Source, Ada Lovelace Centre)

Link functions for non-Gaussian inference with GPs
"""

from jax import jit
import jax.numpy as jnp

class LinkFunction:
    def __init__(self, name):
        self.name = name.title()

class Sigmoid(LinkFunction):
    '''Bounded between 0 and 1'''
    def __init__(self):
        self.name = 'Sigmoid'

    @jit
    def __call__(self, x):
        return 1 / (1 + jnp.exp(x))
    
    @jit
    def Inverse(self, x):
        return -jnp.log(1 / x - 1)

class Log(LinkFunction):
    '''Strictly positive'''
    def __init__(self):
        self.name = 'Log'

    @jit
    def __call__(self, x):
        return jnp.log(x)
    
    @jit
    def Inverse(self, x):
        return jnp.exp(x)