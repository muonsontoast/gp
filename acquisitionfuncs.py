"""
Author: Shaun Preston (John Adams Institute, University of Oxford, Diamond Light Source, Ada Lovelace Centre)

Acquisition functions for bayesian optimisation with GPs
"""
from jax import jit
import jax.numpy as jnp
from functools import partial

class AcquisitionFunction:
    def __init__(self, name):
        self.name = name.title()

class UCB(AcquisitionFunction):
    def __init__(self):
        self.name = 'Upper Confidence Bound (UCB)'

    @partial(jit, static_argnums = 0)
    def __call__(self, mean, variance, beta = 2):
        '''A simple mean-variance trade-off whose greediness is controlled by `beta`'''
        return (mean + beta * jnp.sqrt(variance))