"""
Author: Shaun Preston (John Adams Institute, University of Oxford, Diamond Light Source, Ada Lovelace Centre)

Acquisition functions for bayesian optimisation with GPs
"""
from jax import jit
import jax.numpy as jnp
from functools import partial
from scipy.stats import qmc
import gp
from copy import deepcopy as cp

class AcquisitionFunction:
    def __init__(self, name):
        super().__setattr__('name', name.title())

    def __setattr__(self, name, value):
        if hasattr(self, name):
            super().__setattr__(name, value)

    def __str__(self):
        s = ''
        underline = '=' * len(self.name)
        s += self.name + '\n' + underline
        d = cp(self.__dict__)
        d.pop('name')
        items = d.items()
        for k, v in items:
            s += '\n' + k.title() + ': ' + str(v)
        return s

    def SetHyperparameters(self, value: dict):
        for k, v in value.items():
            setattr(self, k, v)

class UCB(AcquisitionFunction):
    '''Standard global Bayesian optimisation'''
    def __init__(self):
        object.__setattr__(self, 'name', 'Upper Confidence Bound (UCB)')
        object.__setattr__(self, 'beta', 2)

    # @partial(jit, static_argnums = 0)
    def __call__(self, mean, variance):
        '''A simple mean-variance trade-off whose greediness is controlled by `beta`'''
        return mean + self.beta * jnp.sqrt(variance)
    
class UCBI(AcquisitionFunction):
    '''Standard global Bayesian optimisation'''
    def __init__(self):
        object.__setattr__(self, 'name', 'Upper Confidence Bound Interpolate (UCBI)')
        object.__setattr__(self, 'beta', .8)

    # @partial(jit, static_argnums = 0)
    def __call__(self, mean, variance):
        '''A simple mean-variance trade-off whose greediness is controlled by an interpolation `beta`'''
        return self.beta * mean + (1 - self.beta) * jnp.sqrt(variance)

class TuRBO(AcquisitionFunction):
    '''Adaptive local Bayesian optimisation'''
    def __init__(self, l = .25, lmin = .01, lmax = 1, tau = 3):
        self.name = 'Trust Region Bayesian Optimisation (TRBO)'
        self.l, self.lmin, self.lmax = l, lmin, lmax
        self.tau = tau

    def GenerateInitialTrustRegions(self, parentGP, numPoints = 4):
        '''Accepts a GP as input and returns LHS initial samples based on its bounds.'''
        sampler = qmc.LatinHypercube(parentGP.bounds.shape[1])
        self.trustRegionCentres = sampler.random(numPoints)
        self.trustRegionWidths = [self.l for _ in range(numPoints)]
        self.GPs = [gp.GP() for _ in range(numPoints)]
        self.GPInputs = [jnp.empty(0) for _ in range(numPoints)]
        self.GPOutputs = [jnp.empty(0) for _ in range(numPoints)]
        self.GPVariances = [jnp.empty(0) for _ in range(numPoints)]
        
        # initialise local GPs
        for _ in range(numPoints):
            self.GPs[_].bounds = jnp.array([jnp.clip(self.trustRegionCentres[_] - self.trustRegionWidths[_], 0, 1), jnp.clip(self.trustRegionCentres[_] + self.trustRegionWidths[_], 0, 1)])

    def BinMeasurements(self, x, y, variances):
        for _, g in enumerate(self.GPs):
            cond = x >= g.bounds[0] and x <= g.bounds[1]
            self.GPInputs[_], self.GPOutputs[_], self.GPVariances[_] = x[cond], y[cond], variances[cond]