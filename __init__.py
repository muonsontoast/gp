"""
Author: Shaun Preston (John Adams Institute, University of Oxford, Diamond Light Source, Ada Lovelace Centre)

Custom Gaussian Process (GP) code which has been written mostly as a way to become familiar with this probabilistic model
"""

import jax, optax, os
import numpy as np
import jax.numpy as jnp
import jax.random as jrd
from jax import vmap, jit, value_and_grad, lax
from functools import partial
import matplotlib.pyplot as plt

from gp.kernels import *
from gp.acquisitionfuncs import *

os.environ['PATH'] += ':/Library/TeX/texbin'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times'
plt.rcParams['font.size'] = 12

jax.config.update("jax_enable_x64", True)
global_smoothing_factor = 1e-6 # necessary to maintain stability for large covariance matrices

optimisers = dict(
    sgd = optax.sgd,
    adam = optax.adam,
    adamw = optax.adamw,
    nadamw = optax.nadamw,
    rmsprop = optax.rmsprop,
    adadgrad = optax.adagrad,
    noisysgd = optax.noisy_sgd,
    adabelief = optax.adabelief,
)

noiseModels = ["Gaussian", "LogGaussian", "TruncatedGaussian", "Poisson"]

class GP:
    '''Reference attribute names in **camelCase**'''
    def __init__(self, kernel = SE):
        super().__setattr__('name', 'GP')
        super().__setattr__('optimiserName', 'adamw')
        super().__setattr__('optimiser', optimisers[self.optimiserName])
        super().__setattr__('acquisitionFunction', UCB())
        super().__setattr__('noiseModel', 'Gaussian')
        super().__setattr__('kernel', kernel())
        super().__setattr__('bounds', jnp.array([[0], [1]]))
        super().__setattr__('f', lambda x: jnp.zeros(x.shape[0]))

    def __str__(self):
        res = f'\nName: {self.name}'
        res += f'\nKernel: {self.kernel.name}'
        res += f'\nOptimiser: {self.optimiserName.title()}'
        lw = ''.join(f'{round(v_, 2):.2f}' for v_ in jnp.array(self.bounds[0]).flatten())
        up = ''.join(f'{round(v_, 2):.2f}' for v_ in jnp.array(self.bounds[1]).flatten())
        res += f'\nOptimiser Constraints: Lower = {lw}, Upper = {up}'
        res += f'\nAcquisition Function: {self.acquisitionFunction.name}'
        res += f'\nNoise Model: {self.noiseModel}\n'
        return res
    
    def __setattr__(self, name, value):
        try: attr = self.__getattribute__(name)
        except: print("Assignment failed because that attribute doesn\'t exist!"); return

        if name == 'kernel':
            if isinstance(value, Kernel):
                super().__setattr__(name, value)
            else:
                print('Assignment failed because you didn\'t supply an instance of a kernel!')
            return      
        elif name == 'optimiser':
            super().__setattr__(name, value)
            super().__setattr__('optimiserName', [k for k, v in optimisers.items() if v == value][0])
            return
        elif name == 'noiseModel':
            if value not in noiseModels:
                print('Assignment failed because you specified an invalid * Noise Model *. Valid types are %s' % str(noiseModels))
                return
            super().__setattr__(name, value)

        attrType, valueType = type(attr), type(value)
        if attrType != valueType:
            print('Assignment failed because you tried to set * %s * to %s but it must be %s!' % (name, valueType, attrType))
            return

        if valueType == str:
            value = value.title()

        super().__setattr__(name, value)

    def __call__(self, x, y, xnew, variance = jnp.empty(0), fitKernelHyperparams = True):
        lw, up = jnp.min(x), jnp.max(x)
        rng = up - lw
        x, xnew = (x - lw) / rng, (xnew - lw) / rng
        if fitKernelHyperparams:
            self.FitKernelHyperparams(x, y, variance)
        x, xnew = self.FormatInputData(x), self.FormatInputData(xnew)
        if self.DimensionCheck(x):
            S11 = self.ConstructCovarianceMatrix(x, x) + self.ConstructMeasurementNoiseMatrix(x, variance)
            S12 = self.ConstructCovarianceMatrix(x, xnew)
            solved = jax.scipy.linalg.solve(S11, S12, assume_a = 'pos').T
            S22 = self.ConstructCovarianceMatrix(xnew, xnew)
            S2 = S22 - solved @ S12
            evals, evecs = jnp.linalg.eigh(S2)
            S2 = evecs @ jnp.diag(jnp.maximum(evals, 1e-6)) @ evecs.T
            return self.f(xnew) + solved @ (y - self.f(x)), S2
        else:
            print('Input data is the wrong dimension!')
 
    def PickNextPoint(self, x, y, variance, noiseAmplitude = .05, numIterations = 250, numRestarts = 10, returnAll = False):
        '''Takes an optional `kwargs` to pass into the acquisition function.'''
        optimiser = optax.chain(
            self.optimiser(1e-1)
        )
        key = jrd.PRNGKey(np.random.randint(0, 100000))
        x = self.FormatInputData(x)
        xs = jrd.uniform(key, shape = (numRestarts, x.shape[1]), minval = self.bounds[0], maxval = self.bounds[1])

        valueAndGrad = value_and_grad(self.AcquisitionFunctionAtInput, argnums = 2)

        @jit
        def InnerLoop(carry, it):
            h, state = carry
            v, g = valueAndGrad(x, y, h, variance)
            u, state = optimiser.update(-g, state, h)
            return (jnp.clip(h + u, self.bounds[0], self.bounds[1]), state), v
        
        @jit
        def OuterLoop(h):
            state = optimiser.init(h)
            (finalX, state), v = lax.scan(InnerLoop, (h, state), None, length = numIterations)
            return finalX, v
        
        finalXs, vs = vmap(OuterLoop)(xs)

        if returnAll:
            return (jnp.clip(finalXs + noiseAmplitude * jrd.normal(key, shape = finalXs.shape), self.bounds[0], self.bounds[1]), vs)
        return jnp.clip(finalXs[jnp.argmax(vs[:, -1])] + noiseAmplitude * jrd.normal(key), self.bounds[0], self.bounds[1])

    @partial(jit, static_argnums = 0)
    def AcquisitionFunctionAtInput(self, x, y, xnew, variance):
        m, s = self.__call__(x, y, jnp.array([xnew]), variance)
        return self.acquisitionFunction(m, jnp.diag(s))[0]
        
    def FitKernelHyperparams(self, x, y, variance, numIterations = 250, numRestarts = 30, returnAll = False):
        lw, up = jnp.min(x), jnp.max(x)
        x = (x - lw) / (up - lw)
        
        attrs = []
        for attr in self.kernel._orderedAttrs:
            attrs += attr

        hs = jnp.array(np.random.gamma(5, .5, (numRestarts, len(attrs))))
        gradLL = value_and_grad(self.LogLikelihood, argnums = 3)

        optimiser = self.optimiser(.1)

        @jit
        def InnerLoop(carry, it):
            h, state = carry
            ll, g = gradLL(x, y, variance, h)
            u, state = optimiser.update(g, state, h)
            return (jnp.clip(h + u, 5e-2, 1e2), state), ll
        
        @jit
        def OuterLoop(h, key):
            global optimiser
            lr = 10 ** jrd.uniform(key, minval = -4., maxval = -.5)
            optimiser = self.optimiser(lr)
            state = optimiser.init(h)
            (finalH, state), ll = lax.scan(InnerLoop, (h, state), None, length = numIterations)
            return finalH, ll

        rndIdx = np.random.randint(0, 10000)
        keys = jrd.split(jrd.PRNGKey(rndIdx), numRestarts)
        print('Seed #', rndIdx)

        finalHs, lls = vmap(OuterLoop)(hs, keys)
        if returnAll:
            return (finalHs, lls)
        self.kernel.SetAttrs(list(finalHs[jnp.argmin(lls[:, -1])]))

    @partial(jit, static_argnums = 0)
    def LogLikelihood(self, x, y, variance, h = None):
        if h != None:
            S = self.ConstructCovarianceMatrix(x, x, self.kernel.Convert2OrderedAttrs(h)) + self.ConstructMeasurementNoiseMatrix(x, variance)
        else: S = self.ConstructCovarianceMatrix(x, x) + self.ConstructMeasurementNoiseMatrix(x, variance)
        chol = jnp.linalg.cholesky(S)
        Snew = jnp.linalg.solve(chol, y)
        det = 2 * jnp.sum(jnp.log(jnp.diag(chol)))
        return Snew.T @ Snew + det
    
    @partial(jit, static_argnums = 0)
    def ConstructCovarianceMatrix(self, x, y, h = None):
        x, y = self.FormatInputData(x), self.FormatInputData(y)
        if x.shape[1] == y.shape[1]:
            if self.DimensionCheck(x):
                return vmap(lambda x0: vmap( lambda x1: self.kernel._CallWithoutChecks(x0, x1, h))(y))(x)
        else: print('Input data is the wrong dimension!')

    def DisplayCovarianceMatrix(self, m, extent = None):
        fig, ax = plt.subplots()
        im = ax.imshow(m, extent = extent, origin = 'lower')
        cb = plt.colorbar(im)

        return fig
    
    def DrawSamples(self, m, s, numSamples = 1):
        '''Accepts a mean vector `m` and a covariance matrix `s` and returns `numSamples`'''
        return np.random.multivariate_normal(m, s, numSamples)
    
    def DimensionCheck(self, x):
            '''Checks the input data dimension matches that of the kernel'''
            if x.shape[1] == self.kernel.dimension:
                return True
            return False

    def FormatInputData(self, x):
        '''Wraps data in a JAX array'''
        x = [x] if type(x) in [int, float] else x
        return jnp.array(x).reshape(len(x), -1)
    
    def ConstructMeasurementNoiseMatrix(self, x, variance = None):
        variance = jnp.zeros(x.shape[0]) if len(variance) == 0 else variance
        return jnp.diag(variance + global_smoothing_factor)

@jit
def warp(e, y, a = 1):
    return e * (1 - jnp.tanh(a * y))

@jit
def unwarp(e, y, a = 1):
    return e / (1 - jnp.tanh(a * y))