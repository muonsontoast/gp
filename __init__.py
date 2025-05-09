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
from scipy.stats.qmc import LatinHypercube as lhs
from enum import Enum
from copy import deepcopy
from cothread.catools import caget, caput

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
    adagrad = optax.adagrad,
    noisysgd = optax.noisy_sgd,
    adabelief = optax.adabelief,
    lbfgs = optax.scale_by_lbfgs,
)

class OptimiserVariableTypes(Enum):
    CONTINUOUS = 'CONTINUOUS'
    DISCRETE = 'DISCRETE'
    CATEGORICAL = 'CATEGORICAL'

noiseModels = ["Gaussian", "LogGaussian", "TruncatedGaussian", "Poisson"]

class GP:
    '''Reference attribute names in **camelCase**'''
    name: str
    optimiserName: str
    optimiser: optax.GradientTransformation
    noiseModel: str
    kernel: Kernel
    bounds: jax.Array
    f: None
    seed: int
    VOCS: dict
    AF: list

    def __init__(self, dimension = 1, kernel = SE):
        super().__setattr__('name', 'GP')
        super().__setattr__('optimiserName', 'adamw')
        super().__setattr__('optimiser', optimisers[self.optimiserName])
        super().__setattr__('noiseModel', 'Gaussian')
        super().__setattr__('kernel', kernel(dimension))
        super().__setattr__('bounds', jnp.array([[0], [1]]))
        super().__setattr__('f', lambda x: jnp.zeros(x.shape[0]))
        super().__setattr__('seed', 0)
        super().__setattr__('VOCS', {'variables': dict(), 'objectives': dict(), 'constraints': dict()})
        super().__setattr__('AF', [UCB(), {'beta': 2}])

    def __str__(self):
        res = f'\nName: {self.name}'
        res += f'\nKernel: {self.kernel.name}'
        res += f'\nOptimiser: {self.optimiserName.title()}'
        lw = ''.join(f'{round(v_, 2):.2f}' for v_ in jnp.array(self.bounds[0]).flatten())
        up = ''.join(f'{round(v_, 2):.2f}' for v_ in jnp.array(self.bounds[1]).flatten())
        res += f'\nOptimiser Constraints: Lower = {lw}, Upper = {up}'
        res += f'\nAcquisition Function: {self.AF[0].name}'
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

    def fit(self, x, y, xnew, fitKernelHyperparams = True) -> dict:
        '''Fit works on single function output problems.'''
        x, xnew = self.NormaliseDomainPoints(x), self.NormaliseDomainPoints(xnew)
        info = self.NormaliseMeasurements(y)
        info['means'] = info['means'].flatten()
        info['variances'] = info['variances'].flatten()

        if fitKernelHyperparams:
            self.FitKernelHyperparams(x, info['means'], info['variances'])
        if self.DimensionCheck(x):
            S11 = self.ConstructCovarianceMatrix(x, x) + self.ConstructMeasurementNoiseMatrix(x, info['variances'])
            S12 = self.ConstructCovarianceMatrix(x, xnew)
            solved = jax.scipy.linalg.solve(S11, S12, assume_a = 'pos').T
            S22 = self.ConstructCovarianceMatrix(xnew, xnew)
            S2 = S22 - solved @ S12
            evals, evecs = jnp.linalg.eigh(S2)
            S2 = evecs @ jnp.diag(jnp.maximum(evals, 1e-5)) @ evecs.T

            result = self.UnNormaliseMeasurements((self.f(xnew) + solved @ (info['means'] - self.f(x)))[:, None, None], jnp.diag(S2)[:, None], info['rng'], info['mn'])
            A = jnp.diag(info['rng'][0] * jnp.ones(len(xnew)))
            S2 = A @ S2 @ A
            result['covarianceMatrix'] = S2
            return result
        else:
            print('Input data is the wrong dimension!')

    def optimise(self, numEpochs = 1, numInitialRandomSamples = 3, numRepeatMeasurements = 1, mode = 'maximise', **kwargs):
        '''You must first specify a VOCS dictionary with variables and objectives (constraints optional).\n
        `numInitialRandomPoints` = how many initial points to randomly sample (cold/warm start).\n
        `numEpochs` = how many epochs to run optimiser.\n
        `threshold` = improvement threshold for which optimiser will terminate.\n
        `mode` = `maximise` or `minimise`.\n
        set `ignoreVariance` = true to ignore measurement variance (useful for heteroscedastic models).'''
        caput('LI-TI-MTGEN-01:START', 1) # start the LINAC
        ignoreVariance = kwargs.get('ignoreVariance', False)
        numInitialRandomSamples = jnp.maximum(numInitialRandomSamples, 2)
        if mode.lower() == 'maximise':
            selectBest = lambda x: jnp.argmax(x, 0)
        else: # defaults to minimisation
            selectBest = lambda x: jnp.argmin(x, 0)
        bestObjectiveValues = jnp.zeros(0) # only tracks the first objective ('best' meant for 1D obviously)
        x = self.SelectInitialDomainPoints(numInitialRandomSamples)
        # y = jnp.array([jnp.array([jnp.array([objective(x) for repeat in range(numRepeatMeasurements + 1)]).flatten() for objective in self.VOCS['objectives'].values()]) for x in x])
        y = jnp.array([jnp.array([objective(x, numRepeatMeasurements) for objective in self.VOCS['objectives'].values()]) for x in x])
        x = self.NormaliseDomainPoints(x)
        # print(y)
        # if ignoreVariance:
        #     y = jnp.mean(y, -1)[:, None]
        # print(y)
        info = self.NormaliseMeasurements(y)

        # we should track the best value if we have a single objective (always dumbly track the first objective)
        bestObjectiveValues = jnp.append(bestObjectiveValues, info['means'][selectBest(info['means'][:, 0]), 0])

        # temporarily setting to y[:, 0] until full multi objective implemented

        # if ignoreVariance:
        #     info['variances'] = 1e-1 * jnp.ones(info['variances'].shape)

        if ignoreVariance:
            self.FitKernelHyperparams(x, info['means'][:, 0], 1e-3 * jnp.ones(len(x)))
            info['variances'] = 1e-3 * jnp.ones(info['variances'].shape)
        else:
            self.FitKernelHyperparams(x, info['means'][:, 0], info['variances'][:, 0])

        for e in range(numEpochs):
            print(f'Epoch {e + 1}/{numEpochs}')
            self.AF[1]['yBest'] = bestObjectiveValues[-1]
            self.AF[0].SetHyperparameters(self.AF[1])
            xnext = self.PickNextPoint(self.UnNormaliseDomainPoints(x), y)
            x = jnp.concatenate((x, self.NormaliseDomainPoints(xnext)))
            # ynew = jnp.array([jnp.array([jnp.array([objective(xnext[0], numRepeatMeasurements) for repeat in range(numRepeatMeasurements + 1)]).flatten() for objective in self.VOCS['objectives'].values()])])
            ynew = jnp.array([jnp.array([objective(x, numRepeatMeasurements) for objective in self.VOCS['objectives'].values()]) for x in xnext])
            y = jnp.concatenate((y, ynew))
            ynew = self.NormaliseMeasurements(ynew, info['rng'], info['mn'])
            info['y'] = jnp.concatenate((info['y'], ynew['y']))
            info['means'] = jnp.concatenate((info['means'], ynew['means']))
            if ignoreVariance:
                info['variances'] = jnp.concatenate((info['variances'], 1e-3 * jnp.ones(ynew['variances'].shape)))
            else:
                info['variances'] = jnp.concatenate((info['variances'], ynew['variances']))
            bestObjectiveValues = jnp.append(bestObjectiveValues, jnp.maximum(ynew['means'][0, 0], bestObjectiveValues[-1]))
            self.FitKernelHyperparams(x, info['means'][:, 0], info['variances'][:, 0])

        result = self.UnNormaliseMeasurements(info['y'], info['variances'], info['rng'], info['mn'], bestObjectiveValues)

        unnormalisedX = self.UnNormaliseDomainPoints(x)
        result['x'] = unnormalisedX

        AFOld = deepcopy(self.AF[0])
        self.AF[0] = UCBI()
        self.AF[0].beta = 1
        if ignoreVariance:
            xnext = self.PickNextPoint(unnormalisedX, jnp.mean(y, -1)[:, None], 0)
        else:
            xnext = self.PickNextPoint(unnormalisedX, y, 0)
        self.AF[0] = AFOld
        result['xnext'] = xnext[0]
        result['ynext'] = self.fit(unnormalisedX, result['y'], xnext, False)['y'].ravel()
        print('Recommended working point is:', xnext[0])
        print('Optimisation complete!')
        caput('LI-TI-MTGEN-01:STOP', 1) # stop the LINAC

        return result

    def PickNextPoint(self, x, y, noiseAmplitude = 5e-3, numIterations = 150, numRestarts = 20):
        key = jrd.PRNGKey(np.random.randint(0, 100000))
        xs = self.SelectInitialDomainPoints(numRestarts)

        @jit
        def AcquisitionFunctionAtInputWrapper(xi):
            return self.AcquisitionFunctionAtInput(x, y, xi) * -1.

        solver = optax.lbfgs(memory_size = 15)
        valueAndGrad = optax.value_and_grad_from_state(AcquisitionFunctionAtInputWrapper)

        @jit
        def InnerLoop(carry, it):
            xi, optState = carry
            value, grad = valueAndGrad(xi, state = optState)
            updates, optState = solver.update(
                grad, optState, xi, value = value, grad = grad, value_fn = AcquisitionFunctionAtInputWrapper
            )
            return (jnp.clip(xi + updates, self.bounds[0], self.bounds[1]), optState), value

        @jit
        def OuterLoop(xi):
            optState = solver.init(xi)
            (xi, optState), value = lax.scan(InnerLoop, (xi, optState), None, length = numIterations)
            return xi, value[-1]
        
        finalXs, vs = vmap(OuterLoop)(xs[:, None])
        return jnp.clip(finalXs[jnp.argmin(vs)] + noiseAmplitude * jrd.normal(key, shape = 1), self.bounds[0], self.bounds[1])

    def AcquisitionFunctionAtInput(self, x, y, xnew):
        result = self.fit(x, y, xnew, False)
        return self.AF[0](result['means'].ravel(), result['variances'].ravel())[0]
    
    def FitKernelHyperparams(self, x, y, variance, numIterations = 200, numRestarts = 15, returnAll = False):
        attrs = []
        for attr in self.kernel._orderedAttrs:
            attrs += attr

        xv = self.ConstructMeasurementNoiseMatrix(x, variance)

        @jit
        def LogLikelihoodWrapper(xi):
            S = self.ConstructCovarianceMatrix(x, x, self.kernel.Convert2OrderedAttrs(xi)) + xv
            return self.LogLikelihood(y, S)
            

        hs = jnp.array(np.random.randn(numRestarts, len(attrs)))
        # gradLL = value_and_grad(self.LogLikelihood, argnums = 3)
        gradLL = value_and_grad(LogLikelihoodWrapper)
        optimiser = optimisers['adamw'](5e-3)

        @jit
        def InnerLoop(carry, it):
            h, state = carry
            # ll, g = gradLL(x, y, variance, jnp.exp(h))
            ll, g = gradLL(jnp.exp(h))
            u, state = optimiser.update(g, state, h)
            return (h + u, state), ll
        
        @jit
        def OuterLoop(h):
            state = optimiser.init(h)
            (finalH, state), ll = lax.scan(InnerLoop, (h, state), None, length = numIterations)
            return jnp.exp(finalH), ll

        finalHs, lls = vmap(OuterLoop)(hs)
        if returnAll:
            return (finalHs, lls)
        self.kernel.SetAttrs(list(finalHs[jnp.argmin(lls[:, -1])]))

    # def FitKernelHyperparams(self, x, y, variance, numIterations = 100, numRestarts = 30, returnAll = False):
    #     attrs = []
    #     for attr in self.kernel._orderedAttrs:
    #         attrs += attr

    #     @jit
    #     def LogLikelihoodWrapper(xi):
    #         return self.LogLikelihood(x, y, variance, jnp.exp(xi))

    #     xs = jnp.array(np.random.randn(numRestarts, len(attrs)))
    #     solver = optax.lbfgs(memory_size = 15)
    #     valueAndGrad = optax.value_and_grad_from_state(LogLikelihoodWrapper)

    #     @jit
    #     def InnerLoop(carry, it):
    #         xi, optState = carry
    #         value, grad = valueAndGrad(xi, state = optState)
    #         updates, optState = solver.update(
    #             grad, optState, xi, value = value, grad = grad, value_fn = LogLikelihoodWrapper
    #         )
    #         return (xi + updates, optState), value

    #     @jit
    #     def OuterLoop(xi):
    #         optState = solver.init(xi)
    #         (xi, optState), value = lax.scan(InnerLoop, (xi, optState), None, length = numIterations)
    #         return jnp.exp(xi), value
        
    #     finalXs, vs = vmap(OuterLoop)(xs)
    #     # print('ooh!')
    #     # print(finalXs[jnp.argmin(vs)])
    #     self.kernel.SetAttrs(list(finalXs[jnp.argmin(vs[:, -1])]))
    #     # return jnp.clip(finalXs[jnp.argmin(vs)] + noiseAmplitude * jrd.normal(key, shape = 1), self.bounds[0], self.bounds[1])

    # @partial(jit, static_argnums = 0)
    # def LogLikelihood(self, x, y, variance, h = None):
    #     if h != None:
    #         S = self.ConstructCovarianceMatrix(x, x, self.kernel.Convert2OrderedAttrs(h)) + self.ConstructMeasurementNoiseMatrix(x, variance)
    #     else: S = self.ConstructCovarianceMatrix(x, x) + self.ConstructMeasurementNoiseMatrix(x, variance)
    #     chol = jnp.linalg.cholesky(S)
    #     Snew = jnp.linalg.solve(chol, y)
    #     det = 2 * jnp.sum(jnp.log(jnp.diag(chol)))
    #     return Snew.T @ Snew + det

    @partial(jit, static_argnums = 0)
    def LogLikelihood(self, y, S):
        # chol = jnp.linalg.cholesky(S)
        l, lower = jax.scipy.linalg.cho_factor(S)
        Snew = jax.scipy.linalg.cho_solve((l, lower), y)
        return y.T @ Snew + 2 * jnp.sum(jnp.log(jnp.diag(l)))

    @partial(jit, static_argnums = (0)) # should this be kept or not?? remove 1, 2?
    def ConstructCovarianceMatrix(self, x, y, h = None):
        h = self.kernel._orderedAttrs if h == None else h
        return FastConstructCovarianceMatrix(self.kernel.f, x, y, h)

    def DisplayCovarianceMatrix(self, m, extent = None, cmap = 'viridis'):
        fig, ax = plt.subplots()
        im = ax.imshow(m, extent = extent, origin = 'lower', cmap = cmap)
        cb = plt.colorbar(im)
        cb.set_label('Covariance', rotation = 270, labelpad = 15)

        return fig, ax
    
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
    
    @partial(jit, static_argnums = 0)
    def ConstructMeasurementNoiseMatrix(self, x, variance = None):
        variance = jnp.zeros(x.shape[0]) if len(variance) == 0 else variance
        return jnp.diag(variance + global_smoothing_factor)

    @partial(jit, static_argnums = 0)
    def NormaliseDomainPoints(self, x):
        return (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
    
    @partial(jit, static_argnums = 0)
    def UnNormaliseDomainPoints(self, x):
        return x * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

    @partial(jit, static_argnums = 0)
    def NormaliseMeasurements(self, y, _rng = None, _mn = None):
        mx, mn = jnp.max(jnp.max(y, 2), 0), jnp.min(jnp.min(y, 2), 0)
        cond = jnp.equal(mx, mn)
        variances = jnp.var(y, -1, ddof = 1) if len(y[0, 0]) > 1 else jnp.var(y, -1, ddof = 0)
        rng = jnp.where(cond, 1, mx - mn) if _rng is None else _rng
        mn = mn if _mn is None else _mn
        ynew = (y - mn[None, :, None]) / rng[None, :, None]

        result = {
            'y': ynew,
            'means': jnp.mean(ynew, -1),
            'variances': variances / rng ** 2,
            'mn': mn,
            'mx': mx,
            'rng': rng,
        }

        return result
    
    @partial(jit, static_argnums = 0)
    def UnNormaliseMeasurements(self, y, variances, rng, mn, bestObjectiveValues = None):
        ynew = y * rng[None, :, None] + mn[None, :, None]
        newBestObjectiveValues = bestObjectiveValues * rng[None, :, None] + mn[None, :, None] if bestObjectiveValues is not None else jnp.array([])
        result = {
            'y': ynew,
            'bestObjectiveValues': newBestObjectiveValues.ravel(),
            'means': jnp.mean(ynew, -1),
            'variances': variances * rng ** 2,
        }
        return result

    def SelectInitialDomainPoints(self, numPoints = 5):
        '''Use Latin Hypercube Sampling to generate starting points to optimise forwards from.
        '''
        if numPoints == 0:
            return jnp.array([[]])
        samples = lhs(d = self.kernel.dimension).random(n = numPoints)
        return samples * (jnp.array(self.bounds[1]) - jnp.array(self.bounds[0])) + self.bounds[0]

    def GenerateDomainGridPoints(self, numPerDimension = 5, unnormalise = True):
        '''Returns a meshgrid object and a flat array of coordinate tuples'''
        marginalSamples = [jnp.linspace(self.bounds[0][i], self.bounds[1][i], numPerDimension) for i in range(self.kernel.dimension)]
        samples = jnp.meshgrid(*marginalSamples)
        coords = jnp.stack((samples), -1).reshape(-1, len(samples))
        return samples, coords

@jit
def warp(e, y, a = 1):
    return e * (1 - jnp.tanh(a * y))

@jit
def unwarp(e, y, a = 1):
    return e / (1 - jnp.tanh(a * y))

@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

@partial(jit, static_argnums = 0)
def FastConstructCovarianceMatrix(f, x, y, h):
    return vmap(lambda x0: vmap( lambda x1: f(x0, x1, h))(y))(x)