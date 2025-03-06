"""
Author: Shaun Preston (John Adams Institute, University of Oxford, Diamond Light Source, Ada Lovelace Centre)

Custom kernel class
"""

import copy
import jax
import jax.numpy as jnp
from jax import jit
from gp import kernelfuncs
import numpy as np
from functools import partial
from collections.abc import Iterable

attrsToCheck = ['lengthscale', 'amplitude', 'period', 'exponent', 'offset']
incorrectInputDimensionMessage = 'Input data is the wrong dimension!'

class Kernel:
    '''Kernel base class'''

    def __init__(self, dimension = 1):
        super().__setattr__('name', 'General')
        super().__setattr__('dimension', dimension)
        super().__setattr__('description', 'Composition of kernels')
        super().__setattr__('components', [])
        super().__setattr__('componentNames', [])
        super().__setattr__('f', None)
        super().__setattr__('_orderedAttrs', [])
        super().__setattr__('amplitude', [[]])
        super().__setattr__('lengthscale', [[]])
        super().__setattr__('exponent', [[]])
        super().__setattr__('period', [[]])
        super().__setattr__('offset', [[]])

    def __str__(self):
        attrs = vars(self)
        res = f'\nName: {attrs['name']}\n'
        res += f'Dimension: {attrs['dimension']}\n'
        res += f'Description: {attrs['description']}'
        if (len(attrs['components']) > 1):
            idx = 0
            for _, l in enumerate(attrs['components']):
                res += f'\n\n * {self.componentNames[_]} (Type: {l}) - Component {idx} *'
                for k, v in attrs.items():
                    if k in attrsToCheck and len(v[_]) > 0:
                        v = [f'{round(v_, 2):.2f}' for v_ in jnp.array(v).flatten()]
                        res += f'{k.title()}: ' + ', '.join(v) + '\n'
                idx += 1
            res += '\n'
        else:
            res += '\n\n'
            for k, v in attrs.items():
                if k in attrsToCheck and len(v[0]) > 0:
                    v = [f'{round(v_, 2):.2f}' for v_ in jnp.array(v).flatten()]
                    res += f'{k.title()}: ' + ', '.join(v) + '\n'
        return res


    def __setattr__(self, name, value):
        if name in ['description', 'dimension']:
            print('* %s * can\'t be modified after instantiating a kernel!' % name)
            return
        
        if len(self.components) > 1 and name not in ['name']:
            print('Assignment failed, use .UpdateKernel(<attrname>, <attrval>, <componentidx> = 0) instead for composite kernels!')
            return

        try:
            attr = super().__getattribute__(name)
        except: 
            print('Assignment failed because the given attribute name doesn\'t exist!')
            return
        
        attrType, valueType = type(attr), type(value)
        if valueType == attrType:
            if valueType != str:
                if valueType == list and len(attr[0]) > 0:
                    idx = min(len(attr[0]), len(value))
                    attr[0][:idx] = value[:idx]
                    super().__setattr__(name, attr)
                    self._UpdateOrderedAttrs()
            else:
                super().__setattr__(name, value.title())
        elif valueType in [int, float] and attrType == list and len(attr[0]) > 0:
            attr = [value] + attr[1:]
            super().__setattr__(name, [attr])
            self._UpdateOrderedAttrs()
        else:
            print('Assignment failed because you tried to set * %s * to %s but it must be %s!' % (name, valueType, attrType))

    def __call__(self, x0, x1):
        x0, x1 = self.FormatInputData(x0), self.FormatInputData(x1)
        if self.DimensionCheck(x0, x1):
            return self.f(x0, x1, self._orderedAttrs)
        print(f'Input data is the wrong dimension!')

    @partial(jit, static_argnums = 0)
    def _CallWithoutChecks(self, x0, x1, h = None):
        '''Should only be run when called directly by the model as inputs will be formatted a priori'''
        h = self._orderedAttrs if h == None else h
        return self.f(x0, x1, h)

    def __add__(self, kernel):
        '''Custom logic to handle addition of kernels'''
        if self.dimension != kernel.dimension:
            print('Addition failed because the kernels have mismatched dimensions!')
            return self
        new_kernel = Kernel(self.dimension)
        for k, v in vars(self).items():
            if k not in ['name', 'description', '__call__', 'f']:
                object.__setattr__(new_kernel, k, copy.deepcopy(v))
        for k, v in vars(kernel).items():
            if k not in ['name', 'dimension', 'description', '__call__', 'f']:
                if k == 'components':
                    object.__setattr__(new_kernel, k, self.components + kernel.components)
                elif k == 'componentNames':
                    selfName = [self.name] if len(self.components) == 1 else [*self.componentNames]
                    kernelName = [kernel.name] if len(kernel.components) == 1 else [*kernel.componentNames]
                    object.__setattr__(new_kernel, 'componentNames', selfName + kernelName)
                else:
                    try: cur_v = new_kernel.__getattribute__(k)
                    except: cur_v = []
                    object.__setattr__(new_kernel, k, cur_v + copy.deepcopy(v))

        object.__setattr__(new_kernel, 'f', lambda x0, x1, h: self.f(x0, x1, h[:len(self.components)]) + kernel.f(x0, x1, h[len(self.components):]))
        new_kernel._UpdateOrderedAttrs()
        return new_kernel

    def __mul__(self, kernel):
        '''Custom logic to handle multiplication of kernels'''
        if self.dimension != kernel.dimension:
            print('Multiplication failed because the kernels have mismatched dimensions!')
            return self
        new_kernel = Kernel(self.dimension)
        for k, v in vars(self).items():
            if k not in ['name', 'description', '__call__', 'f']:
                object.__setattr__(new_kernel, k, copy.deepcopy(v))
        for k, v in vars(kernel).items():
            if k not in ['name', 'dimension', 'description', '__call__', 'f']:
                if k == 'components':
                    object.__setattr__(new_kernel, k, self.components + kernel.components)
                elif k == 'componentNames':
                    selfName = [self.name] if len(self.components) == 1 else [*self.componentNames]
                    kernelName = [kernel.name] if len(kernel.components) == 1 else [*kernel.componentNames]
                    object.__setattr__(new_kernel, 'componentNames', selfName + kernelName)
                else:
                    try: cur_v = new_kernel.__getattribute__(k)
                    except: cur_v = []
                    object.__setattr__(new_kernel, k, cur_v + copy.deepcopy(v))

        object.__setattr__(new_kernel, 'f', lambda x0, x1, h: self.f(x0, x1, h[:len(self.components)]) * kernel.f(x0, x1, h[len(self.components):]))
        new_kernel._UpdateOrderedAttrs()
        return new_kernel
    
    def UpdateKernel(self, hyperparameterName: str, hyperparameterValue: list, componentIdx: int = 0):
        '''Modify hyperparameters of a sub-kernel component'''
        try:
            name, attrs = hyperparameterName.lower(), self.GetAttrs()
            hyperparameterType = type(hyperparameterValue)
            if name == 'name' and hyperparameterType == str:
                componentNames = self.componentNames
                componentNames[componentIdx] = hyperparameterValue.title()
                super().__setattr__('componentNames', componentNames)
            else:
                attrToUpdate = attrs[hyperparameterName][componentIdx]
                attrType = type(attrToUpdate)
                if attrType == list:
                    if hyperparameterType == list:
                        idx = min(len(attrToUpdate), len(hyperparameterValue))
                        attrToUpdate[:idx] = hyperparameterValue[:idx]
                    elif hyperparameterType in [int, float] and len(attrToUpdate) == 1:
                        attrToUpdate = [hyperparameterValue]
                    else:
                        print('Assignment failed because you tried to set * %s * to %s but it must be %s!' % (name, hyperparameterType, attrType))
                        return
                attrs[hyperparameterName][componentIdx] = attrToUpdate

            for k, v in attrs.items():
                super().__setattr__(k, v)
            self._UpdateOrderedAttrs()
        except Exception as exception:
            print(f'\nAn error occured when updating kernel hyperparameters!\n\n{exception}')
            print('\nRecheck your parameter values!\n')

    def GetAttrs(self) -> dict:
        '''Returns a dict of user-editable attributes'''
        attrs = vars(self).items()
        res = dict()
        for k, v in attrs:
            if k in attrsToCheck:
                res[k] = v
        return res
    
    def _UpdateOrderedAttrs(self, attrs = None):
        numComponents = len(self.components)
        super().__setattr__('_orderedAttrs', [[] for _ in range(numComponents)])
        attrs = self.GetAttrs() if attrs == None else attrs
        for _ in range(numComponents):
            for v in attrs.values():
                self._orderedAttrs[_] += v[_]

    def DimensionCheck(self, x0, x1):
        '''Checks the input data dimension matches that of the kernel'''
        if x0.shape == x1.shape:
            if x0.shape[0] == self.dimension:
                return True
        return False
    
    def FormatInputData(self, x):
        '''Wraps data in a JAX array'''
        if type(x) in [int, float]:
            x = [x]
        elif isinstance(x, Iterable):
            try:
                # this will only trigger if the input is a JAX array or Numpy array
                if len(x.shape) == 0:
                    x = [float(x)]
            except: pass

        return jnp.array(x)

class SE(Kernel):
    def __init__(self, dimension = 1):
        super().__init__(dimension)
        object.__setattr__(self, 'name', 'SE')
        object.__setattr__(self, 'description', 'Squared Exponential with Automatic Relevance Determination')
        object.__setattr__(self, 'amplitude', [[1]])
        object.__setattr__(self, 'lengthscale', [[1] * dimension])
        object.__setattr__(self, 'components', ['SE'])
        object.__setattr__(self, 'componentNames', ['SE'])
        object.__setattr__(self, 'f', kernelfuncs.SE)
        self._UpdateOrderedAttrs()
    
class RQ(Kernel):
    def __init__(self, dimension = 1):
        super().__init__(dimension)
        object.__setattr__(self, 'name', 'RQ')
        object.__setattr__(self, 'description', 'Rational Quadratic which acts like a sum of several SE kernels')
        object.__setattr__(self, 'amplitude', [[1]])
        object.__setattr__(self, 'lengthscale', [[1] * dimension])
        object.__setattr__(self, 'exponent', [[1] * dimension])
        object.__setattr__(self, 'components', ['RQ'])
        object.__setattr__(self, 'componentNames', ['RQ'])
        object.__setattr__(self, 'f', kernelfuncs.RQ)
        self._UpdateOrderedAttrs()

class Per(Kernel):
    def __init__(self, dimension = 1):
        super().__init__(dimension)
        object.__setattr__(self, 'name', 'Per')
        object.__setattr__(self, 'description', 'Periodic with infinite extent')
        object.__setattr__(self, 'amplitude', [[1]])
        object.__setattr__(self, 'lengthscale', [[1] * dimension])
        object.__setattr__(self, 'period', [[1] * dimension])
        object.__setattr__(self, 'components', ['Per'])
        object.__setattr__(self, 'componentNames', ['Per'])
        object.__setattr__(self, 'f', kernelfuncs.Per)
        self._UpdateOrderedAttrs()
    
class M12(Kernel):
    def __init__(self, dimension = 1):
        super().__init__(dimension)
        object.__setattr__(self, 'name', 'M12')
        object.__setattr__(self, 'description', 'Matern 1/2 Kernel for C0-class functions')
        object.__setattr__(self, 'lengthscale', [[1] * dimension])
        object.__setattr__(self, 'components', ['M12'])
        object.__setattr__(self, 'componentNames', ['M12'])
        object.__setattr__(self, 'f', kernelfuncs.M12)
        self._UpdateOrderedAttrs()
    
class M32(Kernel):
    def __init__(self, dimension = 1):
        super().__init__(dimension)
        object.__setattr__(self, 'name', 'M32')
        object.__setattr__(self, 'description', 'Matern 3/2 Kernel for C1-class functions')
        object.__setattr__(self, 'lengthscale', [[1] * dimension])
        object.__setattr__(self, 'components', ['M32'])
        object.__setattr__(self, 'componentNames', ['M32'])
        object.__setattr__(self, 'f', kernelfuncs.M32)
        self._UpdateOrderedAttrs()
    
class M52(Kernel):
    def __init__(self, dimension = 1):
        super().__init__(dimension)
        object.__setattr__(self, 'name', 'M52')
        object.__setattr__(self, 'description', 'Matern 5/2 Kernel for C2-class functions')
        object.__setattr__(self, 'lengthscale', [[1] * dimension])
        object.__setattr__(self, 'components', ['M52'])
        object.__setattr__(self, 'componentNames', ['M52'])
        object.__setattr__(self, 'f', kernelfuncs.M52)
        self._UpdateOrderedAttrs()
    
class Linear(Kernel):
    def __init__(self, dimension = 1):
        super().__init__(dimension)
        object.__setattr__(self, 'name', 'Linear')
        object.__setattr__(self, 'description', 'Best when combined with other kernels')
        object.__setattr__(self, 'amplitude', [[1] * 2 * dimension])
        object.__setattr__(self, 'offset', [[0] * dimension])
        object.__setattr__(self, 'components', ['Linear'])
        object.__setattr__(self, 'componentNames', ['Linear'])
        object.__setattr__(self, 'f', kernelfuncs.Linear)
        self._UpdateOrderedAttrs()