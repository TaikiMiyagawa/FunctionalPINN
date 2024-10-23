# ==========================================================================
# Copyright (c) 2012-2024 Anonymous Authors

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==========================================================================

"""
Elementary and special functions with domain of definition.
x has shape [*, num of points].
"""

from typing import Any, List

import numpy as np
import torch
from torch import Tensor

EPSILON = 1e-10
EPSILON_SIZE = 1e-1
DEFAULT_SIZE = np.pi


def a_b_c_d_e(
        x: Tensor,
        a: float = 2., b: float = 3., c: float = 3.,
        d: float = 3., e: float = -0.5) -> Tensor:
    """ x in (-inf, inf) """
    return (a*x ** b + c) ** d + e


def polynomial(x: Tensor, coeffs: List = [1, 1, 1, 1, 1, 1, 1, 1, 1,]) -> Tensor:
    """ x in (-inf, inf) """
    degree = len(coeffs)
    sum_ = sum([coeffs[i] * x ** i for i in range(degree)])
    return sum_


def exp(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.exp(x)


def log(x: Tensor) -> Tensor:
    """ x in (0, inf) """
    return torch.log(x)  # (0, inf)


def sin(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.sin(x)


def cos(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.cos(x)


def tan(x: Tensor) -> Tensor:
    """ x in (-pi/2, pi/2) is safe; otherwise poles. """
    return torch.tan(x)  # (-pi/2, pi/2) is safe; otherwise poles.


def csc(x: Tensor) -> Tensor:
    return 1. / (torch.sin(x) + EPSILON)


def sec(x: Tensor) -> Tensor:
    return 1. / (torch.cos(x) + EPSILON)


def cot(x: Tensor) -> Tensor:
    return 1. / (torch.tan(x) + EPSILON)


def sinh(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.sinh(x)


def cosh(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.cosh(x)


def tanh(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.tanh(x)


def csch(x: Tensor) -> Tensor:
    return 1. / (torch.sinh(x) + EPSILON)


def sech(x: Tensor) -> Tensor:
    return 1. / (torch.cosh(x) + EPSILON)


def coth(x: Tensor) -> Tensor:
    return 1. / (torch.tanh(x) + EPSILON)


def sinc(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.sinc(x)


def gauss(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.exp(- x ** 2) / np.sqrt(2. * np.pi)


def absolute_value(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.abs(x)


def step(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.heaviside(x, torch.tensor(0.5, device=x.device))


def sawtooth(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    return torch.abs(x % (0.5*np.pi))


def delta_fn1(x: Tensor, N: int = 100):
    """ x in (-inf, inf) """
    terms = [torch.cos(n * x) for n in range(N)]
    output = sum(terms) / torch.pi + 1/(2*torch.pi)
    return output


def delta_fn2(x: Tensor, N: int = 100, L: float = 1.):
    """ x in (-inf, inf) """
    output = torch.sin((2*N + 1) * torch.pi * x / L) / \
        (L * torch.sin(torch.pi * x / L) + EPSILON)
    return output


def delta_fn3(x: Tensor, width=1e-3):
    """ x in (-inf, inf) """
    return gauss(x/2/width) / torch.sqrt(torch.tensor(width, device=x.device))


def log_gamma(x: Tensor) -> Tensor:
    """ x in (0, inf) is safe; otherwise poles. """
    return torch.lgamma(x)


def gamma_fn(x: Tensor) -> Tensor:
    """ x in (0, inf) is safe; otherwise poles. """
    return torch.exp(torch.lgamma(x))


def zeta_fn(x: Tensor) -> Tensor:
    """ x in (1, inf) is safe; otherwise poles. """
    return torch.special.zeta(x, 1.)


def airy(x: Tensor) -> Tensor:
    """ x in (-inf, inf) """
    # https://en.wikipedia.org/wiki/Airy_function
    return torch.special.airy_ai(x)


def polygamma(x: Tensor, n: int = 2) -> Tensor:
    """ x in (0, inf) is safe; otherwise poles. """
    return torch.special.polygamma(n, x)


def erf_inverse(x: Tensor) -> Tensor:
    """ x in (-1, 1) """
    return torch.special.ndtri(x)


def linear(x: Tensor) -> Tensor:
    return x


def quadratic(x: Tensor) -> Tensor:
    return 1. / (1. + (1. * x) ** 2)


def laplacian(x: Tensor) -> Tensor:
    return torch.exp(- torch.abs(x) / 2.)


def expsin(x: Tensor) -> Tensor:
    return torch.exp(- torch.sin(1. * x))


def sinrelu(x: Tensor) -> Tensor:
    return torch.relu(x) + 2. * torch.sin(1. * x)


def sinsilu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x) + 2. * torch.sin(1. * x)


def singauss(x: Tensor) -> Tensor:
    return torch.exp(- 0.5 * x**2 / 0.1**2) + 2. * torch.sin(1. * x)


def elu(x: Tensor) -> Tensor:
    return torch.nn.functional.elu(x)


def hardshrink(x: Tensor) -> Tensor:
    return torch.nn.functional.hardshrink(x)


def hardsigmoid(x: Tensor) -> Tensor:
    return torch.nn.functional.hardsigmoid(x)


def hardswish(x: Tensor) -> Tensor:
    return torch.nn.functional.hardswish(x)


def hardtanh(x: Tensor) -> Tensor:
    return torch.nn.functional.hardtanh(x)


def leakyrelu(x: Tensor) -> Tensor:
    return torch.nn.functional.leaky_relu(x)


def logsigmoid(x: Tensor) -> Tensor:
    return torch.nn.functional.logsigmoid(x)


def relu(x: Tensor) -> Tensor:
    return torch.relu(x)


def relu6(x: Tensor) -> Tensor:
    return torch.nn.functional.relu6(x)


def selu(x: Tensor) -> Tensor:
    return torch.nn.functional.selu(x)


def celu(x: Tensor) -> Tensor:
    return torch.nn.functional.celu(x)


def gelu(x: Tensor) -> Tensor:
    return torch.nn.functional.gelu(x)


def silu(x: Tensor) -> Tensor:
    return torch.nn.functional.silu(x)


def mish(x: Tensor) -> Tensor:
    return torch.nn.functional.mish(x)


def softplus(x: Tensor) -> Tensor:
    return torch.nn.functional.softplus(x)


def softshrink(x: Tensor) -> Tensor:
    return torch.nn.functional.softshrink(x)


def softsign(x: Tensor) -> Tensor:
    return torch.nn.functional.softsign(x)


def b2bsqrt(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.sqrt(1. + torch.abs(x)) - torch.sqrt(torch.tensor(1., device=x.device)))


def ricker(x: Tensor) -> Tensor:
    a = 2.
    return torch.pi * (1 - (x/a)**2) * torch.exp(-(x/a)**2 / 2) / (15 * a)


class FunctionController():
    def __init__(self, name_function: str) -> None:
        self.name_function = name_function

        self.function: Any
        if name_function == "abcde":
            self.function = a_b_c_d_e
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "polynomial":
            self.function = polynomial
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "exp":
            self.function = exp
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "log":
            self.function = log
            self.dod = [0. + EPSILON_SIZE, DEFAULT_SIZE]
        elif name_function == "sin":
            self.function = sin
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "cos":
            self.function = cos
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "tan":
            self.function = tan
            self.dod = [-torch.pi/2. + EPSILON_SIZE,
                        torch.pi/2. - EPSILON_SIZE]
        elif name_function == "csc":
            self.function = csc
            self.dod = [0. + EPSILON_SIZE, torch.pi - EPSILON_SIZE]
        elif name_function == "sec":
            self.function = sec
            self.dod = [-torch.pi/2. + EPSILON_SIZE,
                        torch.pi/2. - EPSILON_SIZE]
        elif name_function == "cot":
            self.function = cot
            self.dod = [0. + EPSILON_SIZE, torch.pi - EPSILON_SIZE]
        elif name_function == "sinh":
            self.function = sinh
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "cosh":
            self.function = cosh
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "tanh":
            self.function = tanh
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "csch":
            self.function = csch
            self.dod = [0. + EPSILON_SIZE, DEFAULT_SIZE]
        elif name_function == "sech":
            self.function = sech
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "coth":
            self.function = coth
            self.dod = [0. + EPSILON_SIZE, DEFAULT_SIZE]
        elif name_function == "sinc":
            self.function = sinc
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "gauss":
            self.function = gauss
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "absolute":
            self.function = absolute_value
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "step":
            self.function = step
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "sawtooth":
            self.function = sawtooth
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "delta1":
            self.function = delta_fn1
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "delta2":
            self.function = delta_fn2
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function in ["delta3", "approx delta fcn"]:
            self.function = delta_fn3
            self.dod = [-0.1, 0.1]
        elif name_function == "log_gamma":
            self.function = log_gamma
            self.dod = [0. + EPSILON_SIZE, DEFAULT_SIZE]
        elif name_function == "gamma":
            self.function = gamma_fn
            self.dod = [0. + EPSILON, DEFAULT_SIZE]
        elif name_function == "zeta":
            self.function = zeta_fn
            self.dod = [1. + EPSILON, 1. + DEFAULT_SIZE]
        elif name_function == "airy":
            self.function = airy
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "polygamma":
            self.function = polygamma
            self.dod = [0. + EPSILON_SIZE, DEFAULT_SIZE]
        elif name_function == "erf_inverse":
            self.function = erf_inverse
            self.dod = [0. + EPSILON, 1. - EPSILON_SIZE]
        elif name_function == "linear":
            self.function = linear
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "quadratic":
            self.function = quadratic
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "laplacian":
            self.function = laplacian
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "expsin":
            self.function = expsin
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "sinrelu":
            self.function = sinrelu
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "sinsilu":
            self.function = sinsilu
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "singauss":
            self.function = singauss
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "elu":
            self.function = elu
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "hardshrink":
            self.function = hardshrink
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "hardsigmoid":
            self.function = hardsigmoid
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "hardswish":
            self.function = hardswish
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "hardtanh":
            self.function = hardtanh
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "leakyrelu":
            self.function = leakyrelu
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "logsigmoid":
            self.function = logsigmoid
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "relu6":
            self.function = relu6
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "selu":
            self.function = selu
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "celu":
            self.function = celu
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "gelu":
            self.function = gelu
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "silu":
            self.function = silu
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "mish":
            self.function = mish
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "softplus":
            self.function = softplus
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "softshrink":
            self.function = softshrink
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "softsign":
            self.function = softsign
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "b2bsqrt":
            self.function = b2bsqrt
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "relu":
            self.function = relu
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        elif name_function == "ricker":
            self.function = ricker
            self.dod = [-DEFAULT_SIZE, DEFAULT_SIZE]
        else:
            raise NotImplementedError(
                f"name_function={name_function} is invalid.")

    def get_function(self):
        return self.function

    def get_domain_of_definition(self):
        return self.dod
