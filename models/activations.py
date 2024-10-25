# ==========================================================================
# Copyright (c) 2012-2024 Taiki Miyagawa

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
# Reference
- Some of activation functions below were proposed in
  'Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs'
  https://arxiv.org/abs/2111.15135
  'Toward Asymptotic Optimality: Sequential Unsupervised Regression of Density Ratio for Early Classification'
  https://github.com/Akinori-F-Ebihara/SPRT-TANDEM-PyTorch/tree/main

# Note
Some functions supports the adaptive slope:
[A. D. Jagtap, K. Kawaguchi, G. E. Karniadakis, Adaptive activation functions accelerate convergence in deep and physics-informed neural networks, Journal of Computational Physics
404 (2020) 109136.]
[A. D. Jagtap, K. Kawaguchi, G. Em Karniadakis, Locally adaptive activation functions with
slope recovery for deep and physics-informed neural networks, Proceedings of the Royal
Society A 476 (2239) (2020) 20200334].
"""
from typing import Any, Union

import torch
from torch import Tensor, nn

MAGIC_NUMBER = 0.27846455574035645
DTYPE_UFP = Union[float, nn.Parameter]


class ActivationController():
    def __init__(self, activation: str, kwargs: dict) -> None:
        self.activation = activation
        self.kwargs = kwargs

    def get_activation(self):
        if self.activation == "Linear":
            return LinearActivation(**self.kwargs)
        elif self.activation == "Gaussian":
            return GaussianActivation(**self.kwargs)
        elif self.activation == "SuperGaussian":
            return SuperGaussianActivation(**self.kwargs)
        elif self.activation == "Quadratic":
            return QuadraticActivation(**self.kwargs)
        elif self.activation == "MultiQuadratic":
            return MultiQuadraticActivation(**self.kwargs)
        elif self.activation == "Laplacian":
            return LaplacianActivation(**self.kwargs)
        elif self.activation == "ExpSin":
            return ExpSinActivation(**self.kwargs)
        elif self.activation == "Sin":
            return SinActivation(**self.kwargs)
        elif self.activation == "SinSquared":
            return SinSquaredActivation(**self.kwargs)
        elif self.activation == "SinReLU":
            return SinReLUActivation(**self.kwargs)
        elif self.activation == "SinSiLU":
            return SinSiLUActivation(**self.kwargs)
        elif self.activation == "SinGaussian":
            return SinGaussianActivation(**self.kwargs)
        elif self.activation == "Sinc":
            return SincActivation(**self.kwargs)
        elif self.activation == "ELU":
            return ELUActivation(**self.kwargs)
        elif self.activation == "Hardshrink":
            return HardshrinkActivation(**self.kwargs)
        elif self.activation == "Hardsigmod":
            return HardsigmodActivation(**self.kwargs)
        elif self.activation == "Hardtanh":
            return HardtanhActivation(**self.kwargs)
        elif self.activation == "Hardswish":
            return HardswishActivation(**self.kwargs)
        elif self.activation == "LeakyReLU":
            return LeakyReLUActivation(**self.kwargs)
        elif self.activation == "LogSigmoid":
            return LogSigmoidActivation(**self.kwargs)
        elif self.activation == "PReLU":
            return PReLUActivation(**self.kwargs)
        elif self.activation == "ReLU":
            return ReLUActivation(**self.kwargs)
        elif self.activation == "ReLU6":
            return ReLU6Activation(**self.kwargs)
        elif self.activation == "SELU":
            return SELUActivation(**self.kwargs)
        elif self.activation == "CELU":
            return CELUActivation(**self.kwargs)
        elif self.activation == "GELU":
            return GELUActivation(**self.kwargs)
        elif self.activation == "Sigmoid":
            return SigmoidActivation(**self.kwargs)
        elif self.activation == "SiLU":
            return SiLUActivation(**self.kwargs)
        elif self.activation == "Mish":
            return MishActivation(**self.kwargs)
        elif self.activation == "Softplus":
            return SoftplusActivation(**self.kwargs)
        elif self.activation == "Softshrink":
            return SoftshrinkActivation(**self.kwargs)
        elif self.activation == "Softsign":
            return SoftsignActivation(**self.kwargs)
        elif self.activation == "Tanh":
            return TanhActivation(**self.kwargs)
        elif self.activation == "Tanhshrink":
            return TanhshrinkActivation(**self.kwargs)
        elif self.activation == "B2Blog":
            return B2BlogActivation(**self.kwargs)
        elif self.activation == "B2Bcbrt":
            return B2BcbrtActivation(**self.kwargs)
        elif self.activation == "B2Bexp":
            return B2BexpActivation(**self.kwargs)
        elif self.activation == "Tanhplus":
            return TanhplusActivation(**self.kwargs)
        elif self.activation == "DullReLU":
            return DullReLUActivation(**self.kwargs)
        elif self.activation == "SinB2BsqrtV2":
            return SinB2BsqrtV2Activation(**self.kwargs)
        elif self.activation == "B2Bsqrt":
            return B2BsqrtActivation(**self.kwargs)
        elif self.activation == "B2BsqrtV2":
            return B2BsqrtV2Activation(**self.kwargs)
        elif self.activation == "Ricker":
            return RickerActivation(**self.kwargs)
        elif self.activation == "ABU":
            return AdaptiveBlendingUnit(**self.kwargs)
        else:
            raise ValueError(
                f"activation={self.activation} is invalid or not implemented.")


class LinearActivation(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return x


class GaussianActivation(nn.Module):
    def __init__(self, a: float = 1., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)
        # stdev of Gaussian

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- 0.5 * x**2 / self.a**2)


class SuperGaussianActivation(nn.Module):
    """ Not that different from GaussianActivation """

    def __init__(self, a: float = 1.,  b: float = 2., trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=trainable)

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- 0.5 * x**2 / self.a**2) ** self.b


class QuadraticActivation(nn.Module):
    def __init__(self, a: DTYPE_UFP = 1., trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)

    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + (self.a * x) ** 2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a: float = 1., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return 1 / torch.sqrt(1 + (self.a * x) ** 2)


class LaplacianActivation(nn.Module):
    def __init__(self, a: float = 1., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- torch.abs(x) / self.a)


class ExpSinActivation(nn.Module):
    def __init__(self, a: float = 1., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- torch.sin(self.a * x))


class SinActivation(nn.Module):
    def __init__(self, a: DTYPE_UFP = 1., trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.a * x)


class SinSquaredActivation(nn.Module):
    """
    Original. Includes all frequencies.
    """

    def __init__(self, a: float = 2., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin((self.a * x) ** 2)


class SincActivation(nn.Module):
    """
    Caution!
    This activation may return nan depending on the inputs.
    Mostly safe, though.

    # Reference
    - "Function approximation using a sinc neural network" [Proceedings of the SPIE, Volume 2760, p. 690-701 (1996).]
      https://www.spiedigitallibrary.org/conference-proceedings-of-spie/2760/1/Function-approximation-using-a-sinc-neural-network/10.1117/12.235959.short?SSO=1

    # Fourier transform of sinc(ax) = sin(ax) / ax
      is a rectangular function between -a and a
      (https://math.stackexchange.com/questions/736749/fourier-transform-of-sinc-function).
    """

    def __init__(self, a: DTYPE_UFP = 1., trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sinc(self.a * x)


class SinReLUActivation(nn.Module):
    """
    Original.
    """

    def __init__(self, a: float = 2., b: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x) + self.b * torch.sin(self.a * x)


class SinSiLUActivation(nn.Module):
    """
    Original.
    """

    def __init__(self, a: float = 2., b: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x) + self.b * torch.sin(self.a * x)


class SinGaussianActivation(nn.Module):
    """
    Original.
    """

    def __init__(self, a: float = 2., b: float = 0.5, c: float = 1., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=False)
        self.c = nn.Parameter(torch.tensor(c), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- 0.5 * x**2 / self.c**2) + self.b * torch.sin(self.a * x)


class ELUActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class HardshrinkActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.Hardshrink()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class HardsigmodActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class HardtanhActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.Hardtanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class HardswishActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.Hardswish()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class LeakyReLUActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class LogSigmoidActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.LogSigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class PReLUActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class ReLUActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class ReLU6Activation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.ReLU6()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class RReLUActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.RReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class SELUActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.SELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class CELUActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.CELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class GELUActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class SigmoidActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class SiLUActivation(nn.Module):
    """
    Swish activation.
    """

    def __init__(self, a: DTYPE_UFP = 1., trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = nn.Parameter(torch.tensor(
            a, dtype=torch.get_default_dtype(), requires_grad=trainable))
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.a * x)


class MishActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.Mish()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class SoftplusActivation(nn.Module):
    """
    """

    def __init__(self, a: DTYPE_UFP = 1., trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = torch.tensor(
            a, dtype=torch.get_default_dtype(), requires_grad=trainable)
        self.act = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.a * x)


class SoftshrinkActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.Softshrink()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class SoftsignActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.Softshrink()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class TanhActivation(nn.Module):
    """
    """

    def __init__(self, a: DTYPE_UFP = 1., trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = nn.Parameter(torch.tensor(
            a, dtype=torch.get_default_dtype(), requires_grad=trainable))
        self.act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.a * x)


class TanhshrinkActivation(nn.Module):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.Tanhshrink()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)


class B2BsqrtV2Activation(nn.Module):
    """ V2 adds a linear component for anti-gradient vanishing. """

    def __init__(self, alpha: float = 1.0, beta: float = 0.01) -> None:
        super().__init__()
        assert alpha >= 0.
        self.alpha = nn.Parameter(torch.tensor(
            alpha, dtype=torch.float), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(
            beta, dtype=torch.float), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(x) * (
            torch.sqrt(self.alpha + torch.abs(x)) - torch.sqrt(self.alpha) + self.beta * torch.abs(x))


class B2BsqrtActivation(nn.Module):
    """ Suitable for nonlinearity2 in LSTM, instead of tanh. """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        assert alpha >= 0.
        self.alpha = nn.Parameter(torch.tensor(
            alpha, dtype=torch.float), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(x) * (torch.sqrt(self.alpha + torch.abs(x)) - torch.sqrt(self.alpha))


class SinB2BsqrtV2Activation(nn.Module):
    """
    V2 adds a linear component for anti-gradient vanishing.
    Also, is multiplied by sin.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.01, gamma: float = 2.0) -> None:
        super().__init__()
        assert alpha >= 0.
        self.alpha = nn.Parameter(torch.tensor(
            alpha, dtype=torch.float), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(
            beta, dtype=torch.float), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(
            gamma, dtype=torch.float), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.gamma * x) * torch.sign(x) * (
            torch.sqrt(self.alpha + torch.abs(x)) - torch.sqrt(self.alpha) + self.beta * torch.abs(x))/self.gamma


class DullReLUActivation(nn.Module):
    """"""

    def __init__(self, beta: float = 0.05) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(
            beta, dtype=torch.float), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.beta * torch.relu(x)


class TanhplusActivation(nn.Module):
    """Adds a linear component to tanh for anti-gradient vanishing """

    def __init__(self, alpha: float = 10.0, beta: float = 0.02, tau: float = 100.) -> None:
        super().__init__()
        assert tau != 0.
        self.alpha = nn.Parameter(torch.tensor(
            alpha, dtype=torch.float), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(
            beta, dtype=torch.float), requires_grad=False)
        self.tau = nn.Parameter(torch.tensor(
            tau, dtype=torch.float), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.alpha * torch.tanh(x / self.tau) + self.beta * x


class B2BexpActivation(nn.Module):
    """"""

    def __init__(self, alpha: float = 0.03, beta: float = 0.1, tau: float = 1000.) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(
            alpha, dtype=torch.float), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(
            beta, dtype=torch.float), requires_grad=False)
        self.tau = nn.Parameter(torch.tensor(
            tau, dtype=torch.float), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(x) * (
            -self.alpha * torch.exp(-torch.abs(x) / self.tau) +
            self.beta * torch.abs(x) + self.alpha
        )


class B2BcbrtActivation(nn.Module):
    """"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(
            alpha, dtype=torch.float), requires_grad=False)
        self.gama = nn.Parameter(torch.tensor(
            gamma, dtype=torch.float), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(x) * (
            torch.pow(self.alpha + torch.abs(x), 2./3.) -
            torch.pow(self.alpha, 2./3.)
        )


class B2BlogActivation(nn.Module):
    """"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(x) * (torch.log(1 + torch.abs(x)))


class RickerActivation(nn.Module):
    """
    Ricker function.
    Its Fourier spectrum is kind of special: it blows up to a certain frequency, while
    the spectrum of other activations, such as ReLU and tanh, monotonically decays.
    Thus, the Ricker function may alleviate the curse of high frequency (spectral bias).
    Note that with smaller a, the Ricker function decays from a higher frequency,
    but is likely to cause gradient vanishing (vanishes x~a).
    See p.13 in https://arxiv.org/abs/2201.07395 for illustration.
    """

    def __init__(self, a: float = 2.) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(
            a, dtype=torch.float), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.pi * (1 - (x/self.a)**2) * torch.exp(-(x/self.a)**2 / 2) / (15 * self.a)


class PositiveSiLU(nn.Module):
    """ Not in ActivationController because usecases are limited. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.silu = torch.nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.silu(x) + MAGIC_NUMBER)
        return x


class AdaptiveBlendingUnit(nn.Module):
    """
    Original paper: Learning Specialized Activation Functions for Physicsinformed Neural Networks
    URL: https://arxiv.org/abs/2308.04073
    We could not find an official implementation of ABU.
    This is our original implementation.

    # Note
    We often saw nan when we included Sinc.
    """

    def __init__(self, num_act: int = 5, *args, **kwargs) -> None:
        """
        # Args
        - num_act: An int. Number of activations.
        """
        super().__init__(*args, **kwargs)
        assert 1 < num_act
        assert num_act < 6

        self.num_act = num_act

        self.weights = nn.Parameter(torch.zeros(
            num_act, dtype=torch.get_default_dtype(), requires_grad=True))
        self.softmax = nn.Softmax(dim=0)

        if num_act == 2:
            self.scale_sin = 1.
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.scale_tanh = 1.
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.acts = lambda x: torch.stack(
                [self.sin(x), self.tanh(x)], dim=-1)

        elif num_act == 3:
            self.scale_sin = 1.
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.scale_tanh = 1.
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.scale_swish = 1.
            self.swish = SiLUActivation(a=self.scale_swish, trainable=True)
            self.acts = lambda x: torch.stack(
                [self.sin(x), self.tanh(x), self.swish(x)], dim=-1)

        elif num_act == 4:
            self.scale_sin = 1.
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.scale_tanh = 1.
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.scale_swish = 1.
            self.swish = SiLUActivation(a=self.scale_swish, trainable=True)
            self.scale_quadratic = 1.
            self.quadratic = QuadraticActivation(
                a=self.scale_quadratic, trainable=True)
            self.acts = lambda x: torch.stack(
                [self.sin(x), self.tanh(x), self.swish(x), self.quadratic(x)], dim=-1)

        elif num_act == 5:
            self.scale_sin = 1.
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.scale_tanh = 1.
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.scale_swish = 1.
            self.swish = SiLUActivation(a=self.scale_swish, trainable=True)
            self.scale_quadratic = 1.
            self.quadratic = QuadraticActivation(
                a=self.scale_quadratic, trainable=True)
            self.scale_softplus = 1.
            self.softplus = SoftplusActivation(
                a=self.scale_softplus, trainable=True)
            self.acts = lambda x: torch.stack(
                [self.sin(x), self.tanh(x), self.swish(x), self.quadratic(x), self.softplus(x)], dim=-1)

        else:
            raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        """
        # Args
        - x: A Tensor with any shape.

        # Returns
        - out: A Tensor with the same shape.
        """
        weights_softmax = self.softmax(self.weights)  # [num_act,]

        # Shape: [*x.shape, num_act] @ [num_act,] = x.shape
        out = torch.matmul(self.acts(x), weights_softmax)

        return out
