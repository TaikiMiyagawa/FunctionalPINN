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
# References
- Orthogonal polynomials (Wikipedia): https://en.wikipedia.org/wiki/Orthogonal_polynomials

# Note
- Other polynomials, e.g., Jacobi polynomial and its special cases (Gegenbauer polynomial etc.)
  are available here: https://github.com/Orcuslc/OrthNet
- Spherical harmonics is here: https://github.com/cheind/torch-spherical-harmonics
- Bessel functions are hard to compute. Only low-order functions are available:
  torch.special.bessel_j0
  torch.special.bessel_j1
  torch.special.bessel_y0
  torch.special.bessel_y1
  torch.special.modified_bessel_i0
  torch.special.modified_bessel_i1
  torch.special.modified_bessel_k0
  torch.special.modified_bessel_k1
  torch.special.spherical_bessel_j0
  torch.special.scaled_modified_bessel_k0
  torch.special.scaled_modified_bessel_k1
  Note that, however, {sqrt(x)j0(alpha_n x)} over [0, 1] can be compelete.
  See https://mathworld.wolfram.com/CompleteOrthogonalSystem.html.
"""

from functools import partial
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from utils.log_config import get_my_logger

logger = get_my_logger(__name__)


INF = 1e1  # TODO: integral from -inf to inf can be done with torch integral.
EPSILON = 1e-12  # default: 1e-12


def smooth_sum(  # TODO: Restrict the format of w (e.g., w: Tensor = torch.ones_like(x)); otherwise smooth_sum gives an unexpected return shape...
        sum_method: str, w: Optional[Union[Tensor, float]], x: Optional[Tensor],
        n: Optional[Union[int, Tensor]], degree: Optional[int], delta: Optional[float]) -> Tensor:
    """
    # What is sum_method?
    Used for counteracting the Gibbs phenomenon
    (see https://en.wikipedia.org/wiki/Gibbs_phenomenon
    & https://en.wikipedia.org/wiki/Runge%27s_phenomenon).
    Riesz mean (https://en.wikipedia.org/wiki/Riesz_mean) and
    sigma-approximation (https://en.wikipedia.org/wiki/Sigma_approximation)
    with a modification (Riesz' delta is applied) are available.

    # Returns
    - w: A Tensor with shape
        - x: [batch, 1] vs. n: [1, degree]                => [1, degree].
        - x: [batch, spacetime dim, 1] vs. n: [1, degree] => [1, degree].
    """
    if sum_method == "riesz":
        logger.warning(
            f"smooth_sum is currently not recommended (2023/11/13). Unexpected shape would be returned.")
        w = w * (1. - n/degree) ** delta

    elif sum_method == "sigma":
        logger.warning(
            f"smooth_sum is currently not recommended (2023/11/13). Unexpected shape would be returned.")
        w = w * torch.sinc(torch.tensor(torch.pi * n/degree,
                           dtype=x.dtype, device=x.device)) ** delta

    elif sum_method is None:
        pass

    else:
        raise NotImplementedError

    return w


def fourier_series(x: Tensor, n: Union[int, Tensor]) -> Tensor:
    """
    {sin(nx), cos(nx)} is complete over [-pi, pi].
    x in (-inf, inf).
    n = 0, 1, 2,...

    # How to compute coefficients
    fc = lambda x: f(x) * cos(i * x)  # i: dummy index
    fs = lambda x: f(x) * sin(i * x)
    An = []  # defining array
    Bn = []
    sum_ = 0

    for i in range(n):
        an = quad(fc, -np.pi, np.pi)[0] * (1.0 / np.pi)
        An.append(an)

    for i in range(n):
        bn = quad(fs, -np.pi, np.pi)[0] * (1.0 / np.pi)
        Bn.append(bn)  # putting value in array Bn

    for i in range(n):
        if i == 0.0:
            sum_ = sum_ + An[i] / 2
        else:
            sum_ = sum_ + (An[i] * np.cos(i * x) + Bn[i] * np.sin(i * x))

    # Args
    - x: A Tensor with shape [batch, 1] or [batch, spacetime dim, 1].
    - n: An int or a Tensor with shape [1, degree], [degree,] or scalar.

    # Returns
    - f: A Tensor with shape
        - x: [batch, 1] vs. n: [1, degree]                => [batch, degree]
        - x: [batch, spacetime dim, 1] vs. n: [1, degree] => [batch, dim, degre]

    # Caution
    This function sometimes returns f without raising any error even when
    the input shapes are wrong.
    """
    if isinstance(n, int):
        nn = n // 2
        if n % 2 == 0:
            f = torch.cos(nn * x)
        elif n % 2 != 0:
            f = torch.sin(nn * x)
    else:
        phase = n//2 * x
        even = torch.cos(phase)
        odd = torch.sin(phase)
        f = torch.where(n % 2 == 0, even, odd)

    return f


def normalized_weight_fourier(x: Tensor, n: Union[int, Tensor], sum_method=None, degree=200, delta=6) -> Tensor:
    """
    # Returns
    A Tensor  with shape
    - x: [batch, 1] vs. n: [1, degree]                => [batch, degree]
    - x: [batch, spacetime dim, 1] vs. n: [1, degree] => [batch, dim, degre]
    """
    w = torch.ones_like(
        x, dtype=x.dtype, device=x.device)
    if isinstance(n, int):
        c_n = (float(n == 0) + 1.) * torch.pi
    else:
        c_n = ((n == 0).type(torch.get_default_dtype()) + 1.) * torch.pi

    w = smooth_sum(sum_method, w, x, n, degree, delta)

    return w / c_n


def fourier_series_no_w(x: Tensor, n: int) -> Tensor:
    """
    Integral interval = [-0.5, 0.5]
    """
    if isinstance(n, int):
        if n == 0:
            f = torch.ones_like(x, device=x.device, dtype=x.dtype)
        elif n % 2 == 0:
            f = 2. ** 0.5 * torch.cos(torch.pi * n * x)
        elif n % 2 != 0:
            f = torch.sqrt(
                torch.tensor(2., device=x.device, dtype=x.dtype)
            ) * torch.sin(torch.pi * (n+1) * x)

    else:
        even = 2. ** 0.5 * torch.cos(torch.pi * n * x)
        odd = 2. ** 0.5 * torch.sin(torch.pi * (n+1) * x)
        f = torch.where(n % 2 == 0, even, odd)
        f = torch.where(n == 0, 1., f)

    return f


def normalized_weight_fourier_no_w(x: Tensor, n: int, sum_method=None, degree=200, delta=6) -> Union[Tensor, float]:
    w = 1.

    w = smooth_sum(sum_method, w, x, n, degree, delta)

    return w


def laguerre_polynomial(x: Tensor, n: Union[int, Tensor]):
    """
    Complete over
    x in [0, inf).
    n = 0, 1, 2, ...
    """
    return torch.special.laguerre_polynomial_l(x, n)


def normalized_weight_laguerre(x: Tensor, n: Union[int, Tensor], sum_method=None, degree=200, delta=6):
    """
    Orthogonal polynomials satisfy:
    \int_a^b w(x) p_m(x) p_n(x) = \delta_{mn} c_n
    and
    c_n = \int_a^b w(x) p_n(x)^2 dx.
    Then, the normalized weight function for {p_n(x)} is defined as:
    \hat{w}(x) := w(x)/c_n.
    Thus, if
    f(x) = \sum_{n=0}^{\infty} A_n p_n(x),
    then,
    A_n = \int_a^b \hat{w}(x) f(x) p_n(x) dx.
    That is, the normalized weight function facilitates computing A_n.
    (Ref: https://mathworld.wolfram.com/OrthogonalPolynomials.html)
    """
    w = torch.exp(-x)

    w = smooth_sum(sum_method, w, x, n, degree, delta)

    return w


def hermite_polynomial1(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by H ("physicist's Hermite polynomial"). x in (-inf, inf).
    n = 0, 1, 2, ...
    """

    return torch.special.hermite_polynomial_h(x, n)


def normalized_weight_hermite1(x: Tensor, n: Union[int, Tensor], sum_method=None, degree=200, delta=6):
    """
    Orthogonal polynomials satisfy:
    \int_a^b w(x) p_m(x) p_n(x) = \delta_{mn} c_n
    and
    c_n = \int_a^b w(x) p_n(x)^2 dx.
    Then, the normalized weight function for {p_n(x)} is defined as:
    \hat{w}(x) := w(x)/c_n.
    Thus, if
    f(x) = \sum_{n=0}^{\infty} A_n p_n(x),
    then,
    A_n = \int_a^b \hat{w}(x) f(x) p_n(x) dx.
    That is, the normalized weight function facilitates computing A_n.
    (Ref: https://mathworld.wolfram.com/OrthogonalPolynomials.html)
    """
    logscale = - x**2 - 0.5*torch.pi - n * \
        torch.log(torch.tensor(2., device=x.device)) - \
        torch.lgamma(torch.tensor(n+1, device=x.device))
    w = torch.exp(logscale)

    w = smooth_sum(sum_method, w, x, n, degree, delta)

    return w


def hermite_polynomial2(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by He ("probabilist's Hermite polynomial"). x in (-inf, inf).
    n = 0, 1, 2, ...
    """

    return torch.special.hermite_polynomial_he(x, n)


def normalized_weight_hermite2(x: Tensor, n: Union[int, Tensor], sum_method=None, degree=200, delta=6):
    """
    Orthogonal polynomials satisfy:
    \int_a^b w(x) p_m(x) p_n(x) = \delta_{mn} c_n
    and
    c_n = \int_a^b w(x) p_n(x)^2 dx.
    Then, the normalized weight function for {p_n(x)} is defined as:
    \hat{w}(x) := w(x)/c_n.
    Thus, if
    f(x) = \sum_{n=0}^{\infty} A_n p_n(x),
    then,
    A_n = \int_a^b \hat{w}(x) f(x) p_n(x) dx.
    That is, the normalized weight function facilitates computing A_n.
    (Ref: https://mathworld.wolfram.com/OrthogonalPolynomials.html)
    """
    logscale = - 0.5 * x**2 - 0.5*torch.pi - n * \
        torch.log(torch.tensor(2., device=x.device)) - \
        torch.lgamma(torch.tensor(n+1, device=x.device))
    w = torch.exp(logscale)

    w = smooth_sum(sum_method, w, x, n, degree, delta)

    return w


def legendre_polynomial(x: Tensor, n: Union[int, Tensor]):
    """
    Complete over [-1, 1]
    x in [-1, 1], but others acceptable.
    n = 0, 1, 2, ...
    """
    return torch.special.legendre_polynomial_p(x, n)


def normalized_weight_legendre(x: Tensor, n: Union[int, Tensor], sum_method=None, degree=200, delta=6):
    """
    Orthogonal polynomials satisfy:
    \int_a^b w(x) p_m(x) p_n(x) = \delta_{mn} c_n
    and
    c_n = \int_a^b w(x) p_n(x)^2 dx.
    Then, the normalized weight function for {p_n(x)} is defined as:
    \hat{w}(x) := w(x)/c_n.
    Thus, if
    f(x) = \sum_{n=0}^{\infty} A_n p_n(x),
    then,
    A_n = \int_a^b \hat{w}(x) f(x) p_n(x) dx.
    That is, the normalized weight function facilitates computing A_n.
    (Ref: https://mathworld.wolfram.com/OrthogonalPolynomials.html)
    """
    w = n + 0.5

    w = smooth_sum(sum_method, w, x, n, degree, delta)

    return w


def legendre_polynomial_no_w(x: Tensor, n: Union[int, Tensor]):
    """
    Complete over [-1, 1]
    x in [-1, 1], but others acceptable.
    n = 0, 1, 2, ...
    """
    if isinstance(n, int):
        prefactor = np.sqrt(n + 0.5)
    else:
        prefactor = torch.sqrt(n + 0.5)
    return prefactor * torch.special.legendre_polynomial_p(x, n)


def normalized_weight_legendre_no_w(x: Tensor, n: Union[int, Tensor], sum_method=None, degree=200, delta=6):
    """
    Orthogonal polynomials satisfy:
    \int_a^b w(x) p_m(x) p_n(x) = \delta_{mn} c_n
    and
    c_n = \int_a^b w(x) p_n(x)^2 dx.
    Then, the normalized weight function for {p_n(x)} is defined as:
    \hat{w}(x) := w(x)/c_n.
    Thus, if
    f(x) = \sum_{n=0}^{\infty} A_n p_n(x),
    then,
    A_n = \int_a^b \hat{w}(x) f(x) p_n(x) dx.
    That is, the normalized weight function facilitates computing A_n.
    (Ref: https://mathworld.wolfram.com/OrthogonalPolynomials.html)
    """
    w = 1.

    w = smooth_sum(sum_method, w, x, n, degree, delta)

    return w


def chebyshev_polynomial1(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by T_n(cos(x)). |x| <= pi, but others acceptable.
    n = 0, 1, 2, ...
    """
    raise DeprecationWarning
    return torch.special.chebyshev_polynomial_t(torch.cos(x), n)


def normalized_weight_chebyshev1(x: Tensor, n: int, sum_method=None, degree=200, delta=6):
    """
    Orthogonal polynomials satisfy:
    \int_a^b w(x) p_m(x) p_n(x) = \delta_{mn} c_n
    and
    c_n = \int_a^b w(x) p_n(x)^2 dx.
    Then, the normalized weight function for {p_n(x)} is defined as:
    \hat{w}(x) := w(x)/c_n.
    Thus, if
    f(x) = \sum_{n=0}^{\infty} A_n p_n(x),
    then,
    A_n = \int_a^b \hat{w}(x) f(x) p_n(x) dx.
    That is, the normalized weight function facilitates computing A_n.
    (Ref: https://mathworld.wolfram.com/OrthogonalPolynomials.html)
    """
    raise DeprecationWarning
    # T_n(cos(x)) is assumed instead of T_n(x) for numerical stability
    w = torch.ones_like(x, dtype=x.dtype, device=x.device)
    c_n = 0.5*torch.pi + float(n != 0)*0.5*torch.pi

    w = smooth_sum(sum_method, w, x, n, degree, delta)

    return w/c_n


def chebyshev_polynomial1_no_w(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by T_n(cos(theta)). 0 <= theta <= pi, but others acceptable.
    n = 0, 1, 2, ...
    Normalization constant c_n is included, i.e., (T_n, T_m) = delta_{n,m}, not c_n * delta_{n,m}

    # Args
    - x: [batch, 1]
    - n: int or [1, degree]

    # Returns
    - [batch, degree]
    """
    if isinstance(n, int):
        prefactor = np.pi * 0.5 if n != 0 else np.pi
    else:
        prefactor = torch.pi * torch.where(n == 0, 1., 0.5)
    return torch.cos(n * x) / prefactor ** 0.5


def normalized_weight_chebyshev1_no_w(x: Tensor, n: Union[int, Tensor], sum_method=None, degree=200, delta=6):
    """
    Orthogonal polynomials satisfy:
    \int_a^b w(x) p_m(x) p_n(x) = \delta_{mn} c_n
    and
    c_n = \int_a^b w(x) p_n(x)^2 dx.
    Then, the normalized weight function for {p_n(x)} is defined as:
    \hat{w}(x) := w(x)/c_n.
    Thus, if
    f(x) = \sum_{n=0}^{\infty} A_n p_n(x),
    then,
    A_n = \int_a^b \hat{w}(x) f(x) p_n(x) dx.
    That is, the normalized weight function facilitates computing A_n.
    (Ref: https://mathworld.wolfram.com/OrthogonalPolynomials.html)
    """
    w = 1.
    w = smooth_sum(sum_method, w, x, n, degree, delta)
    return w


def chebyshev_polynomial2(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by U. |x| <= 1, but others acceptable.
    n = 0, 1, 2, ...
    """
    return torch.special.chebyshev_polynomial_u(x, n)


def normalized_weight_chebyshev2(x: Tensor, n: Union[int, Tensor], sum_method=None, degree=200, delta=6):
    """
    Orthogonal polynomials satisfy:
    \int_a^b w(x) p_m(x) p_n(x) = \delta_{mn} c_n
    and
    c_n = \int_a^b w(x) p_n(x)^2 dx.
    Then, the normalized weight function for {p_n(x)} is defined as:
    \hat{w}(x) := w(x)/c_n.
    Thus, if
    f(x) = \sum_{n=0}^{\infty} A_n p_n(x),
    then,
    A_n = \int_a^b \hat{w}(x) f(x) p_n(x) dx.
    That is, the normalized weight function facilitates computing A_n.
    (Ref: https://mathworld.wolfram.com/OrthogonalPolynomials.html)
    """
    w = (1 - x**2)**0.5
    c_n = 0.5*torch.pi

    w = smooth_sum(sum_method, w, x, n, degree, delta)

    return w/c_n


def chebyshev_polynomial3(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by V_n(2 * cos(x)^2 - 1). |x| <= pi, but others acceptable.
    n = 0, 1, 2, ...
    """
    return torch.special.chebyshev_polynomial_v(transform_chebyshev3(x), n)


def normalized_weight_chebyshev3(x: Tensor, n: Union[int, Tensor], sum_method=None, degree=200, delta=6):
    """
    Orthogonal polynomials satisfy:
    \int_a^b w(x) p_m(x) p_n(x) = \delta_{mn} c_n
    and
    c_n = \int_a^b w(x) p_n(x)^2 dx.
    Then, the normalized weight function for {p_n(x)} is defined as:
    \hat{w}(x) := w(x)/c_n.
    Thus, if
    f(x) = \sum_{n=0}^{\infty} A_n p_n(x),
    then,
    A_n = \int_a^b \hat{w}(x) f(x) p_n(x) dx.
    That is, the normalized weight function facilitates computing A_n.
    (Ref: https://mathworld.wolfram.com/OrthogonalPolynomials.html)
    """
    # V_n(2cos^2(x)-1) is assumed instead of V_n(x) for numerical stability
    w = torch.ones_like(x, dtype=x.dtype, device=x.device)
    c_n = torch.pi

    w = smooth_sum(sum_method, w, x, n, degree, delta)
    return w/c_n


def chebyshev_polynomial4(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by W. |x| <= 1, but others acceptable.
    n = 0, 1, 2, ...
    """
    return torch.special.chebyshev_polynomial_w(x, n)


def normalized_weight_chebyshev4(x: Tensor, n: Union[int, Tensor], sum_method=None, degree=200, delta=6):
    """
    Orthogonal polynomials satisfy:
    \int_a^b w(x) p_m(x) p_n(x) = \delta_{mn} c_n
    and
    c_n = \int_a^b w(x) p_n(x)^2 dx.
    Then, the normalized weight function for {p_n(x)} is defined as:
    \hat{w}(x) := w(x)/c_n.
    Thus, if
    f(x) = \sum_{n=0}^{\infty} A_n p_n(x),
    then,
    A_n = \int_a^b \hat{w}(x) f(x) p_n(x) dx.
    That is, the normalized weight function facilitates computing A_n.
    (Ref: https://mathworld.wolfram.com/OrthogonalPolynomials.html)
    """
    w = ((1 - x)/(1 + x))**0.5
    c_n = torch.pi

    w = smooth_sum(sum_method, w, x, n, degree, delta)

    return w/c_n


def shifted_chebyshev_polynomial1(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by T. x in [0, 1], but others acceptable.
    n = 0, 1, 2, ...
    """
    return torch.special.shifted_chebyshev_polynomial_t(x, n)


def shifted_chebyshev_polynomial2(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by U. x in [0, 1], but others acceptable.
    n = 0, 1, 2, ...
    """
    return torch.special.shifted_chebyshev_polynomial_u(x, n)


def shifted_chebyshev_polynomial3(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by V. x in [0, 1], but others acceptable.
    n = 0, 1, 2, ...
    """
    return torch.special.shifted_chebyshev_polynomial_v(x, n)


def shifted_chebyshev_polynomial4(x: Tensor, n: Union[int, Tensor]):
    """
    Denoted by W. x in [0, 1], but others acceptable.
    n = 0, 1, 2, ...
    """
    return torch.special.shifted_chebyshev_polynomial_w(x, n)


def id_map(x: Tensor) -> Tensor:
    return x


def transform_chebyshev1(x: Tensor) -> Tensor:
    return torch.cos(x)


def transform_chebyshev3(x: Tensor) -> Tensor:
    return 2 * torch.cos(x)**2 - 1.


def comb(n: Tensor, k: Tensor) -> Tensor:
    return ((n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()).exp()


def bernstein_polynomial(x: Tensor, nu: Union[int, Tensor], n: Union[int, Tensor]) -> Tensor:
    """
    The Bernstein polynomial of n, nu as a function of x
    Ref: https://en.wikipedia.org/wiki/Bernstein_polynomial#Approximating_continuous_functions
    This is numerically unstable for n~>200 due to the combinatorial computation
    (it often gives nan or inf).
    """
    if type(n) != Tensor:
        n_ = torch.tensor(n, dtype=x.dtype, device=x.device)
    else:
        n_ = n
    if type(nu) != Tensor:
        nu_ = torch.tensor(nu, dtype=x.dtype, device=x.device)
    else:
        nu_ = nu
    return comb(n_, nu_) * (x) ** (n_ - nu_) * (1 - x) ** nu_


def chebyshev_node(x: Tensor, domain_of_definition: List[float]):
    """ Numerically instable """
    assert domain_of_definition[0] == - domain_of_definition[1]
    return 1/torch.sqrt(domain_of_definition[1] - x**2 + EPSILON)


class BasisFunctionController():
    def __init__(self, name_basis_function: str, degree: int,
                 sum_method: Optional[str] = None, delta: Optional[float] = 1) -> None:
        """
        # What is sum_method?
          Used for counteracting the Gibbs phenomenon
          (see https://en.wikipedia.org/wiki/Gibbs_phenomenon
          & https://en.wikipedia.org/wiki/Runge%27s_phenomenon).
          Riesz mean (https://en.wikipedia.org/wiki/Riesz_mean) and
          sigma-approximation (https://en.wikipedia.org/wiki/Sigma_approximation)
          with a modification (Riesz' delta is applied) are available.

        # Args
        - name_basis_function: Name of the basis function.
        - degree: The degree of the basis function.
        - sum_method: riesz or sigma or None.
          See https://en.wikipedia.org/wiki/Riesz_mean.
        - delta: Used only for Riesz mean and sigma-approximation.

        # Usage
        bfc = BasisFunctionController(..., degree,...)
        bf = bfc.get_basis_function() # function with input (x, n) -> Tensor w/ shape x.shape
        nwf = bfc.get_normalized_weight_function() # function (x, n) -> Tensor w/ shape x.shape
        dod = bfc.get_domain_of_definition() # List[float, float]
        trf = bfc.get_transform() # function (x,) -> Tensor w/ shape x.shape

        fc_gt = ... ground truth function ...
        dod_gt = ... dod for fc_gt ...

        x_bf = torch.linspace(*dod, num_points, device=device)
        x_bf = trf(x_bf)
        x_fc = torch.linspace(*dod_gt, num_points, device=device)
        y_gt = fc_gt(x_fc)

        coeffs = []
        y_approx = 0.
        for n in range(degree):
            y_bf_n = bf(x_bf, n)
            coeff_n = torch.trapezoid(y_gt *  y_bf_n * nwf(x_bf, n), x=x_bf, dim=-1)
            y_approx += coeff_n * y_bf_n
            coeffs.append(coeff_n)

        mse = torch.nn.functional.mse_loss(y_gt, y_approx)
        """
        assert sum_method in ["riesz", "sigma", None]
        logger.info(
            f"BasisFunctionController: name_basis_function = {name_basis_function}")
        logger.info(f"BasisFunctionController: degree              = {degree}")

        self.sum_method = sum_method
        self.name_basis_function = name_basis_function
        self.delta = delta
        # Deleted: 20231204
        # if name_basis_function in ["fourier", "fourier_no_w"]:
        #     degree *= 2  # sin and cos
        #     logger.info(
        #         f"BasisFunctionController: Degree doubles up because name_basis_function={name_basis_function}: Got {degree//2}. Now {degree}.")
        self.degree = degree

        self.dod: List
        if name_basis_function == "fourier":
            self.basis_function = fourier_series
            self.nwf = partial(normalized_weight_fourier,
                               sum_method=sum_method, degree=degree, delta=delta)
            self.dod = [-torch.pi, torch.pi]
            self.transform = id_map
        elif name_basis_function == "fourier_no_w":
            self.basis_function = fourier_series_no_w
            self.nwf = partial(normalized_weight_fourier_no_w,  # type:ignore
                               sum_method=sum_method, degree=degree, delta=delta)
            self.dod = [-0.5, 0.5]
            self.transform = id_map
        elif name_basis_function == "laguerre":
            self.basis_function = laguerre_polynomial
            self.nwf = partial(normalized_weight_laguerre,
                               sum_method=sum_method, degree=degree, delta=delta)
            self.dod = [0., INF]  # bad approximability
            self.transform = id_map
        elif name_basis_function == "hermite1":
            self.basis_function = hermite_polynomial1
            self.nwf = partial(normalized_weight_hermite1,
                               sum_method=sum_method, degree=degree, delta=delta)
            self.dod = [-INF, INF]  # bad approximability
            self.transform = id_map
        elif name_basis_function == "hermite2":
            self.basis_function = hermite_polynomial2
            self.nwf = partial(normalized_weight_hermite2,
                               sum_method=sum_method, degree=degree, delta=delta)
            self.dod = [-INF, INF]  # bad approximability
            self.transform = id_map
        elif name_basis_function == "legendre":
            self.basis_function = legendre_polynomial
            self.nwf = partial(normalized_weight_legendre,
                               sum_method=sum_method, degree=degree, delta=delta)
            self.dod = [-1., 1.]
            self.transform = id_map
        elif name_basis_function == "legendre_no_w":
            self.basis_function = legendre_polynomial_no_w
            self.nwf = partial(normalized_weight_legendre_no_w,
                               sum_method=sum_method, degree=degree, delta=delta)
            self.dod = [-1., 1.]
            self.transform = id_map
        elif name_basis_function == "chebyshev1":
            raise NotImplementedError
            self.basis_function = chebyshev_polynomial1
            self.nwf = partial(normalized_weight_chebyshev1,
                               sum_method=sum_method, degree=degree, delta=delta)  # wrong?
            self.dod = [-torch.pi, torch.pi]
            # <=====TODO: transform is wrong or usage is wrong or when to linspace (before/after transform)?
            self.transform = transform_chebyshev1
        elif name_basis_function == "chebyshev1_no_w":
            self.basis_function = chebyshev_polynomial1_no_w
            self.nwf = partial(normalized_weight_chebyshev1_no_w,
                               sum_method=sum_method, degree=degree, delta=delta)
            self.dod = [0., np.pi]
            self.transform = id_map
        elif name_basis_function == "chebyshev2":
            self.basis_function = chebyshev_polynomial2
            self.nwf = partial(normalized_weight_chebyshev2,
                               sum_method=sum_method, degree=degree, delta=delta)
            self.dod = [-1., 1.]
            self.transform = id_map
        elif name_basis_function == "chebyshev3":
            raise NotImplementedError
            self.basis_function = chebyshev_polynomial3
            self.nwf = partial(normalized_weight_chebyshev3,
                               sum_method=sum_method, degree=degree, delta=delta)  # wrong?
            self.dod = [-torch.pi, torch.pi]
            # <=========TODO: transform is wrong or usage is wrong or when to linspace (before/after transform)?
            self.transform = transform_chebyshev3
        elif name_basis_function == "chebyshev4":
            self.basis_function = chebyshev_polynomial4
            self.nwf = partial(normalized_weight_chebyshev4,
                               sum_method=sum_method, degree=degree, delta=delta)
            self.dod = [-1., 1.]
            self.transform = id_map
        else:
            raise NotImplementedError(
                f"name_basis_function = {name_basis_function} is invalid.")

    def get_basis_function(self) -> Callable:
        return self.basis_function

    def get_normalized_weight_function(self) -> Callable:
        return self.nwf

    def get_domain_of_definition(self) -> List[float]:
        return self.dod

    def get_transform(self) -> Callable:
        return self.transform
