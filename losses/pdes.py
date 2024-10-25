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
# Coding rule for collocation point datasets
All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
and Tensor y with shape [batch, 1+dim_output].
In a mini-batch, X and X_bc (and X_data if available) are included.
X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
They have the shape of [dim_input,].
Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.

# Refs
- Functional derivatives: https://en.wikipedia.org/wiki/Functional_derivative
"""


from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from utils.basis_functions import BasisFunctionController
from utils.log_config import get_my_logger

logger = get_my_logger(__name__)

NUM_POINTS_INTEGRAL = int(1e4)
EPSILON = 1e-12


class GenerateFunctionFromCoefficients():
    """
    Define a function from t and {a_k}, i.e., from X_in.
    Only spacetime dim = 1 is supported currently.
    The domain of definition of the function is assumed to be dod; i.e., it is the same as
    the one of the basis functions.
    Note that the returned values are torch.float64 Tensors.
    """

    def __init__(self, sizes: List, device, degree: int, name_basis_function: str, dim_input: int,
                 sum_method: Optional[str] = None, delta: Optional[float] = 1.) -> None:
        """
        # Args
        - sizes: Ranges of input.
        - device: Device.
        - degree: The degree of the basis function.
        - name_basis_function: Name of the basis function.
        - dim_input: An int. Input dimension. Must be degree + 1, i.e., spacetime dim must be 1.
        - sum_method: riesz or sigma or None. See smooth_sum in utils/basis_funcitons.py.
        - delta: Used only for Riesz mean and sigma-approximation.
        """
        # Initialize
        self.sizes = sizes
        self.device = device
        self.name_basis_function = name_basis_function
        self.sum_method = sum_method
        self.delta = delta

        # Basis function expansion
        self.change_domains_with_sizes = ChangeDomains(
            sizes=sizes, device=device)
        self.bf_controller = BasisFunctionController(
            name_basis_function=name_basis_function,
            degree=degree,
            sum_method=sum_method,
            delta=delta)
        self.degree = self.bf_controller.degree
        self.bf = self.bf_controller.\
            get_basis_function()
        self.nwf = self.bf_controller.\
            get_normalized_weight_function()
        self.dod = self.bf_controller.\
            get_domain_of_definition()  # list[float,float]
        self.trf = self.bf_controller.\
            get_transform()  # function (x,) -> Tensor w/ shape x.shape
        self.degree_tensor = torch.arange(
            0, self.degree, device=device, dtype=torch.float64).unsqueeze(0)  # [1, degree]

        # Error handling
        if dim_input - self.degree != 1:
            raise ValueError(
                f"Currently spacetime dim (:= dim_input - degree) must be 1. Got dim_input={dim_input} and degree={degree}.")

    def __call__(self, X_in: Tensor,
                 t: Optional[Tensor] = None, size_t: Optional[List[float]] = None) -> Tuple[Tensor, Tensor]:
        """
        Define function from t and {a_k}, i.e., from X_in.
        Function values are estimated at X_in[:, 0] if t is None; otherwise, at t.

        # Args
        - X_in: A Tensor with shape [num collocation points, dim_input=degree+1]
        - t: Optional. A Tensor with shape [num collocation points,]. Default is None.
        - size_t: Optional. A list [float, float]. Range of t. Default is None.

        # Returns
        - y: A Tensor with shape [num collocation points,]. function estimated at t.
          The domain of definition of function is assumed to be dod; i.e., it is the same as
          the one of the basis functions.
        - t: A Tensor with shape [num collocation points,]. x in dod.
        """
        if size_t is not None:
            assert t is not None

        X_in = X_in.double()

        # Change range from sizes -> [-1,1] -> dod
        x_normed = self.change_domains_with_sizes.\
            normalize(X_in)  # [num collocation points,dim_input]

        # Define t_change
        t_change: Tensor
        if t is None:
            coeffs = X_in[:, 1:]  # [degree,]
            t_change = change_domains_simple(
                x_normed[:, 0:1], from_=[-1., 1.], to=self.dod)  # [num collocation points, 1]
            t = X_in[:, 0]
        else:
            coeffs = X_in[0, 1:]  # [degree,]
            if size_t is not None:
                t_change = change_domains_simple(
                    t, from_=size_t, to=self.dod).unsqueeze(1)  # [num collocation points, 1]
            else:
                t_change = t.unsqueeze(1)
        t_change = self.trf(t_change)
        t_change = t_change.double()  # [num collocation points, 1]

        # Calc basis functions at tau = X_in[:,0] (rescaled to t_change) with a_k = X_in[:, 1:]
        Fnt = self.bf(
            t_change, self.degree_tensor)  # [num collocation points, degree]
        weights_bf = self.nwf(t_change, self.degree_tensor)

        # Calc function values y
        y = torch.sum(
            coeffs * Fnt * weights_bf,
            dim=1)  # [num collocation points,]

        # Shape = [num collocation points,], [num collocation points,], torch.float64
        return y, t

    def return_callable_function(self, coeffs: Tensor, flag_change_domain: bool = True) -> Callable:
        """
        # Args
        - coeffs: Tensor with shape [degree,]

        # Returns
        - func: A function constructed from basis functions with coefficients=coeffs.
        """
        def func(t: Tensor, flag_change_domain: bool = flag_change_domain):
            """
            - t: Tensor with shape [num points,] or a scalar. The first dimension of the input X_in.
            """
            if flag_change_domain:
                t_dod = change_domains_simple(
                    t, from_=self.sizes[0], to=self.dod)  # [num points] or a scalar
            else:
                t_dod = t  # [num ponts,] or a scalar
            if len(t_dod.shape) != 0:
                t_dod = t_dod.unsqueeze(1)  # [num_points, 1]

            # [num points, degree,] or [degree]
            Fnt = self.bf(t_dod, self.degree_tensor)
            weights_bf = self.nwf(
                t_dod, self.degree_tensor)  # [num points,degree] or [degree]
            y = torch.sum(
                coeffs * Fnt * weights_bf, dim=1)  # [num points, degree] or [degree]- > [num points,] or scalar
            return y
        return func


class GenerateCoefficientsFromFunction():
    """ Deprecated: Use V2 instead (numerical error reduced).
    Calculate expansion coefficients of a function.
    Only spacetime dim = 1 is supported currently.
    The domain of definition of the funciton is assumed to be finite.
    Note that the returned values are torch.float64 Tensors.
    """

    def __init__(self, num_points: int, sizes: List, device, degree: int, name_basis_function: str, dim_input: int,
                 ) -> None:
        """
        # Args
        - num_points: An int. Num of collocation ponts for integral.
        - sizes: A list of lists. Ranges of input.
        - device: Device.
        - degree: The degree of the basis function.
        - name_basis_function: Name of the basis function.
        - dim_input: An int. Input dimension. Must be degree + 1, i.e., spacetime dim must be 1.
        """
        logger.warning(
            "GenerateCoefficientsFromFunction: is deprecated. Use V2 instead.")

        # Initialize
        self.num_points = num_points
        self.sizes = sizes
        self.device = device
        self.name_basis_function = name_basis_function

        # Basis function expansion
        bf_controller = BasisFunctionController(
            name_basis_function=name_basis_function,
            degree=degree)
        self.degree = bf_controller.degree
        bf = bf_controller.\
            get_basis_function()
        nwf = bf_controller.\
            get_normalized_weight_function()
        dod = bf_controller.\
            get_domain_of_definition()  # list[float,float]
        trf = bf_controller.\
            get_transform()  # function (x,) -> Tensor w/ shape x.shape
        degree_tensor = torch.arange(
            0, self.degree,
            device=device).unsqueeze(0)  # [1, degree]

        # Error handling
        if dim_input - self.degree != 1:
            raise ValueError(
                f"Currently spacetime dim (:= dim_input - degree) must be 1. Got dim_input={dim_input} and degree={degree}.")

        # Generate collocation points for integral
        self.cps_sizes = torch.linspace(start=sizes[0][0], end=sizes[0][1], steps=num_points,
                                        device=device, dtype=torch.float64)

        # Change range from sizes[0] to dod
        cps_dod: Tensor = change_domains_simple(
            self.cps_sizes, from_=sizes[0], to=dod)  # [num_points,]
        cps_dod = trf(cps_dod)  # [num_points,]
        self.cps_dod = cps_dod.double()

        # Calculate basis functions of all degrees at self.cps_dod (domain of def = domain of def of basis functions)
        self.weights_bf = nwf(self.cps_dod.unsqueeze(
            1), degree_tensor)  # [num_points, degree]
        self.Pnx = bf(
            self.cps_dod.unsqueeze(1),
            degree_tensor)  # [num_points, degree]

    def __call__(self, function: Callable) -> Tensor:
        """
        # Args
        - function: A function. Tensor -> Tensor.

        # Returns
        - coeffs: A Tensor with shape [degree,]. The coefficients of the
          input function associated the basis functions. torch.float64.
        """
        # Calculate function values at self.cps_sizes (domain of definition is that of function)
        y: Tensor = function(self.cps_sizes)  # [num_points,]
        y = y.unsqueeze(1)  # [num_points, 1]

        # Calculate coefficients. Integral interval is equal to the domain of def of the basis functions (self.cps_dod)
        coeffs = torch.trapezoid(
            y * self.Pnx * self.weights_bf, self.cps_dod.unsqueeze(1),
            dim=0)  # [degree,]

        return coeffs  # [degree,], torch.float64


def GenerateCoefficientsFromFunctionV2(
        function: Callable, sizes: List, name_basis_function: str,
        num_points: int, device, degree: int) -> Tuple[Tensor, int]:
    bfc = BasisFunctionController(
        name_basis_function=name_basis_function, degree=degree)
    degree = bfc.degree
    dod_function = sizes[0]

    fb = bfc.get_basis_function()
    normalized_weight_function = bfc.get_normalized_weight_function()
    dod_fb = bfc.get_domain_of_definition()
    transform = bfc.get_transform()

    x: Tensor = torch.linspace(
        *dod_fb, num_points, device=device, dtype=torch.float64)
    dod_fb_tensor = torch.tensor(dod_fb, device=x.device, dtype=torch.float64)
    x_dod_f: Tensor = (dod_function[1] - dod_function[0]) * ((x - transform(dod_fb_tensor[0])
                                                              ) / transform(dod_fb_tensor[1] - dod_fb_tensor[0])) + dod_function[0]  # from dod to sizes
    y = function(
        x_dod_f.double())  # function values at x_dod_f (domain of def = sizes)

    # y_basis = []
    # yba = y_basis.append
    # nwfs = []
    # nwfs_ap = nwfs.append
    # for n in range(degree):
    #     yba(fb(x, n))
    #     nwfs_ap(normalized_weight_function(x, n))

    # coeffs = []
    # ca = coeffs.append
    # for n in range(degree):
    #     ca(torch.trapezoid(y * y_basis[n] * nwfs[n], x=x, dim=-1))

    coeffs = []
    ca = coeffs.append
    for n in range(degree):
        ca(torch.trapezoid(y * fb(x, n) * normalized_weight_function(x, n), x=x, dim=-1))

    return coeffs, degree  # [degree,], torch.flaot64


def extract_X_bc_from_X_in(X_in: Tensor, y: Tensor, flag_detach: bool = True) -> Tensor:
    """
    Extract X_bc from X_in.
    X_in is composed of X, X_bc, and X_data.
    X_bc has label [1., nan, ...].

    # Coding rule for collocation point datasets
    All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
    and Tensor y with shape [batch, 1+dim_output].
    In a mini-batch, X and X_bc (and X_data if available) are included.
    X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
    X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
    X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
    X, X_bc, and X_data have the shape of [dim_input,].
    Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.

    # Args
    - X_in: A batch of datapoint [batch, dim_input].
    - y: Shape = [batch, 1+dim_output].

    # Returns
    - X_bc: Shape = [num y[:,0]==1., dim_input].
            Collocation points on the boundary.
    """
    idx = torch.where(y[:, 0] == 1.)[0]
    X_bc = X_in[idx]  # [num y[:,0]==1., dim_input]
    if flag_detach:
        X_bc = X_bc.detach()
    return X_bc


def extract_X_bulk_from_X_in(X_in: Tensor, y: Tensor, flag_detach: bool = False) -> Tensor:
    idx = torch.where(y[:, 0] == 0.)[0]
    X_bulk = X_in[idx]  # [num y[:,0]==0., dim_input]
    if flag_detach:
        X_bulk = X_bulk.detach()
    return X_bulk


@torch.enable_grad()
def extract_X_data_from_X_in(X_in: Tensor, y: Tensor) -> Tensor:
    """
    Extract X_data from X_in.
    X_in is composed of X, X_bc, and X_data.
    X_data has label [2., float].

    # Coding rule for collocation point datasets
    All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
    and Tensor y with shape [batch, 1+dim_output].
    In a mini-batch, X and X_bc (and X_data if available) are included.
    X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
    X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
    X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
    They have the shape of [dim_input,].
    Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.

    # Args
    - X_in: A batch of datapoint [batch, dim_input].
    - y: Shape = [batch, 1+dim_output].

    # Returns
    - X_data: Shape = [num X_data = num y[:,0]==2., dim_input].
              Observation data.
    """
    idx = torch.where(y[:, 0] == 2.)[0]
    X_data = X_in[idx]  # [num y[:,0]==2., dim_input]
    return X_data  # [num X_data]


def extract_y_data_from_X_in(y: Tensor) -> Tensor:
    """
    Extract X_data from X_in.
    X_in is composed of X, X_bc, and X_data.
    X_data has label [2., float].

    # Coding rule for collocation point datasets
    All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
    and Tensor y with shape [batch, 1+dim_output].
    In a mini-batch, X and X_bc (and X_data if available) are included.
    X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
    X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
    X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
    They have the shape of [dim_input,].
    Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.

    # Args
    - y: Shape = [batch, 1+dim_output].

    # Returns
    - y_data: Shape = [num y[:,0]==2., dim_out].
    """
    idx = torch.where(y[:, 0] == 2.)[0]
    y_data = y[idx]  # [num y[:,0]==2., dim_input]
    y_data = y_data[:, 1:]  # [num X_data, dim_output]
    return y_data


class PDE(metaclass=ABCMeta):
    """
    Base class of PDEs. Do not forget to define name_pde in __init__.

    # Coding rule for collocation point datasets
    All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
    and Tensor y with shape [batch, 1+dim_output].
    In a mini-batch, X and X_bc (and X_data if available) are included.
    X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
    X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
    X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
    They have the shape of [dim_input,].
    Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.

    # Sizes and boundaries format
    - sizes: Domain of definition. len(sizes) = dim_input
    - boundaries: Regions to which boundary (initial) condiitons to be applied.
      len(boundaries) = number of boundaries
      len(boundaries[*]) = dim_input

    # Examples
    sizes = [
        [0., 1.0],  # t
        [0., 2.0],  # x
    ]

    boundaries = [
        [0., None]        # [t=0, all x]
    ]

    boundaries = [
        [0., None],        # [t=0, all x]
        [None, 2.0]        # [all t, x=2]
    ]

    boundaries = [
        [0., None],        # [t=0, all x]
        [None, [1.0, 2.0]] # [all t, x in [1.0, 2.0]]
    ]
    """
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """ Define the following. """
        self.name_pde: str
        raise NotImplementedError

    @abstractmethod
    def get_name_equation(self, *args, **kwargs) -> str:
        """ Returns self.name_pde. """
        raise NotImplementedError

    @abstractmethod
    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        # Coding rule for collocation point datasets
        All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
        and Tensor y with shape [batch, 1+dim_output].
        In a mini-batch, X and X_bc (and X_data if available) are included.
        X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
        X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
        X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
        They have the shape of [dim_input,].
        Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - y: Shape = [batch, dim_output+1]. Label Tensor.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        raise NotImplementedError

    @abstractmethod
    def get_boundary_condition(self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        raise NotImplementedError

    @abstractmethod
    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        raise NotImplementedError


class PDE1(PDE):
    """
    A time-dependent 1D linear PDE
    Ref: https://www.youtube.com/watch?v=CveZCpDq3Y4

    # Equation
    du/dx - 2du/dt - u = 0.
    u = u(t, x)

    # Boundary (initial) condition
    u(0, x) = 6*e^(-3*x)  for all x.

    # Domain of definition:
      [
          [0., 1.0],  # t
          [0., 2.0],  # x
      ]

    # Analytic solution:
    u(t, x) = 6*exp(-(2*t + 3*x))
    """

    def __init__(self, *args, **kwargs):
        self.name_pde = "PDE1"

    def get_name_equation(self) -> str:
        return self.name_pde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        # Coding rule for collocation point datasets
        All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
        and Tensor y with shape [batch, 1+dim_output].
        In a mini-batch, X and X_bc (and X_data if available) are included.
        X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
        X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
        X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
        They have the shape of [dim_input,].
        Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert len(X_in.shape) == 2
        assert X_in.requires_grad == True

        # Compute du/dt and du/dx
        du_dt_dx_in = torch.autograd.grad(
            u_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]

        # Compute residual
        # Shape = [batch, dim_output]
        lhs = du_dt_dx_in[:, 1:2] - 2 * du_dt_dx_in[:, 0:1]
        rhs = u_in
        res = lhs - rhs
        res_relative = res / (rhs + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)

        return res  # [batch, dim_output]

    def get_boundary_condition(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        assert len(X_in.shape) == 2

        # Extract u_bc from u_in
        # Shape = [num X_bc()<=batch, dim_input]
        u_bc = extract_X_bc_from_X_in(u_in, y, flag_detach=False)

        return u_bc  # [num X_bc, dim_output]

    @torch.no_grad()
    def get_boundary_condition_gt(self, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Hard boundary condition for PDE1.
        u(x, 0) = 6e^(-3x)  for all x.

        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        assert len(X_in.shape) == 2

        # Extract X_bc from X_in
        # Shape = [num X_bc()<=batch, dim_input]
        X_bc = extract_X_bc_from_X_in(X_in, y)

        # Compute u based on BC
        u_bc_gt = 6 * torch.exp(-3 * X_bc[:, 1:2])  # [num X_bc, dim_output]
        u_bc_gt.requires_grad = False

        return u_bc_gt  # [num X_bc, dim_output]

    @torch.no_grad()
    def get_ground_truth(self, X_in: Tensor, *args, **kwargs) -> Tensor:
        """
        X: [batch, dim_input=2]
        Returns analytic solution
        u(t, x1) = 6*exp(-(2*t + 3*x))
        """
        assert len(X_in.shape) == 2
        assert X_in.shape[1] == 2
        u_gt = 6 * torch.exp(-(2*X_in[:, 0] + 3 * X_in[:, 1]))
        return u_gt


class ODE1(PDE):
    """
    Ref: https://www.youtube.com/watch?v=CveZCpDq3Y4
    Simple ODE:
    u(t) s.t. du/dt = -u.
    Initial condition:
    u(0.5) = 1.
    Domain of definition:
      [
          [-1.0, 1.0],  # t
      ]
    Analytic solution:
    u(t) = exp(-(t-0.5))
    """

    def __init__(self, *args, **kwargs):
        """
        """
        self.name_pde = "ODE1"

    def get_name_equation(self) -> str:
        return self.name_pde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        # Coding rule for collocation point datasets
        All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
        and Tensor y with shape [batch, 1+dim_output].
        In a mini-batch, X and X_bc (and X_data if available) are included.
        X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
        X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
        X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
        They have the shape of [dim_input,].
        Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert len(X_in.shape) == 2
        assert X_in.requires_grad == True

        du_dx_in = torch.autograd.grad(
            u_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        lhs = du_dx_in   # [batch, dim_output]
        rhs = - u_in
        res = lhs - rhs
        res_relative = res / (rhs + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)
        return res

    def get_boundary_condition(self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        u_bc = extract_X_bc_from_X_in(u_in, y, flag_detach=False)
        return u_bc  # [num X_bc, dim_output]

    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        assert len(X_in.shape) == 2
        X_bc = extract_X_bc_from_X_in(X_in, y)
        u_bc_gt: Tensor = torch.ones(
            [X_bc.shape[0], 1], device=X_bc.device, requires_grad=False)
        return u_bc_gt  # [num X_bc, dim_output=1]

    @torch.no_grad()
    def get_ground_truth(self, X_in: Tensor, *args, **kwargs) -> Tensor:
        """
        X_in: [batch, dim_input]
        Returns analytic solution
        u(t) = exp(-(t -0.5)) with shape [batch, dim_output]
        """
        assert len(X_in.shape) == 2
        assert X_in.shape[1] == 1
        u_gt = torch.exp(-(X_in[:, 0:1] - 0.5))
        return u_gt


@torch.no_grad()
def uniform_pressure(x: Tensor, p: float) -> Tensor:
    """
    Pressure function measured in [Pa=N/m^2].
    Input shape = [batch, 2]. Output shape = [batch,].

    # Args
    - x: [batch, dim_input]
    - p: Scale of pressure measured in [Pa].

    # Returns
    - Tensor with shape = [batch,] and require_grad=False
    """
    return p * torch.ones_like(x[:, 0])


@torch.no_grad()
def rod_pressure(x: Tensor, p: float, radius: float = 1e-2) -> Tensor:
    """
    Pressure function measured in [Pa=N/m^2].
    Input shape = [batch, 2]. Output shape = [batch,].

        # Args
    - x: [batch, dim_input=2]
    - p: Scale of pressure measured in [Pa].
    - radius: Radius of the rod measured in [m].
      Default is 1e-2 [m] = 1 [cm]

    # Returns
    - Tensor with shape = [batch,] and require_grad=False
    """
    assert x.shape[1] == 2
    r_squared = x.norm(dim=1).norm(dim=1)  # [batch,], x^2+y^2
    pressure = p * \
        torch.where(torch.less_equal(r_squared, radius), 1., 0.)  # [batch,]
    return pressure


class HarmonicOscillator(PDE):
    """
    u(t) s.t. d^2u/dt^2 = - omega u.
    Initial condition:
    u(0) = 1.
    Domain of definition:
      [
          [0., 10.0],  # t
      ]
    Analytic solution:
    u(t) = cos(omega t)
    """

    def __init__(self, omega: float = 2., *args, **kwargs):
        """
        - omega: Angular frequency (spring const/mass) [/s].
          Default is 2.
        """
        self.name_pde = "HarmonicOscillator"
        self.omega = omega

    def get_name_equation(self) -> str:
        return self.name_pde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        # Coding rule for collocation point datasets
        All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
        and Tensor y with shape [batch, 1+dim_output].
        In a mini-batch, X and X_bc (and X_data if available) are included.
        X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
        X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
        X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
        They have the shape of [dim_input,].
        Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert len(X_in.shape) == 2
        assert X_in.requires_grad == True

        du_dt_in = torch.autograd.grad(
            u_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input=1]
        d2u_dt2_in = torch.autograd.grad(
            du_dt_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input=1]

        lhs = d2u_dt2_in  # [batch, dim_output=1]
        rhs = - self.omega**2 * u_in  # [batch, dim_output=1]
        res = lhs - rhs  # [batch, dim_output=1]
        res_relative = res / (res + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)
        return res

    def get_boundary_condition(self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        u_bc = extract_X_bc_from_X_in(u_in, y, flag_detach=False)
        return u_bc  # [num X_bc, dim_output=1]

    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        assert len(X_in.shape) == 2
        X_bc = extract_X_bc_from_X_in(X_in, y)
        u_bc_gt: Tensor = torch.ones(
            [X_bc.shape[0], 1], device=X_bc.device, requires_grad=False)
        return u_bc_gt  # [num X_bc, dim_output=1]

    @torch.no_grad()
    def get_ground_truth(self, X_in: Tensor, *args, **kwargs) -> Tensor:
        """
        X_in: [batch, dim_input]
        Returns analytic solution
        u(t) = exp(-(t -0.5)) with shape [batch, dim_output]
        """
        assert len(X_in.shape) == 2
        assert X_in.shape[1] == 1
        u_gt = torch.cos(self.omega**2 * X_in[:, 0:1])
        return u_gt  # [batch, 1]


class KirchhoffPlateBendingEquation(PDE):
    """
    This is the equation of equilibrium for an isotropic plate bent by external forces acting on it.
    Caution: W and H should be >> 10^3 h because the theory assumes the plate is sufficiently thin.

    # References
    - TensorFlow Notebook (see 'Benchmark PDEs')
      https://towardsdatascience.com/improving-pinns-through-adaptive-loss-balancing-55662759e701
    - Kirchhoff-Love Plate Theory (see 'Isotropic plates')
      https://en.wikipedia.org/wiki/Kirchhoff%E2%80%93Love_plate_theory
    - 'Thery of Elasticity 3ed Edition, Landau and Lifschitz Course of Theoretical Physics Volume 7'
      See equation (12.5) for a detailed derivation and assumptions.

    # Kirchhoff plate bending equation
    - Equation: (d^4/dx^4 + 2d^2/dx^2dy^2 + d^4/dy^4) u(x,y) = p(x,y) / D
                Note that d^4/dx^4 + 2d^2/dx^2dy^2 + d^4/dy^4 = \Delta^2 (Laplacian squared).
    - u(x,y) [m]: the vertical displacement of a point on the neutral surface, i.e., its z corrdinate.
    - p(x,y) [Pa]: the pressure (e.g., by a load) applied on the plate at (x, y).
    - D: Flexural rigidity, cylindrical rigidity, or bending stiffness of the plate.
         The constant D in the equation encapsulates various properties of the plate;
         specifically,
         D = E h^3 / 12(1-nu^2),
         where E [GPa] is Young's modulus
         (see example values here https://en.wikipedia.org/wiki/Young%27s_modulus),
         h [mm] is the thickness of the plate, and nu [dimensionless] is Poisson's ratio
         (see example values here https://en.wikipedia.org/wiki/Poisson%27s_ratio).
    - Domain of definition:
      [
          [0., W],  # x
          [0., H],  # y
      ]
      W and H are the width and height of the plate.
    - Boundary condition for supported edges:
      u(0, y) = u(W, y) = u(x, 0) = u(x, H) = 0 (no displacement at the edge)
      mx(0, y) = mx(W, y) = mx(x, 0) = mx(x, H) = 0 (no moments at the edge)
      where mx := -D * (dudxx + nue * dudyy) and my := -D * (nue * dudxx + dudyy) are moment
      of the stress, and
      W and H are the width and height of the plate.
      For the definition of moments, see here:
      https://en.wikipedia.org/wiki/Kirchhoff%E2%80%93Love_plate_theory#Isotropic_quasistatic_Kirchhoff-Love_plates
      and https://en.wikipedia.org/wiki/Kirchhoff%E2%80%93Love_plate_theory#Equilibrium_equations.
    - Boundary condition for clamped edges:
        u(0, y) = u(W, y) = u(x, 0) = u(x, H) = 0 (no displacement at the edge)
        du/dx(0, y) = du/dx(W, y) = du/dy(x, 0) = du/dy(x, H) = 0 (no bending at the edge)
    - Analytic solution:
      Any reference?

    #  Example parameters
    - Original code
      W = 10 [m]
      H = 10 [m]
      h = 0.2 [mm] (should be much smaller)
      E = 30000 [GPa]
      nu = 0.2 [dimless]
      p(x,y) = 0.15 [Pa]
    - Under gravity
      p(x,y) = rho h g = 0.0980665 [Pa]
      when
      rho: density = 1.0 [g/cm^3 = 10^3 kg/m^3] (water)
      h: thickness  = 0.01 [mm = 10^-3 m]
      g: gravitational acceleration = 9.80665 [m/s^2]

    # Poisson's ratio
    - Large = elongates well in x, hardly compressed in y, when pressed in y-direction.
    - Poisson's ratio of a material defines the ratio of transverse strain (x direction)
      to the axial strain (y direction). Most materials have Poisson's ratio values
      ranging between 0.0 and 0.5. For soft materials, such as rubber, where the bulk modulus
      is much higher than the shear modulus, Poisson's ratio is near 0.5.
      For open-cell polymer foams, Poisson's ratio is near zero
      because the cells tend to collapse in compression. Many typical solids have Poisson's ratios
      in the range of 0.2-0.3.
    - Ref: https://en.wikipedia.org/wiki/Poisson%27s_ratio
    - rubber           0.4999  (density=1.34[g/cm^3])
    - gold             0.42-0.44 (19.3)
    - saturated clay   0.40-0.49 (1.6)
    - magnesium        0.252-0.289 (1.7)
    - titanium         0.265-0.34 (4.5)
    - copper           0.33 (8.3-9.0)
    - aluminium-alloy  0.32 (2.71)
    - clay             0.30-0.45 (1.6)
    - stainless steel  0.30-0.31  (7.5)
    - steel	           0.27-0.30 (7.85)
    - cast iron        0.21-0.26 (7.13)
    - sand             0.20-0.455 (1.32)
    - concrete         0.1-0.2 (2.4)
    - glass	           0.18-0.3 (2.7)
    - metallic glasses 0.276-0.409 (6.882)
    - foam             0.10-0.50 (0.0048-0.42)
    - cork             0.0 (0.7-1.1)

    # Young's modulus (modulus of elasticity) [GPa]
    - Large = hard; Small = soft.
    - It quantifies the relationship between tensile/compressive stress sigma  (force per unit area)
      and axial strain epsilon (proportional deformation) in the linear elastic region
      of a material and is determined using the formula: E = sigma / epsilon.
    - Ref: https://en.wikipedia.org/wiki/Young%27s_modulus#Approximate_values
    - Aluminium (13Al)	68
    - Amino-acid molecular crystals	21 - 44
    - Aramid (for example, Kevlar)	70.5 - 112.4
    - Aromatic peptide-nanospheres	230 - 275
    - Aromatic peptide-nanotubes	19 - 27
    - Bacteriophage capsids	1 - 3
    - Beryllium (4Be)	287
    - Bone, human cortical	14
    - Brass	106
    - Bronze	112
    - Carbon nitride (CN2)	822
    - Carbon-fiber-reinforced plastic (CFRP), 50/50 fibre/matrix, biaxial fabric	30 - 50
    - Carbon-fiber-reinforced plastic (CFRP), 70/30 fibre/matrix, unidirectional, along fibre	181
    - Cobalt-chrome (CoCr)	230
    - Copper (Cu), annealed	110
    - Diamond (C), synthetic	1050 - 1210
    - Diatom frustules, largely silicic acid	0.35 - 2.77
    - Flax fiber	58
    - Float glass	47.7 - 83.6
    - Glass-reinforced polyester (GRP)	17.2
    - Gold	77.2
    - Graphene	1050
    - Hemp fiber	35
    - High-density polyethylene (HDPE)	0.97 - 1.38
    - High-strength concrete	30
    - Lead (82Pb), chemical	13
    - Low-density polyethylene (LDPE), molded	0.228
    - Magnesium alloy	45.2
    - Medium-density fiberboard (MDF)	4
    - Molybdenum (Mo), annealed	330
    - Monel	180
    - Mother-of-pearl (largely calcium carbonate)	70
    - Nickel (28Ni), commercial	200
    - Nylon 66	2.93
    - Osmium (76Os)	525 - 562
    - Osmium nitride (OsN2)	194.99 - 396.44
    - Polycarbonate (PC)	2.2
    - Polyethylene terephthalate (PET), unreinforced	3.14
    - Polypropylene (PP), molded	1.68
    - Polystyrene, crystal	2.5 - 3.5
    - Polystyrene, foam	0.0025 - 0.007
    - Polytetrafluoroethylene (PTFE), molded	0.564
    - Rubber, small strain	0.01 - 0.1
    - Silicon, single crystal, different directions	130 - 185
    - Silicon carbide (SiC)	90 - 137
    - Single-walled carbon nanotube	>1000
    - Steel, A36	200
    - Stinging nettle fiber	87
    - Titanium (22Ti)	116
    - Titanium alloy, Grade 5	114
    - Tooth enamel, largely calcium phosphate	83
    - Tungsten carbide (WC)	600 - 686
    - Wood, American beech	9.5 - 11.9
    - Wood, black cherry	9 - 10.3
    - Wood, red maple	9.6 - 11.3
    - Wrought iron	193
    - Yttrium iron garnet (YIG), polycrystalline	193
    - Yttrium iron garnet (YIG), single-crystal	200
    - Zinc (30Zn)	108
    - Zirconium (40Zr), commercial	95
    """

    def __init__(self, E: float, h: float, nu: float, p: float, type_p: str,
                 type_boundary: str = "supported", *args, **kwargs) -> None:
        """
        # Args
        - E: Young's modulus measured in [GPa].
        - h: Thickness of the plate measured in [mm].
        - nu: Poisson's ratio, which is [dimensionless].
        - p: Pressure function measured in [Pa=N/m^2].
             Input shape = [batch, 2]. Output shape = [batch,].
        - type_p: 'uniform' or 'rod'.
        - type_boundary: 'supported' or 'clamped'. Default is 'supported'.
        """
        assert type_boundary in ["supported", "clamped"]
        self.name_pde = "Kirchhoff"
        self.type_boundary = type_boundary
        self.type_p = type_p

        self.E = E
        self.h = h
        self.nu = nu
        self.p = p
        self.D = E * h**3 / 12 / (1 - nu**2)  # bending stiffness
        if type_p == "uniform":
            self.pressure_fn = partial(uniform_pressure, p=p)
        elif type_p == "rod":
            self.pressure_fn = partial(rod_pressure, p=p)
        else:
            raise NotImplementedError(f"type_p = '{type_p}' is invalid.")

    def get_name_equation(self) -> str:
        return self.name_pde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        # Coding rule for collocation point datasets
        All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
        and Tensor y with shape [batch, 1+dim_output].
        In a mini-batch, X and X_bc (and X_data if available) are included.
        X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
        X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
        X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
        They have the shape of [dim_input,].
        Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert len(X_in.shape) == 2
        assert X_in.requires_grad == True

        # Compute gradients
        du_dx_dy_in = torch.autograd.grad(
            u_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        dudx_in = du_dx_dy_in[:, 0]  # [batch,]
        dudy_in = du_dx_dy_in[:, 1]  # [batch,]
        d2udx_dx_dy_in = torch.autograd.grad(
            dudx_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        d2udy_dx_dy_in = torch.autograd.grad(
            dudy_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        d2udxdx_in = d2udx_dx_dy_in[:, 0]  # [batch,]
        d2udydy_in = d2udy_dx_dy_in[:, 1]  # [batch,]
        d3udxdx_dx_dy_in = torch.autograd.grad(
            d2udxdx_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        d3udydy_dx_dy_in = torch.autograd.grad(
            d2udydy_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        d3udxdxdx_in = d3udxdx_dx_dy_in[:, 0]  # [batch,]
        d3udxdxdy_in = d3udxdx_dx_dy_in[:, 1]  # [batch,]
        d3udydydy_in = d3udydy_dx_dy_in[:, 1]  # [batch,]
        d4udxdxdx_dx_dy_in = torch.autograd.grad(
            d3udxdxdx_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        d4udxdxdy_dx_dy_in = torch.autograd.grad(
            d3udxdxdy_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        d4udydydy_dx_dy_in = torch.autograd.grad(
            d3udydydy_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        d4udxdxdxdx = d4udxdxdx_dx_dy_in[:, 0]  # [batch,]
        d4udxdxdydy = d4udxdxdy_dx_dy_in[:, 1]  # [batch,]
        d4udydydydy = d4udydydy_dx_dy_in[:, 1]  # [batch,]

        lhs = d4udxdxdxdx + 2 * d4udxdxdydy + d4udydydydy
        rhs = self.pressure_fn(X_in) / self.D
        lhs = lhs.unsqueeze(1)  # [batch,1]
        rhs = rhs.unsqueeze(1)  # [batch,1]

        res = lhs - rhs  # [batch,1]
        res_relative = res / (rhs + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)
        return res

    def get_boundary_condition(self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Hard boundary condition.
        u(0, y) = u(W, y) = u(x, 0) = u(x, H) = 0 (no displacement at the edge)
        mx(0, y) = mx(W, y) = mx(x, 0) = mx(x, H) = 0 (no moments at the edge)
        where mx := -D * (dudxx + nue * dudyy) and my := -D * (nue * dudxx + dudyy) are moment
        of the stress, and
        W and H are the width and height of the plate.

        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        assert len(X_in.shape) == 2
        assert X_in.requires_grad == True

        # 0th order boundary condition
        u_bc = extract_X_bc_from_X_in(u_in, y, flag_detach=False)[
            :, 0]  # [num X_bc,]

        # 2nd order boundary condition
        du_dx_dy = torch.autograd.grad(
            u_bc.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        dudx = du_dx_dy[:, 0]  # [batch,]
        dudy = du_dx_dy[:, 1]  # [batch,]
        d2udx_dx_dy = torch.autograd.grad(
            dudx.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        d2udy_dx_dy = torch.autograd.grad(
            dudy.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        d2udxdx = d2udx_dx_dy[:, 0]  # [batch,]
        d2udydy = d2udy_dx_dy[:, 1]  # [batch,]
        mx = -self.D * (d2udxdx + self.nu * d2udydy)  # [batch,]
        my = -self.D * (self.nu * d2udxdx + d2udydy)  # [batch,]
        mx_bc = extract_X_bc_from_X_in(mx, y, flag_detach=False)  # [num X_bc,]
        my_bc = extract_X_bc_from_X_in(my, y, flag_detach=False)  # [num X_bc,]

        # Stack boundary conditions
        um_bc = torch.stack([u_bc, mx_bc, my_bc], dim=1)  # [num X_bc, 3]

        return um_bc

    @torch.no_grad()
    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        assert len(X_in.shape) == 2
        assert X_in.requires_grad == True

        # Compute u based on BC
        X_bc = extract_X_bc_from_X_in(X_in, y)
        u_bc_gt: Tensor = torch.zeros(
            [X_bc.shape[0], 1],
            device=X_bc.device,
            requires_grad=False)  # [num X_bc,1]
        m_bc_gt: Tensor = torch.zeros(
            [X_bc.shape[0], 2],
            device=X_bc.device,
            requires_grad=False)  # [num X_bc,2]
        um_bc_gt = torch.cat([u_bc_gt, m_bc_gt], dim=1)  # [num X_bc,3]
        um_bc_gt.requires_grad = False

        return um_bc_gt  # [num X_bc,3]

    def get_ground_truth(self, X: Tensor) -> Tensor:
        """
        Not available.
        """
        raise NotImplementedError("Not available.")


class HelmholtzEquation(PDE):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError


class BurgerEquation(PDE):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError


class ThomasFermiEquation(PDE):
    pass


class FDE(metaclass=ABCMeta):
    """
    Base class of FDEs (funtional differential equations).
    Do not forget to define name_fde in __init__.

    Ref:
    - https://en.wikipedia.org/wiki/Functional_derivative
    """
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """ Define the following. """
        self.name_fde: str
        self.bfcontroller: BasisFunctionController
        raise NotImplementedError

    @abstractmethod
    def get_name_equation(self, *args, **kwargs) -> str:
        """ Returns self.name_fde. """
        raise NotImplementedError

    @abstractmethod
    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - y: Shape = [batch, dim_output+1]. Label Tensor.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        raise NotImplementedError

    @abstractmethod
    def get_boundary_condition(
            self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        raise NotImplementedError

    @abstractmethod
    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        raise NotImplementedError

    def get_gt_solution(self, X_in: Tensor, *args, **kwargs) -> Tensor:
        """
        Get the ground truth, analytic solution on X_in.
        J(a, t) = J_0(a-vt)
        = rho_0 / L * sum_i=0^degree v_i (a_i - v_i t)
        = v_0 rho_0 / L^2 * sum_i=0^degree a'_i (a_i - v_0 t a'_i / L)

        # Args
        - X_in: A Tensor. A batch of datapoints [batch, dim_input].

        # Returns
        - solution: A Tensor with shape [batch, dim_output].
        """
        raise NotImplementedError


class Integral1(FDE):
    """
    Simple functional learning.

    # Equation
    F([f]) = \int_{-pi/2}^{pi/2} f(x) dx
    x is one-dimensional.

    # Boundary condition
    N/A

    # Domain of definition of x
    [
        [-pi/2, pi/2]
    ]

    # Domain of definition of coefficients
    [
        [-1e2, 1e2]
    ]

    # Orthogonal polynomial expansion
    F([f])  = \int_{-pi/2}^{pi/2} f(x) dx
            = \sum_{k=0}^{degree} a_k,
    where f(x) = \sum_{k=0}^{degree} a_k \phi_k(x).
    {\phi_k} is a set of orthonormal basis functions with domain [-pi/2, pi/2].

    # Reference functions
    - f(x) = sin(x) => F[f] = 0
    - f(x) = cos(x) => F[f] = 2
    - f(x) = x^2 => f[f] = pi^3 /12
    - etc...
    """

    def __init__(self, name_basis_function: str, degree: int, num_points: int, device,
                 a_int: float = - torch.pi/2., b_int: float = torch.pi/2.,
                 sum_method: Optional[str] = None, delta: Union[float, int] = 1.,
                 *args, **kwargs) -> None:
        """
        # Args
        - name_basis_function: Name of the basis function.
        - sum_method: riesz or sigma or None.
          See https://en.wikipedia.org/wiki/Riesz_mean.
        - degree: The degree of the basis function.
        - delta: Used only for Riesz mean and sigma-approximation.
        - num_points: Num of linearly-separated collocation points for numerical integral.
        - device: torch device.
          for numerical integration for coefficients.
        - a_int: Lower bound of the objective integral. Default is -pi/2.
        - b_int: Upper bound of the objective integral. Default is pi/2.
        """
        self.name_basis_function = name_basis_function
        self.sum_method = sum_method
        num_points = int(num_points)
        self.num_points = num_points
        self.delta = delta
        self.a_int = a_int
        self.b_int = b_int
        self.device = device
        self.name_fde = "Integral1"

        # Used for basis functions
        self.bfcontroller = BasisFunctionController(
            name_basis_function=name_basis_function,
            degree=degree,
            sum_method=sum_method, delta=delta)
        self.degree = self.bfcontroller.degree
        self.bf = self.bfcontroller.\
            get_basis_function()  # function with input (x, n) -> Tensor w/ shape x.shape
        # self.nwf = self.bfcontroller.\
        #     get_normalized_weight_function()  # function (x, n) -> Tensor w/ shape x.shape
        # List[float, float]
        self.dod = self.bfcontroller.get_domain_of_definition()
        # function (x,) -> Tensor w/ shape x.shape
        self.trf = self.bfcontroller.get_transform()
        self.x_bf = torch.linspace(
            *self.dod, num_points, device=device)  # type: ignore
        self.x_bf = self.trf(self.x_bf)

        # Used for the objective integral
        self.x_int = torch.linspace(a_int, b_int, num_points, device=device)

        # Calculate basis functions on x_bf
        y_bf_ls = []
        for n in range(self.degree):
            y_bf_n = self.bf(self.x_bf, n)  # [num_points,]
            y_bf_ls.append(y_bf_n)
        self.y_bf = torch.stack(
            y_bf_ls, dim=0).unsqueeze(0).to(device)  # [1, degree, num_points]

    def get_name_equation(self, *args, **kwargs) -> str:
        """ Returns self.name_fde. """
        return self.name_fde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert self.degree == X_in.shape[1], f"degree = {self.degree}, X_in.shape = {X_in.shape}."
        assert len(X_in.shape) == 2, f"len = {len(X_in.shape)}."

        # Computation of y:
        # [batch, dim_input=degree, 1] * [1, degree, num_points]
        # = [batch, degree, num_points]
        # sum with dim=1 => [batch, num_points]
        with torch.no_grad():
            y = torch.sum(
                X_in.unsqueeze(2) * self.y_bf, dim=1)  # [batch, num_points]
            u_gt = torch.trapezoid(y, x=self.x_int, dim=1)  # [batch,]
        res = u_in - u_gt.unsqueeze(1)  # [batch, dim_output=1]
        res_relative = res / (u_gt.unsqueeze(1) + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)
        return res, res_relative

    def get_boundary_condition(
            self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        # N/A
        u_bc = extract_X_bc_from_X_in(u_in, y, flag_detach=False)
        return torch.empty_like(u_bc, device=u_in.device)

    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        # N/A
        return torch.empty([0, 1], device=X_in.device)


class Integral1V2(FDE):
    """
    Simple functional learning. Integral is absent because of the basis function expansion.
    name_basis_function must be in [fourier, legendre]. chebyshev2 is not allowed
    because int chebyshev_k dx is not 0 when k is even.

    # ********************************************************* #
                        Caution!!!!!!
    To stabilize training, we employ self.normalization.
    As a consequence, the learned functional is rescaled.
    The actual functional without this rescaling
    should be u_in * self.normalization instead of u_in itself.
    Pay attention when comparing u_in using analytic solutions.
    See self.get_residual for how self.normalization is used.
    # ********************************************************* #

    # Equation
    F([f]) = \int_{-pi/2}^{pi/2} f(x) dx
    x is one-dimensional.

    # Orthonormal basis function expansion
    F([f])  = \int_{-pi/2}^{pi/2} f(x) dx
            = a_0,
    where f(x) = \sum_{k=0}^{degree} a_k \phi_k(x).
    {\phi_k} is a set of orthonormal basis functions with domain [-pi/2, pi/2].

    # Boundary condition
    N/A

    # Domain of definition of x
    [
        [-pi/2, pi/2]
    ]

    # Domain of definition of coefficients
    [
        [-variable, variable]
    ]

    # Reference functions
    - f(x) = sin(x) => F[f] = 0
    - f(x) = cos(x) => F[f] = 2
    - f(x) = x^2 => f[f] = pi^3 /12
    - etc...
    """

    def __init__(self, name_basis_function: str, degree: int, sizes: List,
                 *args, **kwargs) -> None:
        """
        # Args
        - name_basis_function: Name of the basis function.
        - degree: The degree of the basis function.
        - sizes: Ranges of input.
        """
        if not name_basis_function in ["fourier", "fourier_no_w", "legendre", "legendre_no_w"]:
            raise ValueError(
                f"name_basis_function {name_basis_function} is not allowed.")

        self.name_basis_function = name_basis_function
        self.degree = degree
        self.sizes = sizes

        # Caution!
        # To stabilize training, we employ self.normalization here.
        # As a consequence, the learned functional is rescaled.
        # The actual functional without this rescaling
        # should be u_in * self.normalization instead of u_in itself.
        # See self.get_residual.
        tmp = np.array(sizes)
        tmp = tmp[:, 1] - tmp[:, 0]
        self.normalization = tmp.max() * 1e1
        self.name_fde = "Integral1V2"

    def get_name_equation(self, *args, **kwargs) -> str:
        """ Returns self.name_fde. """
        return self.name_fde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert self.degree == X_in.shape[1], f"degree = {self.degree}, X_in.shape = {X_in.shape}."
        assert len(X_in.shape) == 2, f"len = {len(X_in.shape)}."

        # Note: For the sake of stabilizing training,
        # normalizing the output target is necessary.
        rhs = X_in[:, 0:1].detach()
        rhs = rhs / self.normalization
        res = u_in - rhs  # [batch,dim_output=1]
        res_relative = res / (rhs + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)
        return res, res_relative

    def get_boundary_condition(
            self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        # N/A
        u_bc = extract_X_bc_from_X_in(u_in, y, flag_detach=False)
        return torch.empty_like(u_bc, device=u_in.device)

    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        # N/A
        return torch.empty([0, 1], device=X_in.device)


def change_domains_simple(x: Tensor, from_: List, to: List):
    """ Changes domains of definition: simple version
    # Args
    - x: Shape = any. Float numbers >= from_[0] and <= from_[1].
    - from_: [low, high]
    - to: [low, high]

    # Return
    - x: Shape = any. Float numbers >= to[0] and <=to[1].
    """
    len_to = to[1] - to[0]
    low_to = to[0]
    len_from = from_[1] - from_[0]
    low_from = from_[0]

    x = x - low_from
    x = x / len_from
    x = x * len_to
    x = x + low_to

    return x


class ChangeDomains():
    """
    Normalize to [-1, 1].
    This is considerably important for generalization and stabilization of training.
    """

    def __init__(self, sizes: List, device, *args, **kwargs) -> None:
        """
        # Args
        - x: Shape = [any, dim_input=len(sizes)].
        - sizes: len(sizes) = dim_input. The sizes of the physical system.
        """
        assert sizes is not None
        self.sizes = sizes
        self.device = device

        lengths = [j - i for i, j in sizes]
        lows = [i for i, _ in sizes]
        self.lengths_t = torch.tensor(
            lengths, requires_grad=False, device=device).unsqueeze(0)  # [1, dim_input,]
        self.lows_t = torch.tensor(
            lows, requires_grad=False, device=device).unsqueeze(0)  # [1, dim_input,]

    def normalize(self, x: Tensor):
        """
        # Args
        - x: Shape = [batch, dim_input=len(sizes)].
          Original scale defined in config.

        # Returns
        - x: Shape = [batch, dim_input].
          Normalized to [-1, 1].
        """
        x = ((x - self.lows_t) / self.lengths_t) * 2. - 1.
        return x


class Integral2(FDE):
    """
    Simple functional learning.

    # ********************************************************* #
                        Caution!!!!!!
    To stabilize training, we employ self.normalization.
    As a consequence, the learned functional is rescaled.
    The actual functional without this rescaling
    should be u_in * self.normalization instead of u_in itself.
    Pay attention when comparing u_in using analytic solutions.
    See self.get_residual for how self.normalization is used.
    # ********************************************************* #

    # Equation
    dF([f], t)/dt = f(t)
    t is one-dimensional.

    # Boundary condition
    F([f], -1) = 0

    # Domain of definition of t
    [
        [-1, 1]
    ]

    # Domain of definition of coefficients
    [
        [-variable, variable], ...
    ]

    # Solution
    The solution is F([f], t) = \int_{-1}^{t} f(s) ds.
    """

    def __init__(self, name_basis_function: str, degree: int, device, sizes: List,
                 sum_method: Optional[str] = None, delta: Union[float, int] = 1.,
                 *args, **kwargs) -> None:
        """
        # Args
        - name_basis_function: Name of the basis function.
        - degree: The degree of the basis function.
        - device: torch device.
        - sizes: Ranges of input.
          for numerical integration for coefficients.
        """
        self.name_basis_function = name_basis_function
        self.sum_method = sum_method
        self.delta = delta
        self.device = device
        self.sizes = sizes
        self.name_fde = "Integral2"
        tmp = np.array(sizes)
        tmp = tmp[:, 1] - tmp[:, 0]
        self.normalization = tmp.max() * 1e2
        self.change_domains_with_sizes = ChangeDomains(sizes, device)

        # Used for basis functions
        self.bfcontroller = BasisFunctionController(
            name_basis_function=name_basis_function,
            degree=degree,
            sum_method=sum_method, delta=delta)
        self.degree = self.bfcontroller.degree
        self.bf = self.bfcontroller.\
            get_basis_function()  # function with input (x, n) -> Tensor w/ shape x.shape
        self.nwf = self.bfcontroller.\
            get_normalized_weight_function()  # function (x, n) -> Tensor w/ shape x.shape
        # List[float, float]
        self.dod = self.bfcontroller.get_domain_of_definition()
        # function (x,) -> Tensor w/ shape x.shape
        self.trf = self.bfcontroller.get_transform()

        self.degree_tensor = torch.arange(
            0, self.degree, device=device).unsqueeze(0)  # [1, degree]

    def get_name_equation(self, *args, **kwargs) -> str:
        """ Returns self.name_fde. """
        return self.name_fde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert self.degree + \
            1 == X_in.shape[1], f"degree = {self.degree}, X_in.shape = {X_in.shape}."
        assert len(X_in.shape) == 2, f"len = {len(X_in.shape)}."
        assert X_in.requires_grad == True

        # Left-hand side
        du_dt_da_in = torch.autograd.grad(
            u_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input=1+degree]
        dFdt = du_dt_da_in[:, 0:1]  # [batch, 1]

        # Right-hand side
        with torch.no_grad():
            x = self.change_domains_with_sizes.normalize(
                X_in)  # [batch, dim_input]
            x = change_domains_simple(
                x[:, 0:1], from_=[-1., 1.], to=self.dod)  # [batch, 1]
            x = self.trf(x)  # [batch, 1]
            phi_of_t = self.bf(
                x, self.degree_tensor
            ) * self.nwf(x, self.degree_tensor)  # [batch,degree]
            rhs = X_in[:, 1:] * phi_of_t  # [batch,degree]
            rhs = rhs.sum(dim=1, keepdim=True)  # [batch,1]

        # Residual
        # Note: For the sake of stabilizing training,
        # normalizing the output target (rhs) is necessary.
        res = dFdt - rhs / self.normalization  # [batch,1]
        res_relative = (self.normalization * dFdt - rhs) / (rhs + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)
        return res, res_relative

    def get_boundary_condition(
            self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        u_bc = extract_X_bc_from_X_in(
            u_in, y, flag_detach=False)  # [num X_bc, 1]
        return u_bc

    @torch.no_grad()
    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        X_bc = extract_X_bc_from_X_in(X_in, y)
        u_bc_gt = torch.zeros_like(
            X_bc[:, 0:1], device=X_in.device, dtype=X_in.dtype) / self.normalization
        return u_bc_gt


class LegendreDerivative():
    """ Compute derivatives of Legendre polynomial. """

    def __init__(self, degree: int, device, *args, **kwargs) -> None:
        """
        # Args
        - degree: Degree of Legendre polynomial.
        """
        assert degree > 1
        self.degree = degree
        self.device = device

        # Calc factor matrix
        ls_factor: List = []
        ls_factor_append = ls_factor.append
        for it_col in range(degree//2):
            it_factor = torch.tensor(
                [4 * it_col + 1 + 2*(i % 2) for i in range(degree-1)],
                device=device)  # [degree,]
            it_mask = torch.tensor(
                [1 if i >= it_col*2 else 0 for i in range(degree-1)],
                device=device)  # [degree,]

            ls_factor_append(it_factor * it_mask)
        factor_mat = torch.stack(ls_factor, dim=1)  # [degree-1,degree//2]
        factor_mat = torch.cat([torch.zeros(
            [1, degree//2], device=device), factor_mat], dim=0)  # [degree,degree//2]
        self.factor_mat = factor_mat.unsqueeze(0)  # [1,degree,degree//2]

    @ torch.no_grad()
    def calc_derivative(self, Pnx: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - Pnx: Shape [batch, degree]. Legendre polynomials.

        # Returns
        - Pnx_der: Shape [batch, degree]. Derivatives.
          requires_grad is False.
        """
        Pnx = Pnx.unsqueeze(2)  # [batch, degree, 1]
        Pnx_der_mat = Pnx * self.factor_mat  # [batch, degree, degree//2]
        Pnx_der = Pnx_der_mat.sum(2)  # [batch, degree]
        return Pnx_der


class ArcLength(FDE):
    """
    Arc length.

    # ********************************************************* #
                        Caution!!!!!!
    To stabilize training, we employ self.normalization.
    As a consequence, the learned functional is rescaled.
    The actual functional without this rescaling
    should be u_in * self.normalization instead of u_in itself.
    Pay attention when comparing u_in using analytic solutions.
    See self.get_residual for how self.normalization is used.
    # ********************************************************* #

    # Equation
    dF([f], t)/dt = sqrt(1 + df(t)/dt^2)

    # Boundary condition
    F([f], -1) = 0

    # Domain of definition of t
    [
        [-1, 1]
    ]

    # Domain of definition of coefficients
    [
        [-variable, variable], ...
    ]

    # Solution
    The solution is F([f], t) = \int_{-1}^{t} sqrt(1 + df(s)/ds^2) ds.
    """

    def __init__(self, name_basis_function: str, degree: int, device, sizes: List,
                 sum_method: Optional[str] = None, delta: Union[float, int] = 1.,
                 *args, **kwargs) -> None:
        """
        # Args
        - name_basis_function: Name of the basis function.
        - degree: The degree of the basis function.
        - device: torch device.
        - sizes: Ranges of input.
          for numerical integration for coefficients.
        """
        assert name_basis_function in ["fourier_no_w", "legendre_no_w"]
        self.name_basis_function = name_basis_function
        self.sum_method = sum_method
        self.delta = delta
        self.device = device
        self.sizes = sizes
        self.name_fde = "ArcLength"
        tmp = np.array(sizes)
        tmp = tmp[:, 1] - tmp[:, 0]

        # Used for basis functions
        self.bfcontroller = BasisFunctionController(
            name_basis_function=name_basis_function,
            degree=degree,
            sum_method=sum_method, delta=delta)
        self.degree = self.bfcontroller.degree
        self.bf = self.bfcontroller.\
            get_basis_function()  # function with input (x, n) -> Tensor w/ shape x.shape
        self.nwf = self.bfcontroller.\
            get_normalized_weight_function()  # function (x, n) -> Tensor w/ shape x.shape
        self.dod = self.bfcontroller.get_domain_of_definition(
        )  # List[float, float]
        self.trf = self.bfcontroller.get_transform(
        )  # function (x,) -> Tensor w/ shape x.shape

        self.degree_tensor = torch.arange(
            0, self.degree, device=device).unsqueeze(0)  # [1, degree]
        self.normalization = tmp.max() * self.degree ** 2.5
        self.change_domains_with_sizes = ChangeDomains(
            sizes=sizes, device=device)
        self.legendre_der = LegendreDerivative(self.degree, device)

    def get_name_equation(self, *args, **kwargs) -> str:
        """ Returns self.name_fde. """
        return self.name_fde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        # Scales after self.normalization at initializaiton
        - degree = 10
            rhs.max()
            tensor(0.3458, device='cuda:0', grad_fn=<MaxBackward1>)
            rhs.min()
            tensor(0.0001, device='cuda:0', grad_fn=<MinBackward1>)
            res.max()
            tensor(0.0008, device='cuda:0', grad_fn=<MaxBackward1>)
            res.min()
            tensor(-0.3866, device='cuda:0', grad_fn=<MinBackward1>)
        - degree = 100
            rhs.max()
            tensor(1.1249, device='cuda:0', grad_fn=<MaxBackward1>)
            rhs.min()
            tensor(1.5546e-05, device='cuda:0', grad_fn=<MinBackward1>)
            res.max()
            tensor(0.0138, device='cuda:0', grad_fn=<MaxBackward1>)
            res.min()
            tensor(-1.1167, device='cuda:0', grad_fn=<MinBackward1>)
        - degree = 1000
            rhs.max()
            tensor(3.1853, device='cuda:0', grad_fn=<MaxBackward1>)
            rhs.min()
            tensor(0.0010, device='cuda:0', grad_fn=<MinBackward1>)
            res.max()
            tensor(-0.0076, device='cuda:0', grad_fn=<MaxBackward1>)
            res.min()
            tensor(-3.1936, device='cuda:0', grad_fn=<MinBackward1>)

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert self.degree + \
            1 == X_in.shape[1], f"degree = {self.degree}, X_in.shape = {X_in.shape}."
        assert len(X_in.shape) == 2, f"len = {len(X_in.shape)}."
        assert X_in.requires_grad == True

        # Left-hand side
        du_dt_da_in = torch.autograd.grad(
            u_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input=1+degree]
        dFdt = du_dt_da_in[:, 0:1]  # [batch, 1]

        # Right-hand side
        x = self.change_domains_with_sizes.normalize(
            X_in)  # [batch, dim_input]
        x = change_domains_simple(
            x[:, 0:1], from_=[-1., 1.], to=self.dod)  # [batch, 1]
        x = self.trf(x)  # [batch, 1]
        Pnx = self.bf(
            x, self.degree_tensor) / self.normalization  # [batch, degree]
        Pnx_der = self.legendre_der.calc_derivative(Pnx)  # [batch, degree]
        dfdt = X_in[:, 1:] * Pnx_der * \
            self.nwf(x, self.degree_tensor)  # [batch,degree]
        dfdt = dfdt.sum(dim=1, keepdim=True)  # [batch,1]
        rhs = torch.sqrt(1. / self.normalization ** 2 + dfdt**2)  # [batch,1]

        # Residual
        # Note: For the sake of stabilizing training,
        # normalizing the output target (rhs) is necessary.
        res = dFdt - rhs   # [batch, 1]
        res_relative = (self.normalization * dFdt - rhs) / (rhs + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)
        return res, res_relative

    def get_boundary_condition(
            self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        u_bc = extract_X_bc_from_X_in(
            u_in, y, flag_detach=False)  # [num X_bc, 1]
        return u_bc

    @torch.no_grad()
    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        X_bc = extract_X_bc_from_X_in(X_in, y)
        u_bc_gt = torch.zeros_like(
            X_bc[:, 0:1], device=X_in.device, dtype=X_in.dtype) / self.normalization
        return u_bc_gt


class ThomasFermi(FDE):
    """
    Thomas-Fermi kinetic energy.
    Ref:
    - https://en.wikipedia.org/wiki/Functional_derivative#Thomas%E2%80%93Fermi_kinetic_energy_functional
    - https://en.wikipedia.org/wiki/Thomas%E2%80%93Fermi_model#Kinetic_energy

    # ********************************************************* #
                        Caution!!!!!!
    To stabilize training, we employ self.normalization.
    As a consequence, the learned functional is rescaled.
    The actual functional without this rescaling
    should be u_in * self.normalization instead of u_in itself.
    Pay attention when comparing u_in using analytic solutions.
    See self.get_residual for how self.normalization is used.
    # ********************************************************* #

    # Equation
    d^3 F([rho], x,y,z)/dxdydz = C_kin rho(x,y,z)^{5/3}

    # Boundary condition
    F([rho], x,y,-1) = F([rho], x,-1,z) = F([rho], -1,y,z) = 0

    # Domain of definition of x,y,z
    [
        [-1, 1],
        [-1, 1],
        [-1, 1],
    ]

    # Domain of definition of coefficients
    [
        [-variable, variable], ...
    ]

    # Analytic solution and functional derivative
    The solution is F([rho], x,y,z) = C_kin \int_{-1}^{r} \rho(r)^{5/3} d^3r (r:=(x,y,z)).
    The functional derivative of F at r = (1, 1, 1) is
    \delta F([\rho], (1,1,1)) / \delta \rho (r) = (5 C_kin /3) * rho^{2/3}(r)
    """

    def __init__(self, name_basis_function: str, degree: int, device, sizes: List,
                 C_kin: float = 2e-5,
                 sum_method: Optional[str] = None, delta: Union[float, int] = 1.,
                 *args, **kwargs) -> None:
        """
        # Args
        - name_basis_function: Name of the basis function.
        - degree: The degree of the basis function.
        - device: torch device.
        - sizes: Ranges of input.
          for numerical integration for coefficients.
        - C_kin: Originally, this is equal to 3.5e-38 [J m^2] but can be changed to any value by rescaling F. Default is 2e-5.
        """
        assert name_basis_function == "fourier_no_w"
        self.name_basis_function = name_basis_function
        self.sum_method = sum_method
        self.delta = delta
        self.device = device
        self.sizes = sizes
        self.name_fde = "ThomasFermi"
        self.C_kin = torch.tensor(C_kin, device=device)
        self.change_domains_with_sizes = ChangeDomains(
            sizes=sizes, device=device)

        # Used for basis functions
        self.bfcontroller = BasisFunctionController(
            name_basis_function=name_basis_function,
            degree=degree,
            sum_method=sum_method, delta=delta)
        self.degree = self.bfcontroller.degree
        self.bf = self.bfcontroller.\
            get_basis_function()  # function with input (x, n) -> Tensor w/ shape x.shape
        self.nwf = self.bfcontroller.\
            get_normalized_weight_function()  # function (x, n) -> Tensor w/ shape x.shape
        # List[float, float]
        self.dod = self.bfcontroller.get_domain_of_definition()
        # function (x,) -> Tensor w/ shape x.shape
        self.trf = self.bfcontroller.get_transform()

        self.degree_tensor = torch.arange(
            0, self.degree, device=device).unsqueeze(0)  # [1, degree]
        self.normalization = 10 ** (3*(np.log10(self.degree)-2))

    def get_name_equation(self, *args, **kwargs) -> str:
        """ Returns self.name_fde. """
        return self.name_fde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        # Scales of rhs
        - C_kin=1e-15, degree=10
            rhs.max(), rhs.min()
            (tensor(1.1003, devic...='cuda:0'), tensor(2.4581e-09, d...='cuda:0'))
        - C_kin=1e-15, degree=100
            rhs.max(), rhs.min()
            (tensor(2.2208, devic...='cuda:0'), tensor(1.7503e-07, d...='cuda:0'))
        - C_kin=1e-15, degree=1000
            rhs.max(), rhs.min()
            (tensor(4.3879, devic...='cuda:0'), tensor(4.2704e-10, d...='cuda:0'))

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert self.degree * 3 + \
            3 == X_in.shape[1], f"degree = {self.degree}, X_in.shape = {X_in.shape}."
        assert len(X_in.shape) == 2, f"len = {len(X_in.shape)}."
        assert X_in.requires_grad == True

        # Left-hand side
        dF_dr_da_in = torch.autograd.grad(
            u_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input=1+degree]
        dFdx = dF_dr_da_in[:, 0:1]  # [batch, 1]
        dFdx_dr_da = torch.autograd.grad(
            dFdx.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input=1+degree]
        dFdxdy = dFdx_dr_da[:, 1:2]  # [batch, 1]
        dFdxdy_dr_da = torch.autograd.grad(
            dFdxdy.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input=1+degree]
        dFdxdydz = dFdxdy_dr_da[:, 2:3]  # [batch, 1]

        # Right-hand side
        with torch.no_grad():
            x = self.change_domains_with_sizes.normalize(X_in)
            x = change_domains_simple(
                x[:, 0:3], from_=[-1., 1.], to=self.dod)  # [batch, 3]
            x = self.trf(x).unsqueeze(2)  # [batch, 3, 1]
            basis = self.bf(
                x, self.degree_tensor
            ) * self.nwf(x, self.degree_tensor)  # [batch, 3, degree]
            rho = basis * X_in[:, 3:].reshape(
                -1, 3, self.degree)  # [batch, 3, degree]
            rho = rho.sum(dim=2)  # [batch,3]
            rho = torch.abs(rho)
            rhs = torch.exp(
                torch.log(self.C_kin) + (5/3) *
                torch.log(rho + EPSILON).sum(dim=1, keepdim=True))  # [batch,1]
            # `rhs` is equivalent to:
            # rhs = self.C_kin * torch.prod(rho, dim=1, keepdim=True) ** (5/3) # [batch,1]
            rhs = rhs / self.normalization

        # Residual
        res = dFdxdydz - rhs   # [batch,1]
        res_relative = (dFdxdydz - rhs) / (rhs + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)
        return res, res_relative

    def get_boundary_condition(
            self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 3+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        u_bc = extract_X_bc_from_X_in(
            u_in, y, flag_detach=False)  # [num X_bc, 1]
        return u_bc

    @torch.no_grad()
    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 3+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        X_bc = extract_X_bc_from_X_in(X_in, y)
        u_bc_gt = torch.zeros_like(
            X_bc[:, 0:1], device=X_in.device, dtype=X_in.dtype)  # [num X_bc, 1]
        return u_bc_gt


class GaussianWhiteNoise(FDE):
    """
    Characteristic functional of Gaussian white noise:
    F([theta]) = exp(-int_{-0.5}^{0.5} theta(x)^2 dx / 2).
    Below, W([theta]) := log F([theta]), and u_in is W.
    theta is periodic in [0, 2pi) by definition.

    # ********************************************************* #
                        Caution!!!!!!
    To stabilize training, we employ self.normalization.
    As a consequence, the learned functional is rescaled.
    The actual functional without this rescaling
    should be u_in * self.normalization instead of u_in itself.
    Pay attention when comparing u_in using analytic solutions.
    See self.get_residual for how self.normalization is used.
    # ********************************************************* #

    # Equation
    W(alpha) = - sum_k a_k^2 /2

    # Boundary condition
    N/A

    # Domain of definition of x
    N/A

    # Domain of definition of coefficients
    [
        [-variable, variable], ...
    ]
    """

    def __init__(self, name_basis_function: str, degree: int, device, sizes: List,
                 sum_method: Optional[str] = None, delta: Union[float, int] = 1.,
                 *args, **kwargs) -> None:
        """
        # Args
        - name_basis_function: Name of the basis function.
        - degree: The degree of the basis function.
        - device: torch device.
        - sizes: Ranges of input.
          for numerical integration for coefficients.
        """
        assert name_basis_function == "fourier_no_w", "Only periodic functions in [-0.5, 0.5] are allowed by definition."

        self.name_basis_function = name_basis_function
        self.sum_method = sum_method
        self.delta = delta
        self.device = device
        self.sizes = sizes
        self.name_fde = "GaussianWhiteNoise"
        self.change_domains_with_sizes = ChangeDomains(
            sizes=sizes, device=device)

        # Used for basis functions
        self.bfcontroller = BasisFunctionController(
            name_basis_function=name_basis_function,
            degree=degree,
            sum_method=sum_method, delta=delta)
        self.degree = self.bfcontroller.degree
        self.bf = self.bfcontroller.\
            get_basis_function()
        self.nwf = self.bfcontroller.\
            get_normalized_weight_function()
        self.dod = self.bfcontroller.get_domain_of_definition(
        )  # list. [float, float].
        self.trf = self.bfcontroller.get_transform(
        )

        self.degree_tensor = torch.arange(
            0, self.degree, device=device).unsqueeze(0)  # [1, degree]

        # Define self.normalization
        tmp = np.array(sizes)
        tmp = tmp[:, 1] - tmp[:, 0]
        self.normalization = self.degree * tmp.max() ** 2

    def get_name_equation(self, *args, **kwargs) -> str:
        """ Returns self.name_fde. """
        return self.name_fde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert self.degree == X_in.shape[1] - \
            1, f"degree = {self.degree}, X_in.shape = {X_in.shape}."
        assert len(X_in.shape) == 2, f"len = {len(X_in.shape)}."
        # assert X_in.requires_grad == True # no grad needed for GauWhNoise

        # Left-hand side
        lhs = u_in  # [batch, 1]

        # Right-hand side
        with torch.no_grad():
            rhs = - 0.5 * X_in**2  # [batch, dim_input=degree]
            rhs = rhs.sum(1, keepdim=True)  # [batch, 1]

        # Residual
        res = lhs - rhs / self.normalization  # [batch,1]
        res_relative = (lhs - rhs) / (rhs + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)
        return res, res_relative

    @torch.no_grad()
    def get_boundary_condition(
            self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        N/A

        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        u_bc = extract_X_bc_from_X_in(
            u_in, y, flag_detach=False)  # [num X_bc=0, 1]
        return u_bc

    @torch.no_grad()
    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch=0, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        X_bc = extract_X_bc_from_X_in(X_in, y)
        u_bc_gt = torch.zeros_like(
            X_bc[:, 0:1], device=X_in.device, dtype=X_in.dtype) / self.normalization
        return u_bc_gt  # [num X_bc=0, 1]

    @torch.no_grad()
    def get_gt_solution(self, X_in: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth solution.
        Caution: The ground truth function is normalized with self.normalization.

        # ********************************************************* #
                            Caution!!!!!!
        To stabilize training, we employ self.normalization.
        As a consequence, the learned functional is rescaled.
        The actual functional without this rescaling
        should be u_in * self.normalization instead of u_in itself.
        Pay attention when comparing u_in using analytic solutions.
        See self.get_residual for how self.normalization is used.
        # ********************************************************* #

        # Args
        - X_in: A batch of datapoint [batch, dim_input].

        # Returns
        - gt_solution: A Tensor with shape [batch, dim_output=1,].
          torch.float64.
        """
        rhs = - 0.5 * X_in.double()**2  # [batch, dim_input=degree]
        gt_solution = rhs.sum(1, keepdim=True) / \
            self.normalization  # [batch, 1]

        return gt_solution


class FTransportEquation(FDE):
    """
    Functional transport equation.
    name_basis_function is fourier or legendre. chebyshev2 is not allowed,
    because the weight function is not a constant.

    # Equation
    dJ([X],t)/dt = - int_0^L dy v(y) delta J([X], t) / delta X(y)
    or
    dJ(a,t)/dt = - sum_i=0^degree v_i dJ(a,t)/da_i,
    where v(y) = sum_i=0^degree v_i phi_i(y') and y' is y divided by L=1,
    i.e., y' is dimensionless.

    # Boundary condition
    J([X], 0) = int_0^L dy v(y) rho(X(y), y)
    or
    J(a, 0) = J_0(a)
            = rho_0 / L * sum_i=0^degree v_i a_i
            = v_0 rho_0 / L^2 * sum_i=0^degree a'_i a_i,
    where
    v(y) [m/s] is defined as v_0 y / L,
    rho(x, y) [kg/m^2] is defined as rho_0 x / L, and thus
    v_i (:= int_0^L dy v(y) phi_i(y)) = v_0 a'_i / L (phi_i are orthonormal in [0, L=1]).
    That is, a'_i [m] are the coefficients of the linear function with gradient 1.

    # Domain of definition of t
    [
        [0, 1]
    ]

    # Domain of definition of coefficients a
    [
        [-variable, variable], ...
    ]

    # Analytic solution
    J([X], t) = int_0^L dy v(y) rho(X(y) - v(y)t, y)  [kg/s]
    J(a, t) = J_0(a-vt)
            = rho_0 / L * sum_i=0^degree v_i (a_i - v_i t)
            = v_0 rho_0 / L^2 * sum_i=0^degree a'_i (a_i - v_0 t a'_i / L)
            = v_0 rho_0 / L^2 * (a_1 - v_0 t / L) (if legendre)

    # Deprecated Note: the concern below was addressed. Now a_prime is exact.
    # Note on degree and a_prime
    Too large degree causes numerical error of self.a_prime (the coefficients of the linear function with slope 1).
    10 <~ degree <~ 200 gives almost 0 MSE (function fitting error) for the linear function.
    degree ~ 300 gives 1e-6.
    degree ~ 400 gives 9e-6.
    degree ~ 1000 gives 1e-3.
    """

    def __init__(self, name_basis_function: str, degree: int, device, sizes: List,
                 v_0: float, rho_0: float, L: float = 1., init_cond: str = "linear",
                 sum_method: Optional[str] = None, delta: Union[float, int] = 1.,
                 *args, **kwargs) -> None:
        """
        # Args
        - name_basis_function: Name of the basis function.
        - degree: The degree of the basis function.
        - device: torch device.
        - sizes: Ranges of input.
          for numerical integration for coefficients.
        - sum_method: riesz or sigma or None.
          See https://en.wikipedia.org/wiki/Riesz_mean.
        - degree: The degree of the basis function.
        - v_0: Velocity parameter of the fluid.
        - rho_0: Density parameter of the fluid.
        - L: Dimension of the 2D pipe.
        - init_cond: A str. Specifies the initial condition
        """
        assert L > 0
        assert init_cond in ["linear", "spectrum15"]
        if not name_basis_function in ["legendre_no_w"]:
            raise ValueError(
                "Basis function is assumed to be legendre_no_w for Functional Transport Equation.")

        self.name_basis_function = name_basis_function
        self.v_0 = v_0
        self.rho_0 = rho_0
        self.L = L
        self.device = device
        self.sizes = sizes
        self.name_fde = "FTransportEquation"
        self.init_cond = init_cond
        # tmp = np.array(sizes)
        # tmp = tmp[:, 1] - tmp[:, 0]
        # self.normalization = tmp.max() * 1e1
        # num_points: int = NUM_POINTS_INTEGRAL

        # Used for basis functions
        self.bfcontroller = BasisFunctionController(
            name_basis_function=name_basis_function,
            degree=degree,
            sum_method=sum_method, delta=delta)
        self.degree = self.bfcontroller.degree
        self.bf = self.bfcontroller.\
            get_basis_function()  # function with input (x, n) -> Tensor w/ shape x.shape
        self.nwf = self.bfcontroller.\
            get_normalized_weight_function()  # function (x, n) -> Tensor w/ shape x.shape
        # List[float, float]
        self.dod = self.bfcontroller.get_domain_of_definition()
        # function (x,) -> Tensor w/ shape x.shape
        self.trf = self.bfcontroller.get_transform()

        self.a_prime = torch.zeros(
            [1, self.degree], device=device)  # [1, degree=dim_iput-1]
        if self.init_cond == "linear":
            if name_basis_function == "legendre":
                self.a_prime[:, 1] += 1.
            elif name_basis_function == "legendre_no_w":
                self.a_prime[:, 1] += 1./np.sqrt(1.5)
            else:
                raise ValueError
        elif self.init_cond == "spectrum15":
            if name_basis_function == "legendre":
                for i in range(min(15, self.a_prime.shape[1])):
                    self.a_prime[:, i] += 1.
            elif name_basis_function == "legendre_no_w":
                for i in range(min(15, self.a_prime.shape[1])):
                    self.a_prime[:, i] += 1./np.sqrt(1.5)
            else:
                raise ValueError

    def get_name_equation(self, *args, **kwargs) -> str:
        """ Returns self.name_fde. """
        return self.name_fde

    def get_residual(self, u_in: Tensor, X_in: Tensor,  y: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        dJ(a,t)/dt = - sum_i=0^degree v_i dJ(a,t)/da_i

        When
        v(y) [m/s] is defined as v_0 y / L,
        rho(x, y) [kg/m^2] is defined as rho_0 x / L, and thus
        v_i (:= int_0^L dy v(y) phi_i(y)) = v_0 a'_i / L (phi_i are orthonormal in [0, L=1]),

        then,
        dJ(a,t)/dt = -         sum_i=0^degree  v_i dJ(a,t)/da_i
                   = - v_0/L * sum_i=0^degree a'_i dJ(a,t)/da_i
                   = - v_0/L * dJ(a,t)/da_1 (if legendre)

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert self.degree + \
            1 == X_in.shape[1], f"degree = {self.degree}, X_in.shape = {X_in.shape}."
        assert len(X_in.shape) == 2, f"len = {len(X_in.shape)}."
        assert X_in.requires_grad == True

        # Right-hand side
        dJ_dt_da = torch.autograd.grad(
            u_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]
        dJda = dJ_dt_da[:, 1:]  # [batch, degree=dim_input-1]
        directional_grad = dJda * self.a_prime  # [batch, degree]
        directional_grad = directional_grad.sum(
            dim=1, keepdim=True)  # [batch,1]
        rhs = - self.v_0 / self.L * directional_grad  # [batch,1]

        # Left-hand side
        dJdt = dJ_dt_da[:, 0:1]  # [batch,1]
        lhs = dJdt

        # Residual
        res = lhs - rhs  # [batch, dim_output=1]
        res_relative = (lhs - rhs) / (rhs + EPSILON)
        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)

        return res, res_relative

    def get_boundary_condition(
            self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Boundary condition
        J([X], 0) = int_0^L dy v(y) rho(X(y), y)
        or
        J(a, 0) = J_0(a)
                = rho_0 / L * sum_i=0^degree v_i a_i
                = v_0 rho_0 / L^2 * sum_i=0^degree a'_i a_i,
                = v_0 rho_0 / L^2 * a_1 (if legendre)
        where
        v(y) [m/s] is defined as v_0 y / L,
        rho(x, y) [kg/m^2] is defined as rho_0 x / L, and thus
        v_i (:= int_0^L dy v(y) phi_i(y)) = v_0 a'_i / L (phi_i are orthonormal in [0, L=1]).
        That is, a'_i [m] are the coefficients of the linear function with gradient 1.

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output].
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - bc: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        X_bc = extract_X_bc_from_X_in(
            X_in, y, flag_detach=False)  # [num X_bc, dim_input]
        u_bc = extract_X_bc_from_X_in(
            u_in, y, flag_detach=False)  # [num X_bc, 1]

        lhs = u_bc  # [num X_bc, 1]

        with torch.no_grad():
            if X_bc.shape[0] != 0:
                # [num X_bc, degree=dim_input-1]
                aa = X_bc[:, 1:] * self.a_prime
            else:
                aa = X_bc[:, 1:]
            _sum = aa.sum(dim=1, keepdim=True)  # [num X_bc,1]
            rhs = self.v_0 * self.rho_0 / \
                self.L**2 * _sum  # [num X_bc,1]

        bc = lhs - rhs

        return bc

    @torch.no_grad()
    def get_boundary_condition_gt(self,  X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        # Args
        - X_in: A batch of datapoints [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, num conditions].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        X_bc = extract_X_bc_from_X_in(X_in, y)
        u_bc_gt = torch.zeros_like(
            X_bc[:, 0:1], device=X_in.device, dtype=X_in.dtype)
        return u_bc_gt  # [num X_bc, 1=dim_output=num bc conditions]

    def get_gt_solution(self, X_in: Tensor, flag_no_grad: bool = True, *args, **kwargs) -> Tensor:
        """
        Get the ground truth, analytic solution on X_in.
        J(a, t) = J_0(a-vt)
        = rho_0 / L * sum_i=0^degree v_i (a_i - v_i t)
        = v_0 rho_0 / L^2 * sum_i=0^degree a'_i (a_i - v_0 t a'_i / L)
        = v_0 rho_0 / L^2 * (a_1 - v_0 t / L) (if legendre)

        # Args
        - X_in: A Tensor. A batch of datapoints [batch, dim_input].
        - flag_no_grad: A bool. Detach or not.

        # Returns
        - solution: A Tensor with shape [batch, dim_output].
        """
        prefac = self.v_0 * self.rho_0 / self.L**2
        solution = prefac * torch.sum(
            self.a_prime * (X_in[:, 1:] - self.v_0 *
                            X_in[:, 0:1] * self.a_prime / self.L),
            dim=1,
            keepdim=True)  # [batch, dim_output=1]
        if flag_no_grad:
            return solution.detach()
        else:
            return solution


class BurgersHopfEquation(FDE):
    """
    1-dimensional Hopf equation without the advection term.
    Boundary condition is Gaussian.
    Dimensionless and log-style equation; W := log Phi.
    The basis function is assumed to be fourier_no_w (periodic functions).

    # Equation
    d W(alpha, tau)/d tau = alpha^T C_tilde (d W/d alpha)
    where
    C_tilde is a matrix and the i, j component of C_tilde is given by
    = - pi^2 (i+1)^2 (j=j=odd)
      - pi^2 i^2     (i=j=nonzero even)
      0            (otherwise)

    # Boundary condition (Gaussian initial condition) with "delta"
    W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} \sum_{i=0}^{degree-1} alpha_i^2
    and
    W(0, tau) \equiv 0 (identity).

    # Boundary condition (Gaussian initial condition) with "const"
    W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} alpha_0^2
    and
    W(0, tau) \equiv 0 (identity).

    # Boundary condition (Gaussian initial condition) with "moderate"
    W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} (\sum_{i=0}^{99} \exp(-i) alpha_i)^2
    and
    W(0, tau) \equiv 0 (identity).

    # Domain of definition of t
    [
        [-variable, variable],
    ]

    # Domain of definition of coefficients
    [
        [-variable, variable], ...
    ]

    # Analytic solution
    \Phi ([\theta], t) = \Phi_0 ([\hat{\theta}_t(x)]),
    where\hat{\theta}_t(x) := \frac{1}{\sqrt{4 \pi \nu t}} \int_{-\infty}^{\infth} dx^\prime e^{\frac{(x - x^\prime)^2}{4 \pi \nu t}}\theta(x^\prime),
    and \Phi := exp W and \Phi_0 := exp W_0 with a slight abuse of notation.
    20231209: This expression can be further simplified:
    W([\Theta], \tau) = - \bar{\mu} \Theta_0 + \frac{\sigma^2}{2}\sum_{n=0}^{M-1} (\exp(-8\pi^2 n^2 \tau) \Theta_{2n}^2 + \exp(-8\pi^2 (n+1)^2 \tau) \Theta_{2n+1}^2),
    where degree = 2M (set degree=M in the config file).
    """

    def __init__(self, name_basis_function: str, degree: int, device, sizes: List,
                 mu_bar: float, sigma_squared: float, init_cond: str,
                 sum_method: Optional[str] = None, delta: Union[float, int] = 1.,
                 flag_hard_bc: bool = False,
                 *args, **kwargs) -> None:
        """
        # Args
        - name_basis_function: Name of the basis function.
        - degree: The degree of the basis function.
        - device: torch device.
        - sizes: Ranges of input.
          for numerical integration for coefficients.
        - mu_bar: A float. Mean of the initial Gaussian.
        - sigma_squared: A positive float. Strength ("hight" of the delta function covariance) of the initial Gaussian.
        - init_cond: A str. Specifies the initial condition.
        """
        assert sigma_squared > 0
        assert sizes[0][0] >= 0
        assert init_cond in ["delta", "const", "moderate"]
        if name_basis_function != "fourier_no_w":
            raise ValueError(
                f"name_basis_function must be 'fourier_no_w'. Got {name_basis_function}.")

        self.name_basis_function = name_basis_function
        self.sum_method = sum_method
        self.degree = degree
        self.delta = delta
        self.device = device
        self.sizes = sizes
        self.name_fde = "BurgersHopfEquation"
        self.change_domains_with_sizes = ChangeDomains(
            sizes=sizes, device=device)
        self.mu_bar = mu_bar
        self.sigma_squared = sigma_squared
        self.init_cond = init_cond
        self.flag_hard_bc = flag_hard_bc

        # Initial condition
        if self.init_cond == "delta":
            n_power = torch.tensor(
                [i // 2 for i in range(self.degree)][1:] + [self.degree//2], device=device)
            self.exp_factors_gtsol_delta = - 8 * torch.pi**2 * n_power**2
            self.exp_factors_gtsol_delta = \
                self.exp_factors_gtsol_delta.unsqueeze(0)  # [1, degree]

        elif self.init_cond == "const":
            pass

        elif self.init_cond == "moderate":
            self.num_terms = 100
            assert self.num_terms >= self.degree, f"self.num_terms must be >= self.degree. Got {self.num_terms} and {self.degree}."
            self.exp_factors_moderate = - \
                torch.arange(1, self.num_terms+1, device=device)/20
            self.exp_factors_moderate = self.exp_factors_moderate.unsqueeze(
                0)  # [1, num_terms]
            self.exp_prefactors_gtsol_moderate = \
                self.exp_factors_moderate[:, :self.degree]  # [1,degree]

            n_power = torch.tensor(
                [i // 2 for i in range(self.degree)][1:] + [self.degree//2], device=device)
            self.exp_factors_gtsol_moderate = - 4 * torch.pi**2 * n_power**2
            self.exp_factors_gtsol_moderate = \
                self.exp_factors_gtsol_moderate.unsqueeze(0)  # [1, degree]

        else:
            raise ValueError

        self.calc_init_cond_ready2go = lambda x: self.calc_init_cond_BHE(x)

        # Hard initial and boundary condition
        if self.flag_hard_bc:
            self.heaviside_tau, self.heaviside_coeff = self.calc_smooth_heaviside()

        # Used for basis functions
        self.bfcontroller = BasisFunctionController(
            name_basis_function=name_basis_function,
            degree=degree,
            sum_method=sum_method, delta=delta)
        self.dim_input = self.degree + 1
        self.bf = self.bfcontroller.\
            get_basis_function()  # function with input (x, n) -> Tensor w/ shape x.shape
        self.nwf = self.bfcontroller.\
            get_normalized_weight_function()  # function (x, n) -> Tensor w/ shape x.shape
        self.dod = self.bfcontroller.get_domain_of_definition(
        )  # List[float, float]
        self.trf = self.bfcontroller.get_transform(
        )  # function(x,) -> Tensor w/ shape x.shape

        self.degree_tensor = torch.arange(
            0, self.degree, device=device).unsqueeze(0)  # [1, degree]
        self.C_tilde = self.calc_C_tilde()  # [degree, degree] on GPU.

    def calc_C_tilde(self) -> Tensor:
        """ Covariance matrix associated with the Gaussian initialization.
        # Returns
        - C_tilde: A diag matrix Tensor with shape [degree, degree].
        """
        odd = torch.tensor(
            [torch.pi**2 * (i + 1) ** 2 if i %
             2 == 1 else 0. for i in range(self.degree)],
            device=self.device, dtype=torch.get_default_dtype())  # [degree,]
        even = torch.tensor(
            [torch.pi**2 * i ** 2 if i %
                2 == 0 else 0. for i in range(self.degree)],
            device=self.device, dtype=torch.get_default_dtype())  # [degree,]
        diag = odd + even  # [degree,]
        C_tilde = - torch.diag(diag)  # [degree, degree]

        return C_tilde

    def calc_smooth_heaviside(self) -> Tuple[Callable, Callable]:
        one = torch.tensor([[1.]], device=self.device)  # [1,1]
        interval_tau = self.sizes[0][1] - self.sizes[0][0]
        interval_coeff = self.sizes[1][1] - self.sizes[1][0]
        eps_tau = interval_tau / 100.
        eps_coeff = interval_coeff * self.degree / 100.
        def dist_tau(X_in: Tensor): return torch.abs(X_in[:, 0:1])  # [batch,1]

        def dist_coeff(X_in: Tensor) -> Tensor: return torch.norm(
            X_in[:, 1:], p=1, dim=1, keepdim=True) / self.degree  # [batch,1]

        def heaviside_tau(X_in: Tensor) -> Tensor:
            return torch.where(
                dist_tau(X_in) >= eps_tau,
                one,
                0.5 * (1 - torch.cos(dist_tau(X_in) * torch.pi / eps_tau)))  # [batch,1]

        def heaviside_coeff(X_in: Tensor) -> Tensor:
            return torch.where(
                dist_coeff(X_in) >= eps_coeff,
                one,
                0.5 * (1 - torch.cos(dist_coeff(X_in) * torch.pi / eps_coeff)))  # [batch,1]

        return heaviside_tau, heaviside_coeff

    def wrap_u_in_with_heaviside(self, u_in: Tensor, X_in: Tensor) -> Tensor:
        u_bc_all = self.get_boundary_condition_gt(
            X_in, None, flag_extract_X_bc=False, flag_detach=False)  # [batch,1], requre_grad=True
        assert u_bc_all.requires_grad == True
        h_coeff = self.heaviside_coeff(X_in)  # [batch,1]
        h_tau = self.heaviside_tau(X_in)  # [batch,1]

        u_in_hard = h_tau * u_in + (1. - h_tau) * u_bc_all
        u_in_hard = h_coeff * u_in_hard + (1. - h_coeff) * u_bc_all

        return u_in_hard  # [batch, 1]

    def get_name_equation(self, *args, **kwargs) -> str:
        """ Returns self.name_fde. """
        return self.name_fde

    def get_residual(self, u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        # Equation
        d W(alpha, tau)/d tau = alpha^T C_tilde (d W/d alpha)
        where
        C_tilde is a matrix and the i, j component of C_tilde is given by
        = pi^2 (i+1)^2 (j=j=odd)
          pi^2 i^2     (i=j=nonzero even)
          0            (otherwise)

        # Args
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.
        - X_in: Shape = [batch, dim_input]. X, X_bc, and X_data are in X_in.
        - u_in: A predicted scalar output u_in = model(X_in). Shape=(batch_size, dim_output),
          i.e., a batch of scalar functions evaluated at X_in.

        # Returns
        - res: Residual. L.H.S. of the PDE. Goes to 0 as training proceeds.
          Shape = [batch, dim_output].

        # Remarks
        - create_graph=True is necessary to compute higher order derivatives.
        See https://qiita.com/tmasada/items/9dee38e5bc1482217493.
        """
        assert self.degree == X_in.shape[1] - \
            1, f"degree = {self.degree}, X_in.shape = {X_in.shape}."
        assert len(X_in.shape) == 2, f"len = {len(X_in.shape)}."
        assert X_in.requires_grad == True

        if self.flag_hard_bc:
            u_in = self.wrap_u_in_with_heaviside(u_in, X_in)

        # Calc derivatives
        dW_dtau_dalpha = torch.autograd.grad(
            u_in.sum(dim=0), X_in,
            create_graph=True, allow_unused=False)[0]  # [batch, dim_input]

        # Right-hand side
        dWdalpha = dW_dtau_dalpha[:, 1:]  # [batch, degree=dim_input-1]
        CdW = torch.tensordot(dWdalpha, self.C_tilde, dims=[
                              [1], [1]])  # [batch, degree]
        rhs = torch.sum(X_in[:, 1:] * CdW, dim=1, keepdim=True)  # [batch, 1]

        # Left-hand side
        dWdtau = dW_dtau_dalpha[:, 0:1]  # [batch,1]
        lhs = dWdtau

        # Residual
        res = lhs - rhs  # [batch,1]
        res_relative = (lhs - rhs) / (rhs + EPSILON)

        res = extract_X_bulk_from_X_in(res, y)
        res_relative = extract_X_bulk_from_X_in(res_relative, y)

        return res, res_relative

    def get_boundary_condition(
            self,  u_in: Tensor, X_in: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute boundary conditions.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} \sum_{i=0}^{degree-1} alpha_i^2
        and
        W(0, tau) \equiv 0 (identity).

        # Args
        - u_in: Model output u_in = model(X_in) with shape [batch, dim_output=1].
        - X_in: A batch of datapoint [batch, dim_input=1+degree].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc: Shape = [num X_bc in batch, dim_output=1].
          The ground truth value of u on the boundary.
        """
        if self.flag_hard_bc:
            u_in = self.wrap_u_in_with_heaviside(u_in, X_in)

        u_bc = extract_X_bc_from_X_in(
            u_in, y, flag_detach=False)  # [num X_bc, dim_output=1]
        return u_bc

    def get_boundary_condition_gt(
            self,  X_in: Tensor, y: Tensor,
            flag_extract_X_bc: bool = True, flag_detach: bool = True, *args, **kwargs) -> Tensor:
        """
        Get ground truth boundary values.
        Outputs from get_boundary_condition and get_boundary_condition_gt
        should be equal after training.

        W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} \sum_{i=0}^{degree-1} alpha_i^2
        and
        W(0, tau) \equiv 0 (identity).

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - y: Shape = [batch, 1+dim_output].

        # Returns
        - u_bc_gt: Shape = [num X_bc in batch, dim_output=1].
          The ground truth value of u on the boundary.
          num conditions depend on the boundary condition.
        """
        if flag_extract_X_bc:
            X_bc = extract_X_bc_from_X_in(X_in, y, flag_detach=flag_detach)
        else:
            X_bc = X_in

        if X_bc.shape[0] == 0:
            if flag_detach:
                # Shape=[num X_bc=0, dim_output=1]
                return X_bc[:, 0:1].detach()
            else:
                return X_bc[:, 0:1]

        else:
            it_ic = self.calc_init_cond_ready2go(X_bc)
            u_bc_gt = torch.where(
                X_bc[:, 0:1] == 0.,
                it_ic, 0.)  # [num X_bc, 1]

        if flag_detach:
            return u_bc_gt.detach()
        else:
            return u_bc_gt  # [num X_bc, 1=dim_output]

    def calc_init_cond_BHE(self, X_bc: Tensor) -> Tensor:
        """
        Calculates the initial condition at X_bc.
        delta, const, or moderate.

        # Boundary condition (Gaussian initial condition) with "delta"
        W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} \sum_{i=0}^{degree-1} alpha_i^2
        and
        W(0, tau) \equiv 0 (identity).

        # Boundary condition (Gaussian initial condition) with "const"
        W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} alpha_0^2
        and
        W(0, tau) \equiv 0 (identity).

        # Boundary condition (Gaussian initial condition) with "moderate"
        W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} (\sum_{i=0}^{99} \exp(-i) alpha_i)^2
        and
        W(0, tau) \equiv 0 (identity).

        # Args
        - X_bc: A Tensor with shape [num X_bc in X_in = X_bc.shape[0], dim_input=1+degree].

        # Returns
        - init_cond: A Tensor with shape [X_bc.shape[0], dim_output=1].
        """
        if self.init_cond == "delta":
            init_cond = self.calc_init_cond_delta(X_bc)
        elif self.init_cond == "const":
            init_cond = self.calc_init_cond_const(X_bc)
        elif self.init_cond == "moderate":
            init_cond = self.calc_init_cond_moderate(X_bc)
        else:
            raise ValueError(f"init_cond = {self.init_cond} is not supported.")

        return init_cond  # [num X_bc, dim_output=1]

    def calc_init_cond_delta(self, X_bc: Tensor) -> Tensor:
        """
        # Boundary condition (Gaussian initial condition) with "delta"
        W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} \sum_{i=0}^{degree-1} alpha_i^2
        and
        W(0, tau) \equiv 0 (identity).

        # Args
        - X_bc: A Tensor with shape [num X_bc in X_in = X_bc.shape[0], dim_input=1+degree].

        # Returns
        - init_cond: A Tensor with shape [X_bc.shape[0], dim_output=1].
        """
        init_cond = - self.mu_bar * X_bc[:, 1:2] + 0.5 * \
            self.sigma_squared * \
            torch.sum(X_bc[:, 1:] ** 2, dim=1, keepdim=True)

        return init_cond  # [num X_bc, dim_output=1]

    def calc_init_cond_const(self, X_bc: Tensor) -> Tensor:
        """
        # Boundary condition (Gaussian initial condition) with "const"
        W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} alpha_0^2
        and
        W(0, tau) \equiv 0 (identity).

        # Args
        - X_bc: A Tensor with shape [num X_bc in X_in = X_bc.shape[0], dim_input=1+degree].

        # Returns
        - init_cond: A Tensor with shape [X_bc.shape[0], dim_output=1].
        """
        init_cond = - self.mu_bar * X_bc[:, 1:2] + 0.5 * \
            self.sigma_squared * X_bc[:, 1:2] ** 2

        return init_cond  # [num X_bc, dim_output=1]

    def calc_init_cond_moderate(self, X_bc: Tensor) -> Tensor:
        """
        # Boundary condition (Gaussian initial condition) with "moderate"
        W(alpha, tau=0) = W_0(alpha) := - mu_bar alpha_0 + \frac{sigma_squared}{2} (\sum_{i=0}^{99} \exp(-i) alpha_i)^2
        and
        W(0, tau) \equiv 0 (identity).

        # Remark
        - num_terms: An int. Specifies the maximum degree of basis functions included in \bar{C}.
          Default is 100.

        # Args
        - X_bc: A Tensor with shape [num X_bc in X_in = X_bc.shape[0], dim_input=1+degree].

        # Returns
        - init_cond: A Tensor with shape [X_bc.shape[0], dim_output=1].
        """
        init_cond = - self.mu_bar * X_bc[:, 1:2] + 0.5 * self.sigma_squared * \
            torch.sum(torch.exp(self.exp_factors_moderate[:, :X_bc.shape[1]-1]) * X_bc[:, 1:],
                      dim=1, keepdim=True) ** 2

        return init_cond  # [num X_bc, dim_output=1]

    def get_gt_solution(self, X_in: Tensor, flag_no_grad: bool = True, *args, **kwargs) -> Tensor:
        """
        Calculate functional W0
        with homogeneous velocity mu & diagonal covariance C(x,x') = val delta(x-x')

        \Phi ([\theta], t) = \Phi_0 ([\hat{\theta}_t(x)]), where
        \hat{\theta}_t(x) := \frac{1}{\sqrt{4 \pi \nu t}} \int_{-\infty}^{\infth} dx^\prime e^{\frac{(x - x^\prime)^2}{4 \pi \nu t}}\theta(x^\prime),
        and \Phi := exp W and \Phi_0 := exp W_0 with a slight abuse of notation.

        20231209: This expression can be further simplified:
        W([\Theta], \tau) = - \bar{\mu} \Theta_0 + \frac{\sigma^2}{2}\sum_{n=0}^{M-1} (\exp(-8\pi^2 n^2 \tau) \Theta_{2n}^2 + \exp(-8\pi^2 (n+1)^2 \tau) \Theta_{2n+1}^2),
        where degree = 2M (set degree=M in the config file).

        # Args
        - X_in: A batch of datapoint [batch, dim_input].
        - flag_no_grad: A bool. Detach or not.

        # Returns
        - gt_solution: A Tensor with shape [batch, dim_output=1].
        """
        if flag_no_grad:
            X_in = X_in.detach().double()
        else:
            X_in = X_in.double()

        if self.init_cond == "delta":
            gt_solution = self.calc_gt_solution_delta(X_in)   # [batch,1]
        elif self.init_cond == "const":
            gt_solution = self.calc_gt_solution_const(X_in)   # [batch,1]
        elif self.init_cond == "moderate":
            gt_solution = self.calc_gt_solution_moderate(X_in)   # [batch,1]

        return gt_solution

    def calc_gt_solution_delta(self, X_in: Tensor) -> Tensor:
        coeffs_squared = X_in[:, 1:] ** 2  # [batch, degree]
        exp_factors_gtsol_taued =\
            torch.exp(
                self.exp_factors_gtsol_delta * X_in[:, 0:1])  # [batch, degree]
        sum_ = torch.sum(
            exp_factors_gtsol_taued * coeffs_squared, dim=1, keepdim=True)  # [batch,1]
        gt_solution = - self.mu_bar * \
            X_in[:, 1:2] + 0.5 * self.sigma_squared * sum_  # [batch,1]
        return gt_solution

    def calc_gt_solution_const(self, X_in: Tensor) -> Tensor:
        gt_solution = - self.mu_bar * \
            X_in[:, 1:2] + 0.5 * self.sigma_squared * \
            X_in[:, 1:2]**2  # [batch,1]
        return gt_solution

    def calc_gt_solution_moderate(self, X_in: Tensor) -> Tensor:
        prefactors = \
            torch.exp(
                self.exp_factors_gtsol_moderate * X_in[:, 0:1] + self.exp_prefactors_gtsol_moderate)  # [batch, degree]

        sum_ = torch.sum(
            prefactors * X_in[:, 1:],
            dim=1, keepdim=True)  # [batch, 1]

        gt_solution = - self.mu_bar * X_in[:, 1:2] + \
            0.5 * self.sigma_squared * sum_ ** 2  # [batch,1]

        return gt_solution


class PDELayer(nn.Module):
    """
    Calculate residual of PDE and boundary condiiton.

    # Coding rule for collocation point datasets
    All the collocation point dataloaders output Tensor X_in with shape [batch, dim_input]
    and Tensor y with shape [batch, 1+dim_output].
    In a mini-batch, X and X_bc (and X_data if available) are included.
    X is the collocation point in the bulk (non-boundary area) with label y of [0., torch.nan,...],
    X_bc is the collocation point on the boundary with label y of [1. torch.nan,...], and
    X_data is the observed datapoint (for inverse problems) with label y of [2., float,...].
    They have the shape of [dim_input,].
    Some of mini-batches may lack X, X_bc, and/or X_data because of the mini-batch stochasticity.
    """

    def __init__(self, pde: PDE) -> None:
        """
        # Args
        - pde: Class object that inherits PDE.
        """
        super().__init__()
        self.eq_controller = pde
        self.name_pde = self.eq_controller.get_name_equation()

    def forward(self, u_in: Tensor, X_in: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        # Args
        - u_in: Output from the model. Shape = [batch, dim_output].
        - X_in: Shape = [batch, dim_input].
        - y: Shape = [batch, dim_output+1].

        # Returns
        - res: Residual (L.H.S.) of PDE. Shape = [batch, dim_output].
          require_grad should be True.
        - u_bc: Boundary values. Shape = [num X_bc, num conditions].
          require_grad should be True.
        - u_bc_gt: Ground truth boundary values. Shape = [num X_bc, num conditions].
          require_grad should be False.
        - res_rel: Relative residual. res/rhs. torch.abs not applied to.
        """
        res, res_rel = self.eq_controller.get_residual(
            u_in, X_in, y)  # [B,dim_output]->[B,dim_output]
        u_bc = self.eq_controller.get_boundary_condition(
            u_in, X_in, y)  # [num X_bc, num cond]
        u_bc_gt = self.eq_controller.get_boundary_condition_gt(
            X_in, y)  # [num X_bc, num cond]

        assert res.requires_grad == True
        # assert u_bc.requires_grad == True or u_bc.shape[0] == 0
        assert u_bc_gt.requires_grad == False or u_bc_gt.shape[0] == 0

        return res, u_bc, u_bc_gt, res_rel


def get_pde(name_equation: str, **kwargs) -> Union[PDE, FDE]:
    """
    Add your class when you implement a new equation.
    """
    if name_equation == "PDE1":
        return PDE1()
    elif name_equation == "ODE1":
        return ODE1()
    elif name_equation == "HarmonicOscillator":
        return HarmonicOscillator(**kwargs)
    elif name_equation == "Kirchhoff":
        return KirchhoffPlateBendingEquation(**kwargs)
    elif name_equation == "Helmholtz":
        raise NotImplementedError
    elif name_equation == "Burgers":
        raise NotImplementedError
    elif name_equation == "Integral1":
        return Integral1(**kwargs)
    elif name_equation == "Integral1V2":
        return Integral1V2(**kwargs)
    elif name_equation == "Integral2":
        return Integral2(**kwargs)
    elif name_equation == "ArcLength":
        return ArcLength(**kwargs)
    elif name_equation == "ThomasFermi":
        return ThomasFermi(**kwargs)
    elif name_equation == "GaussianWhiteNoise":
        return GaussianWhiteNoise(**kwargs)
    elif name_equation == "FTransportEquation":
        return FTransportEquation(**kwargs)
    elif name_equation == "BurgersHopfEquation":
        return BurgersHopfEquation(**kwargs)
    else:
        raise NotImplementedError(f"name_equation={name_equation} is invalid.")


class LossWrapperPINN(nn.Module):
    """
    Fits loss functions to the format for PINN trainings.
    Now, loss functions requires four inputs. See self.forward.
    """

    def __init__(self, loss_fn, name_loss: str, pde: PDE, alg_loss_reweight: str = "uniform") -> None:
        """
        # Args
        - loss_fn: Loss funciton (nn.Module).
        - model: Model.
        - pde: PDE object.
        - alg_loss_reweight: Algorithm for loss re-weighting. Default is uniform.
        """
        super().__init__()
        if not alg_loss_reweight in ["uniform", "softmax", "BMTL"]:
            raise NotImplementedError(
                f"alg_loss_reweight = {alg_loss_reweight} is invalid.")

        self.name_loss = name_loss
        self.pde_layer = PDELayer(pde)
        self.loss_fn = loss_fn
        self.alg_loss_reweight = alg_loss_reweight
        self.zero = torch.tensor(
            0., dtype=torch.get_default_dtype(), requires_grad=False)

    def forward(
        self, u_in: Tensor, X_in: Tensor, y: Tensor, flag_no_weighting: bool = False, *args, **kwargs,
    ) -> Tuple[Tensor, Tensor, Tuple[Union[Tensor, None], ...], Tuple[Union[Tensor, None], ...]]:
        """
        Has more arguments than classification losses.

        # Args
        - u_in: [B, dim_output]
        - X_in: [B, dim_input]
        - y: [B, 1+dim_output]

        # Returns
        - loss_sum: Sum of residual loss, boundary loss(, and
          observation data loss).
        """
        # res.shape     = [batch, dim_output]
        # u_bc.shape    = [num X_bc in batch, num conditions]
        # u_bc_gt.shape = [num X_bc in batch, num conditions]
        res, u_bc, u_bc_gt, res_rel = self.pde_layer(u_in, X_in, y)

        # Residual loss
        if res.shape[0] == 0:
            loss_res = self.zero.to(res.device)
        else:
            loss_res = self.loss_fn(
                res, torch.zeros_like(res, device=res.device))

        # Relative error for logging
        res_rel.detach()
        res_relative_error = torch.sum(
            torch.abs(res_rel))  # scalar. not mean but sum.

        # Boundary condition
        if u_bc.shape[0] == 0:
            # torch.tensor(0., device=u_bc.device, requires_grad=False)
            loss_bc = self.zero.to(u_bc.device)
        else:
            loss_bc = self.loss_fn(u_bc, u_bc_gt)

        # Observed data fitting
        # u_data = extract_X_data_from_X_in(u_in, y)
        # if u_data.shape[0] != 0:
        #     y_data = extract_y_data_from_X_in(y)
        #     loss_data = self.loss_fn(u_data, y_data)
        # else:
        #     loss_data = None

        # Calc loss weights
        list_lambdas = calc_loss_coeffs(
            [loss_res,
             loss_bc,
             #  loss_data
             ],
            alg=self.alg_loss_reweight)

        # Calc weighted loss
        if flag_no_weighting:
            loss_sum = loss_res + loss_bc
        else:
            loss_sum = weighted_sum(
                [loss_res, loss_bc,
                 #  loss_data
                 ],
                list_lambdas)

        return loss_sum, res_relative_error, (loss_res, loss_bc,
                                              #   loss_data
                                              ), (list_lambdas[0], list_lambdas[1],
                                                  #   list_lambdas[2]
                                                  )


def calc_loss_coeffs(list_losses: List[Union[Tensor, None]],
                     alg: str) -> List[Union[Tensor, None]]:
    """
    # Args
    - list_losses: List of scalar Tensors or None.
    - alg: Algorithm for loss reweighting.

    # Returns
    - list_coeffs: List of scalar Tensors or None.
    """
    assert alg in ["uniform", "softmax", "BMTL"]
    num_losses = len(list_losses)
    idx_not_None: List[int] = []
    list_losses_not_None: List[Tensor] = []
    idx_not_None_append = idx_not_None.append
    list_losses_not_None_append = list_losses_not_None.append
    for i, v in enumerate(list_losses):
        if v is not None:
            idx_not_None_append(i)
            list_losses_not_None_append(v)
    assert len(idx_not_None) != 0
    assert len(list_losses_not_None) != 0

    if alg == "uniform":
        coeffs_not_None = torch.tensor(
            [1.] * len(idx_not_None),
            device=list_losses_not_None[0].device, requires_grad=False)
    elif alg == "softmax":
        coeffs_not_None = softmax_coeffs(list_losses_not_None)
    elif alg == "BMTL":
        coeffs_not_None = BMTL(list_losses_not_None)
    else:
        raise NotImplementedError

    list_coeffs = [None] * num_losses
    for i, idx in enumerate(idx_not_None):
        list_coeffs[idx] = coeffs_not_None[i]

    return list_coeffs


def weighted_sum(
        summarand: List[Union[Tensor, None]],
        weights: List[Union[Tensor, None]]) -> Tensor:
    """
    # Args
    - summarand: List of None or Tensors. None's index must be the same as weights's.
    - weights: List of None or Tensors. None's index must be the same as summarand's.
    """
    total = 0.
    for s, w in zip(summarand, weights):
        if s is not None:
            assert w is not None
            total += w * s  # type:ignore

    return total  # type:ignore


@torch.no_grad()
def softmax_coeffs(list_tensors: List[Tensor], temperature: float = 0.25) -> Tensor:
    """
    Caution: Pay attention to require_grad of the coefficients!!
    It should be False! This is done by no_grad().

    # Args
    - list_tensors: List of scalar Tensors. None is not allowed.
    - temperature: Temperature parameter. Default is 0.25

    # Returns
    - coeffs: Tensor with shape [len(list_tensors),].
    """
    assert len(list_tensors) != 0
    device = list_tensors[0].device
    ts = torch.tensor(list_tensors, device=device)
    ts /= torch.max(ts) + EPSILON  # generates loss-scale-invariant weight
    coeffs = torch.softmax(ts / temperature, dim=0)
    return coeffs


@torch.no_grad()
def BMTL(list_tensors: List[Tensor], temperature: float = 1.) -> Tensor:
    """
    Caution: Pay attention to require_grad of the coefficients!
    It should be False!

    # Reference
    Balanced Multi-Task Learning
    https://arxiv.org/pdf/2002.04792.pdf

    # Args
    - list_tensors: List of scalar Tensors. None is not allowed.
    - temperature: Temperature parameter. Default is 1 (=50 in orig. paper).

    # Returns
    - coeffs: Tensor with shape [len(list_tensors),].
    """
    assert len(list_tensors) != 0
    device = list_tensors[0].device
    clipped_losses = torch.clamp(torch.tensor(
        list_tensors, device=device), max=10.)
    coeffs = torch.exp(clipped_losses / temperature) / temperature
    return coeffs
