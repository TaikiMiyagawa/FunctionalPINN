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
- Original paper (MS-SIREN)
  "Solving Partial Differential Equations with Point Source Based on Physics-Informed Neural Networks" [IJCAI 2022]
  https://arxiv.org/abs/2111.01394
- MscaleDNN
  "Multi-scale Deep Neural Network (MscaleDNN) for Solving Poisson-Boltzmann Equation in Complex Domains"
  https://arxiv.org/abs/2007.11207
  https://github.com/xuzhiqin1990/mscalednn/tree/1c6c6f69e9ad586ccaea90a8e8fa0d07313460b2
- This file is my original implementation.
"""

from typing import List, Optional, Tuple

import torch
from models.activations import ActivationController, PositiveSiLU
from models.pinns import NormalizeCP
from torch import Tensor, nn


class MLPWithSkipConnectionLayer(nn.Module):
    """ Linear (+ LayerNorm) + Activation. """

    def __init__(self, dim_input: int, dim_output: int, name_activation: str,
                 kwargs_act: dict, flag_normalization: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.name_activation = name_activation
        self.kwargs_act = kwargs_act
        self.flag_normalization = flag_normalization

        self.linear = nn.Linear(dim_input, self.dim_output)
        self.act = ActivationController(
            name_activation, kwargs_act).get_activation()

        if flag_normalization:
            self.l_norm = nn.LayerNorm(self.dim_output)
            self.layer = nn.Sequential(self.linear, self.l_norm, self.act)
        else:
            self.layer = nn.Sequential(self.linear, self.act)

        if dim_input != dim_output:
            self.reshape_layer = nn.Linear(dim_input, dim_output)
        else:
            self.reshape_layer = nn.Identity()  # type:ignore

    def forward(self, x: Tensor) -> Tensor:
        """
        # Args
        - x: [batch, dim_input]

        # Returns
        - output: [batch, dim_output]
        """
        return self.layer(x) + self.reshape_layer(x)  # skip connection


class GateUnit(nn.Module):
    """
    Gate branch for a modified MS-SIREN.
    The orignal MS-SIREN does not have the gate unit, which was proposed in
    https://arxiv.org/abs/2009.03730
    and also used in
    https://mediatum.ub.tum.de/doc/1688403/uic8b0xn1c845e7rac1or092o.Bischof%20et%20Al.%202022.pdf
    """

    def __init__(self, num_layers: int, dim_input: int, dim_hidden: int, name_activation: str,
                 kwargs_act: dict, num_scales: int, flag_normalization: bool = True,) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.name_activation = name_activation
        self.kwargs_act = kwargs_act
        self.num_scales = num_scales
        self.flag_normalization = flag_normalization
        assert num_layers > 1

        self.layers = nn.ModuleList([])
        for i in range(num_layers - 1):
            if i == 0:
                self.layers.append(
                    MLPWithSkipConnectionLayer(
                        dim_input=dim_input, dim_output=dim_hidden,
                        name_activation=name_activation,
                        kwargs_act=kwargs_act, flag_normalization=flag_normalization,))
            else:
                self.layers.append(
                    MLPWithSkipConnectionLayer(
                        dim_input=dim_hidden, dim_output=dim_hidden,
                        name_activation=name_activation,
                        kwargs_act=kwargs_act, flag_normalization=flag_normalization,))

        self.linear_layer_gate_out = nn.Linear(
            dim_hidden, num_scales)
        self.softmax_gate = nn.Softmax(dim=-1)

        self.gate_branch = nn.Sequential(
            *self.layers,
            self.linear_layer_gate_out, self.softmax_gate)

    def forward(self, x: Tensor) -> Tensor:
        """
        # Args
        - x: Shape = [batch, dim_input].

        # Returns
        - output: Shape = [batch, len_scale].
        """
        return self.gate_branch(x)


class ScalingLayer(nn.Module):
    def __init__(self, scaling_factor: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_factor = nn.Parameter(
            torch.tensor(scaling_factor, dtype=torch.get_default_dtype(), requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        """
        # Args
        - x: Arbitrary shape

        # Returns
        - output: Same shape as x.
        """
        return self.scaling_factor * x


class MLPUnitWithScaling(nn.Module):
    """
    MLP branch for MS-SIREN.
    """

    def __init__(self, num_layers: int, dim_input: int, dim_hidden: int, name_activation: str,
                 kwargs_act: dict, scaling_factor: float,
                 flag_normalization: bool = True) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.name_activation = name_activation
        self.kwargs_act = kwargs_act
        self.scaling_factor = scaling_factor
        self.flag_normalization = flag_normalization
        assert num_layers > 1

        self.scaling_layer = ScalingLayer(scaling_factor)
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    MLPWithSkipConnectionLayer(
                        dim_input=dim_input, dim_output=dim_hidden,
                        name_activation=name_activation,
                        kwargs_act=kwargs_act, flag_normalization=flag_normalization,))
            else:
                self.layers.append(
                    MLPWithSkipConnectionLayer(
                        dim_input=dim_hidden, dim_output=dim_hidden,
                        name_activation=name_activation,
                        kwargs_act=kwargs_act, flag_normalization=flag_normalization,))

        self.mlp_branch = nn.Sequential(
            self.scaling_layer, *self.layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        # Args
        - x: Shape = [batch, dim_input].

        # Returns
        - output: Shape = [batch, dim_hidden].
        """
        return self.mlp_branch(x)  # currently no branch-skip connection


class MSSIREN(nn.Module):
    """
    A modified MS-SIREN:
    - Additional skip connections are added at the beginning of each branch.
    - LayerNorms are added.
    - A wide variety of activation functions are available.
    - Gate Unit is available.
    - Can force the output to be positive using PositiveSiLU (original activation).
    """

    def __init__(self, dim_input: int, dim_output: int, sizes: Optional[List],
                 dim_hidden: int = 64, num_layers: int = 7,
                 name_activation: str = "Sin", kwargs_act: dict = dict(),
                 scales: List[float] = [1., 2., 4., 8.],
                 flag_gate_unit: bool = False,
                 flag_positive_output: bool = False,
                 flag_normalization: bool = True,
                 flag_with_head: bool = True) -> None:
        """
        # Args
        - dim_input: Number of dimensions of the input data.
        - dim_output: Number of dimensions of output features.
        - sizes: len(sizes) = dim_input. The sizes of the physical system.
        - dim_hidden: Number of dimensions of hidden features.
          Default is 64.
        - num_layers: Number of layers in each different-scale branch.
          Default is 7.
        - name_activation: Name of activation function.
          Default is "Sin". See ActivationController.
        - kwargs_act: Keyword arguments for activation functions.
          Default is an empty dictionary.
        - scales: Scale factors to multiply the input.
          Default is [1., 2., 4., 8.].
        - flag_gate_unit: Use gate unit when aggregating the outputs
          from different-scale banches. Default is False.
        - flag_positive_output: Positive output or not.
          Default is False.
        - flag_normalization: Use LayerNorm or not.
          Default is True.
        - flag_with_head: Use head or not.
          Default is True.

        # Remarks
        - The orignal MS-SIREN does not have the gate unit, which was proposed in
          https://arxiv.org/abs/2009.03730
          and also used in
          https://mediatum.ub.tum.de/doc/1688403/uic8b0xn1c845e7rac1or092o.Bischof%20et%20Al.%202022.pdf
        """
        super().__init__()
        assert num_layers > 1, f"Got num_layer={num_layers}."
        if not flag_with_head:
            assert dim_hidden == dim_output

        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.name_activation = name_activation
        self.flag_gate_unit = flag_gate_unit
        self.flag_normalization = flag_normalization
        self.flag_positive_output = flag_positive_output
        self.flag_with_head = flag_with_head
        self.num_scales = len(scales)
        self.scales = nn.Parameter(torch.tensor(scales), requires_grad=False)
        self.sizes = sizes

        # Input layer
        if sizes is None:
            self.layer_in = nn.Identity()
        else:
            self.layer_in = NormalizeCP(sizes)

        # Branches
        self.scalewise_layers = nn.ModuleList([])
        for it_scale in scales:  # branch loop
            self.scalewise_layers.append(
                MLPUnitWithScaling(
                    num_layers=num_layers, dim_input=dim_input, dim_hidden=dim_hidden,
                    name_activation=name_activation, kwargs_act=kwargs_act,
                    scaling_factor=it_scale, flag_normalization=True))

        self.para_scalewise_layers = lambda x: torch.stack(
            [it_l(x) for it_l in self.scalewise_layers], dim=1)

        # Gate unit
        if flag_gate_unit:
            self.gate_unit = GateUnit(
                num_layers=num_layers, dim_input=dim_input, dim_hidden=dim_hidden,
                name_activation=name_activation, kwargs_act=kwargs_act, num_scales=self.num_scales,
                flag_normalization=True)

        # Final head layer
        if flag_with_head:
            self.act_head = ActivationController(
                name_activation, kwargs_act).get_activation()
            self.fc_head = nn.Linear(dim_hidden, dim_output)
            if flag_normalization:
                self.l_norm_head = nn.LayerNorm(dim_hidden)
                self.head = nn.Sequential(  # type: ignore
                    self.l_norm_head, self.act_head, self.fc_head)
            else:
                self.head = nn.Sequential(  # type: ignore
                    self.act_head, self.fc_head)
        else:
            self.head = nn.Identity()  # type: ignore

        if self.flag_positive_output:
            self.positive_silu = PositiveSiLU()

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
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
        - x: Shape = [batch, dim_input]. X_in.

        # Returns
        - output: Shape = [batch, dim_output].
        """
        x = self.layer_in(x)

        # Branch loop
        scalewise_feats_tensor = self.para_scalewise_layers(
            x)  # [batch, num_scales, dim_hidden]

        # Gate unit
        if self.flag_gate_unit:
            branch_weights = torch.unsqueeze(
                self.gate_unit(x), dim=2)  # [batch, num_scales, 1]
            x_out = \
                branch_weights * \
                scalewise_feats_tensor  # [batch,num_scales,dim_hidden]
            x_out = x_out.sum(dim=1)  # [batch,dim_hidden]
        else:
            x_out = torch.mean(
                scalewise_feats_tensor, dim=1)  # [batch,dim_hidden]

        # Final layer
        output = self.head(x_out)  # [batch, dim_output]
        if self.flag_positive_output:
            output = self.positive_silu(output)

        return output, None  # output and hidden feat
