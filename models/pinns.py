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
Physics-informed neural networks (PINNs).

# Awesome articles
- "4 Ideas for Physics-Informed Neural Networks that FAILED"
  https://towardsdatascience.com/4-ideas-for-physics-informed-neural-networks-that-failed-ce054270e62a
- "Improving Physics-Informed Neural Networks through Adaptive Loss Balancing"
  https://towardsdatascience.com/improving-pinns-through-adaptive-loss-balancing-55662759e701
- "Mixture of Experts for PINNs (MoE-PINNs)"
  https://towardsdatascience.com/mixture-of-experts-for-pinns-moe-pinns-6520adf32438
- "10 Useful Hints and Tricks for Improving Physics-Informed Neural Networks (PINNs)"
  https://towardsdatascience.com/10-useful-hints-and-tricks-for-improving-pinns-1a5dd7b86001
"""

from typing import List, Optional, Tuple

import torch
from models.activations import ActivationController, PositiveSiLU
from torch import Tensor, nn


class NormalizeCP(nn.Module):
    """
    Normalize to [-1, 1].
    This is considerably important for generalization and stabilization of training.
    """

    def __init__(self, sizes: List, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert sizes is not None
        self.sizes = sizes

        lengths = [j - i for i, j in sizes]
        lows = [i for i, _ in sizes]
        lengths_t = torch.tensor(
            lengths, requires_grad=False).unsqueeze(0)  # [1, dim_input,]
        lows_t = torch.tensor(lows, requires_grad=False).unsqueeze(
            0)  # [1, dim_input,]
        self.lengths_t = nn.Parameter(lengths_t)
        self.lows_t = nn.Parameter(lows_t)

    def forward(self, x: Tensor):
        """
        # Args
        - x: Shape = [batch, dim_input].

        # Returns
        - x: Shape = [batch, dim_input].
        """
        x = ((x + self.lows_t) / self.lengths_t) * 2. - 1.
        return x


class SimplePINN(nn.Module):
    def __init__(self, dim_input: int, dim_hidden: int, dim_output: int,
                 num_layers: int, sizes: Optional[List],
                 name_activation="Sin", kwargs_act: dict = dict(),
                 flag_positive_output: bool = False) -> None:
        """
        The model works correctly even when there is no temporal dimension.

        # Args
        - dim_input: Input dimension. Is equal to spatial dim + 1
          if the ground truth function u has arguments both x and t.
        - dim_hidden: Width.
        - dim_output: Output dimension. 1 for scalar functions, and
          > 1 for vector functions.
        - num_layers: Number of layers.
        - sizes: len(sizes) = dim_input. The sizes of the physical system.
        - activation: Name of activation.
        - kwargs_act: Kwargs for activation.
        - flag_positive_output: Positive output or not.
        """
        assert num_layers >= 2
        super().__init__()

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_layers = num_layers
        self.name_activation = name_activation
        self.kwargs_act = kwargs_act
        self.flag_positive_output = flag_positive_output
        self.sizes = sizes

        layers_in = nn.ModuleList([])
        if sizes is not None:
            layers_in.append(NormalizeCP(sizes=sizes))
        layers_in.append(nn.Linear(dim_input, dim_hidden))
        layers_in.append(ActivationController(
            self.name_activation, kwargs=kwargs_act).get_activation())
        layers_in.append(nn.LayerNorm(dim_hidden))
        self.input_layers = nn.Sequential(*layers_in)

        layers = nn.ModuleList([])
        if num_layers > 2:
            for _ in range(num_layers-2):
                layers.append(nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    ActivationController(
                        self.name_activation, kwargs=kwargs_act).get_activation(),
                    nn.LayerNorm(dim_hidden)))
        self.mid_layers = nn.Sequential(*layers)

        self.output_layer = nn.Linear(dim_hidden, dim_output)

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
        - x: A batch of collocation points.
          Shape=(batch_size, dim_input).

        # Returns:
        - output: A scalar or vector function u(x) with
          shape = [batch_size, dim_input].
        """
        out = self.input_layers(x)

        if self.num_layers > 2:
            out = self.mid_layers(out)

        output = self.output_layer(out)

        if self.flag_positive_output:
            output = self.positive_silu(output)

        return output, None  # output and hidden feat
