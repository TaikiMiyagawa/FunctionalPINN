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

from typing import List, Optional, Tuple

from models.activations import ActivationController, PositiveSiLU
from models.ms_siren import MSSIREN
from models.pinns import NormalizeCP
from torch import Tensor, nn


class MSSIRENBlockWithSkipConnection(nn.Module):
    def __init__(self, dim_input: int, dim_output: int, sizes: Optional[List],
                 dim_hidden: int = 64, num_layers: int = 7,
                 name_activation: str = "Sin", kwargs_act: dict = dict(),
                 scales: List[float] = [1., 2., 4., 8.],
                 flag_gate_unit: bool = False,
                 flag_positive_output: bool = False,
                 flag_normalization: bool = True,
                 flag_with_head: bool = True) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.name_activation = name_activation
        self.kwargs_act = kwargs_act
        self.flag_gate_unit = flag_gate_unit
        self.flag_normalization = flag_normalization
        self.flag_positive_output = flag_positive_output
        self.flag_with_head = flag_with_head
        self.scales = scales
        self.sizes = sizes

        self.ms_siren_block = MSSIREN(
            dim_input=dim_input, dim_hidden=dim_hidden, dim_output=dim_hidden,
            num_layers=num_layers, name_activation=name_activation,
            kwargs_act=kwargs_act, scales=scales, flag_gate_unit=flag_gate_unit,
            flag_positive_output=flag_positive_output, flag_normalization=flag_normalization,
            flag_with_head=flag_with_head, sizes=sizes)

        if dim_input != dim_output:
            self.reshape_layer = nn.Linear(dim_input, dim_output)
        else:
            self.reshape_layer = nn.Identity()  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        """
        # Args
        - x: [batch, dim_input]

        # Returns
        - output: [batch, dim_output]
        """
        output, _ = self.ms_siren_block(x)
        output += self.reshape_layer(x)
        return output


class SteinWeierstrassNeuralNetwork(nn.Module):
    """ Stein-Weierstrass Neural Network (SWNN). """

    def __init__(self, dim_input: int, dim_output: int, sizes: Optional[List],
                 num_layers: int = 7, dim_hidden: int = 64,
                 name_activation: str = "Sin", kwargs_act: dict = dict(),
                 scales: List[float] = [1., 4., 16., 64.],
                 num_blocks: int = 3, flag_positive_output: bool = False,
                 flag_gate_unit: bool = False,
                 flag_normalization: bool = True, *args, **kwargs) -> None:
        """
        # Args
        - dim_input: Number of dimensions of the input data.
        - dim_output: Number of dimensions of output features.
        - sizes: len(sizes) = dim_input. The sizes of the physical system.
        - num_layers: Number of layers in each different-scale branch.
          Default is 7.
        - dim_hidden: Number of dimensions of hidden features.
          Default is 64.
        - name_activation: Name of activation function.
          Default is "Sin". See ActivationController.
        - kwargs_act: Keyword arguments for activation functions.
          Default is an empty dictionary.
        - scales: Scale factors to multiply the input.
          Default is [1., 2., 4., 8.].
        - num_blocks: Number of MS-DNN blocks.
        - flag_positive_output: Positive output or not.
        - flag_gate_unit: Use gate unit when aggregating the outputs
          from different-scale banches. Default is False.
        - flag_normalization: Use LayerNorm or not.
          Default is True.
        """
        super().__init__(*args, **kwargs)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.name_activation = name_activation
        self.kwargs_act = kwargs_act
        self.scales = scales
        self.num_blocks = num_blocks
        self.flag_positive_output = flag_positive_output
        self.flag_gate_unit = flag_gate_unit
        self.flag_normalization = flag_normalization
        self.sizes = sizes

        # Input layer
        if sizes is None:
            self.layer_in = nn.Identity()
        else:
            self.layer_in = NormalizeCP(sizes)

        # MS-SIREN Blocks
        blocks = nn.ModuleList([])
        for i in range(num_blocks):
            if i == 0:
                blocks.append(MSSIRENBlockWithSkipConnection(
                    dim_input=dim_input,  # <== !!
                    dim_hidden=dim_hidden, dim_output=dim_hidden,
                    num_layers=num_layers, name_activation=name_activation,
                    kwargs_act=kwargs_act, scales=scales, flag_gate_unit=flag_gate_unit,
                    flag_positive_output=False, flag_normalization=flag_normalization,
                    flag_with_head=False, sizes=None))
            else:
                blocks.append(MSSIRENBlockWithSkipConnection(
                    dim_input=dim_hidden,  # <== !!
                    dim_hidden=dim_hidden, dim_output=dim_hidden,
                    num_layers=num_layers, name_activation=name_activation,
                    kwargs_act=kwargs_act, scales=scales, flag_gate_unit=flag_gate_unit,
                    flag_positive_output=False, flag_normalization=flag_normalization,
                    flag_with_head=False, sizes=None))
        self.blocks = nn.Sequential(*blocks)

        # Head
        self.act_head = ActivationController(
            name_activation, kwargs_act).get_activation()
        self.fc_head = nn.Linear(dim_hidden, dim_output)
        if flag_normalization:
            self.l_norm_head = nn.LayerNorm(dim_hidden)
            self.head = nn.Sequential(
                self.l_norm_head, self.act_head, self.fc_head)
        else:
            self.head = nn.Sequential(self.act_head, self.fc_head)

        if flag_positive_output:
            self.positive_silu = PositiveSiLU()

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        """
        # Args
        - x: Shape = [batch, dim_input]. ak_in.

        # Returns
        - output: Shape = [batch, dim_output]
        """
        x = self.layer_in(x)
        x = self.blocks(x)
        output = self.head(x)

        if self.flag_positive_output:
            output = self.positive_silu(output)

        return output, None  # output, hidden feat
