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

import numbers
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .activations import ActivationController

MAX_BATCH_SIZE = 1024


class RNNLayer(nn.Module):
    """ One-layer RNN. """

    def __init__(self, dim_input, dim_hidden, bias,
                 nonlinearity, kwargs_act, bidirectional) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.bias = bias
        self.bidirectional = bidirectional
        self.nonlinearity1 = nonlinearity
        self.kwargs_act1 = kwargs_act
        self.factor = 2 if self.bidirectional else 1

        self.fc_in = nn.Sequential(
            nn.Linear(dim_input + dim_hidden, dim_hidden, bias=bias),
            ActivationController(nonlinearity, kwargs_act).get_activation())
        self.fc_out = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden, bias=bias),
            ActivationController(nonlinearity, kwargs_act).get_activation())

    def _rnn_loop(self, input_: Tensor, duration: int, hidden_in: Tensor) -> Tensor:
        """
        Forward or backward RNN process.

        # Args
        - input_: Input sequence. [B, T, dim_input]
        - duration: Sequence length.
        - hidden_in: Initial hidden state. [B, dim_hidden]
        - c_in: Initial memory cell. [B, dim_hidden]

        # Returns
        - hidden: [B, T, factdim_hidden]
        - cell: [B, T, dim_hidden]
        """
        hidden = hidden_in
        output_ls: List[Tensor] = []
        for it_t in range(duration):
            concat = torch.cat(
                [input_[:, it_t, :], hidden],  # type: ignore
                dim=1)  # [B,dim_input+dim_hidden]
            hidden = self.fc_in(concat)   # [B,dim_hidden]
            output_ls.append(self.fc_out(hidden))  # [B,dim_hidden]
        output = torch.stack(output_ls, dim=1)   # [B,T,dim_hidden]
        return output

    def forward(self, x: Tensor, hidden_in: Optional[Tensor] = None) -> Tensor:
        """
        # Args
        - x: Input sequence. [B, T, dim_input]
        - hidden_in: Initial hidden state. [B, factor*dim_hidden]

        # Returns
        - hidden: [B, T, factor*factdim_hidden]
        """
        # Initialization
        batch_size, duration, _ = x.shape
        if hidden_in is None:
            hidden_in = torch.zeros(
                [batch_size, self.factor*self.dim_hidden],
                dtype=x.dtype, device=x.device, requires_grad=False)

        # RNN
        output = self._rnn_loop(
            x, duration, hidden_in[:, :self.dim_hidden])  # [B,T,dim_hidden]
        if self.bidirectional:
            output_reverse = self._rnn_loop(  # [B,T,dim_hidden]
                torch.flip(x, [1]), duration, hidden_in[:, self.dim_hidden:])
            output_reverse = torch.flip(output_reverse, [1])
            output = torch.cat([output, output_reverse],
                               dim=2)  # [B,T,2*dim_hidden]
        return output


class RNNModel(torch.nn.Module):
    """
    Ref: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    """

    def __init__(self, dim_input: int, dim_hidden: int, dim_output: int, with_mask: bool,
                 duration: int, num_layers: int, dropout: float = 0., nonlinearity: str = "ReLU",
                 bias: bool = True, bidirectional: bool = False, kwargs_act: dict = dict()):
        super().__init__()
        """
        # Args
        - dim_input: Number of feature dimensions.
        - dim_hidden: Number of hidden dimensions.
        - num_layers: Number of stacked layers.
        - dim_output: Number of classes for classification tasks.
        - with_mask: Whether input shape is [2*B, T, C] (with mask) or [B, T, C] (without mask).
        - nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'ReLU'.
        - bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
        - dropout: If non-zero, introduces a Dropout layer on the outputs of each RNN layer
          except the last layer, with dropout probability equal to dropout. Default: 0.
          Thus, Dropout will not be activated if num_layers = 1.
        - bidirectional: If True, becomes a bidirectional RNN. Default: False.

        # Remark
        - batch_first: If True, then the input and output tensors are provided as
          (batch, seq, feature) instead of (seq, batch, feature).
          Note that this does not apply to hidden or cell states.
        """
        # Assert
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        # Initialize
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.dim_output = dim_output
        self.with_mask = with_mask
        self.duration = duration
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.kwargs_act = kwargs_act
        self.factor = 2 if self.bidirectional else 1
        self.max_bs = MAX_BATCH_SIZE
        self.x_mark_enc_dummy = torch.nn.Parameter(torch.ones(
            [self.max_bs, duration], dtype=torch.float), requires_grad=False)

        # Layers
        if nonlinearity in ["ReLU", "Tanh"]:
            self.rnn = torch.nn.RNN(
                input_size=dim_input,
                hidden_size=dim_hidden,
                num_layers=num_layers,
                batch_first=True,  # See Remark in docstring.
                nonlinearity="relu" if nonlinearity == "ReLU" else "tanh",
                bias=bias,
                dropout=dropout,
                bidirectional=bidirectional)
        else:
            self.layers_dropout = nn.ModuleList([nn.Identity()])
            self.layers = nn.ModuleList([RNNLayer(
                dim_input=dim_input,
                dim_hidden=dim_hidden,
                bias=bias, nonlinearity=nonlinearity,
                kwargs_act=kwargs_act,
                bidirectional=bidirectional)])
            for _ in range(num_layers-1):
                self.layers_dropout.append(nn.Dropout(dropout))
                self.layers.append(RNNLayer(
                    dim_input=self.factor*dim_hidden,
                    dim_hidden=dim_hidden,
                    bias=bias, nonlinearity=nonlinearity,
                    kwargs_act=kwargs_act, bidirectional=bidirectional))

        self.fc_last = torch.nn.Linear(self.factor * dim_hidden, dim_output)

    def forward(self, X: Tensor) -> Tuple[Tensor, None]:
        """
        - X: Shape = [2*B, T, C]. X[:B] = x_enc (input sequence) and
          X[B:] = x_mark_enc (padding mask).
        - x: [B, T, C]
        - x_mark_enc: [B, T]

        # Args
        - X: Shape=(batch_size, duration, dim_feat).

        # Returns:
        - out: Shape=(batch_size, duration, dim_output).
        """
        batch_size, duration = X.shape[0], X.shape[1]
        if self.with_mask:
            batch_size = batch_size // 2
            x, x_mark_enc = X[:batch_size], X[batch_size:, :, 0]
        else:
            assert self.max_bs >= batch_size
            x = X

        # RNN
        if self.nonlinearity in ["ReLU", "Tanh"]:
            # out.shape = (batch, duration, factor * dim_hidden)
            # hidden.shape = (factor * num_layers, batch, dim_hidden)
            out, _ = self.rnn(x)
        else:
            out = x
            for it_l in range(self.num_layers):
                out = self.layers_dropout[it_l](out)  # shape=out.shape
                # [B,T,factor*dim_hidden]
                out = self.layers[it_l](out)

        # Output layer
        out = out.contiguous().view(-1, self.factor * self.dim_hidden)
        out = self.fc_last(out)  # [B*T, dim_output]
        out = out.contiguous().view(batch_size, duration, -1)

        # Masking
        if self.with_mask:
            out = out * x_mark_enc.unsqueeze(-1)  # [B,T,factor*dim_output]

        # [B,T,factor*dim_output]
        return out, None


class LSTMLayer(nn.Module):
    """ One-layer LSTM. """

    def __init__(self, dim_input, dim_hidden, bias,
                 nonlinearity1, kwargs_act1, nonlinearity2, kwargs_act2, bidirectional) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.bias = bias
        self.bidirectional = bidirectional
        self.nonlinearity1 = nonlinearity1
        self.nonlinearity2 = nonlinearity2
        self.kwargs_act1 = kwargs_act1
        self.kwargs_act2 = kwargs_act2
        self.factor = 2 if self.bidirectional else 1

        self.fc_forget = nn.Sequential(  # forget gate
            nn.Linear(dim_input + dim_hidden, dim_hidden, bias=bias),
            ActivationController(nonlinearity1, kwargs_act1).get_activation())
        self.fc_input = nn.Sequential(  # input gate
            nn.Linear(dim_input + dim_hidden, dim_hidden, bias=bias),
            ActivationController(nonlinearity1, kwargs_act1).get_activation())
        self.fc_output = nn.Sequential(  # output gate
            nn.Linear(dim_input + dim_hidden, dim_hidden, bias=bias),
            ActivationController(nonlinearity1, kwargs_act1).get_activation())
        self.fc_cell = nn.Sequential(  # added to memory cell
            nn.Linear(dim_input + dim_hidden, dim_hidden, bias=bias),
            ActivationController(nonlinearity2, kwargs_act2).get_activation())
        self.act_cell = ActivationController(
            nonlinearity2, kwargs_act2).get_activation()

    def _lstm_loop(self, input_: Tensor, duration: int, hidden_in: Tensor, c_in: Tensor) -> Tensor:
        """
        Forward or backward LSTM process.

        # Args
        - input_: Input sequence. [B, T, dim_input]
        - duration: Sequence length.
        - hidden_in: Initial hidden state. [B, dim_hidden]
        - c_in: Initial memory cell. [B, dim_hidden]

        # Returns
        - hidden: [B, T, factdim_hidden]
        - cell: [B, T, dim_hidden]
        """
        hidden_ls: List[Tensor] = []
        c = c_in
        for it_t in range(duration):
            concat = torch.cat(
                [input_[:, it_t, :], hidden_ls[it_t-1] if it_t != 0 else hidden_in],
                dim=1)  # [B,dim_input+dim_hidden]
            f = self.fc_forget(concat)  # [B,dim_hidden]
            i = self.fc_input(concat)  # [B,dim_hidden]
            c_tilde = self.fc_cell(concat)  # [B,dim_hidden]
            o = self.fc_output(concat)  # [B,dim_hidden]
            c = c * f + c_tilde * i  # [B,dim_hidden]
            hidden_ls.append(o * self.act_cell(c))  # [B,dim_hidden]
        hidden = torch.stack(hidden_ls, dim=1)  # [B,T,dim_hidden]
        return hidden

    def forward(self, x: Tensor, hidden_in: Optional[Tensor] = None,
                c_in: Optional[Tensor] = None) -> Tensor:
        """
        # Args
        - x: Input sequence. [B, T, dim_input]
        - hidden_in: Initial hidden state. [B, factor*dim_hidden]
        - c_in: Initial memory cell. [B, factor*dim_hidden]

        # Returns
        - hidden: [B, T, factor*factdim_hidden]
        """
        # Initialization
        batch_size, duration, _ = x.shape
        if c_in is None:
            c_in = torch.zeros([batch_size, self.factor*self.dim_hidden],
                               dtype=x.dtype, device=x.device, requires_grad=False)
        if hidden_in is None:
            hidden_in = torch.zeros([batch_size, self.factor*self.dim_hidden],
                                    dtype=x.dtype, device=x.device, requires_grad=False)

        # LSTM
        hidden = self._lstm_loop(
            x, duration, hidden_in[:, :self.dim_hidden], c_in[:, :self.dim_hidden])  # [B,T,dim_hidden]
        if self.bidirectional:
            hidden_reverse = self._lstm_loop(
                torch.flip(x, [1]), duration, hidden_in[:, self.dim_hidden:], c_in[:, self.dim_hidden:])
            # [B,T,dim_hidden]
            hidden_reverse = torch.flip(hidden_reverse, [1])
            hidden = torch.cat([hidden, hidden_reverse],
                               dim=2)  # [B,T,2*dim_hidden]

        return hidden


class LSTMModel(nn.Module):
    def __init__(self, dim_input: int, dim_hidden: int, num_layers: int, dim_output: int,
                 with_mask: bool, duration: int,
                 bias: bool = True, dropout: float = 0.,
                 bidirectional: bool = False,
                 nonlinearity1: Union[None, str] = None,
                 nonlinearity2: Union[None, str] = None,
                 kwargs_act1: dict = dict(),
                 kwargs_act2: dict = dict()) -> None:
        """
        My original implementation of LSTM with changeable activations.

        # Reference
          http://colah.github.io/posts/2015-08-Understanding-LSTMs/

        # Args
        - dim_input: The number of expected features in the input x
        - dim_hidden: The number of features in the hidden state h
        - with_mask: Whether input shape is [2*B, T, C] (with mask) or [B, T, C] (without mask).
        - num_layers: Number of recurrent layers. E.g., setting num_layers=2
          would mean stacking two LSTMs together to form a stacked LSTM,
          with the second LSTM taking in outputs of the first LSTM and computing the final results.
        - bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        - dropout: If non-zero, introduces a Dropout layer on the outputs of each
          LSTM layer except the last layer, with dropout probability equal to dropout.
          Thus, Dropout will not be activated when num_layers = 1. Default: 0
        - bidirectional: If True, becomes a bidirectional LSTM. Default: False.
        - nonlinearity1: Nonlinearity used in LSTM. Default is None (sigmoid).
        - nonlinearity2: Nonlinearity used in LSTM. Default is None (tanh).

        # Remark
        - batch_first: If True, then the input and output tensors are provided as
          (batch, seq, feature) instead of (seq, batch, feature).
          Note that this does not apply to hidden or cell states.
        """
        super().__init__()
        # Assert
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        # Initialization
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.dim_output = dim_output
        self.with_mask = with_mask
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nonlinearity1 = nonlinearity1
        self.nonlinearity2 = nonlinearity2
        self.kwargs_act1 = kwargs_act1
        self.kwargs_act2 = kwargs_act2
        self.factor = 2 if self.bidirectional else 1
        self.flag_official_imp = nonlinearity1 in [
            None, "Sigmoid"] and nonlinearity2 in [None, "Tanh"]
        self.max_bs = MAX_BATCH_SIZE
        self.x_mark_enc_dummy = torch.nn.Parameter(torch.ones(
            [self.max_bs, duration], dtype=torch.float), requires_grad=False)

        # Layers
        if self.flag_official_imp:
            self.lstm = torch.nn.LSTM(
                input_size=dim_input,
                hidden_size=dim_hidden,
                num_layers=num_layers,
                batch_first=True,  # See Remark in docstring.
                bias=bias,
                dropout=dropout,
                bidirectional=bidirectional)
        else:
            self.layers_dropout = nn.ModuleList([nn.Identity()])
            self.layers = nn.ModuleList([LSTMLayer(
                dim_input=dim_input,
                dim_hidden=dim_hidden,
                bias=bias, nonlinearity1=nonlinearity1,
                nonlinearity2=nonlinearity2, kwargs_act1=kwargs_act1,
                kwargs_act2=kwargs_act2, bidirectional=bidirectional)])
            for _ in range(num_layers-1):
                self.layers_dropout.append(nn.Dropout(dropout))
                self.layers.append(LSTMLayer(
                    dim_input=self.factor*dim_hidden,
                    dim_hidden=dim_hidden,
                    bias=bias, nonlinearity1=nonlinearity1,
                    nonlinearity2=nonlinearity2, kwargs_act1=kwargs_act1,
                    kwargs_act2=kwargs_act2, bidirectional=bidirectional))

        self.fc_last = torch.nn.Linear(self.factor * dim_hidden, dim_output)

    def forward(self, X: Tensor) -> Tuple[Tensor, None]:
        """
        - X: Shape = [2*B, T, C]. X[:B] = x_enc (input sequence) and
          X[B:] = x_mark_enc (padding mask).
        - x: [B, T, C]
        - x_mark_enc: [B, T]

        # Args
        - X: Shape=(batch_size, duration, dim_feat).

        # Returns:
        - out: Shape=(batch_size, duration, dim_output).
        """
        batch_size, duration = X.shape[0], X.shape[1]
        if self.with_mask:
            batch_size = batch_size // 2
            x, x_mark_enc = X[:batch_size], X[batch_size:, :, 0]
        else:
            assert self.max_bs >= batch_size
            x = X

        # LSTM
        if self.flag_official_imp:
            out, _ = self.lstm(x)  # [B, T, factor * dim_hidden]
        else:
            out = x
            for it_l in range(self.num_layers):
                out = self.layers_dropout[it_l](out)  # shape=out.shape
                # [B,T,factor*dim_hidden]
                out = self.layers[it_l](out)

        # Output layer
        out = out.contiguous().view(-1, self.factor * self.dim_hidden)
        out = self.fc_last(out)  # [B*T, dim_output]
        out = out.contiguous().view(batch_size, duration, -1)

        # Masking
        if self.with_mask:
            out = out * x_mark_enc.unsqueeze(-1)  # [B,T,dim_output]

        return out, None
