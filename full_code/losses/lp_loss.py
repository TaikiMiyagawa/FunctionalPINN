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

import math
from typing import Union

import torch
from torch import Tensor


class LpLoss(torch.nn.Module):
    def __init__(self, p: Union[int, float]) -> None:
        """
        # Args
        - p: An int, float, inf, or -inf.
          If p is not +-inf, loss = Lp norm ** p: else loss = L+-inf norm (max and min norm, resp.).
          float("inf"), float("-inf"), np.inf, and -np.inf can be used for inf and -inf.
        """
        super(LpLoss, self).__init__()
        self.p = p
        self.zero = torch.tensor(
            0., dtype=torch.get_default_dtype(), requires_grad=False)

    def forward(self, output: Tensor, target: Tensor):
        """
        # Args
        - output: Shape=(batch_size, *). output = model(X)
        - target: Shape=(batch_size, *).

        # Returns
        - loss: A scalar Tensor or 0 if output is an empty Tensor.
        """
        if output.shape[0] == 0:
            return self.zero.to(output.device)

        batch_size = output.shape[0]
        if math.isinf(self.p):
            loss = torch.norm(output - target, p=self.p)
        else:
            loss = torch.norm(output - target, p=self.p) ** self.p / batch_size

        return loss
