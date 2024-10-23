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
A LossController class is defined with methods for configuring and managing different loss functions
used during the training of machine learning models.
"""

from typing import List, Union

from torch import nn

from .lp_loss import LpLoss
from .pdes import FDE, PDE, LossWrapperPINN, get_pde


class LossController():
    def __init__(self, list_name_losses: List[str], model: nn.Module,
                 list_kwargs: List[dict], kwargs_dataset: dict, task: str,
                 name_loss_reweight: str, device
                 ) -> None:
        """
        # Args
        - list_name_losses: List of loss names.
        - model: Model.
        - list_kwargs: List of kwargs dict for loss functions.
        - kwargs_dataset: KWARGS_NAME_DATASET[NAME_DATASET] in config.
        - task: Task, e.g., image_classification, audio_classification, or pinn.
        - name_loss_reweight: Currently supported only for task = 'pinn'.
        """
        if "name_equation" in kwargs_dataset.keys():
            self.name_equation = kwargs_dataset["name_equation"]
        else:
            self.name_equation = None

        self.list_name_losses = list_name_losses
        self.model = model
        self.list_kwargs = list_kwargs
        self.kwargs_dataset = kwargs_dataset
        self.task = task
        self.name_loss_reweight = name_loss_reweight
        self.device = device
        self.kwargs_dataset["device"] = device
        self.__f = False

        self.list_losses = []
        for itr_name_loss, itr_kwargs in zip(list_name_losses, list_kwargs):
            if itr_name_loss == "MSELoss":
                self.list_losses.append(self._get_MSELoss(**itr_kwargs))
            elif itr_name_loss == "L1Loss":
                self.list_losses.append(self._get_L1Loss(**itr_kwargs))
            elif itr_name_loss == "SmoothL1Loss":
                self.list_losses.append(self._get_SmoothL1Loss(**itr_kwargs))
            elif itr_name_loss == "CrossEntropyLoss":
                self.list_losses.append(
                    self._get_CrossEntropyLoss(**itr_kwargs))
            elif itr_name_loss == "BCELoss":
                self.list_losses.append(self._get_BCELoss(**itr_kwargs))
            elif itr_name_loss == "LpLoss":
                self.list_losses.append(self._get_LpLoss(**itr_kwargs))
            else:
                raise ValueError(f"name_loss={itr_name_loss} is invalid.")

    def get_list_losses(self) -> List:
        self.__f = True
        return self.list_losses

    def wrap4pinns(self):
        """ Change losses for PINN training """
        # Assertion
        assert self.task == "pinn"
        if self.__f:
            raise AssertionError(
                "Please call wrap4pinns before get_list_losses.")

        # Define PDE controller
        self.pde: Union[PDE, FDE] = get_pde(**self.kwargs_dataset)

        # Wrap loss functions
        losses = []
        for it_n, it_l in zip(self.list_name_losses, self.list_losses):

            # =============listing up is error-prone?
            if it_n in ["CrossEntropyLoss", "BCELoss"]:
                continue
            losses.append(LossWrapperPINN(
                it_l, it_n, self.pde, self.name_loss_reweight))

        # Overwrite losses
        self.list_losses = losses

    def _get_MSELoss(self, **kwargs):
        return nn.MSELoss(**kwargs)

    def _get_L1Loss(self, **kwargs):
        return nn.L1Loss(**kwargs)

    def _get_SmoothL1Loss(self, **kwargs):
        return nn.SmoothL1Loss(**kwargs)

    def _get_CrossEntropyLoss(self, **kwargs):
        return nn.CrossEntropyLoss(**kwargs)

    def _get_BCELoss(self, **kwargs):
        return nn.BCELoss(**kwargs)

    def _get_LpLoss(self, **kwargs):
        return LpLoss(**kwargs)
