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

from typing import Any, Generator

import torch

from .custom_schedulers import (
    get_CosineAnnealingWarmRestartsLinearStart,
    get_CosineAnnealingWarmRestartsLinearStartWeightDecay)


class SchedulerController():
    def __init__(self, name_scheduler: str, optimizer: torch.optim.Optimizer, verbose: bool = False, **kwargs) -> None:
        """
        # Args
        - name_scheduler: Name of scheduler.
        - optimzier: E.g., torch.optim.SGD(model.parameters())
        - kwargs: Used for the scheduler.
        """
        self.name_scheduler = name_scheduler
        self.optimizer = optimizer
        self.kwargs = kwargs
        self.verbose = verbose

        self.scheduler: Any
        if name_scheduler == "Constant":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda _: 1., verbose=verbose)
        elif name_scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "ConstantLR":
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "LinearLR":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "PolynomialLR":
            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "CyclicLR":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "CosineAnnealingWarmRestartsLinearStart":
            self.scheduler = get_CosineAnnealingWarmRestartsLinearStart(
                optimizer, **kwargs, verbose=verbose)
        elif name_scheduler == "CosineAnnealingWarmRestartsLinearStartWeightDecay":
            self.scheduler = get_CosineAnnealingWarmRestartsLinearStartWeightDecay(
                optimizer, **kwargs, verbose=verbose)
        else:
            raise ValueError(f"name_scheduler={name_scheduler} is invalid.")

    def get_scheduler(self):
        return self.scheduler


class OptimizerController():
    def __init__(self, name_optimizer: str, params: Generator, **kwargs) -> None:
        """
        # Args
        - name_optimizer: Name of optimizer.
        - params: torch.nn.Module.parameters().
        - kwargs: Used for optimizer.
        """
        self.name_optimizer = name_optimizer
        self.params = params
        self.kwargs = kwargs

        self.optimizer = Any
        if name_optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(params, **kwargs)
        elif name_optimizer == 'SGDW':
            raise NotImplementedError
            self.optimizer = torch_optimizer.SGDW(params, **kwargs)
        elif name_optimizer == 'SGDP':
            raise NotImplementedError
            self.optimizer = torch_optimizer.SGDP(params, **kwargs)
        elif name_optimizer == 'NAdam':
            self.optimizer = torch.optim.NAdam(params, **kwargs)
        elif name_optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(params, **kwargs)
        elif name_optimizer == 'Yogi':
            raise NotImplementedError
            self.optimizer = torch_optimizer.Yogi(params, **kwargs)
        elif name_optimizer == 'PID':
            raise NotImplementedError
            self.optimizer = torch_optimizer.PID(params, **kwargs)
        elif name_optimizer == 'Lamb':
            raise NotImplementedError
            self.optimizer = torch_optimizer.Lamb(params, **kwargs)
        elif name_optimizer == 'QHM':
            raise NotImplementedError
            self.optimizer = torch_optimizer.QHM(params, **kwargs)
        elif name_optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params, **kwargs)
        elif name_optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(params, **kwargs)
        elif name_optimizer == 'RAdam':
            self.optimizer = torch.optim.RAdam(params, **kwargs)
        elif name_optimizer == 'AdaBelief':
            raise NotImplementedError
            self.optimizer = torch_optimizer.AdaBelief(params, **kwargs)
        elif name_optimizer == 'Rprop':
            self.optimizer = torch.optim.Rprop(params, **kwargs)
        elif name_optimizer == 'QHAdam':
            raise NotImplementedError
            self.optimizer = torch_optimizer.QHAdam(params, **kwargs)
        elif name_optimizer == 'Lion':
            raise NotImplementedError
            self.optimizer = Lion(params, **kwargs)
        elif name_optimizer == 'AdamP':
            raise NotImplementedError
            self.optimizer = torch_optimizer.AdamP(params, **kwargs)
        elif name_optimizer == 'AdaMod':
            raise NotImplementedError
            self.optimizer = torch_optimizer.AdaMod(params, **kwargs)
        elif name_optimizer == 'DiffGrad':
            raise NotImplementedError
            self.optimizer = torch_optimizer.DiffGrad(params, **kwargs)
        elif name_optimizer == 'SWATS':
            raise NotImplementedError
            self.optimizer = torch_optimizer.SWATS(params, **kwargs)
        elif name_optimizer == 'Adai':
            raise NotImplementedError
            self.optimizer = Adai(params, **kwargs)
        elif name_optimizer == 'AdaiV2':
            raise NotImplementedError
            self.optimizer = AdaiV2(params, **kwargs)
        elif name_optimizer == 'Apollo':
            raise NotImplementedError("Tends to cause OOM.")
            self.optimizer = torch_optimizer.Apollo(params, **kwargs)
        elif name_optimizer == 'RangerVA':
            raise NotImplementedError
            self.optimizer = torch_optimizer.RangerVA(params, **kwargs)
        elif name_optimizer == 'RangerQH':
            raise NotImplementedError
            self.optimizer = torch_optimizer.RangerQH(params, **kwargs)
        elif name_optimizer == 'Ranger':
            raise NotImplementedError
            self.optimizer = torch_optimizer.Ranger(params, **kwargs)
        elif name_optimizer == 'MADGRAD':
            raise NotImplementedError(
                "MADGRAD is buggy. You will encounter:\n" +
                "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!")
            self.optimizer = torch_optimizer.MADGRAD(params, **kwargs)
        elif name_optimizer == 'ASGD':
            self.optimizer = torch.optim.ASGD(params, **kwargs)
        elif name_optimizer == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(params, **kwargs)
        elif name_optimizer == 'Adagrad':
            raise NotImplementedError(
                "Adagrad is buggy. You will encounter:\n" +
                "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!")
            self.optimizer = torch.optim.Adagrad(params, **kwargs)
        elif name_optimizer == 'Adamax':
            self.optimizer = torch.optim.Adamax(params, **kwargs)
        elif name_optimizer == 'A2GradExp':
            raise NotImplementedError
            self.optimizer = torch_optimizer.A2GradExp(params, **kwargs)
        elif name_optimizer == 'A2GradInc':
            raise NotImplementedError
            self.optimizer = torch_optimizer.A2GradInc(params, **kwargs)
        elif name_optimizer == 'A2GradUni':
            raise NotImplementedError
            self.optimizer = torch_optimizer.A2GradUni(params, **kwargs)
        elif name_optimizer == 'AccSGD':
            raise NotImplementedError
            self.optimizer = torch_optimizer.AccSGD(params, **kwargs)
        elif name_optimizer == 'AdaBound':
            raise NotImplementedError
            self.optimizer = torch_optimizer.AdaBound(params, **kwargs)
        elif name_optimizer == 'NovoGrad':
            raise NotImplementedError
            self.optimizer = torch_optimizer.NovoGrad(params, **kwargs)
        elif name_optimizer == 'AggMo':
            raise NotImplementedError
            self.optimizer = torch_optimizer.AggMo(params, **kwargs)
        elif name_optimizer == "Sophia":
            raise NotImplementedError
            self.optimizer = SophiaG(params, **kwargs)
        else:
            raise ValueError(f"name_optimizer={name_optimizer} is invalid.")

    def get_optimizer(self):
        return self.optimizer
