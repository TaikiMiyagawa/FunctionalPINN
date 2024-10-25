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

import datetime
import glob
import os
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from losses.pdes import FDE, PDE
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.log_config import get_my_logger
from utils.metrics import MetricController, RelativeErrorCalculator
from utils.misc import flatten_sequence, remove_padded_timestamps

logger = get_my_logger(__name__)

INIT_BEST_VALUE = -1e3
INIT_METRIC = 1e3  # For ReduceLROnPlateau


class TrainingController:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        batch_size: int,
        num_classes: int,
        top_k: int,
        list_losses: Any,
        list_name_losses: List[str],
        sampler_tr: Any,
        task: str,
        num_iterations: int,
        log_interval: int,
        validation_interval: int,
        start_pruning: int,
        tblog_interval: int,
        dir_tblog: str,
        dir_ckpt: str,
        flag_save_ckpt_trytuning: bool,
        exp_phase: str,
        rank: int,
        world_size: int,
        flag_multigpu: bool,
        now: str,
        device: str,
        trial,
        dataloader_tr: Optional[DataLoader],
        dataloader_va: Optional[DataLoader] = None,
        dataloader_te: Optional[DataLoader] = None,
        pde: Optional[Union[PDE, FDE]] = None,
        degree: Optional[int] = None,
        relerr_calculator: Optional[RelativeErrorCalculator] = None,
        *args, **kwargs
    ) -> None:
        """
        # Args
        - model: On CPU. model.to() is done in __init__.
        - optimizer: Optimizer.
        - scheduler: A child of torch.optim.lr_scheduler.
        - batch_size: Batch size.
        - num_classes: Number of classes.
        - top_k: Used to calculate metrics.
        - sampler_tr: Sampler for the training dataloader.
        - task: "image_classification", "audio_classification", "pinn", etc.
        - start_pruning: Make a pruning flag only when the number of iterations >= start_pruning iterations.
        - rank: Rank. Set 0 if FLAG_MULTIGPU = False.
        - world_size: Number of available GPUs.
        - flag_multigpu: 1 GPU or more.
        - now: Current timestamp.
        - device: "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        - dataloader_tr: Training dataloader.
        - dataloader_va: Optional. Validation dataloader.
        - dataloader_te: Optional. Test dataloader. Currently not used in training_controller.
        - trial: Optuna trial.
        - pde: Optional. For PINN training.
        - degree: Optional. For PINN training of PDEs with functional derivatives.
        - relerr_calculator: Optional. For regression.
        """
        self.rank = rank
        self.world_size = world_size
        self.flag_multigpu = flag_multigpu
        self.device = device
        self.task = task
        self.trial = trial

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.top_k = top_k
        self.dataloader_tr = dataloader_tr
        self.dataloader_va = dataloader_va
        self.dataloader_te = dataloader_te

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.list_losses = list_losses
        self.list_name_losses = list_name_losses
        self.sampler_tr = sampler_tr

        self.num_iterations = num_iterations
        self.log_interval = log_interval
        self.validation_interval = validation_interval
        self.start_pruning = start_pruning
        self.tblog_interval = tblog_interval
        self.dir_tblog = dir_tblog
        self.dir_ckpt = dir_ckpt
        self.flag_save_ckpt_trytuning = flag_save_ckpt_trytuning
        self.exp_phase = exp_phase
        self.now = now

        self.pde = pde  # = it_loss.pde_layer.eq_controller
        self.degree = degree
        self.relerr_calculator = relerr_calculator

        self.global_step = 0
        self.best_value = INIT_BEST_VALUE
        if rank == 0:
            self.writer = SummaryWriter(dir_tblog)
        if self.task in ["image_classification", "audio_classification"]:
            self.metric_controller = MetricController(
                num_classes=num_classes,
                top_k=top_k)
        elif self.task == "pinn":
            assert dataloader_va is not None
        else:
            raise ValueError(f"Invalid task name: Got {self.task}")

        # model.to()
        if flag_multigpu:
            self.model = model.to(rank)
            self.model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank])
        else:
            self.model = model.to(device)

        # Calc num_epochs
        if self.flag_multigpu:
            self.num_epochs = np.ceil(
                self.num_iterations / np.ceil(len(self.dataloader_tr.sampler) / self.batch_size))  # type: ignore
        else:
            self.num_epochs = np.ceil(
                self.num_iterations / np.ceil(len(self.dataloader_tr.dataset) / self.batch_size))  # type: ignore
        self.num_epochs = int(self.num_epochs)

        # # trorch.distributed: synchronize
        # if self.flag_multigpu:
        #     torch.distributed.barrier()

    def _calc_global_step(self, epoch: int, itr_b: int) -> int:
        """
        Calculate global step based on epoch and itr_b.
        """
        if self.flag_multigpu:
            global_step_all_worker = epoch * \
                (np.ceil(len(
                    self.dataloader_tr.sampler) / self.batch_size)) + itr_b + 1
        else:
            global_step_all_worker = epoch * \
                (np.ceil(len(
                    self.dataloader_tr.dataset) / self.batch_size)) + itr_b + 1
        global_step_all_worker = int(global_step_all_worker)
        return global_step_all_worker

    def _add_graph_tb(self, x: Tensor, verbose: bool) -> None:
        assert self.rank == 0
        self.writer.add_graph(self.model, input_to_model=x, verbose=verbose)
        logger.info("TensorBoard: Computation graph added.")

    def _add_scalar_tb(self, main_tag: str, name: str, scalar: float, global_step: int):
        assert self.rank == 0
        assert scalar is not None
        self.writer.add_scalar(
            tag=main_tag, scalar_value=scalar,
            # tag_scalar_dict={name: scalar},
            global_step=global_step)
        logger.info(f"TensorBoard: {main_tag} = {scalar} added.")

    def _add_images_tb(self,):
        """See https://pytorch.org/docs/stable/tensorboard.html"""
        raise NotImplementedError
        assert self.rank == 0
        self.writer.add_images('four_fashion_mnist_images',
                               img_grid)  # ===============
        logger.info("TensorBoard: Image added.")

    def _add_histogram_tb(self,):
        """See https://pytorch.org/docs/stable/tensorboard.html"""
        raise NotImplementedError
        assert self.rank == 0
        self.writer.add_histogram()  # ==================
        # logger.info("TensorBoard: Histogram added.")

    def _add_figure_tb(self, tag: str, figure: plt.figure, global_step: int):
        assert self.rank == 0
        self.writer.add_figure(tag=tag, figure=figure, global_step=global_step)
        # logger.info("TensorBoard: Figure added.")

    def _save_checkpoint(self, best_value: float, dir_ckpt: str):
        # Set path
        os.makedirs(dir_ckpt, exist_ok=True)
        path_model = dir_ckpt +\
            f"/ckpt{datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]}_Step{self.global_step}_Best{best_value:0.6f}.pt"

        # Save a ckpt
        ckpt = {
            "model_sd": self.model.state_dict(),
            "optimizer_sd": self.optimizer.state_dict(),
            "scheduler_sd": self.scheduler.state_dict(),
        }
        loss_ckpt = dict()
        for n, l in zip(self.list_name_losses, self.list_losses):
            loss_ckpt[n+"_sd"] = l.state_dict()
        ckpt.update(loss_ckpt)
        torch.save(ckpt, path_model)
        logger.info(
            f"Training checkpoint saved at {path_model}.\n")

        # Remove an old model
        old_model = sorted(glob.glob(dir_ckpt + "/*.pt"))
        if len(old_model) > 1:
            os.remove(old_model[0])

    def _metric_average(self, tensor: Tensor) -> Tensor:
        """
        All reduce.
        """
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor / self.world_size

    def _metric_sum(self, tensor: Tensor) -> Tensor:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor

    def _metric_max(self, tensor: Tensor) -> Tensor:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return tensor

    def _to_device(self, *args, flag_rcdpv3: bool = False):
        """
        # Args:
        - args: Tuple of Tensor on CPU. (itr_x, itr_y).
        - flag_rcdpv3: If True, remove the first redundat dim with size 1.
        """
        if self.flag_multigpu:
            args = [v.to(self.rank, non_blocking=True) for v in args]
        else:
            args = [v.to(self.device, non_blocking=True) for v in args]

        if self.task == "pinn":
            if flag_rcdpv3:
                args = [v[0] for v in args]
            args[0].requires_grad = True

        return args

    def _tblog_train(
            self,
            total_loss: Tensor,
            losses: List[Tensor],
            logits: Tensor,
            itr_x: Tensor,
            itr_y: Tensor,
            losses_pinn: List,
            coeffs_pinn: List,
            res_relative_error_tr: Tensor,
            *args, **kwargs):
        logger.info(
            f"Global step {self.global_step}: TensorBoard Logging ...")

        # Scalar
        self._add_scalar_tb(
            main_tag="TrainLoss/TotalLoss",
            name=self.now,
            scalar=total_loss.item(),
            global_step=self.global_step)

        for itr_v, itr_name in zip(losses, self.list_name_losses):
            self._add_scalar_tb(
                main_tag=f"TrainLoss/{itr_name}",
                name=self.now,
                scalar=itr_v.item(),
                global_step=self.global_step)

        if self.task in ["image_classification", "audio_classification"]:
            with torch.no_grad():
                preds, logits, itr_y = self.metric_controller.calc_preds(
                    logits, itr_y)
                preds, logits, itr_y = preds.cpu(), logits.cpu(), itr_y.cpu()
                self._add_scalar_tb(
                    main_tag="TrainMetric/Accuracy",
                    name=self.now,
                    scalar=self.metric_controller.get_accuracy(
                        preds=preds, target=itr_y),
                    global_step=self.global_step)
                self._add_scalar_tb(
                    main_tag="TrainMetric/Recall",
                    name=self.now,
                    scalar=self.metric_controller.get_recall(
                        preds=preds, target=itr_y),
                    global_step=self.global_step)
                self._add_scalar_tb(
                    main_tag="TrainMetric/F1Score",
                    name=self.now,
                    scalar=self.metric_controller.get_F1Score(
                        preds=preds, target=itr_y),
                    global_step=self.global_step)
                # self._add_scalar_tb(
                #     main_tag="TrainMetric/AUROC",
                #     name=self.now,
                #     scalar=self.metric_controller.get_AUROC(
                #         preds=logits, target=itr_y),
                #     global_step=self.global_step)
                # self._add_scalar_tb(
                #     main_tag="TrainMetric/AvePrec",
                #     name=self.now,
                #     scalar=self.metric_controller.get_AveragePrecision(
                #         preds=logits, target=itr_y),
                #     global_step=self.global_step)
                if self.num_classes > 2:
                    self._add_scalar_tb(
                        main_tag=f"TrainMetric/Top{self.top_k}Accuracy",
                        name=self.now,
                        scalar=self.metric_controller.get_top_k_accuracy(
                            preds=logits, target=itr_y),
                        global_step=self.global_step)
                    self._add_scalar_tb(
                        main_tag=f"TrainMetric/Top{self.top_k}F1Score",
                        name=self.now,
                        scalar=self.metric_controller.get_accuracy(
                            preds=logits, target=itr_y),
                        global_step=self.global_step)
                    self._add_scalar_tb(
                        main_tag=f"TrainMetric/Top{self.top_k}Recall",
                        name=self.now,
                        scalar=self.metric_controller.get_accuracy(
                            preds=logits, target=itr_y),
                        global_step=self.global_step)
                logger.info("Confusion matrix:\n", self.metric_controller.get_confmx(
                    preds=preds, target=itr_y))

        elif self.task == "pinn":
            # Losses
            _tmp = ["residual", "boundary", "data"]
            for i, v in enumerate(losses_pinn):
                if v is None:
                    continue
                self._add_scalar_tb(
                    main_tag=f"TrainLoss/loss_{_tmp[i]}",
                    name=self.now,
                    scalar=v,
                    global_step=self.global_step)
            for i, v in enumerate(coeffs_pinn):
                if v is None:
                    continue
                self._add_scalar_tb(
                    main_tag=f"TrainLoss/coeffs_{_tmp[i]}",
                    name=self.now,
                    scalar=v,
                    global_step=self.global_step)

            # Errors
            idx_bulk = torch.where(itr_y[:, 0] == 0)[0]
            idx_bdy = torch.where(itr_y[:, 0] == 1)[0]
            it_re_bulk, it_wcre_bulk = self.relerr_calculator.calc_batch_re_wcre(
                X_in=itr_x[idx_bulk], preds=logits[idx_bulk])  # [num bulk,], scalar
            it_re_bdy, it_wcre_bdy = self.relerr_calculator.calc_batch_re_wcre(
                X_in=itr_x[idx_bdy], preds=logits[idx_bdy])  # [num bdy,], scalar
            if it_re_bulk.shape[0] != 0:
                self._add_scalar_tb(
                    main_tag=f"TrainError/RelativeError_bulk",
                    name=self.now,
                    scalar=torch.mean(it_re_bulk).item(),
                    global_step=self.global_step)
                self._add_scalar_tb(
                    main_tag=f"TrainError/Worst-CaseRelativeError_bulk",
                    name=self.now,
                    scalar=it_wcre_bulk.item(),
                    global_step=self.global_step)
            if it_re_bdy.shape[0] != 0:
                self._add_scalar_tb(
                    main_tag=f"TrainError/RelativeError_bdy",
                    name=self.now,
                    scalar=torch.mean(it_re_bdy).item(),
                    global_step=self.global_step)
                self._add_scalar_tb(
                    main_tag=f"TrainError/Worst-CaseRelativeError_bdy",
                    name=self.now,
                    scalar=it_wcre_bdy.item(),
                    global_step=self.global_step)
        else:
            raise ValueError

    def _run_batch(self, x: Tensor, y: Tensor
                   ) -> Tuple[Tensor, List[Tensor], Tensor, Tensor, Tensor, List, List]:
        """
        # Returns
        - total_loss: Scalar Tensor.
        - losses: List of scalar Tensors.
        - output: Output Tensor.
        - y: Lable Tensor.
        - relative_error: Scalar Tensor or None.
        - losses_pinn: Tuple of Tensors.
        - coeffs_pinn: Tuple of Tensors.
        """
        # Forward/Backward propagation
        self.optimizer.zero_grad()
        output, _ = self.model(x)  # logits, feat

        # For sequential data (output.shape=[B,T,num_classes])
        if len(output.shape) > 2:  # [B,T,num_classes]
            output, y = flatten_sequence(
                output, y, self.num_classes)  # [B*T,*]

        # Calculate losses
        losses: List = [0.] * len(self.list_losses)
        total_loss = 0.
        losses_pinn = [0., 0.]
        for itr_i, itr_loss in enumerate(self.list_losses):
            # Calc this loss (itr_loss)
            if self.task == "pinn":
                loss, res_relative_error, it_losses_pinn, coeffs_pinn = itr_loss(
                    output, x, y, flag_no_weighting=False)
                losses_pinn = [v1 + v2 for v1, v2 in zip(
                    losses_pinn, it_losses_pinn)]
            else:
                loss = itr_loss(output, y)
                res_relative_error = None
                losses_pinn = [None]
                coeffs_pinn = [None]

            # Add this loss to total loss
            total_loss += loss
            losses[itr_i] += loss

        total_loss.backward()
        self.optimizer.step()

        # Increment global_step here
        if self.rank == 0:
            self.global_step += 1

        return total_loss, losses, output, y, res_relative_error, losses_pinn, coeffs_pinn

    def _run_epoch(self, epoch: int) -> None:
        """
        Run an training epoch.
        """
        scheduler_name = type(self.scheduler).__name__
        if self.flag_multigpu:
            # Otherwise gives same load order in every epoch
            self.dataloader_tr.sampler.set_epoch(epoch)  # type: ignore

        # f1 = open("./weights1.csv", "a")
        # f2 = open("./weights2.csv", "a")
        # Training loop
        self.model.train()
        for itr_b, (itr_x, itr_y) in enumerate(self.dataloader_tr):
            itr_x, itr_y = self._to_device(itr_x, itr_y, flag_rcdpv3=True)

            # Run batch
            total_loss, losses, logits, itr_y, res_relative_error_tr, losses_pinn, coeffs_pinn =\
                self._run_batch(itr_x, itr_y)
            global_step_all_worker = self._calc_global_step(epoch, itr_b)

            # Change learning rate
            if scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(self.metric)  # type:ignore
            else:
                self.scheduler.step()

            # If len(logits.shape) > 2
            if self.task == "audio_classification":
                logits, itr_y = remove_padded_timestamps(logits, itr_y)

            # Verbose
            if (global_step_all_worker == 1 or global_step_all_worker % self.validation_interval == 0) and self.rank == 0:
                logger.info(
                    f"[GPU{self.rank}] Epoch {epoch+1}/{self.num_epochs}" +
                    f"| Global step {self.global_step}/{self.num_iterations} [({100*self.global_step/self.num_iterations:.0f}%)]")
                logger.info("In-epoch iteration: {}/{} [({:.0f}%)]\tTotal loss: {:.6f}\tLosses: {}".format(
                    itr_b + 1, len(self.dataloader_tr),  # type: ignore
                    100 * (itr_b+1) / len(self.dataloader_tr),  # type: ignore
                    total_loss.item(),
                    [v.item() for v in losses],
                ))

            # TensorBoard logging
            if (global_step_all_worker == 1 or global_step_all_worker % self.validation_interval == 0) and self.rank == 0:
                self._tblog_train(
                    total_loss=total_loss,
                    losses=losses,
                    logits=logits,
                    itr_x=itr_x,
                    itr_y=itr_y,
                    losses_pinn=[v for v in losses_pinn if v is not None],
                    coeffs_pinn=[v for v in coeffs_pinn if v is not None],
                    res_relative_error_tr=res_relative_error_tr,)

                if scheduler_name != "ReduceLROnPlateau":
                    self._add_scalar_tb(
                        main_tag="LearningRate",
                        name=self.now,
                        scalar=self.scheduler.get_last_lr()[0],
                        global_step=self.global_step)

            # Validation loop
            if (global_step_all_worker == 1 or global_step_all_worker % self.validation_interval == 0) \
                    and self.dataloader_va is not None:
                if self.task in ["image_classification", "audio_classification"]:
                    with torch.no_grad():
                        value, self.metric = self.test(
                            self.dataloader_va,
                            self.dataloader_va.sampler)
                elif self.task == "pinn":
                    value, self.metric = self.test(
                        self.dataloader_va,
                        self.dataloader_va.sampler)

                if self.rank == 0:
                    logger.info(
                        f"Current best value: {self.best_value}\nGot {value}.")

                # Save checkpoint if best updated
                if self.best_value < value and self.rank == 0:
                    self.best_value = value
                    logger.info(
                        f"Best value updated (Global step {self.global_step})!!")
                    if (self.flag_save_ckpt_trytuning and self.exp_phase in ["try", "tuning"]) or self.exp_phase == "stat":
                        self._save_checkpoint(self.best_value, self.dir_ckpt)

                # Pruning flag
                global_step_all_worker = self._calc_global_step(
                    epoch, itr_b)
                if self.exp_phase != "stat":
                    self.trial.report(self.best_value, global_step_all_worker)
                    if self.trial.should_prune() and global_step_all_worker >= self.start_pruning:
                        raise optuna.exceptions.TrialPruned()

                self.model.train()

            # Termination flag
            global_step_all_worker = self._calc_global_step(epoch, itr_b)
            if global_step_all_worker >= self.num_iterations:
                self.flag_terminate = True
                if self.rank == 0:
                    self.writer.close()
                break
        # f1.close()
        # f2.close()

    def training(self):
        """
        Run training loop.
        """
        self.metric: float = INIT_METRIC
        self.flag_terminate = False
        for epoch in range(self.num_epochs):
            # Run an epoch
            self._run_epoch(epoch)

            # Termination flag
            if self.flag_terminate:
                break

        # torch.distributed: synchronize
        if self.flag_multigpu:
            torch.distributed.barrier()

        return self.best_value

    def test(self, dataloader: DataLoader, sampler) -> Tuple[float, float]:
        """ Run test loop. """
        if self.rank == 0:
            logger.info("Start evaluation loop...")

        # Initialization
        self.model.eval()
        confmx = torch.zeros(  # type: ignore
            [self.num_classes, self.num_classes])
        relative_error = 0.
        _dev = self.rank if self.flag_multigpu else self.device
        wc_relative_error = torch.tensor(
            0., dtype=torch.get_default_dtype(), device=_dev)
        total_loss = 0.
        losses = [0.] * len(self.list_losses)
        losses_pinn = [0., 0.]

        # Validation loop
        for itr_b, (itr_x, itr_y) in enumerate(dataloader):
            itr_x, itr_y = self._to_device(itr_x, itr_y, flag_rcdpv3=False)
            batch_size = itr_x.shape[0]

            # 1. Forward propagation
            outputs, _ = self.model(itr_x)

            # For sequential data (logit.shape=[B,T,num_classes])
            # to outputs.shape=[B*T, num_classes] (reshape)
            #  & itr_y.shape=[B*T] (torch.tile))
            if self.task == "audio_classification":
                duration = outputs.shape[1]
                outputs, itr_y = flatten_sequence(
                    outputs, itr_y, self.num_classes)
                outputs, itr_y = remove_padded_timestamps(outputs, itr_y)
            elif self.task in ["image_classification", "pinn"]:
                duration = 1.

            if self.flag_multigpu:
                # len(validation_sampler)  is the number of examples in this worker's partition.
                # For image classification and pinn, duration is set to 1.
                divisor = len(sampler) * duration
            else:
                divisor = len(dataloader.dataset)
            balance: float = batch_size / divisor

            # 2. Calculate  loss
            for itr_i, itr_lossf in enumerate(self.list_losses):
                if self.task in ["image_classification", "audio_classification"]:
                    # Calc loss
                    it_loss = itr_lossf(outputs, itr_y).detach()

                    # Stack
                    total_loss += it_loss * balance
                    losses[itr_i] += it_loss * balance

                elif self.task == "pinn":
                    # Calc loss
                    it_loss, _, it_losses_pinn, _ = itr_lossf(
                        outputs, itr_x, itr_y, flag_no_weighting=True)
                    it_loss = it_loss.detach() * balance
                    it_losses_pinn = [
                        v.detach() * balance for v in it_losses_pinn if v is not None]

                    # Stack
                    total_loss += it_loss
                    losses[itr_i] += it_loss
                    if it_losses_pinn != []:
                        losses_pinn = [
                            losses_pinn[i] + v for i, v in enumerate(it_losses_pinn)]

                else:
                    raise NotImplementedError

            # 3. Calculate metrics
            if self.task in ["image_classification", "audio_classification"]:
                pred = outputs.data.max(1, keepdim=True)[1]  # [batch,1]
                if itr_b == 0:
                    # target = itr_y  # [B,]
                    # preds = pred[:, 0]  # [B,]
                    pass
                else:
                    # target = torch.cat([target, itr_y])
                    # preds = torch.cat([preds, pred[:, 0]])
                    pass

                it_p = nn.functional.one_hot(
                    pred[:, 0], num_classes=self.num_classes)
                it_t = nn.functional.one_hot(
                    itr_y, num_classes=self.num_classes)
                it_p = it_p.unsqueeze(2)  # [B, num_classes, 1]
                it_t = it_t.unsqueeze(1)  # [B, 1, num_classes]
                prod = it_p * it_t  # [B,num_classes,num_classes]time-consuming
                confmx += prod.sum(dim=0).cpu()

            elif self.task == "pinn":
                # Calc relative error
                it_re, it_wcre = self.relerr_calculator.calc_batch_re_wcre(
                    X_in=itr_x, preds=outputs)  # [batch,], scalar
                relative_error += torch.sum(it_re)
                wc_relative_error = torch.maximum(wc_relative_error, it_wcre)

            else:
                raise ValueError

            # Verbose
            if (itr_b+1) % self.log_interval == 0 and self.rank == 0:
                logger.info(
                    f"Validation iteration {itr_b+1}/{len(dataloader)}")

        relative_error /= self.relerr_calculator.num_data

        # 4. Normalize along ranks
        if self.flag_multigpu:
            # Average metric values across workers.
            total_loss = self._metric_average(total_loss)
            losses = [self._metric_average(v) for v in losses]

            losses_pinn = [self._metric_average(v) for v in losses_pinn]
            relative_error = self._metric_sum(
                relative_error)
            wc_relative_error = self._metric_max(wc_relative_error)

            if self.task in ["image_classification", "audio_classification"]:
                ls_confmx = [torch.zeros(confmx.shape, dtype=torch.float32).to(
                    self.rank) for _ in range(self.world_size)]
                torch.distributed.all_gather(
                    ls_confmx, confmx.to(self.rank))
                confmx = torch.sum(torch.stack(ls_confmx, dim=0), dim=0)

            elif self.task == "pinn":
                pass

            else:
                raise NotImplementedError
        else:
            pass

        # 5. Compute metrics & Tensorboard logging on rank 0
        if self.task in ["image_classification", "audio_classification"]:
            rec = self.metric_controller.get_recall_from_confmx(confmx)
        elif self.task == "pinn":
            pass
        else:
            raise ValueError(f"Invalid task name: {self.task}")

        if self.rank == 0:
            norms = [torch.norm(v, p=2) for v in self.model.parameters()]
            norm = sum(norms)
            norm = norm.item()
            self._add_scalar_tb(
                main_tag="ValLosses/TotalLoss",
                name=self.now,
                scalar=total_loss.item(),
                global_step=self.global_step)
            self._add_scalar_tb(
                main_tag="WeightDecay",
                name=self.now,
                scalar=norm,
                global_step=self.global_step)

            if self.task in ["image_classification", "audio_classification"]:
                self._add_scalar_tb(
                    main_tag="ValMetric/Accuracy",
                    name=self.now,
                    scalar=self.metric_controller.get_accuracy_from_confmx(
                        confmx),
                    global_step=self.global_step)
                self._add_scalar_tb(
                    main_tag="ValMetric/Recall",
                    name=self.now,
                    scalar=rec,
                    global_step=self.global_step)
                self._add_scalar_tb(
                    main_tag="ValMetric/F1Score",
                    name=self.now,
                    scalar=self.metric_controller.get_F1Score_from_confmx(
                        confmx),
                    global_step=self.global_step)
                logger.info("Confusion matrix:\n", confmx)

            elif self.task == "pinn":
                _tmp = ["residual", "boundary", "data"]
                for i, v in enumerate(losses_pinn):
                    if v is None:
                        continue
                    self._add_scalar_tb(
                        main_tag=f"ValidationLoss/PINNLoss_{_tmp[i]}",
                        name=self.now,
                        scalar=v,
                        global_step=self.global_step)

                self._add_scalar_tb(
                    main_tag=f"ValidationError/RelativeError",
                    name=self.now,
                    scalar=relative_error.item(),
                    global_step=self.global_step)
                self._add_scalar_tb(
                    main_tag=f"ValidationError/Worst-Case_RelativeError",
                    name=self.now,
                    scalar=wc_relative_error.item(),
                    global_step=self.global_step)

            else:
                raise NotImplementedError

        # 7. Verbose
        tmp_losses = {k: v.item()
                      for k, v in zip(self.list_name_losses, losses)}
        if self.rank == 0:
            logger.info(
                "\nValidation set: Average loss: {:.5f}\n".format(total_loss) +
                f"Validation losses: {tmp_losses}\n")

        # trorch.distributed: synchronize
        if self.flag_multigpu:
            torch.distributed.barrier()

        # 8. Define outputs
        if self.task in ["image_classification", "audio_classification"]:
            value = rec  # For Optuna (maximize)
        elif self.task == "pinn":
            value = - relative_error.item()  # For Optuna (maximize)
        else:
            raise NotImplementedError
        metric = total_loss.item()  # For ReduceLROnPlateau (minimize)
        return value, metric
