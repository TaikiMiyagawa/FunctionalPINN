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

import torch
import torchmetrics
from losses.pdes import FDE
from torch import Tensor
from torch.utils.data import DataLoader

EPSILON = 1e-12


class MetricController():
    def __init__(self, num_classes: int, top_k: int, *args, **kwargs) -> None:
        assert num_classes >= 2
        assert num_classes > top_k >= 1
        self.num_classes = num_classes
        self.top_k = top_k
        self.task = "multiclass"

        # Instantiate metrics
        self.confmx = torchmetrics.ConfusionMatrix(
            task=self.task,  # type:ignore
            num_classes=self.num_classes,
            normalize=None)  # type: ignore

        self.accuracy = torchmetrics.Accuracy(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average='micro')

        self.recall = torchmetrics.Recall(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average='macro')

        self.F1Score = torchmetrics.F1Score(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average='macro')

        self.AUROC = torchmetrics.AUROC(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average='macro')

        self.AveragePrecision = torchmetrics.AveragePrecision(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average="macro")

        if num_classes > 2:
            self.top_k_accuracy = torchmetrics.Accuracy(
                task=self.task,  # type: ignore
                num_classes=num_classes,
                average='micro',
                top_k=top_k)

            self.top_k_recall = torchmetrics.Recall(
                task=self.task,  # type: ignore
                num_classes=num_classes,
                average='macro',
                top_k=top_k)

            self.top_k_F1Score = torchmetrics.F1Score(
                task=self.task,  # type: ignore
                num_classes=num_classes,
                average='macro',
                top_k=top_k)

    def calc_preds(self, logits: Tensor, targets: Tensor, *args, **kwargs):
        """ Utility function to distinguish sequential and non-sequential data """
        dim = len(logits.shape)
        if dim == 2:  # [B,num_classes]
            pass
        elif dim == 3:  # [B,T,num_classes]
            duration = logits.shape[1]
            logits = logits.reshape(-1, self.num_classes)
            targets = torch.tile(targets.unsqueeze(1), [
                                 1, duration]).reshape(-1)
        else:
            raise ValueError(
                f"Invalid logit dimension: logits.shape={logits.shape}.")
        return logits.data.max(1, keepdim=True)[1][:, 0], logits, targets

    def get_confmx(self, preds: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - preds: Integers. Shape=[batch size,]. Non-one-hot.
        - target: Ingeters. Shape=[batch size,]. Non-one-hot.

        # Returns
        - Confusion matrix Tensor.
        """
        return self.confmx(preds, target)

    def get_accuracy(self, preds: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - preds: Integers. Shape=[batch size,]. Non-one-hot.
        - target: Integers. Shape=[batch size,]. Non-one-hot.

        # Returns
        - Micro-averaged accuracy = Macro-averaged accuracy
          = (1/N * sum_i I(target_i=pred_i)).
        """
        return self.accuracy(preds, target)

    def get_accuracy_from_confmx(self, confmx: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - confmx: Shape = [num_classes, num_classes].

        # Returns
        - accuracy: Scalar.
        """
        s = confmx.sum()
        assert s.item() != 0
        accuracy = torch.diagonal(confmx).sum() / s
        return accuracy

    def get_top_k_accuracy(self, preds: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - preds: Float logits or softmax. Shape=[batch size, num classes]. Non-one-hot.
        - target: Ingeters. Shape=[batch size,]. Non-one-hot.

        # Returns
        - Micro-averaged top-k accuracy.
        """
        if self.num_classes == 2:
            raise ValueError(
                "get_top_k_accuracy cannot be used for num_classes=2.")
        assert len(preds.shape) == 2, f"Got {len(preds.shape)}."
        return self.top_k_accuracy(preds, target)

    def get_recall(self, preds: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - preds: Integers. Shape=[batch size,]. Non-one-hot.
        - target: Integers. Shape=[batch size,]. Non-one-hot.

        # Returns
        - Macro-averaged recall.
        """
        return self.recall(preds, target)

    def get_recall_from_confmx(self, confmx: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - confmx: Shape = [num_classes, num_classes].

        # Returns
        - rec: Macro-averaged recall.
        """
        diag = torch.diagonal(confmx)  # [num_classes,]
        classwise_sample_size = confmx.sum(dim=1)  # [num_classes,]
        recalls = diag / (classwise_sample_size + EPSILON)  # [num_classes,]
        rec = torch.mean(recalls)
        return rec

    def get_top_k_recall(self, preds: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - preds: Float logits or softmax. Shape=[batch size, num classes]. Non-one-hot.
        - target: Ingeters. Shape=[batch size,]. Non-one-hot.

        # Returns
        - Macro-averaged top-k recall.
        """
        if self.num_classes == 2:
            raise ValueError(
                "get_top_k_recall cannot be used for num_classes=2.")
        assert len(preds.shape) == 2, f"Got {len(preds.shape)}."
        return self.top_k_recall(preds, target)

    def get_F1Score(self, preds: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - preds: Integers. Shape=[batch size,]. Non-one-hot.
        - target: Integers. Shape=[batch size,]. Non-one-hot.

        # Returns
        - Macro-averaged F1 score.
        """
        return self.F1Score(preds, target)

    def get_F1Score_from_confmx(self, confmx: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - confmx: Shape = [num_classes, num_classes].

        # Returns
        - rec: Macro-averaged F1 score.
        """
        diag = torch.diagonal(confmx)  # [num_classes,]
        classwise_sample_size = confmx.sum(dim=1)  # [num_classes,]
        classwise_prediction = confmx.sum(dim=0)  # [num_classes,]
        recalls = diag / (classwise_sample_size + EPSILON)  # [num_classes,]
        precisions = diag / (classwise_prediction + EPSILON)  # [num_classes,]
        f1 = 2 * recalls * precisions / (recalls + precisions + EPSILON)
        f1 = torch.mean(f1)
        return f1

    def get_top_k_F1Score(self, preds: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - preds: Float logits or softmax. Shape=[batch size, num classes]. Non-one-hot.
        - target: Ingeters. Shape=[batch size,]. Non-one-hot.

        # Returns
        - Macro-averaged top-k F1 score.
        """
        if self.num_classes == 2:
            raise ValueError(
                "get_top_k_F1Score cannot be used for num_classes=2.")
        assert len(preds.shape) == 2, f"Got {len(preds.shape)}."
        return self.top_k_F1Score(preds, target)

    def get_AUROC(self, preds: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        # Args
        - preds: Float logits or softmax. Shape=[batch size, num classes]. Non-one-hot.
        - target: Ingeters. Shape=[batch size,]. Non-one-hot.

        # Returns
        - Macro-averaged AUROC.
        """
        assert len(preds.shape) == 2, f"Got {len(preds.shape)}."
        return self.AUROC(preds, target)

    def get_AveragePrecision(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        # Args
        - preds: Float logits or softmax. Shape=[batch size, num classes]. Non-one-hot.
        - target: Ingeters. Shape=[batch size,]. Non-one-hot.

        # Returns
        - Macro-averaged average precision (AUC-PR).
        """
        assert len(preds.shape) == 2, f"Got {len(preds.shape)}."
        return self.AveragePrecision(preds, target)


class RelativeErrorCalculator():
    def __init__(self, model, dl_test: DataLoader, pde: FDE, degree: int, num_data: int, device, *args, **kwargs) -> None:
        """
        See losses\pdes.py for the arguments (class 'FDE' and the definiton of 'degree').
        """
        self.model = model
        self.dl_test = dl_test
        self.pde = pde
        self.degree = degree
        self.num_data = num_data
        self.device = device
        if "flag_hard_bc" in vars(pde).keys():
            self.flag_hard_bc = pde.flag_hard_bc
        else:
            self.flag_hard_bc = False

    def _to_device(self, *args):
        """
        # Args:
        - args: Tuple of Tensor on CPU. (itr_x, itr_y).
        """
        args = [v.to(self.device, non_blocking=True) for v in args]
        args[0].requires_grad = True

        return args

    def calc_batch_gt_solution(self, X_in: Tensor, *args, **kwargs) -> Tensor:
        """
        Calculate the ground truth solution at X_in.

        # Args
        - X_in: A Tensor with shape [batch, dim_input].

        # Returns
        - error: A Tensor with shape [batch,]
        """
        gt_solution = self.pde.get_gt_solution(X_in=X_in)
        return gt_solution

    def calc_batch_relative_errors(self, X_in: Tensor, preds: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
        """
        Base method for the other error calculators.
        Calculate a batch of relative errors:
        | (F([theta], t) - \hat{F}([theta], t) ) / F([theta], t) |.

        # Args
        - X_in: A Tensor with shape [batch, dim_input].
        - preds: Optional. A Tensor with shape [batch, dim_output]. Prediction from model.

        # Returns
        - error: A Tensor with shape [batch,]
        """
        if preds is None:
            preds, _ = self.model(X_in)  # [batch, dim_output]

        if self.flag_hard_bc:
            preds = self.pde.wrap_u_in_with_heaviside(u_in=preds, X_in=X_in)

        gt_solution = self.calc_batch_gt_solution(X_in)  # [batch, dim_output]
        error_candid = torch.abs(
            (preds - gt_solution) / (gt_solution + EPSILON))  # [batch, dim_output]
        error = torch.where(
            gt_solution == 0.,
            torch.abs(preds),
            error_candid)  # [batch, dim_output]
        error = torch.sum(
            error, dim=1)  # [batch,]
        error = error.detach()
        return error  # [batch,]

    def calc_batch_re_wcre(self, X_in: Tensor, preds: Optional[Tensor] = None, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """ Computes relative error and worst-case relative error.
        Relative error is
        $\mathcal{E}_q :=
            \f{1}{|D(F, \mathcal{X}, q)|} \sum_{(t^\prime, \theta^\prime) \in D(F, \mathcal{X}, q)}
                \left|
                    (F([\theta^\prime], t^\prime) - \hat{F}([\theta^\prime], t^\prime) ) / F([\theta^\prime], t^\prime)
                \right|$, and
        worst-case relative error is
        $\mathcal{E}_q^{\rm w} :=
            \sup_{t^\prime \in D(\mathcal{X}), \theta^\prime \in D(F, q)}
                \left|
                    (F([\theta^\prime], t^\prime) - \hat{F}([\theta^\prime], t^\prime) ) / F([\theta^\prime], t^\prime)
                \right|$,
        where $D(F, \mathcal{X}, q) := D(F, q) \times D(\mathcal{X})$ and
        $D(F, q) := \theta \in D(F) | \th(x) = \sum_{k=0}^{q} b_k \phi_k (t)$, and $D(\mathcal{X})$ is the domain of $t^\prime$.
        $D(F,\mathcal{X}, q)$ is approximated by linearly-spaced grid points of $(t, b_1,...,b_q)$.

        # Returns
        - error: A Tensor with shape [batch,]. Relative error.
        - wc_error: A scalar Tensor. Worst-case relative error.
        """
        if X_in.shape[0] == 0:
            return X_in[:, 0].detach(), X_in[:, 0].detach()
        relerr = self.calc_batch_relative_errors(
            X_in=X_in, preds=preds)  # [batch,]
        relerr = relerr.detach()
        error = relerr  # [batch,]
        wc_error = torch.max(relerr)  # scalar
        return error, wc_error

    def relative_error(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """ Computes relative error and worst-case relative error.
        Relative error is
        $\mathcal{E}_q :=
            \f{1}{|D(F, \mathcal{X}, q)|} \sum_{(t^\prime, \theta^\prime) \in D(F, \mathcal{X}, q)}
                \left|
                    (F([\theta^\prime], t^\prime) - \hat{F}([\theta^\prime], t^\prime) ) / F([\theta^\prime], t^\prime)
                \right|$, and
        worst-case relative error is
        $\mathcal{E}_q^{\rm w} :=
            \sup_{t^\prime \in D(\mathcal{X}), \theta^\prime \in D(F, q)}
                \left|
                    (F([\theta^\prime], t^\prime) - \hat{F}([\theta^\prime], t^\prime) ) / F([\theta^\prime], t^\prime)
                \right|$,
        where $D(F, \mathcal{X}, q) := D(F, q) \times D(\mathcal{X})$ and
        $D(F, q) := \theta \in D(F) | \th(x) = \sum_{k=0}^{q} b_k \phi_k (t)$, and $D(\mathcal{X})$ is the domain of $t^\prime$.
        $D(F,\mathcal{X}, q)$ is approximated by linearly-spaced grid points of $(t, b_1,...,b_q)$.

        # Returns
        - error: A scalar Tensor. Relative error.
        - wc_error: A scalar Tensor. Worst-case relative error.
        """
        ls_relerr: List = []
        ls_relerr_ap = ls_relerr.append
        ls_wc_relerr: List = []
        ls_wc_relerr_ap = ls_wc_relerr.append
        for it_x, it_y in self.dl_test:
            it_x, _ = self._to_device(it_x, it_y)
            it_re, it_wcre = self.calc_batch_re_wcre(X_in=it_x)
            ls_relerr_ap(it_re)
            ls_wc_relerr_ap(it_wcre)

        ts_relerr = torch.stack(ls_relerr, dim=0)  # [num_data]
        error = torch.mean(ts_relerr)  # scalar
        ts_wc_relerr = torch.stack(ls_wc_relerr, dim=0)  # [len(ls_wc_relerr),]
        wc_error = torch.max(ts_wc_relerr)  # scalar

        return error, wc_error

    def pointwise_RE(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """ Computes pointwise relative error and pointwise worst-case relative error.
        Pointwise relative error at \theta:
        For a given $\th(t) \in D(F)$,
        $\mathcal{E}_q([\theta]) :=
            \f{1}{|D(\mathcal{X})|}\sum_{t^\prime \in D(\mathcal{X})}
                \left|
                    (F([\theta], t^\prime) - \hat{F}([\theta], t^\prime) ) / F([\theta], t^\prime)
                \right|$, and
        poitwise worst-case relative error is
        $\mathcal{E}_q^{\rm w}([\theta]) :=
            \sup_{t^\prime \in D(\mathcal{X})}
                \left|
                    (F([\theta], t^\prime) - \hat{F}([\theta], t^\prime) ) / F([\theta], t^\prime)
                \right|$,
        where $D(\mathcal{X})$ is the domain of $t^\prime$.

        # Returns
        - error: A scalar Tensor. Pointwise relative error.
        - wc_error: A scalar Tensor. Pointwise worst-case relative error.
        """
        error, wc_error = self.relative_error()
        return error, wc_error

    def time_dependent_RE(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """ Computes time-dependent relative error and time-dependent worst-case relative error.
        Time-dependent relative error at t is
        $\mathcal{E}_q(t) :=
            \f{1}{|D(F, q)|}\sum_{\theta \in D(F, q)}
                \left|
                    (F([\theta], t) - \hat{F}([\theta], t) ) / F([\theta], t)
                \right|$, and
        time-dependent worst-case relative error is
        $\mathcal{E}_q^{\rm w}([\theta]) :=
            \sup_{t^\prime \in D(\mathcal{X})}
                \left|
                    (F([\theta], t^\prime) - \hat{F}([\theta], t^\prime) ) / F([\theta], t^\prime)
                \right|$.

        # Returns
        - error: A scalar Tensor. Time-dependent relative error.
        - wc_error: A scalar Tensor. Time-dependent worst-case relative error.
        """
        error, wc_error = self.relative_error()
        return error, wc_error

    def time_dependent_pointwise_RE(self, *args, **kwargs) -> Tensor:
        """ Computes time-dependent pointwise relative error.
        Time-dependent pointwise relative error at \theta and t is
        $\mathcal{E}([\theta], t) :=
            \left|
                (F([\theta], t) - \hat{F}([\theta], t) ) / F([\theta], t)
            \right|$,
        where $ \theta \in D(F)$ and $t \in D(\mathcal{X})$.

        # Note
        - The output shape is different from other functions.

        # Returns
        - error: A Tensor with shape [num_data,]. Time-dependent pointwise relative error.
        """
        ls_relerr: List = []
        ls_relerr_ap = ls_relerr.append
        for it_x, it_y in self.dl_test:
            it_x, _ = self._to_device(it_x, it_y)
            it_relerr = self.calc_batch_relative_errors(X_in=it_x)  # [batch,]
            it_relerr = it_relerr.detach()
            ls_relerr_ap(it_relerr)

        ts_relerr = torch.cat(ls_relerr, dim=0)  # [num_data,]

        return ts_relerr
