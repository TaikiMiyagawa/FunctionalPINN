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

import functools
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchvision
from filelock import FileLock
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from utils.log_config import get_my_logger

from .random_cp_dataset import RCPDv3, get_RCPD_for_va_te
from .uitls import (L2ISpeechCommands, collate_fn_SpeechCommands,
                    transform_flatten, transpose_sequence)

logger = get_my_logger(__name__)


class SpeechCommandsWrapper(Dataset):
    """
    Defines preprocesses.
    """

    def __init__(self, dataset: Dataset,
                 transform: Union[Callable, None] = None,
                 target_transform: Union[Callable, None] = None) -> None:
        """
        # Args
        - dataset: torchaudio.datasets.SPEECHCOMMANDS(...)
        - transform: Additional preprocess function for waveform.
          Default is None.
        - target_transform: Additional preprocess function for labels.
          Default is None.
        """
        super().__init__()
        self.ds = dataset

        # Additional preprocesses
        self.transform = transform
        self.target_transform = target_transform

        # Preprocesses
        self.transform_default = torchvision.transforms.Compose(
            [torchaudio.transforms.Resample(
                orig_freq=16000, new_freq=8000),
             torchvision.transforms.Lambda(transpose_sequence),
             ])
        self.target_transform_default = L2ISpeechCommands()

    def __getitem__(self, index):
        waveform, _, label, _, _ = self.ds[index]
        waveform = self.transform_default(waveform)
        label = self.target_transform_default(label)

        if self.transform is not None:
            waveform = self.transform(waveform)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return waveform, label

    def __len__(self):
        return len(self.ds)


class YesNoWrapper(Dataset):
    """
    Defines preprocesses.
    """

    def __init__(self, dataset: Dataset,
                 transform: Union[Callable, None] = None,
                 target_transform: Union[Callable, None] = None) -> None:
        """
        # Args
        - dataset: torchaudio.datasets.YESNO(...)
        - transform: Additional preprocess function for waveform.
          Default is None.
        - target_transform: Additional preprocess function for labels.
          Default is None.
        """
        raise NotImplementedError(
            "Add padding process (collate_fn) to _get_YesNo")
        super().__init__()
        self.ds = dataset

        # Additional preprocesses
        self.transform = transform
        self.target_transform = target_transform

        # Preprocesses
        self.transform_default = lambda x: x
        self.target_transform_default = lambda x: torch.tensor(
            x, dtype=torch.int)

    def __getitem__(self, index):
        waveform, _, labels = self.ds[index]
        waveform = self.transform_default(waveform)
        labels = self.target_transform_default(labels)

        if self.transform is not None:
            waveform = self.transform(waveform)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return waveform, labels

    def __len__(self):
        return len(self.ds)


class DataAugmentationController():
    """
    Controls 'transorm'. To be implemented.
    """

    def __init__(self) -> None:
        raise NotImplementedError


class DataController():
    """
    Controls Dataset, Sampler, and DataLoader.
    """

    def __init__(self, name_dataset: str, name_sampler: str, batch_size: int,
                 flag_shuffle: bool, pin_memory: bool,
                 num_workers: int,  flag_multigpu: bool,
                 world_size: int, rank: int,
                 rootdir_ptdatasets: str, task: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 kwargs_dataset: Optional[dict] = None) -> None:
        """
        # Args
        - name_dataset: Name of dataset.
        - name_sampler: Name of sampler.
        - batch_size: Batch size.
        - flag_shuffle: Shuffle training set or not.
        - pin_memory: If True, consumes CPU memory more.
        - flag_multigpu: Use more than one GPUs or not.
        - world_size: Num of GPUs. Used only when flag_multigpu=True.
        - rank: Rank.
        - task: Task, e.g., 'image_classification', 'audio_classification', or 'pinn'.
        - rootdir_ptdataset: Directory path under which pytorch datasets are downloaded
          E.g.,
          rootdir_ptdataset/MNIST/raw/t10k-images-idx3-ubyte
          rootdir_ptdataset/SpeechCommands/speech_commands_v0.02/dog/ffbb695d_nohash_0.wav
          etc.

        # What is collate_fn?
        It enables us to do batch-wise transforms.
        DataLoader supports automatically collating individual fetched data samples
        into batches via arguments batch_size, drop_last, batch_sampler,
        and collate_fn.
        See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn .
        self.collate_fn is None by default (nothing applied).
        Collate functions are defined in self._get_DATASET_NAME method if necessary.
        See, e.g., self._get_SpeechCommands.
        """
        self.dataset_tr: Dataset[Any]
        self.dataset_va: Dataset[Any]
        self.dataset_te: Dataset[Any]
        self.sampler_tr: Any = None
        self.sampler_va: Any = None
        self.sampler_te: Any = None
        self.dataloader_tr: Union[DataLoader, None] = None
        self.dataloader_va: Union[DataLoader, None] = None
        self.dataloader_te: Union[DataLoader, None] = None
        self.collate_fn: Union[Callable, None] = None

        self.name_dataset = name_dataset
        self.name_sampler = name_sampler
        self.batch_size = batch_size
        self.flag_shuffle = flag_shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.flag_multigpu = flag_multigpu
        self.world_size = world_size
        self.rank = rank
        self.task = task
        self.rootdir_ptdatasets = rootdir_ptdatasets
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs_dataset = kwargs_dataset

        # For PINNs
        if task == "pinn":
            self.flag_shuffle = False
            if rank == 0:
                logger.info(
                    "DataController: flag_shuffle is automatically set to False because task='pinn'.")

        # Construct datasets, samplers, and dataloaders
        if self.rootdir_ptdatasets is not None:
            with FileLock(self.rootdir_ptdatasets + "/.ddp_lock_datasets"):  # torch.distributed
                self._prepare_dataset()
        else:
            self._prepare_dataset()
        self._prepare_sampler()
        self._prepare_dataloader()

    def _prepare_dataset(self) -> None:
        """
        Sets dataset.
        """

        if self.name_dataset == "MNIST":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_MNIST()
        elif self.name_dataset == "FashionMNIST":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_FashionMNIST()
        elif self.name_dataset == "CIFAR10":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_CIFAR10()
        elif self.name_dataset == "CIFAR100":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_CIFAR100()
        elif self.name_dataset == "CIFAR10LT":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_CIFAR10LT()
        elif self.name_dataset == "CIFAR100LT":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_CIFAR100LT()
        elif self.name_dataset == "iNaturalist2017":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_iNaturalist2017()
        elif self.name_dataset == "iNaturalist2018":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_iNaturalist2018()
        elif self.name_dataset == "SpeechCommands":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_SpeechCommands()
        elif self.name_dataset == "YesNo":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_YesNo()
        elif self.name_dataset == "CIFAR10Flat":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_CIFAR10Flat()
        elif self.name_dataset == "CIFAR100Flat":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_CIFAR100Flat()
        elif self.name_dataset == "RCPDPDE1":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCPDPDE1()
        elif self.name_dataset == "RCPDODE1":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCPDODE1()
        elif self.name_dataset == "RCPDHarmonicOscillator":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCPDHarmonicOscillator()
        elif self.name_dataset == "RCPDKirchhoff":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCPDKirchhoff()
        elif self.name_dataset == "RCDIntegral1":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCDIntegral1()
        elif self.name_dataset == "RCDIntegral1V2":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCDIntegral1V2()
        elif self.name_dataset == "RCDIntegral2":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCDIntegral2()
        elif self.name_dataset == "RCDArcLength":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCDArcLength()
        elif self.name_dataset == "RCDThomasFermi":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCDThomasFermi()
        elif self.name_dataset == "RCDGaussianWhiteNoise":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCDGaussianWhiteNoise()
        elif self.name_dataset == "RCDFTransportEquation":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCDFTransportEquation()
        elif self.name_dataset == "RCDBurgersHopfEquation":
            self.dataset_tr, self.dataset_va, self.dataset_te =\
                self._get_RCDBurgersHopfEquation()
        else:
            raise ValueError(f"name_dataset = {self.name_dataset} is invalid.")

    def _prepare_sampler(self) -> None:
        """
        Sets dataset sampler (self.sampler).
        """
        # 1. Random sampler
        if self.name_sampler == "RandomSampler":
            self.sampler_tr, self.sampler_va, self.sampler_te =\
                self._get_RandomSampler()

        # 2. OverSampler
        elif self.name_sampler == "OverSampler":
            raise NotImplementedError
            self.sampler_tr, self.sampler_va, self.sampler_te =\
                self._get_OverSampler()

        else:
            raise ValueError(
                f"name_sampler={self.name_sampler} is not available.")

    def _prepare_dataloader(self) -> None:
        """
        Sets dataloaders.
        """
        # RCDPv3 compatibility
        # def worker_init_fn(worker_id):
        #     np.random.seed(np.random.get_state()[1][0] + worker_id)
        def worker_init_fn(x): return np.random.seed()

        if self.task == "pinn":
            dummy_batch_size = 1
            dummy_wif = worker_init_fn
        else:
            dummy_batch_size = self.batch_size
            dummy_wif = None

        # 1. For multi-gpu
        # shuffle=False & sampler=DistributedSampler(sfhuffle=self.flag_shuffle)
        if self.flag_multigpu:
            # Training dataloader
            self.dataloader_tr = DataLoader(
                dataset=self.dataset_tr,
                batch_size=dummy_batch_size,
                pin_memory=self.pin_memory,
                shuffle=False,
                sampler=self.sampler_tr,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                worker_init_fn=dummy_wif)

            # Validation dataloader
            if self.dataset_va is not None:
                self.dataloader_va = DataLoader(
                    self.dataset_va,
                    batch_size=self.batch_size,
                    pin_memory=self.pin_memory,
                    shuffle=False,
                    sampler=self.sampler_va,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn)

            # Test dataloader
            if self.dataset_te is not None:
                self.dataloader_te = DataLoader(
                    self.dataset_te,
                    batch_size=self.batch_size,
                    pin_memory=self.pin_memory,
                    shuffle=False,
                    sampler=self.sampler_te,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn)

        # 2. For single GPU
        # shuffle=self.flag_shuffle
        else:
            # Training dataloader
            self.dataloader_tr = DataLoader(
                self.dataset_tr,
                batch_size=dummy_batch_size,
                pin_memory=self.pin_memory,
                shuffle=self.flag_shuffle,
                sampler=self.sampler_tr,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                worker_init_fn=dummy_wif)

            # Validation dataloader
            if self.dataset_va is not None:
                self.dataloader_va = DataLoader(
                    self.dataset_va,
                    batch_size=self.batch_size,
                    pin_memory=self.pin_memory,
                    shuffle=False,
                    sampler=self.sampler_va,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn)
            else:
                self.dataloader_va = None

            # Test dataloader
            if self.dataset_te is not None:
                self.dataloader_te = DataLoader(
                    self.dataset_te,
                    batch_size=self.batch_size,
                    pin_memory=self.pin_memory,
                    shuffle=False,
                    sampler=self.sampler_te,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn)
            else:
                self.dataloader_te = None

    def get_datasets(self):
        """
        Returns datasets.
        self.dataset_tr is the same as self.dataloader_tr.dataset.
        """
        return self.dataset_tr, self.dataset_va, self.dataset_te

    def get_samplers(self):
        """
        Returns samplers.
        """
        return self.sampler_tr, self.sampler_va, self.sampler_te

    def get_dataloaders(self):
        """
        Returns dataloaders.
        """
        return self.dataloader_tr, self.dataloader_va, self.dataloader_te

    def _get_DistributedSampler(self) -> Tuple[
            torch.utils.data.distributed.DistributedSampler,
            Union[torch.utils.data.distributed.DistributedSampler, None],
            Union[torch.utils.data.distributed.DistributedSampler, None]]:
        # Training sampler
        sampler_tr: torch.utils.data.distributed.DistributedSampler\
            = torch.utils.data.distributed.DistributedSampler(
                self.dataset_tr,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=self.flag_shuffle)

        # Validation sampler
        sampler_va: Union[torch.utils.data.distributed.DistributedSampler, None]
        if self.dataset_va:
            sampler_va = torch.utils.data.distributed.DistributedSampler(
                self.dataset_va,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False)
        else:
            sampler_va = None

        # Test sampler
        sampler_te: Union[torch.utils.data.distributed.DistributedSampler, None]
        if self.dataset_te:
            sampler_te = torch.utils.data.distributed.DistributedSampler(
                self.dataset_te,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False)
        else:
            sampler_te = None

        return sampler_tr, sampler_va, sampler_te

    def _get_RandomSampler(self) -> Union[
            Tuple[Any, Union[Any, None], Union[Any, None]],
            Tuple[None, None, None]]:
        if self.flag_multigpu:
            return self._get_DistributedSampler()
        else:
            return None, None, None

    # def _get_OverSampler(self) -> Tuple[
    #         Union[DistributedWeightedSampler, ImbalancedDatasetSampler], None, None]:
    #     raise NotImplementedError
    #     if self.flag_multigpu:
    #         raise NotImplementedError("DistributedWeightedSampler")
    #         sampler_tr = DistributedWeightedSampler(self.dataset_tr)
    #         if self.dataset_va:
    #             sampler_va = None
    #         if self.dataset_te:
    #             sampler_te = None
    #     else:
    #         sampler_tr = ImbalancedDatasetSampler(self.dataset_tr)
    #         if self.dataset_va:
    #             sampler_va = None
    #         if self.dataset_te:
    #             sampler_te = None
    #     return sampler_tr, sampler_va, sampler_te

    def _get_MNIST(self):
        dataset = torchvision.datasets.MNIST(
            root=self.rootdir_ptdatasets,
            train=True,
            transform=self.transform,
            target_transform=self.target_transform,
            download=True)
        dataset_tr, dataset_va = torch.utils.data.random_split(
            dataset=dataset, lengths=[50000, 10000],
            generator=torch.Generator().manual_seed(777))
        dataset_te = torchvision.datasets.MNIST(
            root=self.rootdir_ptdatasets,
            train=False,
            transform=self.transform,
            target_transform=self.target_transform,
            download=True)
        return dataset_tr, dataset_va, dataset_te

    def _get_FashionMNIST(self):
        dataset = torchvision.datasets.FashionMNIST(
            root=self.rootdir_ptdatasets,
            train=True,
            transform=self.transform,
            target_transform=self.target_transform,
            download=True)
        dataset_tr, dataset_va = torch.utils.data.random_split(
            dataset=dataset, lengths=[50000, 10000],
            generator=torch.Generator().manual_seed(777))
        dataset_te = torchvision.datasets.FashionMNIST(
            root=self.rootdir_ptdatasets,
            train=False,
            transform=self.transform,
            target_transform=self.target_transform,
            download=True)
        return dataset_tr, dataset_va, dataset_te

    def _get_CIFAR10(self):
        dataset = torchvision.datasets.CIFAR10(
            root=self.rootdir_ptdatasets,
            train=True,
            transform=self.transform,
            target_transform=self.target_transform,
            download=True)
        dataset_tr, dataset_va = torch.utils.data.random_split(
            dataset=dataset, lengths=[40000, 10000],
            generator=torch.Generator().manual_seed(777))
        dataset_te = torchvision.datasets.CIFAR10(
            root=self.rootdir_ptdatasets,
            train=False,
            transform=self.transform,
            target_transform=self.target_transform,
            download=True)
        return dataset_tr, dataset_va, dataset_te

    def _get_CIFAR100(self):
        dataset = torchvision.datasets.CIFAR100(
            root=self.rootdir_ptdatasets,
            train=True,
            transform=self.transform,
            target_transform=self.target_transform,
            download=True)
        dataset_tr, dataset_va = torch.utils.data.random_split(
            dataset=dataset, lengths=[40000, 10000],
            generator=torch.Generator().manual_seed(777))
        dataset_te = torchvision.datasets.CIFAR100(
            root=self.rootdir_ptdatasets,
            train=False,
            transform=self.transform,
            target_transform=self.target_transform,
            download=True)
        return dataset_tr, dataset_va, dataset_te

    def _get_CIFAR10Flat(self):
        """
        - x: [B, T, C]
        - y: int
        """
        if self.transform is not None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(transform_flatten),
                torchvision.transforms.Lambda(transpose_sequence),
                self.transform
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(transform_flatten),
                torchvision.transforms.Lambda(transpose_sequence),
            ])
        dataset = torchvision.datasets.CIFAR10(
            root=self.rootdir_ptdatasets,
            train=True,
            transform=transform,
            target_transform=self.target_transform,
            download=True)
        dataset_tr, dataset_va = torch.utils.data.random_split(
            dataset=dataset, lengths=[40000, 10000],
            generator=torch.Generator().manual_seed(777))
        dataset_te = torchvision.datasets.CIFAR10(
            root=self.rootdir_ptdatasets,
            train=False,
            transform=transform,
            target_transform=self.target_transform,
            download=True)
        return dataset_tr, dataset_va, dataset_te

    def _get_CIFAR100Flat(self):
        """
        - x: [B, T, C]
        - y: int
        """
        if self.transform is not None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(transform_flatten),
                torchvision.transforms.Lambda(transpose_sequence),
                self.transform
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(transform_flatten),
                torchvision.transforms.Lambda(transpose_sequence),
            ])
        dataset = torchvision.datasets.CIFAR100(
            root=self.rootdir_ptdatasets,
            train=True,
            transform=transform,
            target_transform=self.target_transform,
            download=True)
        dataset_tr, dataset_va = torch.utils.data.random_split(
            dataset=dataset, lengths=[40000, 10000],
            generator=torch.Generator().manual_seed(777))
        dataset_te = torchvision.datasets.CIFAR100(
            root=self.rootdir_ptdatasets,
            train=False,
            transform=transform,
            target_transform=self.target_transform,
            download=True)
        return dataset_tr, dataset_va, dataset_te

    def _get_CIFAR10LT(self):
        raise NotImplementedError

    def _get_CIFAR100LT(self):
        raise NotImplementedError

    def _get_iNaturalist2017(self):
        raise NotImplementedError

    def _get_iNaturalist2018(self):
        raise NotImplementedError

    def _get_RCPDPDE1(self):
        assert self.kwargs_dataset is not None
        assert "sizes" in self.kwargs_dataset.keys()
        assert "boundaries" in self.kwargs_dataset.keys()
        assert "num_points_max" in self.kwargs_dataset.keys()
        assert "dim_output" in self.kwargs_dataset.keys()
        assert "batch_size_ratio" in self.kwargs_dataset.keys()
        # assert "sampler" in self.kwargs_dataset.keys() # deprecated since RCDPv3

        dataset_tr = RCPDv3(
            sizes=self.kwargs_dataset["sizes"],
            boundaries=self.kwargs_dataset["boundaries"],
            batch_size_ratio=self.kwargs_dataset["batch_size_ratio"],
            num_points=self.kwargs_dataset["num_points_max"],
            dim_output=self.kwargs_dataset["dim_output"],
            batch_size=self.batch_size)

        dataset_va, dataset_te = get_RCPD_for_va_te(
            sizes=self.kwargs_dataset["sizes"],
            boundaries=self.kwargs_dataset["boundaries"],
            dim_output=self.kwargs_dataset["dim_output"])

        return dataset_tr, dataset_va, dataset_te

    def _get_RCPDODE1(self):
        return self._get_RCPDPDE1()

    def _get_RCPDHarmonicOscillator(self):
        return self._get_RCPDPDE1()

    def _get_RCPDKirchhoff(self):
        return self._get_RCPDPDE1()

    def _get_RCDIntegral1(self):
        return self._get_RCPDPDE1()

    def _get_RCDIntegral1V2(self):
        return self._get_RCPDPDE1()

    def _get_RCDIntegral2(self):
        return self._get_RCPDPDE1()

    def _get_RCDArcLength(self):
        return self._get_RCPDPDE1()

    def _get_RCDThomasFermi(self):
        return self._get_RCPDPDE1()

    def _get_RCDGaussianWhiteNoise(self):
        return self._get_RCPDPDE1()

    def _get_RCDFTransportEquation(self):
        return self._get_RCPDPDE1()

    def _get_RCDBurgersHopfEquation(self):
        assert self.kwargs_dataset is not None
        assert "sizes" in self.kwargs_dataset.keys()
        assert "boundaries" in self.kwargs_dataset.keys()
        assert "num_points_max" in self.kwargs_dataset.keys()
        assert "dim_output" in self.kwargs_dataset.keys()
        assert "batch_size_ratio" in self.kwargs_dataset.keys()
        assert "degree" in self.kwargs_dataset.keys()
        # assert "sampler" in self.kwargs_dataset.keys() # deprecated since RCDPv3

        def _calc_coeff_scale_BHE(degree) -> Tensor:
            """ Covariance matrix associated with the Gaussian initialization.
            # Returns
            - diag_inverse: Shape [1, degree]. Tensor on CPU. Cancels C_tilde factor in B-H eq.
            """
            odd = torch.tensor(
                [torch.pi**2 * (i + 1) ** 2 if i %
                    2 == 1 else 0. for i in range(degree)],
                dtype=torch.get_default_dtype())  # [degree,]
            even = torch.tensor(
                [torch.pi**2 * i ** 2 if i %
                    2 == 0 else 0. for i in range(degree)],
                dtype=torch.get_default_dtype())  # [degree,]
            diag = odd + even  # [degree,]
            diag[0] = 1.
            diag = torch.cat([torch.tensor(
                [1.], dtype=torch.get_default_dtype()), diag], dim=0)  # [degree+1,]
            diag_inverse = diag**-1  # [degree+1,]
            diag_inverse.unsqueeze(0)  # [1, degree+1]
            return diag_inverse

        diag_inverse = _calc_coeff_scale_BHE(self.kwargs_dataset["degree"])
        transform_factor = torchvision.transforms.Lambda(
            lambda X_in: X_in * diag_inverse
        )

        dataset_tr = RCPDv3(
            sizes=self.kwargs_dataset["sizes"],
            boundaries=self.kwargs_dataset["boundaries"],
            batch_size_ratio=self.kwargs_dataset["batch_size_ratio"],
            num_points=self.kwargs_dataset["num_points_max"],
            dim_output=self.kwargs_dataset["dim_output"],
            batch_size=self.batch_size,
            transform=transform_factor)

        dataset_va, dataset_te = get_RCPD_for_va_te(
            sizes=self.kwargs_dataset["sizes"],
            boundaries=self.kwargs_dataset["boundaries"],
            dim_output=self.kwargs_dataset["dim_output"],
            transform=transform_factor)

        return dataset_tr, dataset_va, dataset_te

    def _get_SpeechCommands(self):
        """
        Paper: https://arxiv.org/abs/1804.03209

        Statistics:
        36 classes (including '_unknown_' class with index = 0)
        'train'	85,511 wav files (min & max duration = 4096, 16000)
        'validation' 10,102 wav files (min & max duration = 5461, 16000)
        'test' 4,890 wav files (min & max duration= 3413, 16000))
        Sample rate is 16kHz. To be resampled to 8kHz (see SpeechCommandsWrapper).

        A datapoint is a tuple
        (waveform: Float32 Tensor with shape=(1, variable num of frames),
         sample rate: int,
         label: str,
         speaker ID: str,
         utterance number: int).

        A wrapped datapoint is a tuple
        (waveform: Float32 Tensor with shape=(self.kwargs_dataset["size"] or max duration in a batch, 1),
         label: int)

        # Labels
        ['_unknown_','backward','bed','bird','cat','dog','down','eight',
         'five','follow','forward','four','go','happy','house','learn',
         'left','marvin','nine','no','off','on','one','right','seven',
         'sheila','six','stop','three','tree','two','up','visual','wow','yes','zero']
        These str is transformed to int after the wrapping.
        """
        assert "duration" in self.kwargs_dataset.keys(), \
            f"kwargs={self.kwargs_dataset}"

        dataset_tr = SpeechCommandsWrapper(torchaudio.datasets.SPEECHCOMMANDS(
            self.rootdir_ptdatasets, subset="training", download=True))
        dataset_va = SpeechCommandsWrapper(torchaudio.datasets.SPEECHCOMMANDS(
            self.rootdir_ptdatasets, subset="validation", download=False))
        dataset_te = SpeechCommandsWrapper(torchaudio.datasets.SPEECHCOMMANDS(
            self.rootdir_ptdatasets, subset="testing", download=False))
        self.collate_fn = functools.partial(
            collate_fn_SpeechCommands, size=self.kwargs_dataset["duration"])

        return dataset_tr, dataset_va, dataset_te

    def _get_YesNo(self):
        """
        Sample rate is 8kHz.
        An original (unwrapped) datapoint is a tuple
        (waveform: float32 Tensor with shape=(1, variable num of frames),
         sample_rate: int,
         labels: list[python int] with len=8).
        1 for 'yes' and 0 for 'no'.
        A wrapped datapoint is a tuple
        (waveform: float Tensor with shape=(1, variable num of frames),
         labels: int Tensor with shape=(8,)).
        """
        raise NotImplementedError("What task is this dataset for?")
        dataset = YesNoWrapper(torchaudio.datasets.YESNO(
            root=self.rootdir_ptdatasets,
            download=True))
        dataset_tr, dataset_va = torch.utils.data.random_split(
            dataset=dataset, lengths=[50, 10],
            generator=torch.Generator().manual_seed(777))
        dataset_te = None

        return dataset_tr, dataset_va, dataset_te
