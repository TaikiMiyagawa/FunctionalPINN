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
This is a config file.

# Remark
- Most of parameters will be overwritten
  when EXP_PHASE = "stat" (see utils.misc.load_config_stat).
- When you add/remove parameters from this file,
  check and update utils.misc.load_config_stat if necessary.

# Contents:
- 1. Project name
- 2. Important
- 3. Less important
- 4. Search spaces (for hparam tuning)
- 5. Optuna params
- 6. Directory names
- 7. Other dependencies (parameters that are automatically determined)
- 8. Sanity checks (AssertError, ValueError, NotImplementedError)
- 9. Make config dictionary (pack all variables to a dict)

# Naming Rules (Always complied unless otherwise noted  )
- FLAG_*: bool
- NUM_*: int
- LIST_*: list
- NAME_*: str
- KWARGS_*: dict
- SS_* : Search spaces for Optuna
- SS_CAT_*: list. Categorical search space, e.g., ["Adam", "SGD", "RMSprop"].
- SS_FLT_*: list. Float search space, e.g., [0.1, 0.9] ([low, high)).
- SS_INT_*: list. Int search space, e.g., [5, 10] ([low, high)).
- SS_FLL_*: list. Log float search space, e.g., [1e-8, 1e-1] ([low, high)).
- SS_INL_*: list. Log int search space, e.g.,  [1, 1000] ([low, high)).
- SS_KWARGS_*: dict, where
  key = a suggested param from SS_CAT_* and
  value = a keyword argument dict for *.
  A keyword argument dict has
  key = "arg_{method name}{CAT, FLT, FLL, INT, or INL}" and
  value = a search space list.
  E.g., SS_KWARGS_OPTIMIZER["SGD"] is {"lr_SGDFLL": [low, high], "momentum_SGDFLT": [low, high]}.

# Directories (See 6. Directory names)
- TensorBoard log of a trial: ROOTDIR_DATA/NAME_PROJECT/tblogs/NAME_SUBPROJECT/COMMENT_NOW
- Text log of a trial: ROOTDIR_DATA/NAME_PROJECT/txlogs/NAME_SUBPROJECT/COMMENT_NOW
- Checkpoint log of a trial: ROOTDIR_DATA/NAME_PROJECT/ckpts/NAME_SUBPROJECT/COMMENT_NOW
- Config log of a trial: ROOTDIR_DATA/NAME_PROJECT/configs/NAME_SUBPROJECT/COMMENT_NOW
- Optuna database of a subproject: ROOTDIR_DATA/NAME_PROJECT/dbfiles/NAME_SUBPROJECT
- Pytorch datasets: ROOTDIR_PTDATASET = ROOTDIR_DATA + "/datasets_pytorch"
- Torchvision model weights: /home/USERNAME/.cache/torch/hub (official default).
"""
import os
import warnings
from typing import Any, Dict

import torch
from configs.info_available import (LIST_AVAILABLE_ACTIVATIONS,
                                    LIST_AVAILABLE_DATASETS,
                                    LIST_AVAILABLE_LOSSES,
                                    LIST_AVAILABLE_MODELS,
                                    LIST_AVAILABLE_OPTIMIZERS,
                                    LIST_AVAILABLE_OPTUNA_PRUNERS,
                                    LIST_AVAILABLE_OPTUNA_SAMPLERS,
                                    LIST_AVAILABLE_SAMPLERS,
                                    LIST_AVAILABLE_SCHEDULERS)
from utils.log_config import get_my_logger

logger = get_my_logger(__name__)

""" 1. Project name """
USER_NAME = "anonymous"
ROOTDIR_DATA = "/data/" + USER_NAME
NAME_PROJECT = "nfn"

NAME_SUBPROJECT = "FTE_deg10_width2048_iter800k_nonlinearIC"
COMMENT = "_"
# COMMENT will be added to the end of the (cktp, tblog, txlog) directory name of the trial.


""" 2. Important """
# General
EXP_PHASE = "stat"  # try, tuning, or stat # When tuning, see kwargs of MedianPruner
# If DEBUG_MODE=False, RuntimeError (OOM) trials will not kill subsequent trials in the same process.
DEBUG_MODE = False
CUDA_VISIBLE_DEVICES = "0"
NUM_TRIALS = 1
BATCH_SIZE = 1024  # ignored when tuning and stat
NUM_ITERATIONS = 800_000  # 500_000
LOG_INTERVAL = NUM_ITERATIONS // 400
TBLOG_INTERVAL = NUM_ITERATIONS // 400
VALIDATION_INTERVAL = NUM_ITERATIONS // 40
NAME_MODEL = "SimplePINN"  # ignored when stat
DEGREE = 10  # For task="pinn" # ignored when stat

NAME_DATASET = "RCDFTransportEquation"
NAME_BASIS_FUNCTION = "legendre_no_w"  # ignored when stat
idx_init_cond_fte = 1  # Only for FTransport. # ignored when stat

# NAME_DATASET = "RCDBurgersHopfEquation"
# NAME_BASIS_FUNCTION = "fourier_no_w"  # ignored when stat
idx_init_cond_bhe = 2  # Only for BurgersHopf. # ignored when stat

flag_hard_bc = False  # Only for BurgersHopf. # ignored when stat
INDEX_TRIAL_STAT = None   # None (best trial) or int # used only when stat
DEFAULT_DTYPE = torch.float32

if NAME_BASIS_FUNCTION in ["fourier", "fourier_no_w"]:  # Added: 20231204
    DEGREE *= 2
    logger.info(
        f"Degree doubles up because name_basis_function={NAME_BASIS_FUNCTION}: Got {DEGREE//2}. Now {DEGREE}.")
assert DEGREE > 1  # Use at least polynomial_0 and polynomial_1.
if NAME_DATASET == "Integral2":
    assert NAME_BASIS_FUNCTION in ["legendre_no_w", "fourier_no_w"]
if NAME_DATASET == "RCDArcLength":
    assert NAME_BASIS_FUNCTION in ["legendre_no_w", "fourier_no_w"]
if NAME_DATASET == "RCDThomasFermi":
    assert NAME_BASIS_FUNCTION == "fourier_no_w"  # Periodic BC
if NAME_DATASET == "RCDGaussianWhiteNoise":
    assert NAME_BASIS_FUNCTION == "fourier_no_w"  # Periodic BC by definition
if NAME_DATASET == "RCDFTransportEquation":
    assert NAME_BASIS_FUNCTION == "legendre_no_w"
if NAME_DATASET == "RCDBurgersHopfEquation":
    assert NAME_BASIS_FUNCTION == "fourier_no_w"

# Loss
LIST_LOSSES = [
    # "CrossEntropyLoss",
    # "BCELoss",
    # "MSELoss",
    # "L1Loss",
    "SmoothL1Loss",
    # "LpLoss",
]
NAME_LOSS_REWEIGHT = [
    "uniform", "softmax", "BMTL"][1]  # currently used only for pinn
WEIGHT_DECAY = 5e-07  # Ignored when tuning
KWARGS_LOSSES = {
    "CrossEntropyLoss": {},
    "BCELoss": {},
    "MSELoss": {},
    "L1Loss": {},
    "SmoothL1Loss": {"beta": 1.},
    "LpLoss": {"p": float("inf"), }
}

# Optimizer
NAME_OPTIMIZER = "AdamW"  # Ignored when tuning
NAME_SCHEDULER = "CosineAnnealingWarmRestartsLinearStart"  # "ConstantLR"
# Ignored when tuning

# Data
NAME_SAMPLER = "RandomSampler"  # "OverSampler"
# PINN_SAMPLER = ["random", "latin_hypercube"][1]
BATCH_SIZE_RATIO = [5, 5]
KWARGS_NAME_SAMPLER: dict = {
    "RandomSampler": {},
    # "OverSampler": {},
}
init_cond_bhe = ["delta", "const", "moderate"][idx_init_cond_bhe]
# size_tau_bhe = [1e-4, 1e-3, 1e-3][idx_init_cond_bhe]
# mu_bar_bhe = [8., 16., 2.][idx_init_cond_bhe]
# sigma_squared_bhe = [2., 8., 16.][idx_init_cond_bhe]
size_tau_bhe = [1e-3, 1e-3, 1e-3][idx_init_cond_bhe]
mu_bar_bhe = [8., 8., 8.][idx_init_cond_bhe]
sigma_squared_bhe = [1e1, 1e1, 1e1][idx_init_cond_bhe]
# size_tau_bhe = [1e-1, 1e-1, 1e-1][idx_init_cond_bhe]
# mu_bar_bhe = [1e-1, 1e-1, 1e-1][idx_init_cond_bhe]
# sigma_squared_bhe = [1e4, 1e4, 1e4][idx_init_cond_bhe]
init_cond_fte = ["linear", "spectrum15"][idx_init_cond_fte]
KWARGS_NAME_DATASETS = {
    "MNIST": {
        "num_classes": 10,
        "task": "image_classification",
    },
    "FashionMNIST": {
        "num_classes": 10,
        "task": "image_classification",
    },
    "CIFAR10": {
        "num_classes": 10,
        "task": "image_classification",
    },
    "CIFAR100": {
        "num_classes": 100,
        "task": "image_classification",
    },
    "CIFAR10Flat": {
        "num_classes": 10,
        "task": "image_classification",
        "duration": 1024,  # seq len
        "num_channels": 3,
        "with_mask": False,
    },
    "CIFAR100Flat": {
        "num_classes": 100,
        "task": "image_classification",
        "duration": 1024,  # seq len
        "num_channels": 3,
        "with_mask": False,
    },
    # "CIFAR10LT": {
    #     "num_classes": 10,
    #     "task": "image_classification",
    # },
    # "CIFAR100LT": {
    #     "num_classes": 100,
    #     "task": "image_classification",
    # },
    # "iNaturalist2017": {
    #     "num_classes": 5089,
    #     "task": "image_classification",
    # },
    # "iNaturalist2018": {
    #     "num_classes": 8142,
    #     "task": "image_classification",
    # },
    "SpeechCommands": {
        "num_classes": 36,  # includes "_unknown_" class (index=0)
        "task": "audio_classification",
        "classwise_sample_sizes_tr": [408, 1346, 1594, 1697, 1657, 1711, 3134, 3033, 3240, 1275, 1256, 2955, 3106, 1632, 1727, 1286, 3037, 1710, 3170, 3130, 2970, 3086, 3140, 3019, 3205, 1606, 3088, 3111, 2966, 1407, 3111, 2948, 1288, 1724, 3228, 3250],
        "classwise_sample_sizes_va": [0, 153, 213, 182, 180, 197, 377, 346, 367, 132, 146, 373, 372, 219, 195, 128, 352, 195, 356, 406, 373, 363, 351, 363, 387, 204, 378, 350, 356, 159, 345, 350, 139, 193, 397, 384],
        "classwise_sample_sizes_te": [0, 165, 207, 185, 194, 220, 406, 408, 445, 172, 155, 400, 402, 203, 191, 161, 412, 195, 408, 405, 402, 396, 399, 396, 406, 212, 394, 411, 405, 193, 424, 425, 165, 206, 419, 418],
        "with_mask": True,  # x.shape=[2*B,C,T]
        "num_channels": 1,
        "duration": 8000,  # See below:
        # What is 'duration'?
        # duration: Union[int, None].
        # If None, sequence length = max length in the batch. If int,
        # sequence length = size (size must be larger than the max sequence
        # length in the batch). Default is None.
        # RNNs can be trained with size=None.
    },
    "RCPDPDE1": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "PDE1",
        "dim_input": 2,  # (t,x)
        "dim_output": 1,  # scalar function
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        "sizes": [[0., 1.], [0., 2.0]],  # domain of definition
        "boundaries": [[0., None]],  # for boundary conditions.
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": BATCH_SIZE_RATIO,  # = X:X_bc in a batch
        # See dataprocesses.random_cp_dataset.RandomCPDataset for definition of sizes and boundaries.
    },
    "RCPDODE1": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "ODE1",
        "dim_input": 1,  # (t,)
        "dim_output": 1,  # scalar function
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        "sizes": [[-1., 1.],],  # domain of definition
        "boundaries": [[0.5,]],  # for boundary conditions.
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": BATCH_SIZE_RATIO,  # = X:X_bc in a batch
        # See dataprocesses.random_cp_dataset.RandomCPDataset for definition of sizes and boundaries.
    },
    "RCPDHarmonicOscillator": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "HarmonicOscillator",
        "dim_input": 1,  # (t,)
        "dim_output": 1,  # scalar function
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        "sizes": [[0., 10.],],  # domain of definition
        "boundaries": [[0.,]],  # for boundary conditions.
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": BATCH_SIZE_RATIO,  # = X:X_bc in a batch
        "omega": 2.  # angular frequency [s]
        # See dataprocesses.random_cp_dataset.RandomCPDataset for definition of sizes and boundaries.
    },
    "RCPDKirchhoff": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "Kirchhoff",
        "dim_input": 2,  # (x,y)
        "dim_output": 1,  # scalar function
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        "sizes": [[0., 1.], [0., 1.]],  # domain of definition
        "boundaries": [
            [0., None],
            [1., None],
            [None, 0.],
            [None, 1.]],  # For boundary conditions. Be consistent with sizes.
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": BATCH_SIZE_RATIO,  # = X:X_bc in a batch
        "E": 68.,  # [GPa] # Aluminum
        "h": 1.,  # [mm]
        "nu": 0.32,  # [dimless]
        "p": 0.0980665 * 2.71,  # [Pa]
        "type_p": "uniform",  # uniform, rod
        "type_boundary": "supported",  # supported,clamped
        # D, p/D = 6.3, 0.042
        # See dataprocesses.random_cp_dataset.RandomCPDataset for definition of sizes and boundaries.
        # See losses.pdes.py for definition of other parameters.
    },
    "RCDIntegral1": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "Integral1",
        "name_basis_function": NAME_BASIS_FUNCTION,
        "dim_input": DEGREE,  # (a0, a1, ...)
        "dim_output": 1,  # scalar function
        "degree": DEGREE,
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        "sizes": [[-10., 10.]] * DEGREE,  # domain of definition
        "boundaries": [],  # no boundary conditions.
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": [1., 0.],  # = X:X_bc in a batch
        "num_points": 1e6  # for numerial integral
    },
    "RCDIntegral1V2": {  # Caution: u_in is rescaled! See pdes.py.
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "Integral1V2",
        "name_basis_function": NAME_BASIS_FUNCTION,
        "dim_input": DEGREE,  # (a0, a1, ...)
        "dim_output": 1,  # scalar function
        "degree": DEGREE,
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        "sizes": [[-10., 10.]] * DEGREE,  # domain of definition
        "boundaries": [],  # no boundary conditions.
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": [1., 0.],  # = X:X_bc in a batch
    },
    "RCDIntegral2": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "Integral2",
        "name_basis_function": NAME_BASIS_FUNCTION,
        "dim_input": 1 + DEGREE,  # (t, a0, a1, ...)
        "dim_output": 1,  # scalar function
        "degree": DEGREE,
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        # domain of definition
        "sizes": [[-1., 1.]] + [[-100., 100.]] * DEGREE,
        "boundaries": [[-1.] + [None] * DEGREE],  # for boundary conditions.
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": BATCH_SIZE_RATIO,  # = X:X_bc in a batch
    },
    "RCDArcLength": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "ArcLength",
        "name_basis_function": NAME_BASIS_FUNCTION,
        "dim_input": 1 + DEGREE,  # (t, a0, a1, ...)
        "dim_output": 1,  # scalar function
        "degree": DEGREE,
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        # domain of definition
        "sizes": [[-1., 1.]] + [[-100., 100.]] * DEGREE,
        "boundaries": [[-1.] + [None] * DEGREE],  # for boundary conditions.
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": BATCH_SIZE_RATIO,  # = X:X_bc in a batch
    },
    "RCDThomasFermi": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "ThomasFermi",
        "name_basis_function": NAME_BASIS_FUNCTION,
        "dim_input": 3 + 3*DEGREE,  # (x, y, z, a0, a1, ...)
        "dim_output": 1,  # scalar function
        "degree": DEGREE,
        "C_kin": 1e-15,
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        # domain of definition
        "sizes": [[-1., 1.], [-1., 1.], [-1., 1.]] + [[-100., 100.]] * 3*DEGREE,
        "boundaries": [
            [-1., None, None] + [None] * 3*DEGREE,
            [None, -1., None] + [None] * 3*DEGREE,
            [None, None, -1.] + [None] * 3*DEGREE,
        ],  # for boundary conditions.
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": BATCH_SIZE_RATIO,  # = X:X_bc in a batch
    },
    "RCDGaussianWhiteNoise": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "GaussianWhiteNoise",
        "name_basis_function": NAME_BASIS_FUNCTION,
        "dim_input": DEGREE,  # (a0, a1, ...)
        "dim_output": 1,  # scalar function
        "degree": DEGREE,
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        # domain of definition
        "sizes": [[-1., 1.]] * DEGREE,
        "boundaries": [],  # no boundary conditions
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": BATCH_SIZE_RATIO,  # = X:X_bc in a batch
    },
    "RCDFTransportEquation": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "FTransportEquation",
        "name_basis_function": NAME_BASIS_FUNCTION,
        "dim_input": 1 + DEGREE,  # (t, a0, a1, ...)
        "dim_output": 1,  # scalar function
        "degree": DEGREE,
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        # domain of definition
        "sizes": [[0., 1.]] + [[-1e0, 1e0]] * DEGREE,
        "boundaries": [[0.] + [None] * DEGREE],  # for boundary conditions.
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": BATCH_SIZE_RATIO,  # = X:X_bc in a batch
        "v_0": 1.,  # [m/s]
        "rho_0": 1.,  # [kg/m^2]
        "L": 1.,  # [m]
        "init_cond": init_cond_fte,
    },
    "RCDBurgersHopfEquation": {
        "num_classes": 0,  # Don't remove this key.
        "task": "pinn",
        "name_equation": "BurgersHopfEquation",
        "init_cond": init_cond_bhe,
        "name_basis_function": NAME_BASIS_FUNCTION,
        "dim_input": 1 + DEGREE,  # (t, a0, a1, ...)
        "dim_output": 1,  # scalar function
        "degree": DEGREE,
        "num_points_max": 1_000_000,  # num of train colloc. pts. in one epoch
        # domain of definition
        # to be rescaled in data_controller
        "sizes": [[0., size_tau_bhe]] + [[-1e-1, 1e-1]] * DEGREE,
        # bdy cond
        # "boundaries": [[0.] + [None] * DEGREE, [None] + [0.] * DEGREE],
        "boundaries": [[0.] + [None] * DEGREE,],
        # "sampler": PINN_SAMPLER,  # random, latin_hypercube
        "batch_size_ratio": BATCH_SIZE_RATIO,  # = X:X_bc in a batch
        "mu_bar": mu_bar_bhe,  # [dimless]
        "sigma_squared": sigma_squared_bhe,  # [dimless]
        "flag_hard_bc": flag_hard_bc,
    },
}

# Model
KWARGS_NAME_MODEL: Dict[str, Dict[str, Any]] = {  # Used only for self-made models
    'ConvNextTiny': {}, 'ConvNextSmall': {}, 'ConvNextBase': {}, 'ConvNextLarge': {}, 'DenseNet121': {}, 'DenseNet161': {}, 'DenseNet169': {}, 'DenseNet201': {}, 'EfficientNetB0': {}, 'EfficientNetB1': {}, 'EfficientNetB2': {}, 'EfficientNetB3': {}, 'EfficientNetB4': {}, 'EfficientNetB5': {}, 'EfficientNetB6': {}, 'EfficientNetB7': {}, 'EfficientNetV2S': {}, 'EfficientNetV2M': {}, 'EfficientNetV2L': {}, 'MaxViT': {}, 'MobileNetV3Small': {}, 'MobileNetV3Large': {}, 'ResNet18': {}, 'ResNet34': {}, 'ResNet50': {}, 'ResNet101': {}, 'ResNet152': {}, 'WideResNet50_2': {}, 'WideResNet101_2': {}, 'ResNeXt50_32X4D': {}, 'ResNeXt101_32X8D': {}, 'RegNetY_400MF': {}, 'RegNetY_800MF': {}, 'RegNetY_1_6GF': {}, 'RegNetY_3_2GF': {}, 'RegNetY_8GF': {}, 'RegNetY_16GF': {}, 'RegNetY_32GF': {}, 'RegNetY_128GF': {}, 'ShuffleNetV2X0_5': {}, 'ShuffleNetV2X1_0': {}, 'ShuffleNetV2X1_5': {}, 'ShuffleNetV2X2_0': {}, 'ViT_B16': {}, 'ViT_L16': {}, 'ViT_H16': {}, 'SwinV2_T': {}, 'SwinV2_S': {}, 'SwinV2_B': {}, 'VGG11_BN': {}, 'VGG13_BN': {}, 'VGG16_BN': {}, 'VGG19_BN': {}, 'S3D': {}, 'MC3_18': {}, 'R3D_18': {}, 'R2Plus1D_18': {}, 'MViT_V2_S': {}, 'MViT_V1_B': {}, 'Swin3D_T': {}, 'Swin3D_S': {}, 'Swin3D_B': {},
    'SimplePINN': {  # num params = ?
        # "dim_hidden": 1024,  # 1024,2048
        "dim_hidden": 2048,  # 1024,2048
        "num_layers": 4,  # 8,
        "name_activation": "Sin",
        "flag_positive_output": None},  # placeholder
    'MSSIREN': {  # num params = 1668618 (dim_input=2)
        "dim_hidden": 256,
        "num_layers": 6,
        "name_activation": "Sin",
        "scales": [0.1, 1., 10., 100, 1000],
        "flag_gate_unit": False,
        "flag_positive_output": None},  # placeholder
    'SWNN': {  # num params = 1668899 (dim_input=2)
        "dim_hidden": 128,
        "num_layers": 6,
        "name_activation": "Sin",
        "scales": [0.1, 1., 10., 100, 1000],
        "flag_gate_unit": True,
        "num_blocks": 3,
        "flag_positive_output": None},  # placeholder
    'TimesNet': {  # See docstring of TimesNet for details.
        "d_model": 32,  # dim_hidden
        "d_ff": 32,  # dim_hidden
        "e_layers": 3,  # num of TimesBlocks
        "dropout": 0.1,  # dropout rate
        "top_k": 6,  # top k frequencies
        "num_kernels": 6,  # Inception kernels
        "name_block": "ConvNeXt"  # "InvertedResidual"
    },  # "InceptionV1" "ConvNeXt" "InvertedResidual"
    'RNN': {
        "dim_hidden": 256,
        "num_layers": 1,
        "dropout":  0.05,
        "nonlinearity":  "B2Bsqrt",
        "bias": True,
        "bidirectional": False
    },
    'LSTM': {
        "dim_hidden": 256,
        "num_layers": 2,
        "dropout":  0.05,
        "nonlinearity1": "Sigmoid",  # sigmoid
        "nonlinearity2": "Tanh",  # "B2Bsqrt",  # tanh
        "bias": True,
        "bidirectional": False
    },
}
if NAME_MODEL in ["SimplePINN", "MSSIREN", "SIREN", "SWNN"]:
    KWARGS_NAME_MODEL[NAME_MODEL]["dim_input"] = KWARGS_NAME_DATASETS[NAME_DATASET]["dim_input"]
    KWARGS_NAME_MODEL[NAME_MODEL]["dim_output"] = KWARGS_NAME_DATASETS[NAME_DATASET]["dim_output"]
    KWARGS_NAME_MODEL[NAME_MODEL]["sizes"] = KWARGS_NAME_DATASETS[NAME_DATASET]["sizes"]

# Keyword arguments: optimizer, scheduler
KWARGS_NAME_OPTIMIZER = {  # Ignored when tuning
    "SGD": {"lr": 1e-4, "momentum": 0.9, "dampening": 0, "weight_decay": WEIGHT_DECAY, "nesterov": True},
    "SGDW": {"lr": 1e-6, "momentum": 0.9, "dampening": 0, "weight_decay": WEIGHT_DECAY, "nesterov": True, },
    "SGDP": {"lr": 1e-4, "momentum": 0.9, "dampening": 0, "weight_decay": WEIGHT_DECAY, "nesterov": True, "delta": 0.1, "wd_ratio": 0.1},
    "NAdam": {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": WEIGHT_DECAY, "momentum_decay": 0.004},
    "RMSprop": {"lr": 0.01, "alpha": 0.99, "eps": 1e-08, "weight_decay": WEIGHT_DECAY, "momentum": 0, "centered": False},
    "Yogi": {"lr": 1e-2, "betas": (0.9, 0.999), "eps": 1e-3, "initial_accumulator": 1e-6, "weight_decay": WEIGHT_DECAY, },
    "PID": {"lr": 1e-3, "momentum": 0, "dampening": 0, "weight_decay": WEIGHT_DECAY, "integral": 5.0, "derivative": 10.0, },
    "Lamb": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": WEIGHT_DECAY, },
    "QHM": {"lr": 1e-3, "momentum": 0, "nu": 0.7, "weight_decay": WEIGHT_DECAY, "weight_decay_type": 'grad', },
    "Adam": {"lr": 1e-7, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": WEIGHT_DECAY, "amsgrad": False},
    "AdamW": {"lr": 1e-5, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": WEIGHT_DECAY, "amsgrad": False},
    "RAdam": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": WEIGHT_DECAY},
    "AdaBelief": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-3, "weight_decay": WEIGHT_DECAY, "amsgrad": False, "weight_decouple": False, "fixed_decay": False, "rectify": False, },
    "Rprop": {"lr": 0.01, "etas": (0.5, 1.2), "step_sizes": (1e-06, 50)},
    "QHAdam": {"lr": 1e-3, "betas": (0.9, 0.999), "nus": (1.0, 1.0), "weight_decay": WEIGHT_DECAY, "decouple_weight_decay": False, "eps": 1e-8, },
    "Lion": {"lr": 1e-4, "betas": (0.9, 0.99), "weight_decay": WEIGHT_DECAY},
    "AdamP": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": WEIGHT_DECAY, "delta": 0.1, "wd_ratio": 0.1},
    "AdaMod": {"lr": 3.2560046984800586e-06, "betas": (0.9, 0.999), "beta3": 0.999, "eps": 1e-8, "weight_decay": WEIGHT_DECAY, },
    "DiffGrad": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": WEIGHT_DECAY, },
    "SWATS": {"lr": 1e-1, "betas": (0.9, 0.999), "eps": 1e-3, "weight_decay": WEIGHT_DECAY, "amsgrad": False, "nesterov": False, },
    "Adai": {"lr": 1e-3, "betas": (0.1, 0.99), "eps": 1e-03, "weight_decay": WEIGHT_DECAY, "decoupled": False},
    "AdaiV2": {"lr": 1e-3, "betas": (0.1, 0.99), "eps": 1e-03, "weight_decay": WEIGHT_DECAY, "dampening": 1., "decoupled": True},
    # "Apollo": {"lr": 1e-2, "beta": 0.9, "eps": 1e-4, "warmup": 0, "init_lr": 0.01, "weight_decay": WEIGHT_DECAY, },
    "RangerVA": {"lr": 1e-3, "alpha": 0.5, "k": 6, "n_sma_threshhold": 5, "betas": (.95, 0.999), "eps": 1e-5, "weight_decay": WEIGHT_DECAY, "amsgrad": True, "transformer": 'softplus', "smooth": 50, "grad_transformer": 'square'},
    "RangerQH": {"lr": 1e-3, "betas": (0.9, 0.999), "nus": (.7, 1.0), "weight_decay": WEIGHT_DECAY, "k": 6, "alpha": .5, "decouple_weight_decay": False, "eps": 1e-8, },
    "Ranger": {"lr": 1e-3, "alpha": 0.5, "k": 6, "N_sma_threshhold": 5, "betas": (.95, 0.999), "eps": 1e-5, "weight_decay": WEIGHT_DECAY},
    # "MADGRAD": {"lr": 1e-2, "momentum": 0.9, "weight_decay": WEIGHT_DECAY, "eps": 1e-6, },
    "ASGD": {"lr": 0.01, "lambd": 0.0001, "alpha": 0.75, "t0": 1e6, "weight_decay": WEIGHT_DECAY, },
    "Adadelta": {"lr": 1.0, "rho": 0.9, "eps": 1e-06, "weight_decay": WEIGHT_DECAY},
    # "Adagrad": {"lr": 0.01, "lr_decay": 0, "weight_decay": WEIGHT_DECAY, "initial_accumulator_value": 0, "eps": 1e-10},
    "Adamax": {"lr": 0.002, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": WEIGHT_DECAY},
    "A2GradExp": {"beta": 10.0, "lips": 10.0, "rho": 0.5, },
    "A2GradInc": {"beta": 10.0, "lips": 10.0, },
    "A2GradUni": {"beta": 10.0, "lips": 10.0, },
    "AccSGD": {"lr": 1e-3, "kappa": 1000.0, "xi": 10.0, "small_const": 0.7, "weight_decay": WEIGHT_DECAY},
    "AdaBound": {"lr": 1e-3, "betas": (0.9, 0.999), "final_lr": 0.1, "gamma": 1e-3, "eps": 1e-8, "weight_decay": WEIGHT_DECAY, "amsbound": False, },
    "NovoGrad": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": WEIGHT_DECAY, "grad_averaging": False, "amsgrad": False, },
    "AggMo": {"lr": 1e-3, "betas": (0.0, 0.9, 0.99), "weight_decay": WEIGHT_DECAY, },
    "Sophia": {"lr": 1e-7, 'betas': (0.965, 0.99), 'rho': 0.04, 'weight_decay': WEIGHT_DECAY, },
}
KWARGS_NAME_SCHEDULER = {  # Ignored when tuning
    # constant
    "Constant": {},
    # decays at every step_size
    "StepLR": {"step_size": NUM_ITERATIONS//3, "gamma": 0.1},
    # decays at mulestones
    "MultiStepLR": {"milestones": [NUM_ITERATIONS//4, NUM_ITERATIONS//3, NUM_ITERATIONS//2], "gamma": 0.1},
    # decays at total_iters
    "ConstantLR": {"factor": 0.3264627354902301, "total_iters": NUM_ITERATIONS//2},
    # linear warmup
    "LinearLR": {"start_factor": 1e-10, "end_factor": 1.0, "total_iters": NUM_ITERATIONS//10},
    # decays every epoch
    "ExponentialLR": {"gamma": 0.9},
    # decay every epoch to 0
    "PolynomialLR": {"total_iters": NUM_ITERATIONS//3, "power": 1.0},
    # cosine down, rapid up, cosine up, ...
    "CosineAnnealingLR": {"T_max": NUM_ITERATIONS, "eta_min": 0.},
    # decays on plateau
    "ReduceLROnPlateau": {"factor": 0.5, "patience": 3, "threshold": 1e-4},
    # up-down, up-down, ...
    "CyclicLR": {"base_lr": 0., "max_lr": 1e-3, "step_size_up": NUM_ITERATIONS//10, "gamma": 1.0, "cycle_momentum": False, "base_momentum": 0.8, "max_momentum": 0.9},
    # cosine down, rapid up, cosine down, ... (T_0 means const LR)
    "CosineAnnealingWarmRestarts": {"T_0": NUM_ITERATIONS//2, "T_mult": 1, "eta_min": 0., },
    # linear warmup, cosine down, rapid up, cosine down, ... (T_0 means const LR)
    "CosineAnnealingWarmRestartsLinearStart": {"T_0": NUM_ITERATIONS//2, "T_mult": 1, "eta_min": 0., "start_factor": 1e-12,  "milestone": NUM_ITERATIONS//100},
    "CosineAnnealingWarmRestartsLinearStartWeightDecay": {"T_0": NUM_ITERATIONS//2, "T_mult": 1, "eta_min": 0., "start_factor": 1e-12,  "milestone": NUM_ITERATIONS//100, "num_iter": NUM_ITERATIONS, "gamma": 0.5},
}  # Ref: https://timm.fast.ai/SGDR#t_mul


""" 3. Less Important"""
# General
NUM_THREADS_PER_PROCESS = 1  # Num of CPUs used for a single process
FLAG_FIX_SEED = False
SEED = 777  # Used only when FLAG_FIX_SEED = True
FLAG_SAVE_CKPT_TRYTUNING = False  # Save ckpt when exp phase = try or tuning
FLAG_OUTPUT_CONSOLE = True

# Data
FLAG_SHUFFLE = True  # Shuffle training dataset
FLAG_PIN_MEMORY = False  # True requires more CPU memory
NUM_WORKERS = 10  # Compare 1 vs. 2 vs. 20.

# Model
FLAG_INIT_LAST_LINEAR = True
FLAG_PRETRAINED = True
# FLAG_TRAINABLE_BASE = True # Trainable flag # NotImplemented
# FLAG_USE_BIAS = True  # Use bias of the last Linear layer or not # NotImplemented


""" 4. Search Spaces """
FLL_LR = [1e-6, 1e-3]  # Do not change name: out of naming rule.
FLL_WEIGHT_DECAY = [1e-7, 1e-4]  # Do not change name: out of naming rule.
CAT_ACTIVATION = [  # Do not change name: out of naming rule.
    # "Linear",  # =
    # "Gaussian",
    # "SuperGaussian",  # =
    # "Quadratic",
    # "MultiQuadratic", # =
    # "Laplacian",
    # "ExpSin",
    "Sin",
    # "SinSquared",
    # "Sinc",
    # "SinReLU",
    # "SinSiLU",
    # "SinGaussian",
    # "ELU",
    # "Hardshrink",
    # "Hardsigmoid",  # =
    # "Hardtanh",
    # "Hardswish",
    # "LeakyReLU",
    # "LogSigmoid",
    # "PReLU",
    # "ReLU",
    # "ReLU6",
    # "RReLU",
    # "SELU",
    # "CELU",
    # "GELU",
    # "Sigmoid",  # =
    # "SiLU",  # swish
    # "Mish",
    # "Softplus",
    # "Softshrink",
    # "Softsign",
    # "Tanh",
    # "Tanhshrink",
    # "B2Blog",
    # "B2Bcbrt",
    # "B2Bexp",
    # "Tanhplus",
    # "DullReLU", #=
    # "SinB2BsqrtV2",
    # "B2Bsqrt",
    # "B2BsqrtV2", #=
    # "Ricker",
    # "ABU",
]
if KWARGS_NAME_DATASETS[NAME_DATASET]["task"] == "pinn":
    SS_CAT_NAME_LOSS_REWEIGHT = [
        # "uniform",
        "softmax",
        # "BMTL"
    ]
else:
    SS_CAT_NAME_LOSS_REWEIGHT = ["uniform"]
SS_CAT_BATCH_SIZE = [BATCH_SIZE]
# SS_CAT_PINN_SAMPLER = [
#     "random",
#     # "latin_hypercube"
# ]
if "batch_size_ratio" in KWARGS_NAME_DATASETS[NAME_DATASET].keys() and\
        KWARGS_NAME_DATASETS[NAME_DATASET]["batch_size_ratio"][1] != 0:  # type:ignore
    SS_CAT_BATCH_SIZE_RATIO = [[5, 5]]
elif "batch_size_ratio" in KWARGS_NAME_DATASETS[NAME_DATASET].keys() and\
        KWARGS_NAME_DATASETS[NAME_DATASET]["batch_size_ratio"][1] == 0:  # type:ignore
    SS_CAT_BATCH_SIZE_RATIO = [[1, 0]]
SS_CAT_NAME_OPTIMIZER = [
    # 'SGD',
    # 'SGDW',
    # 'SGDP',
    # 'NAdam',
    # 'RMSprop',
    # 'Yogi',
    # 'PID',
    # 'Lamb',
    # 'QHM',
    # 'Adam',
    'AdamW',
    # 'RAdam',
    # # 'AdaBelief',  # =
    # 'Rprop',
    # #'QHAdam',  # =
    # 'Lion',
    # 'AdamP',
    # 'AdaMod',
    # 'DiffGrad',
    # 'SWATS',
    # 'Adai',
    # #'AdaiV2',  # =
    # # 'Apollo', # OOM
    # 'RangerVA',
    # #'RangerQH',  # =
    # #'Ranger',  # =
    # # 'MADGRAD',# buggy: RuntimeError
    # 'ASGD',
    # #'Adadelta',  # =
    # # 'Adagrad',  # buggy: RuntimeError
    # #'Adamax',  # =
    # #'A2GradExp', # =
    # 'A2GradInc',
    # #'A2GradUni', # =
    # 'AccSGD',
    # 'AdaBound',
    # 'NovoGrad',
    # 'AggMo',
    # 'Sophia',
]
SS_CAT_NAME_SCHEDULER = [
    # "Constant",
    # "MultiplicativeLR",
    # "StepLR",
    # "MultiStepLR",
    # "ConstantLR",
    # "LinearLR",
    # "ExponentialLR",
    # "PolynomialLR",
    # "CosineAnnealingLR",
    # "ReduceLROnPlateau",
    # "CyclicLR",
    # "CosineAnnealingWarmRestarts",
    "CosineAnnealingWarmRestartsLinearStart",
    # "CosineAnnealingWarmRestartsLinearStartWeightDecay",
]
SS_KWARGS_NAME_MODEL = {
    'ConvNextTiny': {}, 'ConvNextSmall': {}, 'ConvNextBase': {}, 'ConvNextLarge': {}, 'DenseNet121': {}, 'DenseNet161': {}, 'DenseNet169': {}, 'DenseNet201': {}, 'EfficientNetB0': {}, 'EfficientNetB1': {}, 'EfficientNetB2': {}, 'EfficientNetB3': {}, 'EfficientNetB4': {}, 'EfficientNetB5': {}, 'EfficientNetB6': {}, 'EfficientNetB7': {}, 'EfficientNetV2S': {}, 'EfficientNetV2M': {}, 'EfficientNetV2L': {}, 'MaxViT': {}, 'MobileNetV3Small': {}, 'MobileNetV3Large': {}, 'ResNet18': {}, 'ResNet34': {}, 'ResNet50': {}, 'ResNet101': {}, 'ResNet152': {}, 'WideResNet50_2': {}, 'WideResNet101_2': {}, 'ResNeXt50_32X4D': {}, 'ResNeXt101_32X8D': {}, 'RegNetY_400MF': {}, 'RegNetY_800MF': {}, 'RegNetY_1_6GF': {}, 'RegNetY_3_2GF': {}, 'RegNetY_8GF': {}, 'RegNetY_16GF': {}, 'RegNetY_32GF': {}, 'RegNetY_128GF': {}, 'ShuffleNetV2X0_5': {}, 'ShuffleNetV2X1_0': {}, 'ShuffleNetV2X1_5': {}, 'ShuffleNetV2X2_0': {}, 'ViT_B16': {}, 'ViT_L16': {}, 'ViT_H16': {}, 'SwinV2_T': {}, 'SwinV2_S': {}, 'SwinV2_B': {}, 'VGG11_BN': {}, 'VGG13_BN': {}, 'VGG16_BN': {}, 'VGG19_BN': {}, 'S3D': {}, 'MC3_18': {}, 'R3D_18': {}, 'R2Plus1D_18': {}, 'MViT_V2_S': {}, 'MViT_V1_B': {}, 'Swin3D_T': {}, 'Swin3D_S': {}, 'Swin3D_B': {},
    'SimplePINN': {
        # "dim_hidden_SimplePINNCAT": [1024],  # [16, 64, 256, 512],
        "dim_hidden_SimplePINNCAT": [2048],  # [16, 64, 256, 512],
        "num_layers_SimplePINNINT": [4],  # [4, 10],
        "name_activation_SimplePINNCAT": CAT_ACTIVATION, },
    'TimesNet': {
        "d_model_TimesNetCAT": [32, 64],  # dim_hidden
        "d_ff_TimesNetCAT": [32, 64],  # dim_hidden
        "e_layers_TimesNetINT": [2, 5],  # num of TimesBlocks
        "dropout_TimesNetFLT": [0., 0.2],  # dropout rate
        "top_k_TimesNetINT": [2, 8],  # top k frequencies
        "num_kernels_TimesNetINT": [6],  # Inception kernels
        "name_block_TimesNetCAT": ["ConvNeXt", "InvertedResidual"]
    },
    'MSSIREN': {
        "dim_hidden_MSSIRENCAT": [256],  # [16, 64, 256, 512],
        "num_layers_MSSIRENINT": [6],  # [4, 8],
        "name_activation_MSSIRENCAT": CAT_ACTIVATION,
        "scales_MSSIRENCAT": [0.1, 1., 10., 100, 1000],
        "flag_gate_unit_MSSIRENCAT": [False], },
    'SWNN': {
        "dim_hidden_SWNNCAT": [128],  # [16, 64, 256, 512],
        "num_layers_SWNNINT": [6],  # [4, 8],
        "name_activation_SWNNCAT": CAT_ACTIVATION,
        "scales_SWNNCAT": [0.1, 1., 10., 100, 1000],
        "flag_gate_unit_SWNNCAT": [True],  # [True, False],
        "num_blocks_SWNNINT": [3,], },
}
SS_KWARGS_NAME_OPTIMIZER = {  # Under constraction: Will come back after completing optuna stuff.
    'SGD': {'lr_SGDFLL': FLL_LR, 'momentum_SGDFLT': [0., 0.999], 'dampening_SGDFLT': [0.], 'weight_decay_SGDFLL': FLL_WEIGHT_DECAY, 'nesterov_SGDCAT': [True, False]},
    'SGDW': {'lr_SGDWFLL': FLL_LR, 'momentum_SGDWFLT': [0., 0.999], 'dampening_SGDWFLT': [0.], 'weight_decay_SGDWFLL': FLL_WEIGHT_DECAY, 'nesterov_SGDWCAT': [True, False]},
    'SGDP': {'lr_SGDPFLL': FLL_LR, 'momentum_SGDPFLT': [0., 0.999], 'dampening_SGDPFLT': [0.], 'weight_decay_SGDPFLL': FLL_WEIGHT_DECAY, 'nesterov_SGDPCAT': [True, False], 'delta_SGDPFLT': [0.1], 'wd_ratio_SGDPFLT': [0.1]},
    'NAdam': {'lr_NAdamFLL': FLL_LR, 'betas_NAdamCAT': [[0.9, 0.999]], 'eps_NAdamFLL': [1e-08], 'weight_decay_NAdamFLL': FLL_WEIGHT_DECAY, 'momentum_decay_NAdamFLT': [0.004]},
    'RMSprop': {'lr_RMSpropFLL': FLL_LR, 'alpha_RMSpropFLT': [0.99], 'eps_RMSpropFLL': [1e-08], 'weight_decay_RMSpropFLL': FLL_WEIGHT_DECAY, 'momentum_RMSpropFLT': [0., 0.999], 'centered_RMSpropCAT': [False]},
    'Yogi': {'lr_YogiFLL': FLL_LR, 'betas_YogiCAT': [[0.9, 0.999]], 'eps_YogiFLL': [1e-3], 'initial_accumulator_YogiFLL': [1e-06], 'weight_decay_YogiFLL': FLL_WEIGHT_DECAY},
    'PID': {'lr_PIDFLL': FLL_LR, 'momentum_PIDFLT': [0., 0.999], 'dampening_PIDFLT': [0], 'weight_decay_PIDFLL': FLL_WEIGHT_DECAY, 'integral_PIDFLT': [5.0], 'derivative_PIDFLT': [10.0]},
    'Lamb': {'lr_LambFLL': FLL_LR, 'betas_LambCAT': [[0.9, 0.999]], 'eps_LambFLL': [1e-08], 'weight_decay_LambFLL': FLL_WEIGHT_DECAY},
    'QHM': {'lr_QHMFLL': FLL_LR, 'momentum_QHMFLT': [0., 0.999], 'nu_QHMFLT': [0.7], 'weight_decay_QHMFLL': FLL_WEIGHT_DECAY, 'weight_decay_type_QHMCAT': ['grad']},
    'Adam': {'lr_AdamFLL': FLL_LR, 'betas_AdamCAT': [[0.9, 0.999]], 'eps_AdamFLL': [1e-08], 'weight_decay_AdamFLL': FLL_WEIGHT_DECAY, 'amsgrad_AdamCAT': [False]},
    'AdamW': {'lr_AdamWFLL': FLL_LR, 'betas_AdamWCAT': [[0.9, 0.999]], 'eps_AdamWFLL': [1e-08], 'weight_decay_AdamWFLL': FLL_WEIGHT_DECAY, 'amsgrad_AdamWCAT': [False]},
    'RAdam': {'lr_RAdamFLL': FLL_LR, 'betas_RAdamCAT': [[0.9, 0.999]], 'eps_RAdamFLL': [1e-08], 'weight_decay_RAdamFLL': FLL_WEIGHT_DECAY},
    'AdaBelief': {'lr_AdaBeliefFLL': FLL_LR, 'betas_AdaBeliefCAT': [[0.9, 0.999]], 'eps_AdaBeliefFLL': [1e-3], 'weight_decay_AdaBeliefFLL': FLL_WEIGHT_DECAY, 'amsgrad_AdaBeliefCAT': [True, False], 'weight_decouple_AdaBeliefCAT': [True, False], 'fixed_decay_AdaBeliefCAT': [True, False], 'rectify_AdaBeliefCAT': [True, False]},
    'Rprop': {'lr_RpropFLL': FLL_LR, 'etas_RpropCAT': [[0.5, 1.2]], 'step_sizes_RpropCAT': [[1e-06, 50]]},
    'QHAdam': {'lr_QHAdamFLL': FLL_LR, 'betas_QHAdamCAT': [[0.9, 0.999]], 'nus_QHAdamCAT': [[1.0, 1.0]], 'weight_decay_QHAdamFLL': FLL_WEIGHT_DECAY, 'decouple_weight_decay_QHAdamCAT': [True, False], 'eps_QHAdamFLL': [1e-08]},
    'Lion': {'lr_LionFLL': FLL_LR, 'betas_LionCAT': [[0.9, 0.99]], 'weight_decay_LionFLL': FLL_WEIGHT_DECAY},
    'AdamP': {'lr_AdamPFLL': FLL_LR, 'betas_AdamPCAT': [[0.9, 0.999]], 'eps_AdamPFLL': [1e-08], 'weight_decay_AdamPFLL': FLL_WEIGHT_DECAY, 'delta_AdamPFLT': [0.1], 'wd_ratio_AdamPFLT': [0.1]},
    'AdaMod': {'lr_AdaModFLL': FLL_LR, 'betas_AdaModCAT': [[0.9, 0.999]], 'beta3_AdaModFLT': [0.999], 'eps_AdaModFLL': [1e-08], 'weight_decay_AdaModFLL': FLL_WEIGHT_DECAY},
    'DiffGrad': {'lr_DiffGradFLL': FLL_LR, 'betas_DiffGradCAT': [[0.9, 0.999]], 'eps_DiffGradFLL': [1e-08], 'weight_decay_DiffGradFLL': FLL_WEIGHT_DECAY},
    'SWATS': {'lr_SWATSFLL': FLL_LR, 'betas_SWATSCAT': [[0.9, 0.999]], 'eps_SWATSFLL': [1e-3], 'weight_decay_SWATSFLL': FLL_WEIGHT_DECAY, 'amsgrad_SWATSCAT': [True, False], 'nesterov_SWATSCAT': [True, False]},
    'Adai': {'lr_AdaiFLL': FLL_LR, 'betas_AdaiCAT': [[0.1, 0.99]], 'eps_AdaiFLL': [0.001], 'weight_decay_AdaiFLL': FLL_WEIGHT_DECAY, 'decoupled_AdaiCAT': [False]},
    'AdaiV2': {'lr_AdaiV2FLL': FLL_LR, 'betas_AdaiV2CAT': [[0.1, 0.99]], 'eps_AdaiV2FLL': [0.001], 'weight_decay_AdaiV2FLL': FLL_WEIGHT_DECAY, 'dampening_AdaiV2FLT': [1.0], 'decoupled_AdaiV2CAT': [True, False]},
    # 'Apollo': {'lr_ApolloFLL': FLL_LR, 'beta_ApolloFLT': [0.9], 'eps_ApolloFLL': [0.0001], 'warmup_ApolloINT': [0], 'init_lr_ApolloFLL': [0.01], 'weight_decay_ApolloFLL': FLL_WEIGHT_DECAY},
    'RangerVA': {'lr_RangerVAFLL': FLL_LR, 'alpha_RangerVAFLT': [0.5], 'k_RangerVAINT': [6], 'n_sma_threshhold_RangerVAINT': [5], 'betas_RangerVACAT': [[0.95, 0.999]], 'eps_RangerVAFLL': [1e-05], 'weight_decay_RangerVAFLL': FLL_WEIGHT_DECAY, 'amsgrad_RangerVACAT': [True, False], 'transformer_RangerVACAT': ['softplus'], 'smooth_RangerVAINT': [50], 'grad_transformer_RangerVACAT': ['square']},
    'RangerQH': {'lr_RangerQHFLL': FLL_LR, 'betas_RangerQHCAT': [[0.9, 0.999]], 'nus_RangerQHCAT': [[0.7, 1.0]], 'weight_decay_RangerQHFLL': FLL_WEIGHT_DECAY, 'k_RangerQHINT': [6], 'alpha_RangerQHFLT': [0.5], 'decouple_weight_decay_RangerQHCAT': [True, False], 'eps_RangerQHFLL': [1e-08]},
    'Ranger': {'lr_RangerFLL': FLL_LR, 'alpha_RangerFLT': [0.5], 'k_RangerINT': [6], 'N_sma_threshhold_RangerINT': [5], 'betas_RangerCAT': [[0.95, 0.999]], 'eps_RangerFLL': [1e-05], 'weight_decay_RangerFLL': FLL_WEIGHT_DECAY},
    # 'MADGRAD': {'lr_MADGRADFLL': FLL_LR, 'momentum_MADGRADFLT': [0.9], 'weight_decay_MADGRADFLL': FLL_WEIGHT_DECAY, 'eps_MADGRADFLL': [1e-06]},
    'ASGD': {'lr_ASGDFLL': FLL_LR, 'lambd_ASGDFLL': [0.0001], 'alpha_ASGDFLT': [0.75], 't0_ASGDFLL': [1e3, 1e6], 'weight_decay_ASGDFLL': FLL_WEIGHT_DECAY},
    'Adadelta': {'lr_AdadeltaFLL': FLL_LR, 'rho_AdadeltaFLT': [0.9], 'eps_AdadeltaFLL': [1e-06], 'weight_decay_AdadeltaFLL': FLL_WEIGHT_DECAY},
    # 'Adagrad': {'lr_AdagradFLL': FLL_LR, 'lr_decay_AdagradFLT': [0], 'weight_decay_AdagradFLL': FLL_WEIGHT_DECAY, 'initial_accumulator_value_AdagradFLT': [0], 'eps_AdagradFLL': [1e-10]},
    'Adamax': {'lr_AdamaxFLL': FLL_LR, 'betas_AdamaxCAT': [[0.9, 0.999]], 'eps_AdamaxFLL': [1e-08], 'weight_decay_AdamaxFLL': FLL_WEIGHT_DECAY},
    'A2GradExp': {'beta_A2GradExpFLT': [1., 100.0], 'lips_A2GradExpFLT': [1., 100.0], 'rho_A2GradExpFLT': [0.1, 0.9]},
    'A2GradInc': {'beta_A2GradIncFLT': [1., 100.0], 'lips_A2GradIncFLT': [1., 100.0]},
    'A2GradUni': {'beta_A2GradUniFLT': [1., 100.0], 'lips_A2GradUniFLT': [1., 100.0]},
    'AccSGD': {'lr_AccSGDFLL': FLL_LR, 'kappa_AccSGDFLT': [1000.0], 'xi_AccSGDFLT': [10.0], 'small_const_AccSGDFLT': [0.7], 'weight_decay_AccSGDFLL': FLL_WEIGHT_DECAY},
    'AdaBound': {'lr_AdaBoundFLL': FLL_LR, 'betas_AdaBoundCAT': [[0.9, 0.999]], 'final_lr_AdaBoundFLL': [0.1], 'gamma_AdaBoundFLL': [0.001], 'eps_AdaBoundFLL': [1e-08], 'weight_decay_AdaBoundFLL': FLL_WEIGHT_DECAY, 'amsbound_AdaBoundCAT': [True, False]},
    'NovoGrad': {'lr_NovoGradFLL': FLL_LR, 'betas_NovoGradCAT': [[0.9, 0.999]], 'eps_NovoGradFLL': [1e-08], 'weight_decay_NovoGradFLL': FLL_WEIGHT_DECAY, 'grad_averaging_NovoGradCAT': [True, False], 'amsgrad_NovoGradCAT': [True, False]},
    'AggMo': {'lr_AggMoFLL': FLL_LR, 'betas_AggMoCAT': [[0.0, 0.9, 0.99]], 'weight_decay_AggMoFLL': FLL_WEIGHT_DECAY},
    'Sophia': {"lr_SophiaFLL": FLL_LR, 'betas_SophiaCAT': [[0.965, 0.99]], 'rho_SophiaFLT': [0.03, 0.04], 'weight_decay_SophiaFLL': FLL_WEIGHT_DECAY, },
}
SS_KWARGS_NAME_SCHEDULER = {
    'Constant': {},
    'StepLR': {'step_size_StepLRINT': [NUM_ITERATIONS//3], 'gamma_StepLRFLT': [0.1, 0.99]},
    'MultiStepLR': {'milestones_MultiStepLRCAT': [[NUM_ITERATIONS//4, NUM_ITERATIONS//3, NUM_ITERATIONS//2]], 'gamma_MultiStepLRFLT': [0.1, 0.9]},
    'ConstantLR': {'factor_ConstantLRFLT': [1./10., 1./2.], 'total_iters_ConstantLRINT': [NUM_ITERATIONS//2]},
    'LinearLR': {'start_factor_LinearLRFLT': [1e-10], 'end_factor_LinearLRFLT': [1.0], 'total_iters_LinearLRINT': [NUM_ITERATIONS//10]},
    'ExponentialLR': {'gamma_ExponentialLRFLT': [0.1, 0.9999]},
    'PolynomialLR': {'total_iters_PolynomialLRINT': [NUM_ITERATIONS//3], 'power_PolynomialLRFLT': [1.0]},
    'CosineAnnealingLR': {'T_max_CosineAnnealingLRINT': [NUM_ITERATIONS], 'eta_min_CosineAnnealingLRFLT': [0.]},
    'ReduceLROnPlateau': {'factor_ReduceLROnPlateauFLT': [0.1, 0.9], 'patience_ReduceLROnPlateauINT': [1, 3], 'threshold_ReduceLROnPlateauFLL': [1e-5, 1e-2]},
    'CyclicLR': {'base_lr_CyclicLRFLL': [0.], 'max_lr_CyclicLRFLL': [1e-3], 'step_size_up_CyclicLRINT': [NUM_ITERATIONS//3], 'gamma_CyclicLRFLT': [1.0], 'cycle_momentum_CyclicLRCAT': [False], 'base_momentum_CyclicLRFLT': [0.8], 'max_momentum_CyclicLRFLT': [0.9]},
    'CosineAnnealingWarmRestarts': {
        'T_0_CosineAnnealingWarmRestartsCAT': [NUM_ITERATIONS//5, NUM_ITERATIONS//2, NUM_ITERATIONS//1],
        'T_mult_CosineAnnealingWarmRestartsINT': [1, 2],
        'eta_min_CosineAnnealingWarmRestartsFLT': [0.]},
    "CosineAnnealingWarmRestartsLinearStart": {
        "T_0_CosineAnnealingWarmRestartsLinearStartCAT": [NUM_ITERATIONS//5, NUM_ITERATIONS//2, NUM_ITERATIONS//1],
        "T_mult_CosineAnnealingWarmRestartsLinearStartINT": [1, 2],
        "eta_min_CosineAnnealingWarmRestartsLinearStartFLT": [0.],
        "start_factor_CosineAnnealingWarmRestartsLinearStartFLT": [1e-10],
        "milestone_CosineAnnealingWarmRestartsLinearStartCAT": [0, NUM_ITERATIONS//10, NUM_ITERATIONS//100]},
    "CosineAnnealingWarmRestartsLinearStartWeightDecay": {
        "T_0_CosineAnnealingWarmRestartsLinearStartWeightDecayCAT": [NUM_ITERATIONS//5, NUM_ITERATIONS//2, NUM_ITERATIONS//1],
        "T_mult_CosineAnnealingWarmRestartsLinearStartWeightDecayINT": [1, 2],
        "eta_min_CosineAnnealingWarmRestartsLinearStartWeightDecayFLT": [0.],
        "start_factor_CosineAnnealingWarmRestartsLinearStartWeightDecayFLT": [1e-12],
        "milestone_CosineAnnealingWarmRestartsLinearStartWeightDecayCAT": [0, NUM_ITERATIONS//10, NUM_ITERATIONS//100],
        "num_iter_CosineAnnealingWarmRestartsLinearStartWeightDecayCAT": [NUM_ITERATIONS,],
        "gamma_CosineAnnealingWarmRestartsLinearStartWeightDecayFLT": [0.1, 0.9]},
}
SS_KWARGS_NAME_SAMPLER: dict = {
    "RandomSampler": {},
    # "OverSampler": {},
}
SS_KWARGS_LOSSES = {
    "MSELoss": {},
    "L1Loss": {},
    "SmoothL1Loss": {},
    "CrossEntropyLoss": {},
    "BCELoss": {},
    "LpLoss": {"p_LpLossCAT": [float("inf")]}
}


""" 5. Optuna params"""
NAME_OPTUNA_SAMPLER = "TPESampler"  # "RandomSampler"   # Ignored when stat
if EXP_PHASE in ["try", "stat"]:
    NAME_OPTUNA_PRUNER = "NopPruner"
elif EXP_PHASE == "tuning":
    NAME_OPTUNA_PRUNER = "MedianPruner"
else:
    raise ValueError()
KWARGS_OPTUNA_SAMPLER: dict = {
    "TPESampler": {},
    "CmaEsSampler": {},
    "RandomSampler": {},
}
# Pruning flag becomes True only when the number of iterations >= start_pruning iterations.
START_PRUNING = NUM_ITERATIONS // 4
KWARGS_OPTUNA_PRUNER = {  # Ignored when stat
    "NopPruner": {},
    "MedianPruner": {
        "n_startup_trials": 25,  # default: 5
        "n_warmup_steps": NUM_ITERATIONS // 4,  # default: 0
        "interval_steps": 1,  # default: 1
        "n_min_trials": 25,  # default: 1
    },
    "SuccessiveHalvingPruner": {
        "min_resource": 'auto',  # default: 'auto'
        "reduction_factor": 4,  # default: 4
        "min_early_stopping_rate": 0,  # default: 0
        "bootstrap_count": 5,  # default: 0
    },
    "HyperbandPruner": {
        "min_resource": 1,  # default: 1
        "max_resource": 'auto',  # default: 'auto'
        "reduction_factor": 3,  # default: 3
        "bootstrap_count": 5,  # default: 0
    },
    "ThresholdPruner": {
        "lower": None,  # default: None
        "upper": 0.02,  # default: None
        "n_warmup_steps": 5,  # default: 0
        "interval_steps": 1,  # default: 1
    },
    "PercentilePruner": {
        "percentile": 75.0,    # top X% is kept.
        "n_startup_trials": 25,  # default: 5
        "n_warmup_steps": NUM_ITERATIONS // 4,  # default: 0
        "interval_steps": 1,   # default: 1
        "n_min_trials": 25,     # default: 1
    }
}


""" 6. Directory names """
if EXP_PHASE != "stat":
    NAME_SUBPROJECT = NAME_SUBPROJECT + "_" + EXP_PHASE  # Overwrite
    ROOTDIR_PROJECT = ROOTDIR_DATA + f"/{NAME_PROJECT}"
    ROOTDIR_TBLOGS = ROOTDIR_PROJECT + "/tblogs"
    ROOTDIR_TXLOGS = ROOTDIR_PROJECT + "/txlogs"
    ROOTDIR_CKPTS = ROOTDIR_PROJECT + "/ckpts"
    ROOTDIR_CONFIGS = ROOTDIR_PROJECT + "/configs"
    ROOTDIR_PTDATASETS = ROOTDIR_DATA + "/datasets_pytorch"
    ROOTDIR_DBFILES = ROOTDIR_PROJECT + "/dbfiles"
    DIR_DBFILE = ROOTDIR_DBFILES + "/" + NAME_SUBPROJECT
    PATH_DBFILE = DIR_DBFILE + "/optuna.db"
else:
    _tmp = NAME_SUBPROJECT + "_tuning"
    NAME_SUBPROJECT = NAME_SUBPROJECT + "_" + EXP_PHASE  # Overwrite
    ROOTDIR_PROJECT = ROOTDIR_DATA + f"/{NAME_PROJECT}"
    ROOTDIR_TBLOGS = ROOTDIR_PROJECT + "/tblogs"
    ROOTDIR_TXLOGS = ROOTDIR_PROJECT + "/txlogs"
    ROOTDIR_CKPTS = ROOTDIR_PROJECT + "/ckpts"
    ROOTDIR_CONFIGS = ROOTDIR_PROJECT + "/configs"
    ROOTDIR_PTDATASETS = ROOTDIR_DATA + "/datasets_pytorch"
    ROOTDIR_DBFILES = ROOTDIR_PROJECT + "/dbfiles"
    DIR_DBFILE = ROOTDIR_DBFILES + "/" + _tmp
    PATH_DBFILE = DIR_DBFILE + "/optuna.db"


""" 7. Other Dependencies """
PATH_CONFIG = os.path.abspath(__file__)
FLAG_MULTIGPU = True if len(CUDA_VISIBLE_DEVICES) > 2 else False
if KWARGS_NAME_DATASETS[NAME_DATASET]["num_classes"] > 5:  # type:ignore
    TOP_K = 5
else:
    TOP_K = KWARGS_NAME_DATASETS[  # type:ignore
        NAME_DATASET]["num_classes"] - 1
if NAME_MODEL in ["TimesNet", "RNN", "LSTM"]:
    assert "duration" in KWARGS_NAME_DATASETS[NAME_DATASET].keys()
    assert "num_channels" in KWARGS_NAME_DATASETS[NAME_DATASET].keys()
    KWARGS_NAME_MODEL[NAME_MODEL][  # type:ignore
        "duration"] = KWARGS_NAME_DATASETS[NAME_DATASET]["duration"]
    KWARGS_NAME_MODEL[NAME_MODEL][  # type:ignore
        "dim_input"] = KWARGS_NAME_DATASETS[NAME_DATASET]["num_channels"]


""" 8. Sanity Checks """
assert EXP_PHASE in ["try", "tuning", "stat"]
# Available or not
if not NAME_OPTIMIZER in LIST_AVAILABLE_OPTIMIZERS:
    raise ValueError(
        f"OPTIMIZER={NAME_OPTIMIZER} not in LIST_AVAILABLE_OPTIMIZERS.")
if not set(KWARGS_NAME_OPTIMIZER.keys()) == set(LIST_AVAILABLE_OPTIMIZERS):
    raise ValueError(
        f"set(KWARGS_NAME_OPTIMIZER.keys()) != set(LIST_AVAILABLE_OPTIMIZERS). " +
        f"Got {set(KWARGS_NAME_OPTIMIZER.keys())} and {set(LIST_AVAILABLE_OPTIMIZERS)}.")
if not set(SS_KWARGS_NAME_OPTIMIZER.keys()).issubset(set(LIST_AVAILABLE_OPTIMIZERS)):
    raise ValueError(
        f"set(SS_KWARGS_NAME_OPTIMIZER.keys()) is not subset of set(LIST_AVAILABLE_OPTIMIZERS). " +
        f"Got {set(SS_KWARGS_NAME_OPTIMIZER.keys())} and {set(LIST_AVAILABLE_OPTIMIZERS)}.")

if not NAME_SCHEDULER in LIST_AVAILABLE_SCHEDULERS:
    raise ValueError(
        f"SCHEDULER={NAME_SCHEDULER} not in LIST_AVAILABLE_SCHEDULERS.")
if not set(KWARGS_NAME_SCHEDULER.keys()) == set(LIST_AVAILABLE_SCHEDULERS):
    raise ValueError(
        f"set(KWARGS_NAME_SCHEDULER.keys()) != set(LIST_AVAILABLE_SCHEDULERS). " +
        f"Got {set(KWARGS_NAME_SCHEDULER.keys())} and {set(LIST_AVAILABLE_SCHEDULERS)}.")
if not set(SS_KWARGS_NAME_SCHEDULER.keys()).issubset(set(LIST_AVAILABLE_SCHEDULERS)):
    raise ValueError(
        f"set(SS_KWARGS_NAME_SCHEDULER.keys()) is not subset of set(LIST_AVAILABLE_SCHEDULERS). " +
        f"Got {set(SS_KWARGS_NAME_SCHEDULER.keys())} and {set(LIST_AVAILABLE_SCHEDULERS)}.")

if not NAME_SAMPLER in LIST_AVAILABLE_SAMPLERS:
    raise ValueError(
        f"SAMPLER={NAME_SAMPLER} not in LIST_AVAILABLE_SAMPLERS.")
if not set(KWARGS_NAME_SAMPLER.keys()) == set(LIST_AVAILABLE_SAMPLERS):
    raise ValueError(
        f"set(KWARGS_NAME_SAMPLER.keys()) != set(LIST_AVAILABLE_SAMPLERS). " +
        f"Got {set(KWARGS_NAME_SAMPLER.keys())} and {set(LIST_AVAILABLE_SAMPLERS)}.")
if not set(SS_KWARGS_NAME_SAMPLER.keys()).issubset(set(LIST_AVAILABLE_SAMPLERS)):
    raise ValueError(
        f"set(SS_KWARGS_NAME_SAMPLER.keys()) is not subset of set(LIST_AVAILABLE_SAMPLERS). " +
        f"Got {set(SS_KWARGS_NAME_SAMPLER.keys())} and {set(LIST_AVAILABLE_SAMPLERS)}.")

if not NAME_MODEL in LIST_AVAILABLE_MODELS:
    raise ValueError(f"MODEL={NAME_MODEL} not in LIST_AVAILABLE_MODELS.")
if not set(KWARGS_NAME_MODEL.keys()) == set(LIST_AVAILABLE_MODELS):
    raise ValueError(
        f"set(KWARGS_NAME_MODEL.keys()) != set(LIST_AVAILABLE_MODELS). " +
        f"Got {set(KWARGS_NAME_MODEL.keys())} and {set(LIST_AVAILABLE_MODELS)}.")

for itr_loss in LIST_LOSSES:
    if not itr_loss in LIST_AVAILABLE_LOSSES:
        raise ValueError(f"LOSS={itr_loss} not in LIST_AVAILABLE_LOSSES.")
if not set(KWARGS_LOSSES.keys()) == set(LIST_AVAILABLE_LOSSES):
    raise ValueError(
        f"set(KWARGS_LOSSES.keys()) != set(LIST_AVAILABLE_LOSSES). " +
        f"Got {set(KWARGS_LOSSES.keys())} and {set(LIST_AVAILABLE_LOSSES)}.")
if not set(SS_KWARGS_LOSSES.keys()).issubset(set(LIST_AVAILABLE_LOSSES)):
    raise ValueError(
        f"set(SS_KWARGS_LOSSES.keys()) is not subset of set(LIST_AVAILABLE_LOSSES). " +
        f"Got {set(SS_KWARGS_LOSSES.keys())} and {set(LIST_AVAILABLE_LOSSES)}.")

if not NAME_OPTUNA_PRUNER in LIST_AVAILABLE_OPTUNA_PRUNERS:
    raise ValueError(
        f"PRUNER={NAME_OPTUNA_PRUNER} not in LIST_AVAILABLE_OPTUNA_PRUNERS.")
if not set(KWARGS_OPTUNA_PRUNER.keys()) == set(LIST_AVAILABLE_OPTUNA_PRUNERS):
    raise ValueError(
        f"set(KWARGS_OPTUNA_PRUNER.keys()) != set(LIST_AVAILABLE_OPTUNA_PRUNERS). " +
        f"Got {set(KWARGS_OPTUNA_PRUNER.keys())} and {set(LIST_AVAILABLE_OPTUNA_PRUNERS)}.")

if not NAME_OPTUNA_SAMPLER in LIST_AVAILABLE_OPTUNA_SAMPLERS:
    raise ValueError(
        f"SAMPLER={NAME_OPTUNA_SAMPLER} not in LIST_AVAILABLE_OPTUNA_SAMPLERS.")
if not set(KWARGS_OPTUNA_SAMPLER.keys()) == set(LIST_AVAILABLE_OPTUNA_SAMPLERS):
    raise ValueError(
        f"set(KWARGS_OPTUNA_SAMPLER.keys()) != set(LIST_AVAILABLE_OPTUNA_SAMPLERS). " +
        f"Got {set(KWARGS_OPTUNA_SAMPLER.keys())} and {set(LIST_AVAILABLE_OPTUNA_SAMPLERS)}.")

if not NAME_DATASET in LIST_AVAILABLE_DATASETS:
    raise ValueError(
        f"DATASET={NAME_DATASET} not in LIST_AVAILABLE_DATASETS.")
if not set(KWARGS_NAME_DATASETS.keys()) == set(LIST_AVAILABLE_DATASETS):
    raise ValueError(
        f"set(KWARGS_NAME_DATASETS.keys()) != set(LIST_AVAILABLE_DATASETS). " +
        f"Got {set(KWARGS_NAME_DATASETS.keys())} and {set(LIST_AVAILABLE_DATASETS)}.")

assert set(CAT_ACTIVATION).issubset(set(LIST_AVAILABLE_ACTIVATIONS))
assert "num_classes" in KWARGS_NAME_DATASETS[NAME_DATASET].keys()
assert "task" in KWARGS_NAME_DATASETS[NAME_DATASET].keys()

if NAME_DATASET == "RCDIntegral1V2":
    if not NAME_BASIS_FUNCTION in ["fourier", "legendre", "fourier_no_w", "legendre_no_w"]:
        raise ValueError(
            f"name_basis_function {NAME_BASIS_FUNCTION} is not allowed when NAME_DATASET is RCDIntegral1V2.")

if EXP_PHASE == "stat" and not os.path.exists(PATH_DBFILE):
    raise ValueError(
        f"[EXP_PHASE: stat] DB file {PATH_DBFILE} does not exist." + "Wrong NAME_SUBPROJECT or...?")

# Pruner check
if EXP_PHASE == "stat":
    logger.info(
        f"EXP_PHASE='stat' is detected, and\nNAME_OPTUNA_PRUNER is set from {NAME_OPTUNA_PRUNER} to NopPruner.")
    NAME_OPTUNA_PRUNER = "NopPruner"
if EXP_PHASE == "tuning" and NAME_OPTUNA_PRUNER == "NopPruner":
    warnings.warn("Warning: No pruning applied while EXP_PHASE='tuning'.")
    # raise ValueError("No pruning applied while EXP_PHASE='tuning'.")
    logger.info("Set NAME_OPTUNA_PRUNER from 'NopPruner' to 'MedianPruner'.")
    NAME_OPTUNA_PRUNER = "MedianPruner"

# Console output check
# if EXP_PHASE == "tuning":
#     FLAG_OUTPUT_CONSOLE = False
#     print("EXP_PHASE='tuning' detected. FLAG_OUTPUT_CONSOLE is set False.")

# Make directories if not exist
os.makedirs(ROOTDIR_TBLOGS, exist_ok=True)
os.makedirs(ROOTDIR_TXLOGS, exist_ok=True)
os.makedirs(ROOTDIR_CKPTS, exist_ok=True)
os.makedirs(ROOTDIR_CONFIGS, exist_ok=True)
os.makedirs(DIR_DBFILE, exist_ok=True)

# Directory check
ROOTDIR_PTMODEL = "/home/" + USER_NAME + "/.cache/torch/hub"
if not ROOTDIR_PTMODEL == os.path.expanduser("~") + "/.cache/torch/hub":
    raise ValueError(
        f"ROOTDIR_PYMODEL={ROOTDIR_PTMODEL} and " +
        f"os.path.expanduser('~')={os.path.expanduser('~')} do not match.\n" +
        "Pytorch model weights are most probably downloaded under os.path.expanduser('~') by default.\n" +
        "If it's OK, comment out this raise-error sentence.\n" +
        "If you are to change the download directory, modify class ModelController.")


""" 9. Make Config Dictionary """
config = {
    "NAME_PROJECT": NAME_PROJECT,
    "NAME_SUBPROJECT": NAME_SUBPROJECT,
    "COMMENT": COMMENT,
    "EXP_PHASE": EXP_PHASE,
    "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
    "NUM_TRIALS": NUM_TRIALS,
    "NUM_ITERATIONS": NUM_ITERATIONS,
    "VALIDATION_INTERVAL": VALIDATION_INTERVAL,
    "LIST_LOSSES": LIST_LOSSES,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "KWARGS_LOSSES": KWARGS_LOSSES,
    "NAME_DATASET": NAME_DATASET,
    "FLAG_SHUFFLE": FLAG_SHUFFLE,
    "BATCH_SIZE": BATCH_SIZE,
    "NAME_SAMPLER": NAME_SAMPLER,
    "NAME_OPTIMIZER": NAME_OPTIMIZER,
    "NAME_SCHEDULER": NAME_SCHEDULER,
    "NAME_MODEL": NAME_MODEL,
    "KWARGS_NAME_OPTIMIZER": KWARGS_NAME_OPTIMIZER,
    "KWARGS_NAME_SCHEDULER": KWARGS_NAME_SCHEDULER,
    "NUM_THREADS_PER_PROCESS": NUM_THREADS_PER_PROCESS,
    "FLAG_FIX_SEED": FLAG_FIX_SEED,
    "SEED": SEED,
    "FLAG_PIN_MEMORY": FLAG_PIN_MEMORY,
    "NUM_WORKERS": NUM_WORKERS,
    "FLAG_INIT_LAST_LINEAR": FLAG_INIT_LAST_LINEAR,
    "FLAG_PRETRAINED": FLAG_PRETRAINED,
    "LOG_INTERVAL": LOG_INTERVAL,
    "TBLOG_INTERVAL": TBLOG_INTERVAL,
    "SS_CAT_BATCH_SIZE": SS_CAT_BATCH_SIZE,
    "SS_CAT_NAME_OPTIMIZER": SS_CAT_NAME_OPTIMIZER,
    "SS_CAT_NAME_SCHEDULER": SS_CAT_NAME_SCHEDULER,
    "SS_KWARGS_NAME_OPTIMIZER": SS_KWARGS_NAME_OPTIMIZER,
    "SS_KWARGS_NAME_SCHEDULER": SS_KWARGS_NAME_SCHEDULER,
    "SS_KWARGS_NAME_MODEL": SS_KWARGS_NAME_MODEL,
    "SS_KWARGS_NAME_SAMPLER": SS_KWARGS_NAME_SAMPLER,
    "SS_KWARGS_LOSSES": SS_KWARGS_LOSSES,
    "NAME_OPTUNA_SAMPLER": NAME_OPTUNA_SAMPLER,
    "NAME_OPTUNA_PRUNER": NAME_OPTUNA_PRUNER,
    "KWARGS_OPTUNA_SAMPLER": KWARGS_OPTUNA_SAMPLER,
    "KWARGS_OPTUNA_PRUNER": KWARGS_OPTUNA_PRUNER,
    "USER_NAME": USER_NAME,
    "ROOTDIR_DATA": ROOTDIR_DATA,
    "ROOTDIR_PROJECT": ROOTDIR_PROJECT,
    "ROOTDIR_TBLOGS": ROOTDIR_TBLOGS,
    "ROOTDIR_TXLOGS": ROOTDIR_TXLOGS,
    "ROOTDIR_CKPTS": ROOTDIR_CKPTS,
    "ROOTDIR_CONFIGS": ROOTDIR_CONFIGS,
    "ROOTDIR_CONFIGS": ROOTDIR_CONFIGS,
    "ROOTDIR_PTDATASETS": ROOTDIR_PTDATASETS,
    "FLAG_MULTIGPU": FLAG_MULTIGPU,
    "KWARGS_NAME_MODEL": KWARGS_NAME_MODEL,
    "KWARGS_NAME_DATASETS": KWARGS_NAME_DATASETS,
    "KWARGS_NAME_SAMPLER": KWARGS_NAME_SAMPLER,
    "FLAG_OUTPUT_CONSOLE": FLAG_OUTPUT_CONSOLE,
    "PATH_CONFIG": PATH_CONFIG,
    "ROOTDIR_DBFILES": ROOTDIR_DBFILES,
    "DIR_DBFILE": DIR_DBFILE,
    "PATH_DBFILE": PATH_DBFILE,
    "TOP_K": TOP_K,
    "FLAG_SAVE_CKPT_TRYTUNING": FLAG_SAVE_CKPT_TRYTUNING,
    "NAME_LOSS_REWEIGHT": NAME_LOSS_REWEIGHT,
    "SS_CAT_NAME_LOSS_REWEIGHT": SS_CAT_NAME_LOSS_REWEIGHT,
    # "PINN_SAMPLER": PINN_SAMPLER,
    "BATCH_SIZE_RATIO": BATCH_SIZE_RATIO,
    # "SS_CAT_PINN_SAMPLER": SS_CAT_PINN_SAMPLER,
    "SS_CAT_BATCH_SIZE_RATIO": SS_CAT_BATCH_SIZE_RATIO,
    "INDEX_TRIAL_STAT": INDEX_TRIAL_STAT,
    "DEBUG_MODE": DEBUG_MODE,
    "DEFAULT_DTYPE": DEFAULT_DTYPE,
    "DEGREE": DEGREE,
    "START_PRUNING": START_PRUNING,
}  # Do not forget to check utils.misc.load_config_stat after you add smth here.
