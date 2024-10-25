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

LIST_AVAILABLE_OPTIMIZERS = [
    "SGD",
    "SGDW",
    "SGDP",
    "NAdam",
    "RMSprop",
    "Yogi",
    "PID",
    "Lamb",
    "QHM",
    "Adam",
    "AdamW",
    "RAdam",
    "AdaBelief",
    "Rprop",
    "QHAdam",
    "Lion",
    "AdamP",
    "AdaMod",
    "DiffGrad",
    "SWATS",
    "Adai",
    "AdaiV2",
    # "Apollo",  # OOM
    "RangerVA",
    "RangerQH",
    "Ranger",
    # "MADGRAD",# buggy: RuntimeError
    "ASGD",
    "Adadelta",
    # "Adagrad", # buggy: RuntimeError
    "Adamax",
    "A2GradExp",
    "A2GradInc",
    "A2GradUni",
    "AccSGD",
    "AdaBound",
    "NovoGrad",
    "AggMo",
    "Sophia",
]

# Scheduler details:
# https://pytorch.org/docs/stable/optim.html
# https://timm.fast.ai/SGDR#t_mul
LIST_AVAILABLE_SCHEDULERS = [
    "Constant",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "PolynomialLR",
    "CosineAnnealingLR",
    "ReduceLROnPlateau",
    "CyclicLR",
    "CosineAnnealingWarmRestarts",
    "CosineAnnealingWarmRestartsLinearStart",
    "CosineAnnealingWarmRestartsLinearStartWeightDecay",
]

# Model details: https://pytorch.org/vision/stable/models.html
# "ViT_H16" (too large: only batch size=1 is possible): Use _L!!
# "RegNetY_128GF" (too large: only batch size=1 is possible)
LIST_AVAILABLE_MODELS = [
    "ConvNextTiny",
    "ConvNextSmall",
    "ConvNextBase",
    "ConvNextLarge",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "DenseNet201",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
    "EfficientNetB7",
    "EfficientNetV2S",
    "EfficientNetV2M",
    "EfficientNetV2L",
    "MaxViT",
    "MobileNetV3Small",
    "MobileNetV3Large",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "WideResNet50_2",
    "WideResNet101_2",
    "ResNeXt50_32X4D",
    "ResNeXt101_32X8D",
    "RegNetY_400MF",
    "RegNetY_800MF",
    "RegNetY_1_6GF",
    "RegNetY_3_2GF",
    "RegNetY_8GF",
    "RegNetY_16GF",
    "RegNetY_32GF",
    "RegNetY_128GF",
    "ShuffleNetV2X0_5",
    "ShuffleNetV2X1_0",
    "ShuffleNetV2X1_5",
    "ShuffleNetV2X2_0",
    "ViT_B16",
    "ViT_L16",
    "ViT_H16",
    "SwinV2_T",
    "SwinV2_S",
    "SwinV2_B",
    "VGG11_BN",
    "VGG13_BN",
    "VGG16_BN",
    "VGG19_BN",
    "S3D",
    "MC3_18",
    "R3D_18",
    "R2Plus1D_18",
    "MViT_V2_S",
    "MViT_V1_B",
    "Swin3D_T",
    "Swin3D_S",
    "Swin3D_B",
    "RNN",
    "LSTM",
    "TimesNet",
    "MSSIREN",  # modified model
    "SimplePINN",  # modified model
    "SWNN",  # original model
]

LIST_AVAILABLE_LOSSES = [
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "CrossEntropyLoss",
    "BCELoss",
    "LpLoss",
]

LIST_AVAILABLE_DATASETS = [
    "MNIST",
    "FashionMNIST",
    "CIFAR10",
    "CIFAR100",
    # "CIFAR10LT",
    # "CIFAR100LT",
    # "iNaturalist2017",
    # "iNaturalist2018",
    "CIFAR10Flat",
    "CIFAR100Flat",
    "SpeechCommands",
    # "YesNo",
    "RCPDPDE1",
    "RCPDODE1",
    "RCPDHarmonicOscillator",
    "RCPDKirchhoff",
    "RCDIntegral1",  # large GPU memory
    "RCDIntegral1V2",  # super-light
    "RCDIntegral2",
    "RCDArcLength",
    "RCDThomasFermi",
    "RCDGaussianWhiteNoise",
    "RCDFTransportEquation",
    "RCDBurgersHopfEquation",
]

LIST_AVAILABLE_SAMPLERS = [
    "RandomSampler",
    # "OverSampler",
]

# Sampler details: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
LIST_AVAILABLE_OPTUNA_SAMPLERS = [
    "TPESampler",  # default
    "CmaEsSampler",
    "RandomSampler",
]

# Pruner details: https://optuna.readthedocs.io/en/stable/reference/pruners.html
LIST_AVAILABLE_OPTUNA_PRUNERS = [
    "SuccessiveHalvingPruner",
    "HyperbandPruner",
    "MedianPruner",  # default
    "ThresholdPruner",
    "NopPruner",
    "PercentilePruner",
]

# Activation functions
LIST_AVAILABLE_ACTIVATIONS = [
    "Linear",
    "Gaussian",
    "SuperGaussian",
    "Quadratic",
    "MultiQuadratic",
    "Laplacian",
    "ExpSin",
    "Sin",
    "SinSquared",
    "Sinc",
    "SinReLU",
    "SinSiLU",
    "SinGaussian",
    "ELU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardtanh",
    "Hardswish",
    "LeakyReLU",
    "LogSigmoid",
    "PReLU",
    "ReLU",
    "ReLU6",
    "RReLU",
    "SELU",
    "CELU",
    "GELU",
    "Sigmoid",
    "SiLU",  # =swish
    "Mish",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "Tanhshrink",
    "B2Blog",
    "B2Bcbrt",
    "B2Bexp",
    "Tanhplus",
    "DullReLU",
    "SinB2BsqrtV2",
    "B2Bsqrt",
    "B2BsqrtV2",
    "Ricker",
    "ABU",
]
