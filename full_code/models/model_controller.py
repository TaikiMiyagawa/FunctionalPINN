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
Model details: https://pytorch.org/vision/stable/models.html
Model wrappers enables models to output hidden features,
typically called hidden (bottleneck) features just before the last Linear.
"""
import os

import torch
import torchvision
from filelock import FileLock
from torch import Tensor, nn
from utils.log_config import get_my_logger

from .ms_siren import MSSIREN
from .pinns import SimplePINN
from .rnns import LSTMModel, RNNModel
from .swnn import SteinWeierstrassNeuralNetwork

logger = get_my_logger(__name__)


class NeuralNetwork1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class KwargsTimesNet():
    def __init__(self, kwargs: dict) -> None:
        self.seq_len = kwargs["duration"]
        self.enc_in = kwargs["dim_input"]
        self.d_model = kwargs["d_model"]
        self.d_ff = kwargs["d_ff"]
        self.top_k = kwargs["top_k"]
        self.e_layers = kwargs["e_layers"]
        self.num_kernels = kwargs["num_kernels"]
        self.dropout = kwargs["dropout"]
        self.name_block = kwargs["name_block"]
        self.task_name = 'classification'  # fix for classification
        self.label_len = 0  # fix for classification
        self.pred_len = 0  # fix for classification
        self.embed = "fixed"  # fix for classification
        self.freq = "h"  # fix for classification
        self.c_out = 0  # Never used for classification


class ModelWrapperV1(torch.nn.Module):
    """
    The wrapped model becomes able to outputs
    - x1: hidden feature before the last Liner
    - x2: logits.
    # Remark
    - V1: model.children must be, e.g.,
      ...
            )
          )
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (permute): Permute()
          (avgpool): AdaptiveAvgPool2d(output_size=1)
          (flatten): Flatten(start_dim=1, end_dim=-1)
          (head): Linear(in_features=768, out_features=10, bias=True) # <= head
        )
      )>.

    # Usage
    weights = Swin_V2_T_Weights.IMAGENET1K_V1
    model = swin_v2_t(weights=weights)
    model = SwinWrapper(model, num_classes)
    preprocess = weights.transforms()
    model.to(device)

    # Can be used for
    - SwinV2
    """

    def __init__(self, model, num_classes):
        super(ModelWrapperV1, self).__init__()
        self.num_classes = num_classes
        self.dim_feat = model.head.in_features

        model.head = torch.nn.Sequential()
        self.FeatureExtractor = model
        self.fc_last = torch.nn.Linear(self.dim_feat, num_classes)

    def forward(self, x):
        feat = self.FeatureExtractor(x)
        output = self.fc_last(feat)
        return output, feat


class ModelWrapperV2(torch.nn.Module):
    """
    The wrapped model becomes able to outputs
    - x1: hidden feature before the last Liner
    - x2: logits.
    # Remark
    - V2: model.children must be, e.g.,
      ...
        (avgpool): AdaptiveAvgPool2d(output_size=1)
        (classifier): Sequential(
            (0): LayerNorm2d((1536,), eps=1e-06, elementwise_affine=True)
            (1): Flatten(start_dim=1, end_dim=-1)
            (2): Linear(in_features=1536, out_features=1000, bias=True)
        )
        )>.

      or

        (avgpool): AdaptiveAvgPool2d(output_size=1)
        (classifier): Sequential(
            (0): Dropout(p=0.2, inplace=True)
            (1): Linear(in_features=1280, out_features=1000, bias=True)
        )
        )>
      etc.
    # Can be used for
    - ConvNext
    - EfficientNet
    - EfficientNetV2
    - MaxViT
    - MobileNetV3
    - VGG
    """

    def __init__(self, model, num_classes):
        super(ModelWrapperV2, self).__init__()
        self.num_classes = num_classes
        self.dim_feat = model.classifier[-1].in_features

        model.classifier[-1] = torch.nn.Sequential()
        self.FeatureExtractor = model
        self.fc_last = torch.nn.Linear(self.dim_feat, num_classes)

    def forward(self, x):
        feat = self.FeatureExtractor(x)
        output = self.fc_last(feat)
        return output, feat


class ModelWrapperV3(torch.nn.Module):
    """
    The wrapped model becomes able to outputs
    - x1: hidden feature before the last Liner
    - x2: logits.
    # Remark
    - V3: model.children must be, e.g.,
      ...
            )
            )
            (norm5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (classifier): Linear(in_features=1024, out_features=1000, bias=True)
        )>

    # Can be used for
    - DenseNet
    """

    def __init__(self, model, num_classes):
        super(ModelWrapperV3, self).__init__()
        self.num_classes = num_classes
        self.dim_feat = model.classifier.in_features

        model.classifier = torch.nn.Sequential()
        self.FeatureExtractor = model
        self.fc_last = torch.nn.Linear(self.dim_feat, num_classes)

    def forward(self, x):
        feat = self.FeatureExtractor(x)
        output = self.fc_last(feat)
        return output, feat


class ModelWrapperV4(torch.nn.Module):
    """
    The wrapped model becomes able to outputs
    - x1: hidden feature before the last Liner
    - x2: logits.
    # Remark
    - V4: model.children must be, e.g.,
      ...
        )
        (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        (fc): Linear(in_features=512, out_features=1000, bias=True)
        )>

    # Usage
    weights = Swin_V2_T_Weights.IMAGENET1K_V1
    model = swin_v2_t(weights=weights)
    model = SwinWrapper(model, num_classes)
    preprocess = weights.transforms()
    model.to(device)

    # Can be used for
    - ResNet
    - WideResNet
    - ResNeXt
    - RegNetY
    - ShuffleNetV2
    - MC3_18
    - R3D_18
    - R2Plus1D_18
    """

    def __init__(self, model, num_classes):
        super(ModelWrapperV4, self).__init__()
        self.num_classes = num_classes
        self.dim_feat = model.fc.in_features

        model.fc = torch.nn.Sequential()
        self.FeatureExtractor = model
        self.fc_last = torch.nn.Linear(self.dim_feat, num_classes)

    def forward(self, x):
        feat = self.FeatureExtractor(x)
        output = self.fc_last(feat)
        return output, feat


class ModelWrapperV5(torch.nn.Module):
    """
    The wrapped model becomes able to outputs
    - x1: hidden feature before the last Liner
    - x2: logits.
    # Remark
    - V5: model.children must be, e.g.,
      ...
            (ln): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        )
        (heads): Sequential(
            (head): Linear(in_features=768, out_features=1000, bias=True)
        )
        )>

    # Can be used for
    - ViT
    """

    def __init__(self, model, num_classes):
        super(ModelWrapperV5, self).__init__()
        self.num_classes = num_classes
        self.dim_feat = model.heads[-1].in_features

        model.heads[-1] = torch.nn.Sequential()
        self.FeatureExtractor = model
        self.fc_last = torch.nn.Linear(self.dim_feat, num_classes)

    def forward(self, x):
        feat = self.FeatureExtractor(x)
        output = self.fc_last(feat)
        return output, feat


class ModelWrapperV6(torch.nn.Module):
    """
    The wrapped model becomes able to outputs
    - x1: hidden feature before the last Liner
    - x2: logits.
    # Remark
    - V6:  model.children must be, e.g.,
      ...
        (avgpool): AvgPool3d(kernel_size=(2, 7, 7), stride=1, padding=0)
        (classifier): Sequential(
            (0): Dropout(p=0.2, inplace=False)
            (1): Conv3d(1024, 400, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        )
        )>
    # Can be used for
    - S3D
    """

    def __init__(self, model, num_classes):
        super(ModelWrapperV6, self).__init__()
        self.num_classes = num_classes
        self.dim_feat = model.classifier[-1].out_channels

        self.FeatureExtractor = model
        self.fc_last = torch.nn.Linear(self.dim_feat, num_classes)

    def forward(self, x):
        feat = self.FeatureExtractor(x)
        output = self.fc_last(feat)
        return output, feat


class ModelWrapperV7(torch.nn.Module):
    """
    The wrapped model becomes able to outputs
    - x1: hidden feature before the last Liner
    - x2: logits.
    # Remark
    - V7: model.children must be, e.g.,
      ...
        )
        (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (head): Sequential(
            (0): Dropout(p=0.5, inplace=True)
            (1): Linear(in_features=768, out_features=400, bias=True)
        )
        )>

    # Can be used for
    - MViT
    """

    def __init__(self, model, num_classes):
        super(ModelWrapperV7, self).__init__()
        self.num_classes = num_classes
        self.dim_feat = model.head[-1].in_features

        model.head[-1] = torch.nn.Sequential()
        self.FeatureExtractor = model
        self.fc_last = torch.nn.Linear(self.dim_feat, num_classes)

    def forward(self, x):
        feat = self.FeatureExtractor(x)
        output = self.fc_last(feat)
        return output, feat


class ModelController():
    """
    # Remarks
    - Model details: https://pytorch.org/vision/stable/models.html
    - Pth will be downleaded under /home/USER/.cache/torch/hub by default.

    """

    def __init__(self, name_model: str, num_classes: int, device: str,
                 flag_pretrained: bool, flag_init_last_linear: bool = True,
                 kwargs_dataset: dict = {},
                 **kwargs_model) -> None:
        """
        # Args
        - name_model: Name of model.
        - num_classes: Number of classes.
        - device: "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        - flag_pretrained: Random initialization or pretrained model.
        - flag_init_last_linear: Used only when flag_pretrained=True.
          If True, the last linear layer will be (added or ) initialized and class size
          will be adjusted to num_classes.
        - kwargs_dataset: KWARGS_NAME_DATASET[NAME_DATASET]. Used for, e.g., TimesNet, whose structure
          depends on dataset format,
          i.e., x.shape = [B, C, T] (CIFAR10Flat) or [2*B (with mask), C, T] (SpeechCommands).
        """
        # Initialization
        self.name_model = name_model
        self.num_classes = num_classes
        self.device = device
        self.flag_pretrained = flag_pretrained
        self.flag_init_last_linear = flag_init_last_linear
        self.kwargs_dataset = kwargs_dataset
        self.kwargs_model = kwargs_model

        # Define model
        # torch.distributed
        with FileLock(os.path.expanduser('~')+"/.cache/torch/hub/.ddp_lock_models"):
            if name_model == 'ConvNextTiny':
                model, self.preprocess, self.weights = self._get_ConvNextTiny()
            elif name_model == 'ConvNextSmall':
                model, self.preprocess, self.weights = self._get_ConvNextSmall()
            elif name_model == 'ConvNextBase':
                model, self.preprocess, self.weights = self._get_ConvNextBase()
            elif name_model == 'ConvNextLarge':
                model, self.preprocess, self.weights = self._get_ConvNextLarge()
            elif name_model == 'DenseNet121':
                model, self.preprocess, self.weights = self._get_DenseNet121()
            elif name_model == 'DenseNet161':
                model, self.preprocess, self.weights = self._get_DenseNet161()
            elif name_model == 'DenseNet169':
                model, self.preprocess, self.weights = self._get_DenseNet169()
            elif name_model == 'DenseNet201':
                model, self.preprocess, self.weights = self._get_DenseNet201()
            elif name_model == 'EfficientNetB0':
                model, self.preprocess, self.weights = self._get_EfficientNetB0()
            elif name_model == 'EfficientNetB1':
                model, self.preprocess, self.weights = self._get_EfficientNetB1()
            elif name_model == 'EfficientNetB2':
                model, self.preprocess, self.weights = self._get_EfficientNetB2()
            elif name_model == 'EfficientNetB3':
                model, self.preprocess, self.weights = self._get_EfficientNetB3()
            elif name_model == 'EfficientNetB4':
                model, self.preprocess, self.weights = self._get_EfficientNetB4()
            elif name_model == 'EfficientNetB5':
                model, self.preprocess, self.weights = self._get_EfficientNetB5()
            elif name_model == 'EfficientNetB6':
                model, self.preprocess, self.weights = self._get_EfficientNetB6()
            elif name_model == 'EfficientNetB7':
                model, self.preprocess, self.weights = self._get_EfficientNetB7()
            elif name_model == 'EfficientNetV2S':
                model, self.preprocess, self.weights = self._get_EfficientNetV2S()
            elif name_model == 'EfficientNetV2M':
                model, self.preprocess, self.weights = self._get_EfficientNetV2M()
            elif name_model == 'EfficientNetV2L':
                model, self.preprocess, self.weights = self._get_EfficientNetV2L()
            elif name_model == 'MaxViT':
                model, self.preprocess, self.weights = self._get_MaxViT()
            elif name_model == 'MobileNetV3Small':
                model, self.preprocess, self.weights = self._get_MobileNetV3Small()
            elif name_model == 'MobileNetV3Large':
                model, self.preprocess, self.weights = self._get_MobileNetV3Large()
            elif name_model == 'ResNet18':
                model, self.preprocess, self.weights = self._get_ResNet18()
            elif name_model == 'ResNet34':
                model, self.preprocess, self.weights = self._get_ResNet34()
            elif name_model == 'ResNet50':
                model, self.preprocess, self.weights = self._get_ResNet50()
            elif name_model == 'ResNet101':
                model, self.preprocess, self.weights = self._get_ResNet101()
            elif name_model == 'ResNet152':
                model, self.preprocess, self.weights = self._get_ResNet152()
            elif name_model == 'WideResNet50_2':
                model, self.preprocess, self.weights = self._get_WideResNet50_2()
            elif name_model == 'WideResNet101_2':
                model, self.preprocess, self.weights = self._get_WideResNet101_2()
            elif name_model == 'ResNeXt50_32X4D':
                model, self.preprocess, self.weights = self._get_ResNeXt50_32X4D()
            elif name_model == 'ResNeXt101_32X8D':
                model, self.preprocess, self.weights = self._get_ResNeXt101_32X8D()
            elif name_model == 'RegNetY_400MF':
                model, self.preprocess, self.weights = self._get_RegNetY_400MF()
            elif name_model == 'RegNetY_800MF':
                model, self.preprocess, self.weights = self._get_RegNetY_800MF()
            elif name_model == 'RegNetY_3_2GF':
                model, self.preprocess, self.weights = self._get_RegNetY_3_2GF()
            elif name_model == 'RegNetY_1_6GF':
                model, self.preprocess, self.weights = self._get_RegNetY_1_6GF()
            elif name_model == 'RegNetY_8GF':
                model, self.preprocess, self.weights = self._get_RegNetY_8GF()
            elif name_model == 'RegNetY_16GF':
                model, self.preprocess, self.weights = self._get_RegNetY_16GF()
            elif name_model == 'RegNetY_32GF':
                model, self.preprocess, self.weights = self._get_RegNetY_32GF()
            elif name_model == 'RegNetY_128GF':
                model, self.preprocess, self.weights = self._get_RegNetY_128GF()
            elif name_model == 'ShuffleNetV2X0_5':
                model, self.preprocess, self.weights = self._get_ShuffleNetV2X0_5()
            elif name_model == 'ShuffleNetV2X1_0':
                model, self.preprocess, self.weights = self._get_ShuffleNetV2X1_0()
            elif name_model == 'ShuffleNetV2X1_5':
                model, self.preprocess, self.weights = self._get_ShuffleNetV2X1_5()
            elif name_model == 'ShuffleNetV2X2_0':
                model, self.preprocess, self.weights = self._get_ShuffleNetV2X2_0()
            elif name_model == 'ViT_B16':
                model, self.preprocess, self.weights = self._get_ViT_B16()
            elif name_model == 'ViT_L16':
                model, self.preprocess, self.weights = self._get_ViT_L16()
            elif name_model == 'ViT_H16':
                model, self.preprocess, self.weights = self._get_ViT_H16()
            elif name_model == 'SwinV2_T':
                model, self.preprocess, self.weights = self._get_SwinV2_T()
            elif name_model == 'SwinV2_S':
                model, self.preprocess, self.weights = self._get_SwinV2_S()
            elif name_model == 'SwinV2_B':
                model, self.preprocess, self.weights = self._get_SwinV2_B()
            elif name_model == 'VGG11_BN':
                model, self.preprocess, self.weights = self._get_VGG11_BN()
            elif name_model == 'VGG13_BN':
                model, self.preprocess, self.weights = self._get_VGG13_BN()
            elif name_model == 'VGG16_BN':
                model, self.preprocess, self.weights = self._get_VGG16_BN()
            elif name_model == 'VGG19_BN':
                model, self.preprocess, self.weights = self._get_VGG19_BN()
            elif name_model == 'S3D':
                model, self.preprocess, self.weights = self._get_S3D()
            elif name_model == 'MC3_18':
                model, self.preprocess, self.weights = self._get_MC3_18()
            elif name_model == 'R3D_18':
                model, self.preprocess, self.weights = self._get_R3D_18()
            elif name_model == 'R2Plus1D_18':
                model, self.preprocess, self.weights = self._get_R2Plus1D_18()
            elif name_model == 'MViT_V2_S':
                model, self.preprocess, self.weights = self._get_MViT_V2_S()
            elif name_model == 'MViT_V1_B':
                model, self.preprocess, self.weights = self._get_MViT_V1_B()
            elif name_model == 'Swin3D_T':
                model, self.preprocess, self.weights = self._get_Swin3D_T()
            elif name_model == 'Swin3D_S':
                model, self.preprocess, self.weights = self._get_Swin3D_S()
            elif name_model == 'Swin3D_B':
                model, self.preprocess, self.weights = self._get_Swin3D_B()
            elif name_model == 'SimplePINN':
                model, self.preprocess, self.weights = self._get_SimplePINN()
            elif name_model == 'MSSIREN':
                model, self.preprocess, self.weights = self._get_MSSIREN()
            elif name_model == 'SWNN':
                model, self.preprocess, self.weights = self._get_SWNN()
            elif name_model == 'TimesNet':
                model, self.preprocess, self.weights = self._get_TimesNet()
            elif name_model == 'RNN':
                model, self.preprocess, self.weights = self._get_RNN()
            elif name_model == 'LSTM':
                model, self.preprocess, self.weights = self._get_LSTM()
            else:
                raise ValueError(f"name_model={name_model} is unavailable.")

        # Initialize and reshape or add the last linear layer if necessary
        if not flag_init_last_linear:
            self.model = model
        else:
            if name_model in ["SwinV2_T", "SwinV2_S", "SwinV2_B", "Swin3B_T", "Swin3B_S", "Swin3B_B",]:
                self.model = ModelWrapperV1(model, self.num_classes)
            elif name_model in ["VGG11_BN", "VGG13_BN", "VGG16_BN", "VGG19_BN",
                                "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3",
                                "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7",
                                "EfficientNetV2S", "EfficientNetV2M", "EfficientNetV2L", "MaxViT",
                                "MobileNetV3Small", "MobileNetV3Large", "ConvNextTiny", "ConvNextSmall",
                                "ConvNextBase", "ConvNextLarge"]:
                self.model = ModelWrapperV2(model, self.num_classes)
            elif name_model in ["DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201",]:
                self.model = ModelWrapperV3(model, self.num_classes)
            elif name_model in ["MC3_18", "R3D_18", "R2Plus1D_18", "ResNet18", "ResNet34",
                                "ResNet50", "ResNet101", "ResNet152", "WideResNet50_2",
                                "WideResNet101_2", "ResNeXt50_32X4D", "ResNeXt101_32X8D",
                                "RegNetY_400MF", "RegNetY_800MF", "RegNetY_3_2GF", "RegNetY_1_6GF",
                                "RegNetY_8GF", "RegNetY_16GF", "RegNetY_32GF", "RegNetY_128GF",
                                "ShuffleNetV2X0_5", "ShuffleNetV2X1_0", "ShuffleNetV2X1_5", "ShuffleNetV2X2_0",]:
                self.model = ModelWrapperV4(model, self.num_classes)
            elif name_model in ["ViT_B16", "ViT_L16", "ViT_H16",]:
                self.model = ModelWrapperV5(model, self.num_classes)
            elif name_model in ["S3D",]:
                self.model = ModelWrapperV6(model, self.num_classes)
            elif name_model in ["MViT_V2_S", "MViT_V1_B",]:
                self.model = ModelWrapperV7(model, self.num_classes)
            else:
                self.model = model
                logger.info(
                    f"flag_init_last_linear=True is ignored when name_model={name_model}.")

    def get_model(self):
        return self.model

    def get_preprocess(self):
        return self.preprocess

    def get_weights(self):
        return self.weights

    def _get_ConvNextTiny(self):
        if self.flag_pretrained:
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.convnext_tiny(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ConvNextSmall(self):
        if self.flag_pretrained:
            weights = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.convnext_small(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ConvNextBase(self):
        if self.flag_pretrained:
            weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.convnext_base(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ConvNextLarge(self):
        if self.flag_pretrained:
            weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.convnext_large(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_DenseNet121(self):
        if self.flag_pretrained:
            weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.densenet121(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_DenseNet161(self):
        if self.flag_pretrained:
            weights = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.densenet161(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_DenseNet169(self):
        if self.flag_pretrained:
            weights = torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.densenet169(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_DenseNet201(self):
        if self.flag_pretrained:
            weights = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.densenet201(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetB0(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.efficientnet_b0(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetB1(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.efficientnet_b1(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetB2(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.efficientnet_b2(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetB3(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.efficientnet_b3(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetB4(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.efficientnet_b4(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetB5(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.efficientnet_b5(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetB6(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.efficientnet_b6(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetB7(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.efficientnet_b7(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetV2S(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.efficientnet_v2_s(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetV2M(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.efficientnet_v2_m(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_EfficientNetV2L(self):
        if self.flag_pretrained:
            weights = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.efficientnet_v2_l(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_MaxViT(self):
        if self.flag_pretrained:
            weights = torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.maxvit_t(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_MobileNetV3Small(self):
        if self.flag_pretrained:
            weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_MobileNetV3Large(self):
        if self.flag_pretrained:
            weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.mobilenet_v3_large(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ResNet18(self):
        if self.flag_pretrained:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.resnet18(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ResNet34(self):
        if self.flag_pretrained:
            weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.resnet34(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ResNet50(self):
        if self.flag_pretrained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.resnet50(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ResNet101(self):
        if self.flag_pretrained:
            weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.resnet101(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ResNet152(self):
        if self.flag_pretrained:
            weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.resnet152(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_WideResNet50_2(self):
        if self.flag_pretrained:
            weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.wide_resnet50_2(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_WideResNet101_2(self):
        if self.flag_pretrained:
            weights = torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.wide_resnet101_2(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ResNeXt50_32X4D(self):
        if self.flag_pretrained:
            weights = torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.resnext50_32x4d(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ResNeXt101_32X8D(self):
        if self.flag_pretrained:
            weights = torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.resnext101_32x8d(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ResNeXt101_64X4D(self):
        if self.flag_pretrained:
            weights = torchvision.models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.resnext101_64x4d(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_RegNetY_400MF(self):
        if self.flag_pretrained:
            weights = torchvision.models.RegNet_Y_400MF_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.regnet_y_400mf(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_RegNetY_800MF(self):
        if self.flag_pretrained:
            weights = torchvision.models.RegNet_Y_800MF_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.regnet_y_800mf(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_RegNetY_3_2GF(self):
        if self.flag_pretrained:
            weights = torchvision.models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.regnet_y_3_2gf(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_RegNetY_1_6GF(self):
        if self.flag_pretrained:
            weights = torchvision.models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.regnet_y_1_6gf(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_RegNetY_8GF(self):
        if self.flag_pretrained:
            weights = torchvision.models.RegNet_Y_8GF_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = torchvision.models.regnet_y_8gf(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_RegNetY_16GF(self):
        if self.flag_pretrained:
            weights = torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
        else:
            weights = None
        model = torchvision.models.regnet_y_16gf(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_RegNetY_32GF(self):
        if self.flag_pretrained:
            weights = torchvision.models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1
        else:
            weights = None
        model = torchvision.models.regnet_y_32gf(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_RegNetY_128GF(self):
        if self.flag_pretrained:
            weights = torchvision.models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
        else:
            weights = None
        model = torchvision.models.regnet_y_128gf(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ShuffleNetV2X0_5(self):
        if self.flag_pretrained:
            weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.shufflenet_v2_x0_5(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ShuffleNetV2X1_0(self):
        if self.flag_pretrained:
            weights = torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.shufflenet_v2_x1_0(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ShuffleNetV2X1_5(self):
        if self.flag_pretrained:
            weights = torchvision.models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.shufflenet_v2_x1_5(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ShuffleNetV2X2_0(self):
        if self.flag_pretrained:
            weights = torchvision.models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.shufflenet_v2_x2_0(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ViT_B16(self):
        if self.flag_pretrained:
            weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        else:
            weights = None
        model = torchvision.models.vit_b_16(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ViT_L16(self):
        if self.flag_pretrained:
            weights = torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
        else:
            weights = None
        model = torchvision.models.vit_l_16(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_ViT_H16(self):
        if self.flag_pretrained:
            weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
        else:
            weights = None
        model = torchvision.models.vit_h_14(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_SwinV2_T(self):
        if self.flag_pretrained:
            weights = torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.swin_v2_t(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_SwinV2_S(self):
        if self.flag_pretrained:
            weights = torchvision.models.Swin_V2_S_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.swin_v2_s(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_SwinV2_B(self):
        if self.flag_pretrained:
            weights = torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.swin_v2_b(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_VGG11_BN(self):
        if self.flag_pretrained:
            weights = torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.vgg11_bn(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_VGG13_BN(self):
        if self.flag_pretrained:
            weights = torchvision.models.VGG13_BN_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.vgg13_bn(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_VGG16_BN(self):
        if self.flag_pretrained:
            weights = torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.vgg16_bn(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_VGG19_BN(self):
        if self.flag_pretrained:
            weights = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.vgg19_bn(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_S3D(self):
        if self.flag_pretrained:
            weights = torchvision.models.video.S3D_Weights.KINETICS400_V1
        else:
            weights = None
        model = torchvision.models.video.s3d(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_MC3_18(self):
        if self.flag_pretrained:
            weights = torchvision.models.video.MC3_18_Weights.KINETICS400_V1
        else:
            weights = None
        model = torchvision.models.video.mc3_18(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_R3D_18(self):
        if self.flag_pretrained:
            weights = torchvision.models.video.R3D_18_Weights.KINETICS400_V1
        else:
            weights = None
        model = torchvision.models.video.r3d_18(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_R2Plus1D_18(self):
        if self.flag_pretrained:
            weights = torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
        else:
            weights = None
        model = torchvision.models.video.r2plus1d_18(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_MViT_V2_S(self):
        if self.flag_pretrained:
            weights = torchvision.models.video.MViT_V2_S_Weights.KINETICS400_V1
        else:
            weights = None
        model = torchvision.models.video.mvit_v2_s(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_MViT_V1_B(self):
        if self.flag_pretrained:
            weights = torchvision.models.video.MViT_V1_B_Weights.KINETICS400_V1
        else:
            weights = None
        model = torchvision.models.video.mvit_v1_b(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_Swin3D_T(self):

        if self.flag_pretrained:
            weights = torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1
        else:
            weights = None
        model = torchvision.models.video.swin3d_t(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_Swin3D_S(self):

        if self.flag_pretrained:
            weights = torchvision.models.video.Swin3D_S_Weights.KINETICS400_V1
        else:
            weights = None
        model = torchvision.models.video.swin3d_s(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_Swin3D_B(self):
        if self.flag_pretrained:
            weights = torchvision.models.video.Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        else:
            weights = None
        model = torchvision.models.video.swin3d_b(weights=weights)
        preprocess = weights.transforms() if weights is not None else None
        return model, preprocess, weights

    def _get_SimplePINN(self):
        model = SimplePINN(
            dim_input=self.kwargs_model["dim_input"],
            dim_hidden=self.kwargs_model["dim_hidden"],
            dim_output=self.kwargs_model["dim_output"],
            num_layers=self.kwargs_model["num_layers"],
            name_activation=self.kwargs_model["name_activation"],
            kwargs_act=dict(),
            flag_positive_output=self.kwargs_model["flag_positive_output"],
            sizes=self.kwargs_model["sizes"])

        def preprocess(x): return torch.tensor(x, dtype=torch.float)
        return model, preprocess, None

    def _get_MSSIREN(self):
        model = MSSIREN(
            dim_input=self.kwargs_model["dim_input"],
            dim_hidden=self.kwargs_model["dim_hidden"],
            dim_output=self.kwargs_model["dim_output"],
            num_layers=self.kwargs_model["num_layers"],
            name_activation=self.kwargs_model["name_activation"],
            kwargs_act=dict(),
            scales=self.kwargs_model["scales"],
            flag_gate_unit=self.kwargs_model["flag_gate_unit"],
            flag_positive_output=self.kwargs_model["flag_positive_output"],
            sizes=self.kwargs_model["sizes"])

        def preprocess(x): return torch.tensor(x, dtype=torch.float)
        return model, preprocess, None

    def _get_SWNN(self):
        model = SteinWeierstrassNeuralNetwork(
            dim_input=self.kwargs_model["dim_input"],
            dim_hidden=self.kwargs_model["dim_hidden"],
            dim_output=self.kwargs_model["dim_output"],
            num_layers=self.kwargs_model["num_layers"],
            name_activation=self.kwargs_model["name_activation"],
            kwargs_act=dict(),
            scales=self.kwargs_model["scales"],
            flag_gate_unit=self.kwargs_model["flag_gate_unit"],
            num_blocks=self.kwargs_model["num_blocks"],
            flag_positive_output=self.kwargs_model["flag_positive_output"],
            sizes=self.kwargs_model["sizes"])

        def preprocess(x): return torch.tensor(x, dtype=torch.float)
        return model, preprocess, None

    def _get_TimesNet(self):
        """
        # Necessary kwargs
        See time_series_library.models.TimesNet.Model for details.
        - seq_len
        - enc_in
        - d_model
        - d_ff
        - top_k
        - e_layers
        - num_kernels
        - dropout
        - num_class
        - task_name
        - label_len
        - pred_len
        - embed
        - freq
        - c_out
        """
        raise NotImplementedError
        assert "with_mask" in self.kwargs_dataset.keys()
        kwargs_timesnet = KwargsTimesNet(self.kwargs_model)
        kwargs_timesnet.num_class: int = self.num_classes
        model = CustomTimesNet(
            kwargs_timesnet, self.kwargs_dataset["with_mask"])
        # def preprocess(x): return torch.tensor(x, dtype=torch.float)
        preprocess = None
        weights = None
        return model, preprocess, weights

    def _get_RNN(self):
        model = RNNModel(
            dim_input=self.kwargs_model["dim_input"],
            dim_hidden=self.kwargs_model["dim_hidden"],
            dim_output=self.num_classes,
            num_layers=self.kwargs_model["num_layers"],
            dropout=self.kwargs_model["dropout"],
            nonlinearity=self.kwargs_model["nonlinearity"],
            bias=self.kwargs_model["bias"],
            bidirectional=self.kwargs_model["bidirectional"],
            with_mask=self.kwargs_dataset["with_mask"],
            duration=self.kwargs_dataset["duration"])
        preprocess = None
        weights = None
        return model, preprocess, weights

    def _get_LSTM(self):
        model = LSTMModel(
            dim_input=self.kwargs_model["dim_input"],
            dim_hidden=self.kwargs_model["dim_hidden"],
            num_layers=self.kwargs_model["num_layers"],
            dim_output=self.num_classes,
            bias=self.kwargs_model["bias"],
            dropout=self.kwargs_model["dropout"],
            bidirectional=self.kwargs_model["bidirectional"],
            nonlinearity1=self.kwargs_model["nonlinearity1"],
            nonlinearity2=self.kwargs_model["nonlinearity2"],
            with_mask=self.kwargs_dataset["with_mask"],
            duration=self.kwargs_dataset["duration"])
        preprocess = None
        weights = None
        return model, preprocess, weights
