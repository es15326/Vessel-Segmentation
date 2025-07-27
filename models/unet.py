# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""U-Net model for segmentation."""

import torch.nn as nn
import segmentation_models_pytorch as smp


def Unet(backbone: str = 'resnet34',
         in_channels: int = 1,
         classes: int = 1,
         encoder_weights: str = 'imagenet') -> nn.Module:
    """
    Initializes a U-Net model from the segmentation-models-pytorch library.

    This wrapper function provides a consistent interface for model creation.
    It is configured to output raw logits (no final activation function), which
    is the standard practice for training with numerically stable loss functions
    like BCEWithLogitsLoss.

    Args:
        backbone (str): Name of the classification model to use as an encoder.
        in_channels (int): Number of input channels in the image.
        classes (int): Number of output classes (for segmentation).
        encoder_weights (str): 'imagenet' for pre-trained weights or None.

    Returns:
        A PyTorch model instance (smp.Unet).
    """
    model = smp.Unet(
        encoder_name=backbone,
        in_channels=in_channels,
        classes=classes,
        encoder_weights=encoder_weights,
        # CORRECTED: Removed 'activation' argument. The model will now output
        # raw logits, which is required for BCEWithLogitsLoss.
        activation=None
    )
    return model

