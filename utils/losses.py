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

"""Collection of loss functions for segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    A wrapper for PyTorch's BCEWithLogitsLoss.

    This loss is numerically more stable than a standard BCELoss followed by a
    sigmoid layer, making it safe for automatic mixed precision (AMP) training.
    It expects raw logits from the model as input.
    """

    def __init__(self):
        super().__init__()
        # CORRECTED: Switched to the numerically stable BCEWithLogitsLoss.
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(inputs, targets)


class DiceLoss(nn.Module):
    """
    Computes the Dice Loss. This version expects raw logits as input and
    applies a sigmoid internally to convert them to probabilities.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CORRECTED: Apply sigmoid to convert logits to probabilities.
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice_coeff


class CE_DiceLoss(nn.Module):
    """
    A combined loss function (BCEWithLogits + Dice).
    This version is safe for mixed precision training as it uses the stable
    BCEWithLogitsLoss internally.
    """

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        # This will now correctly use our updated, stable BCELoss wrapper.
        self.bce_loss = BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 'inputs' are raw logits here.
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return (1 - self.dice_weight) * bce + self.dice_weight * dice


class FocalLoss(nn.Module):
    """
    Focal Loss. This version is adapted to accept raw logits as input,
    making it safe for mixed precision training.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CORRECTED: Use the stable 'with_logits' version of the loss.
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # The rest of the logic remains the same.
        # p_t is the probability of the correct class.
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t)**self.gamma * bce_loss
        
        return focal_loss.mean()

