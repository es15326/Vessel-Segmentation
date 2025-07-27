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

"""Evaluation metrics and related helper classes."""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class AverageMeter:
    """Computes and stores the average and current value of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all the statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The new value to add.
            n (int): The number of samples associated with the value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculates a set of standard binary segmentation metrics.

    Args:
        predictions (np.ndarray): The raw, continuous model predictions (probabilities),
                                  flattened into a 1D array.
        targets (np.ndarray): The binary ground truth labels, flattened into a 1D array.
        threshold (float): The threshold to binarize the predictions.

    Returns:
        A dictionary containing the calculated metrics: AUC, F1, Accuracy,
        Sensitivity (Recall), Specificity, and Precision.
    """
    # Binarize predictions based on the threshold
    binary_preds = (predictions >= threshold).astype(np.uint8)
    targets = targets.astype(np.uint8)

    # Calculate confusion matrix components
    tp = (binary_preds * targets).sum()
    tn = ((1 - binary_preds) * (1 - targets)).sum()
    fp = ((1 - targets) * binary_preds).sum()
    fn = ((1 - binary_preds) * targets).sum()

    # Calculate metrics, handling potential division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Also known as Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    try:
        auc = roc_auc_score(targets, predictions)
    except ValueError:
        # This can happen if only one class is present in targets.
        auc = 0.5

    metrics = {
        'AUC': round(auc, 4),
        'F1': round(f1_score(targets, binary_preds), 4),
        'Accuracy': round(accuracy_score(targets, binary_preds), 4),
        'Sensitivity': round(recall, 4),
        'Specificity': round(specificity, 4),
        'Precision': round(precision, 4),
    }
    return metrics

