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

"""Base configuration for all experiments."""

import ml_collections


def get_config():
  """Returns the base configuration for vessel segmentation experiments."""
  config = ml_collections.ConfigDict()

  # General experiment parameters.
  config.experiment_name = 'vessel_segmentation'

  # Data parameters.
  config.data = ml_collections.ConfigDict()
  config.data.patch_size = 64
  config.data.stride = 16
  config.data.dataset_name = 'DRIVE'
  config.data.dataset_path = ''  # Must be set by the user in a specific config.
  config.data.validation_split = 0.1 # Percentage of training data for validation.

  # Model parameters.
  config.model = ml_collections.ConfigDict()
  config.model.name = 'Unet'
  config.model.args = ml_collections.ConfigDict()
  config.model.args.backbone = 'se_resnet101'
  config.model.args.in_channels = 1
  config.model.args.classes = 1
  config.model.args.encoder_weights = 'imagenet'

  # Training parameters.
  config.train = ml_collections.ConfigDict()

  config.train.optimizer = ml_collections.ConfigDict()
  config.train.optimizer.name = 'Adam'
  config.train.optimizer.args = ml_collections.ConfigDict()
  # Slightly reduced learning rate for more stability.
  config.train.optimizer.args.lr = 1e-4
  config.train.optimizer.args.weight_decay = 1e-5

  # CORRECTED: Switched to a more stable learning rate scheduler.
  # ReduceLROnPlateau will decrease the LR when validation loss plateaus.
  config.train.lr_scheduler = ml_collections.ConfigDict()
  config.train.lr_scheduler.name = 'ReduceLROnPlateau'
  config.train.lr_scheduler.args = ml_collections.ConfigDict()
  config.train.lr_scheduler.args.mode = 'min'
  config.train.lr_scheduler.args.factor = 0.2
  config.train.lr_scheduler.args.patience = 5
  config.train.lr_scheduler.args.verbose = True

  config.train.loss = ml_collections.ConfigDict()
  config.train.loss.name = 'CE_DiceLoss'
  config.train.loss.args = ml_collections.ConfigDict()
  config.train.loss.args.dice_weight = 0.5

  config.train.num_epochs = 50 # Increased epochs for the new scheduler.
  config.train.batch_size = 128
  config.train.num_workers = 8
  config.train.pin_memory = True
  config.train.use_amp = True  # Automatic Mixed Precision.

  # Evaluation parameters.
  config.eval = ml_collections.ConfigDict()
  config.eval.batch_size = 1
  config.eval.val_per_epochs = 1
  config.eval.threshold = 0.5
  config.eval.use_tta = True  # Test-Time Augmentation.

  # Logging and saving parameters.
  config.log = ml_collections.ConfigDict()
  config.log.save_period = 5  # Save checkpoint every N epochs.
  config.log.log_to_tensorboard = True

  return config

