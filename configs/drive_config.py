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

"""Configuration for training on the DRIVE dataset."""

from configs import base_config


def get_config():
  """Returns the configuration for DRIVE dataset."""
  config = base_config.get_config()

  # ----- Override base config with dataset-specific settings -----

  # Data parameters
  config.data.dataset_name = 'DRIVE'
  config.data.dataset_path = './datasets/DRIVE'

  # Evaluation parameters
  # Original DRIVE images are 565x584. They are padded to 608x608
  # during preprocessing for consistent patching. We need the original
  # dimensions to crop the final predictions for accurate evaluation.
  config.eval.original_size = (565, 584)
  config.eval.padded_size = 608

  # Model parameters (can be customized for the dataset)
  config.model.args.backbone = 'se_resnet101'

  # Training parameters
  config.train.num_epochs = 50
  config.train.batch_size = 128
  
  # CORRECTED: The line setting 'T_max' has been removed, as it's not
  # a valid argument for the new ReduceLROnPlateau scheduler.
  # The scheduler will now correctly use the arguments defined in the base config.

  return config

