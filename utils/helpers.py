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

"""Core utility functions for model training and evaluation."""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from absl import logging


def get_instance(module: Any, name: str, config: dict, *args, **kwargs) -> Any:
    """
    Dynamically creates an instance of a class from a module.

    This is a flexible way to instantiate objects like models, optimizers,
    and loss functions based on configuration settings.

    Args:
        module: The module where the class is defined (e.g., `models`, `torch.optim`).
        name (str): The name of the class to instantiate.
        config: A configuration object (like a dict or ml_collections.ConfigDict)
                containing the arguments for the class constructor under an 'args' key.
        *args: Additional positional arguments to pass to the constructor.
        **kwargs: Additional keyword arguments to pass to the constructor.

    Returns:
        An instance of the specified class.
    """
    constructor_args = dict(config.get('args', {}))
    constructor_args.update(kwargs)
    return getattr(module, name)(*args, **constructor_args)


def setup_logging(workdir: Path):
    """
    Configures absl logging to file and console.

    Args:
        workdir (Path): The working directory where the log file will be saved.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    log_file = workdir / 'experiment.log'
    # The default is to log to stderr. The following line adds a file handler.
    logging.get_absl_handler().use_absl_log_file(log_file.name, workdir.as_posix())
    logging.set_verbosity(logging.INFO)
    logging.info('Logging to console and file: %s', log_file)


def seed_torch(seed: int = 42):
    """
    Sets the seed for reproducibility in PyTorch, NumPy, and Python's random module.

    Args:
        seed (int): The seed to use for all random number generators.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # The following two lines are crucial for deterministic results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info('Set all seeds to %d for reproducibility.', seed)

