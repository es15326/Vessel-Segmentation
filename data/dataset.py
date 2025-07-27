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

"""PyTorch Dataset class for loading preprocessed vessel segmentation data."""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip


class FixRandomRotation:
    """Applies a random rotation of 0, 90, 180, or 270 degrees."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        angle = torch.randint(0, 4, (1,)).item()
        if angle > 0:
            img = torch.rot90(img, k=angle, dims=[-2, -1])
        return img


class VesselDataset(Dataset):
    """
    Dataset for loading preprocessed retinal image patches.

    This class loads image and ground truth patches that have been preprocessed
    and saved as pickle files. It supports different modes ('training', 'test')
    and applies data augmentation for the training set.
    """

    def __init__(self,
                 path: str,
                 mode: str = 'training',
                 file_list: Optional[List[str]] = None):
        """
        Initializes the VesselDataset.

        Args:
            path (str): The root path to the preprocessed dataset folder
                        (e.g., './datasets/DRIVE/training_pro').
            mode (str): The mode of the dataset, either 'training' or 'test'.
            file_list (Optional[List[str]]): A specific list of image files to use.
                                             If None, all 'img_*.pkl' files in the
                                             path are used. This is useful for
                                             creating train/validation splits.
        """
        self.data_path = Path(path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

        if file_list:
            self.image_files = file_list
        else:
            self.image_files = sorted(
                [f.name for f in self.data_path.glob('img_*.pkl')])

        self.is_training = (mode == 'training')

        if self.is_training:
            self.transforms = Compose([
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                FixRandomRotation(),
            ])
        else:
            self.transforms = None

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the image and ground truth mask for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A tuple containing the image tensor and the ground truth mask tensor.
        """
        img_file_path = self.data_path / self.image_files[idx]
        with open(img_file_path, 'rb') as f:
            img = torch.from_numpy(pickle.load(f)).float()

        gt_file_name = self.image_files[idx].replace('img_', 'gt_')
        gt_file_path = self.data_path / gt_file_name
        with open(gt_file_path, 'rb') as f:
            gt = torch.from_numpy(pickle.load(f)).float()

        if self.is_training and self.transforms:
            # Stack image and mask to apply the same random transform.
            stacked = torch.cat((img, gt), dim=0)
            stacked = self.transforms(stacked)
            img, gt = torch.chunk(stacked, chunks=2, dim=0)

        return img, gt

