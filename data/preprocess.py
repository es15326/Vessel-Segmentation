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

"""Processes raw retinal vessel datasets into patches for training."""

import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from absl import app
from absl import flags
from absl import logging
from PIL import Image
from torchvision import transforms

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', None, 'Path to the root of the dataset.', required=True)
# CORRECTED LINE: Replaced DEFINE_string with DEFINE_enum for controlled vocabulary.
flags.DEFINE_enum('dataset_name', None, ['DRIVE', 'CHASEDB1', 'STARE'], 
                  'Name of the dataset to process.', required=True)
flags.DEFINE_integer('patch_size', 64, 'The height and width of the patches.')
flags.DEFINE_integer('stride', 16, 'The stride for patch extraction.')
flags.DEFINE_boolean('clear_existing', True, 'Clear existing processed files.')


def _normalize_and_convert_to_tensor(image_list: List[Image.Image]
                                     ) -> List[torch.Tensor]:
    """
    Normalizes a list of PIL images and converts them to PyTorch tensors.

    Args:
        image_list: A list of images as PIL.Image objects.

    Returns:
        A list of normalized images as PyTorch tensors.
    """
    tensor_list = [transforms.ToTensor()(img) for img in image_list]
    # For retinal images, it's common to normalize across the entire dataset.
    stacked_tensors = torch.cat(tensor_list, dim=0)

    mean = torch.mean(stacked_tensors)
    std = torch.std(stacked_tensors)

    logging.info('Normalizing with Mean: %.4f, Std: %.4f', mean, std)
    normalizer = transforms.Normalize([mean], [std])

    # Also perform a min-max scaling to [0, 1] range after standardization.
    # This can sometimes help with model stability.
    normalized_list = []
    for tensor in tensor_list:
        norm_tensor = normalizer(tensor)
        norm_tensor = ((norm_tensor - norm_tensor.min()) /
                       (norm_tensor.max() - norm_tensor.min() + 1e-8))
        normalized_list.append(norm_tensor)

    return normalized_list


def _extract_patches(image_tensors: List[torch.Tensor],
                     gt_tensors: List[torch.Tensor],
                     patch_size: int,
                     stride: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extracts corresponding patches from image and ground truth tensors.

    Args:
        image_tensors: A list of image tensors (C, H, W).
        gt_tensors: A list of ground truth tensors (C, H, W).
        patch_size: The size of the square patches to extract.
        stride: The stride between patches.

    Returns:
        A tuple containing (all_image_patches, all_gt_patches).
    """
    all_img_patches, all_gt_patches = [], []
    for img_tensor, gt_tensor in zip(image_tensors, gt_tensors):
        _, h, w = img_tensor.shape
        # Calculate padding needed to make dimensions perfectly divisible.
        pad_h = (stride - (h - patch_size) % stride) % stride
        pad_w = (stride - (w - patch_size) % stride) % stride

        padded_img = F.pad(img_tensor, (0, pad_w, 0, pad_h), "constant", 0)
        padded_gt = F.pad(gt_tensor, (0, pad_w, 0, pad_h), "constant", 0)

        # Use unfold to create sliding window views.
        img_patches = padded_img.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride)
        gt_patches = padded_gt.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride)

        # Reshape to (num_patches, C, patch_size, patch_size)
        img_patches = img_patches.contiguous().view(-1, 1, patch_size, patch_size)
        gt_patches = gt_patches.contiguous().view(-1, 1, patch_size, patch_size)

        # Filter out patches that are mostly background from the training set.
        for img_patch, gt_patch in zip(img_patches, gt_patches):
          # Keep patches that have at least a small amount of vessel.
          if gt_patch.sum() > 10:
              all_img_patches.append(img_patch)
              all_gt_patches.append(gt_patch)

    return all_img_patches, all_gt_patches


def _save_data(data_list: List[torch.Tensor], save_dir: Path, prefix: str):
    """Saves a list of tensors to individual pickle files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(data_list):
        filename = save_dir / f'{prefix}_{i}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(item.numpy(), f)
    logging.info('Saved %d files with prefix "%s" to %s', len(data_list),
                 prefix, save_dir)

def _process_drive(dataset_path: Path, patch_size: int, stride: int):
    """Handles preprocessing for the DRIVE dataset."""
    # Define padded size for consistent processing.
    padded_size = 608
    for mode in ['training', 'test']:
        logging.info('Processing DRIVE %s set...', mode)
        img_dir = dataset_path / mode / 'images'
        gt_dir = dataset_path / mode / '1st_manual'
        save_dir = dataset_path / f'{mode}_pro'

        if FLAGS.clear_existing and save_dir.exists():
            for f in save_dir.glob('*.pkl'):
                f.unlink()
        save_dir.mkdir(parents=True, exist_ok=True)


        img_files = sorted(list(img_dir.glob('*.tif')))
        gt_files = sorted(list(gt_dir.glob('*.gif')))

        images = [Image.open(p).convert('L') for p in img_files]
        gts = [Image.open(p).convert('L') for p in gt_files]

        norm_images = _normalize_and_convert_to_tensor(images)
        gt_tensors = [transforms.ToTensor()(gt) for gt in gts]

        if mode == 'training':
            img_patches, gt_patches = _extract_patches(
                norm_images, gt_tensors, patch_size, stride)
            _save_data(img_patches, save_dir, 'img_patch')
            _save_data(gt_patches, save_dir, 'gt_patch')
        else:  # test mode
            padded_images = [F.pad(t, (0, padded_size - t.shape[2], 0, padded_size - t.shape[1])) for t in norm_images]
            padded_gts = [F.pad(t, (0, padded_size - t.shape[2], 0, padded_size - t.shape[1])) for t in gt_tensors]
            _save_data(padded_images, save_dir, 'img')
            _save_data(padded_gts, save_dir, 'gt')

def main(_):
    """Main entry point for the preprocessing script."""
    dataset_path = Path(FLAGS.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset path not found: {dataset_path}')

    logging.info('Starting preprocessing for dataset: %s', FLAGS.dataset_name)
    logging.info('Patch size: %d, Stride: %d', FLAGS.patch_size, FLAGS.stride)

    if FLAGS.dataset_name == 'DRIVE':
        _process_drive(dataset_path, FLAGS.patch_size, FLAGS.stride)
    # Add other datasets here as elif blocks.
    # elif FLAGS.dataset_name == 'CHASEDB1':
    #     _process_chasedb1(dataset_path, FLAGS.patch_size, FLAGS.stride)
    else:
        raise ValueError(f'Unsupported dataset: {FLAGS.dataset_name}')

    logging.info('Preprocessing finished successfully.')


if __name__ == '__main__':
    app.run(main)

