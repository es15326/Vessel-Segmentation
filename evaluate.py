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

"""Script for evaluating a trained vessel segmentation model on the test set."""

import json
from pathlib import Path

import numpy as np
import torch
import ttach as tta
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_dict
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop
from tqdm import tqdm

import models
from data.dataset import VesselDataset
from utils.helpers import get_instance, setup_logging
from utils.metrics import get_metrics

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory containing the trained model checkpoint and config.')
flags.DEFINE_string('checkpoint', 'latest', 'Checkpoint to use for evaluation (e.g., "checkpoint_epoch_40.pth" or "latest").')
flags.DEFINE_boolean('show_predictions', False, 'If True, save predicted images.')
flags.mark_flags_as_required(['workdir'])


class Evaluator:
    """Encapsulates the model evaluation logic."""

    def __init__(self, workdir, checkpoint_name, show_predictions):
        self.workdir = Path(workdir)
        self.show_predictions = show_predictions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        setup_logging(self.workdir)

        # Load config from the training run
        config_path = self.workdir / 'config.json'
        with open(config_path) as f:
            self.config = config_dict.ConfigDict(json.load(f))

        # Data Loader
        logging.info('Initializing test dataloader...')
        test_data_path = Path(self.config.data.dataset_path) / 'test_pro'
        test_dataset = VesselDataset(path=test_data_path, mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Load Model
        logging.info('Loading model...')
        self.model = get_instance(models, self.config.model.name, self.config.model).to(self.device)
        
        if checkpoint_name == 'latest':
            checkpoints = sorted(self.workdir.glob('checkpoint_epoch_*.pth'), 
                                 key=lambda x: int(x.stem.split('_')[-1]))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {self.workdir}")
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = self.workdir / checkpoint_name

        logging.info('Loading checkpoint from: %s', checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Set up Test-Time Augmentation (TTA) if enabled
        if self.config.eval.use_tta:
            logging.info("Using Test-Time Augmentation.")
            self.model = tta.SegmentationTTAWrapper(self.model, tta.aliases.d4_transform(), merge_mode='mean')

        if self.show_predictions:
            self.pred_dir = self.workdir / 'predictions'
            self.pred_dir.mkdir(exist_ok=True)
            logging.info('Predictions will be saved to %s', self.pred_dir)

    def run(self):
        """Runs the evaluation loop."""
        self.model.eval()
        all_raw_preds, all_gts = [], []

        with torch.no_grad():
            for i, (images, masks) in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Evaluating"):
                images = images.to(self.device)
                outputs = self.model(images)
                
                # Crop predictions and masks to original image size for accurate metrics.
                orig_h, orig_w = self.config.eval.original_size
                outputs = crop(outputs, 0, 0, orig_h, orig_w)
                masks = crop(masks, 0, 0, orig_h, orig_w)

                all_raw_preds.append(outputs.cpu().numpy().flatten())
                all_gts.append(masks.cpu().numpy().flatten())

                if self.show_predictions:
                    pred_binary = (outputs[0, 0].cpu().numpy() > self.config.eval.threshold).astype(np.uint8)
                    pred_img = Image.fromarray(pred_binary * 255, 'L')
                    pred_img.save(self.pred_dir / f'prediction_{i}.png')

        all_raw_preds = np.concatenate(all_raw_preds)
        all_gts = np.concatenate(all_gts)

        metrics = get_metrics(all_raw_preds, all_gts, threshold=self.config.eval.threshold)
        self._log_metrics(metrics)

    def _log_metrics(self, metrics: dict):
        """Logs evaluation metrics to console and file."""
        logging.info('--- Test Set Evaluation Metrics ---')
        for key, value in metrics.items():
            logging.info('%-12s: %.4f', key, value)
        logging.info('-----------------------------------')

        # Save metrics to a file
        metrics_path = self.workdir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info('Saved evaluation metrics to %s', metrics_path)


def main(_):
    evaluator = Evaluator(
        workdir=FLAGS.workdir,
        checkpoint_name=FLAGS.checkpoint,
        show_predictions=FLAGS.show_predictions
    )
    evaluator.run()


if __name__ == '__main__':
    app.run(main)

