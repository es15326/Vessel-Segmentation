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

"""Main script for training a vessel segmentation model."""

import json
from pathlib import Path
import random

import numpy as np
import torch
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models
import utils.losses as losses
from data.dataset import VesselDataset
from utils.helpers import get_instance, setup_logging, seed_torch
from utils.metrics import AverageMeter, get_metrics

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Path to the configuration file.', lock_config=True)
flags.DEFINE_string('workdir', None, 'Directory to store logs and checkpoints.')
flags.DEFINE_integer('seed', 42, 'Random seed for reproducibility.')
flags.mark_flags_as_required(['config', 'workdir'])


class Trainer:
    """Encapsulates the training and validation logic."""

    def __init__(self, config, workdir):
        self.config = config
        self.workdir = Path(workdir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup logging and reproducibility
        setup_logging(self.workdir)
        seed_torch(FLAGS.seed)

        if self.config.log.log_to_tensorboard:
            self.writer = SummaryWriter(log_dir=self.workdir / 'tensorboard')

        # --- Data Loaders ---
        logging.info('Initializing datasets and dataloaders...')
        train_pro_dir = Path(config.data.dataset_path) / 'training_pro'
        all_files = sorted([f.name for f in train_pro_dir.glob('img_*.pkl')])
        random.shuffle(all_files)
        split_idx = int(len(all_files) * (1 - config.data.validation_split))
        train_files, val_files = all_files[:split_idx], all_files[split_idx:]

        train_dataset = VesselDataset(path=train_pro_dir, mode='training', file_list=train_files)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            pin_memory=self.config.train.pin_memory)
        
        val_dataset = VesselDataset(path=train_pro_dir, mode='validation', file_list=val_files)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        logging.info('Training samples: %d, Validation samples: %d', len(train_dataset), len(val_dataset))

        # --- Model, Optimizer, Scheduler, Loss ---
        logging.info('Initializing model, optimizer, and loss function...')
        self.model = get_instance(models, self.config.model.name, self.config.model).to(self.device)
        
        self.optimizer = get_instance(torch.optim, self.config.train.optimizer.name, 
                                      self.config.train.optimizer, self.model.parameters())
        
        self.scheduler = get_instance(torch.optim.lr_scheduler, self.config.train.lr_scheduler.name, 
                                      self.config.train.lr_scheduler, self.optimizer)
        
        self.loss_fn = get_instance(losses, self.config.train.loss.name, 
                                    self.config.train.loss).to(self.device)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.train.use_amp)
        self.start_epoch = 1

        # Save config to workdir for reproducibility
        with open(self.workdir / 'config.json', 'w') as f:
            f.write(self.config.to_json_best_effort())

    def _train_epoch(self, epoch):
        """Runs a single training epoch."""
        self.model.train()
        loss_meter = AverageMeter()
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.train.num_epochs}', leave=False)
        for images, masks in progress_bar:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config.train.use_amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter.update(loss.item(), images.size(0))
            progress_bar.set_postfix(loss=loss_meter.avg)

        logging.info('Epoch %d - Training Loss: %.4f', epoch, loss_meter.avg)
        if self.config.log.log_to_tensorboard:
            self.writer.add_scalar('Loss/train', loss_meter.avg, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
        
        return loss_meter.avg

    def _valid_epoch(self, epoch):
        """Runs a single validation epoch."""
        self.model.eval()
        loss_meter = AverageMeter()
        all_preds, all_gts = [], []

        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                loss_meter.update(loss.item(), images.size(0))
                
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_gts.append(masks.cpu().numpy())
        
        all_preds = np.concatenate(all_preds).flatten()
        all_gts = np.concatenate(all_gts).flatten()
        metrics = get_metrics(all_preds, all_gts, self.config.eval.threshold)

        logging.info('Epoch %d - Validation Loss: %.4f, AUC: %.4f, F1: %.4f', 
                     epoch, loss_meter.avg, metrics['AUC'], metrics['F1'])
        if self.config.log.log_to_tensorboard:
            self.writer.add_scalar('Loss/validation', loss_meter.avg, epoch)
            for key, val in metrics.items():
                self.writer.add_scalar(f'Metrics/{key}', val, epoch)
        
        return loss_meter.avg

    def _save_checkpoint(self, epoch):
        """Saves a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict()
        }
        checkpoint_path = self.workdir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logging.info('Saved checkpoint to %s', checkpoint_path)

    def run(self):
        """Starts the training process."""
        logging.info('Starting training for %d epochs on device: %s', 
                     self.config.train.num_epochs, self.device)
        for epoch in range(self.start_epoch, self.config.train.num_epochs + 1):
            self._train_epoch(epoch)
            
            val_loss = 0
            if self.val_loader:
                if epoch % self.config.eval.val_per_epochs == 0:
                    val_loss = self._valid_epoch(epoch)

            # CORRECTED: Pass the validation loss to the scheduler's step method.
            # This is required for ReduceLROnPlateau.
            self.scheduler.step(val_loss)

            if epoch % self.config.log.save_period == 0 or epoch == self.config.train.num_epochs:
                self._save_checkpoint(epoch)

        if self.config.log.log_to_tensorboard:
            self.writer.close()
        logging.info('Training finished successfully.')

def main(_):
    trainer = Trainer(config=FLAGS.config, workdir=FLAGS.workdir)
    trainer.run()

if __name__ == '__main__':
    app.run(main)

