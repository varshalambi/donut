"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import logging
import math
import random
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance
from pytorch_lightning.utilities import rank_zero_only
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from donut import DonutConfig, DonutModel


def get_logger():
    """Get the logger instance for lightning module"""
    return logging.getLogger('donut_training')


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = get_logger()

        self.logger.info("Initializing DonutModelPLModule...")
        
        if self.config.get("pretrained_model_name_or_path", False):
            self.logger.info(f"Loading pretrained model from: {self.config.pretrained_model_name_or_path}")
            self.model = DonutModel.from_pretrained(
                self.config.pretrained_model_name_or_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                ignore_mismatched_sizes=True,
            )
            self.logger.info("Pretrained model loaded successfully")
        else:
            self.logger.info("Creating new Donut model from scratch")
            self.model = DonutModel(
                config=DonutConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                    # with DonutConfig, the architecture customization is available, e.g.,
                    # encoder_layer=[2,2,14,2], decoder_layer=4, ...
                )
            )
            self.logger.info("New model created successfully")
            
        self.pytorch_lightning_version_is_1 = int(pl.__version__[0]) < 2
        self.num_of_loaders = len(self.config.dataset_name_or_paths)
        self.logger.info(f"Number of dataloaders: {self.num_of_loaders}")
        
        # Log model architecture info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total model parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Model initialization completed")

    def training_step(self, batch, batch_idx):
        # Log training step info periodically
        if batch_idx % 100 == 0:
            self.logger.debug(f"Training step {batch_idx}: Processing batch")
            
        image_tensors, decoder_input_ids, decoder_labels = list(), list(), list()
        for batch_data in batch:
            image_tensors.append(batch_data[0])
            decoder_input_ids.append(batch_data[1][:, :-1])
            decoder_labels.append(batch_data[2][:, 1:])
        image_tensors = torch.cat(image_tensors)
        decoder_input_ids = torch.cat(decoder_input_ids)
        decoder_labels = torch.cat(decoder_labels)
        
        # Log batch shapes periodically
        if batch_idx % 100 == 0:
            self.logger.debug(f"Batch shapes - images: {image_tensors.shape}, input_ids: {decoder_input_ids.shape}, labels: {decoder_labels.shape}")
            
        loss = self.model(image_tensors, decoder_input_ids, decoder_labels)[0]
        
        # Log loss periodically
        if batch_idx % 50 == 0:
            self.logger.info(f"Training step {batch_idx}: Loss = {loss.item():.4f}")
            
        self.log_dict({"train_loss": loss}, sync_dist=True)
        if not self.pytorch_lightning_version_is_1:
            self.log('loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.validation_step_outputs = [[] for _ in range(self.num_of_loaders)]
        self.logger.info(f"Starting validation epoch. Number of dataloaders: {self.num_of_loaders}")
        return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Log validation step info periodically
        if batch_idx % 50 == 0:
            self.logger.debug(f"Validation step {batch_idx} (dataloader {dataloader_idx}): Processing batch")
            
        image_tensors, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )

        # Log batch info periodically
        if batch_idx % 50 == 0:
            self.logger.debug(f"Validation batch shapes - images: {image_tensors.shape}, prompts: {decoder_prompts.shape}")

        preds = self.model.inference(
            image_tensors=image_tensors,
            prompt_tensors=decoder_prompts,
            return_json=False,
            return_attentions=False,
        )["predictions"]

        scores = list()
        for pred, answer in zip(preds, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.model.decoder.tokenizer.eos_token, "")
            score = edit_distance(pred, answer) / max(len(pred), len(answer))
            scores.append(score)

            if self.config.get("verbose", False) and len(scores) == 1:
                self.logger.info(f"Sample prediction: {pred}")
                self.logger.info(f"Sample answer: {answer}")
                self.logger.info(f"Sample normalized edit distance: {score:.4f}")

        self.validation_step_outputs[dataloader_idx].append(scores)

        return scores

    def on_validation_epoch_end(self):
        assert len(self.validation_step_outputs) == self.num_of_loaders
        cnt = [0] * self.num_of_loaders
        total_metric = [0] * self.num_of_loaders
        val_metric = [0] * self.num_of_loaders
        
        self.logger.info("Computing validation metrics...")
        
        for i, results in enumerate(self.validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.logger.info(f"Dataset {i}: {cnt[i]} samples, average metric: {val_metric[i]:.4f}")
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
            
        overall_metric = np.sum(total_metric) / np.sum(cnt)
        self.logger.info(f"Overall validation metric: {overall_metric:.4f} (across {np.sum(cnt)} samples)")
        self.log_dict({"val_metric": overall_metric}, sync_dist=True)

    def configure_optimizers(self):
        self.logger.info("Configuring optimizers and schedulers...")

        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )
            self.logger.info(f"Max iterations based on epochs: {max_iter:.0f}")

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps
            self.logger.info(f"Max iterations after considering max_steps: {max_iter:.0f}")

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        self.logger.info(f"Optimizer: Adam with learning rate {self.config.lr}")
        
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        self.logger.info(f"Scheduler: Cosine with {self.config.warmup_steps} warmup steps")
        
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.config.result_path) / self.config.exp_name / self.config.exp_version
        self.logger.info(f"Saving model checkpoint to: {save_path}")
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)
        self.logger.info("Model checkpoint saved successfully")


class DonutDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = get_logger()
        
        self.logger.info("Initializing DonutDataPLModule...")
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)
        
        self.logger.info(f"Train batch sizes: {self.train_batch_sizes}")
        self.logger.info(f"Validation batch sizes: {self.val_batch_sizes}")
        self.logger.info(f"Number of workers: {self.config.num_workers}")
        self.logger.info("Data module initialization completed")

    def train_dataloader(self):
        self.logger.info("Setting up training dataloaders...")
        loaders = list()
        for i, (train_dataset, batch_size) in enumerate(zip(self.train_datasets, self.train_batch_sizes)):
            self.logger.info(f"Creating training dataloader {i+1}/{len(self.train_datasets)} with batch size {batch_size}")
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True,
                worker_init_fn=self.seed_worker,
                generator=self.g,
                shuffle=True,
            )
            loaders.append(loader)
            self.logger.info(f"Training dataloader {i+1} created with {len(train_dataset)} samples")
        self.logger.info(f"All training dataloaders created: {len(loaders)} loaders")
        return loaders

    def val_dataloader(self):
        self.logger.info("Setting up validation dataloaders...")
        loaders = list()
        for i, (val_dataset, batch_size) in enumerate(zip(self.val_datasets, self.val_batch_sizes)):
            self.logger.info(f"Creating validation dataloader {i+1}/{len(self.val_datasets)} with batch size {batch_size}")
            loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                pin_memory=True,
                shuffle=False,
            )
            loaders.append(loader)
            self.logger.info(f"Validation dataloader {i+1} created with {len(val_dataset)} samples")
        self.logger.info(f"All validation dataloaders created: {len(loaders)} loaders")
        return loaders

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
