"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import logging
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
from donut.util import PerformanceMonitor, optimize_batch_processing, get_gpu_memory_info, clear_gpu_cache

# Pre-compiled regex patterns for efficiency
TAG_CLEANUP_PATTERN = re.compile(r'<[^>]+>')
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')


def get_logger():
    """Get the logger instance for lightning module"""
    return logging.getLogger('donut_training')


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._logger = get_logger()
        
        self._logger.info("🔧 Initializing DonutModelPLModule...")
        
        if self.config.get("pretrained_model_name_or_path", False):
            self._logger.info(f"📥 Loading pretrained model: {self.config.pretrained_model_name_or_path}")
            self.model = DonutModel.from_pretrained(
                self.config.pretrained_model_name_or_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                ignore_mismatched_sizes=True,
            )
            self._logger.info("✅ Pretrained model loaded successfully")
        else:
            self._logger.info("🏗️  Creating new Donut model from scratch")
            self.model = DonutModel(
                config=DonutConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                    # with DonutConfig, the architecture customization is available, e.g.,
                    # encoder_layer=[2,2,14,2], decoder_layer=4, ...
                )
            )
            self._logger.info("✅ New model created successfully")
            
        self.pytorch_lightning_version_is_1 = int(pl.__version__[0]) < 2
        self.num_of_loaders = len(self.config.dataset_name_or_paths)
        self._logger.info(f"📊 Number of dataloaders: {self.num_of_loaders}")
        
        # Log model architecture info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._logger.info(f"🧮 Total parameters: {total_params:,}")
        self._logger.info(f"🎯 Trainable parameters: {trainable_params:,}")
        self._logger.info(f"✅ Model initialization completed")

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor("donut_model", self._logger)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.performance_monitor.start_epoch()
        current_epoch = self.current_epoch
        self._logger.info(f"🚀 Epoch {current_epoch} | Starting training...")
        
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        epoch_duration = self.performance_monitor.end_epoch(self.current_epoch)
        stats = self.performance_monitor.get_stats()
        
        self._logger.info(f"✅ Epoch {self.current_epoch} completed in {epoch_duration:.2f}s")
        if stats.get('avg_batch_time'):
            memory_info = f" | GPU: {stats['avg_gpu_memory']:.1f}MB" if stats.get('avg_gpu_memory') else ""
            self._logger.info(f"📊 Performance - Avg batch: {stats['avg_batch_time']:.3f}s{memory_info}")
        
        # Clear GPU cache periodically to prevent memory buildup
        if self.current_epoch % 5 == 0:
            clear_gpu_cache()
        
    def training_step(self, batch, batch_idx):
        batch_start_time = time.time()
        
        # Use optimized batch processing utility
        image_tensors, decoder_input_ids, decoder_labels = optimize_batch_processing(
            batch, self._logger if batch_idx % 100 == 0 else None
        )
        
        # Debug tensor shapes for first few batches to catch issues early
        if batch_idx < 3:
            self._logger.info(f"DEBUG Batch {batch_idx} - images: {image_tensors.shape}, input_ids: {decoder_input_ids.shape}, labels: {decoder_labels.shape}")
            
        loss = self.model(image_tensors, decoder_input_ids, decoder_labels)[0]
        
        # Track batch time and GPU memory
        batch_time = time.time() - batch_start_time
        gpu_memory_info = get_gpu_memory_info()
        gpu_memory_mb = gpu_memory_info.get('max_allocated', 0) * 1024 if gpu_memory_info else None
        self.performance_monitor.record_batch(batch_time, gpu_memory_mb)
        
        # Log loss periodically with better formatting
        if batch_idx % 20 == 0:
            memory_info = f" | GPU: {gpu_memory_mb:.1f}MB" if gpu_memory_mb else ""
            self._logger.info(f"Step {batch_idx:3d} | Loss: {loss.item():.4f} | Time: {batch_time:.3f}s{memory_info}")
            
        self.log_dict({"train_loss": loss}, sync_dist=True)
        if not self.pytorch_lightning_version_is_1:
            self.log('loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.validation_step_outputs = [[] for _ in range(self.num_of_loaders)]
        current_epoch = self.current_epoch
        self._logger.info(f"🔄 Epoch {current_epoch} | Starting validation...")
        return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Log validation step info periodically
        if batch_idx % 50 == 0:
            self._logger.debug(f"Validation step {batch_idx} (dataloader {dataloader_idx}): Processing batch")
            
        image_tensors, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )

        # Log batch info periodically
        if batch_idx % 50 == 0:
            self._logger.debug(f"Validation batch shapes - images: {image_tensors.shape}, prompts: {decoder_prompts.shape}")

        preds = self.model.inference(
            image_tensors=image_tensors,
            prompt_tensors=decoder_prompts,
            return_json=False,
            return_attentions=False,
        )["predictions"]

        scores = list()
        for pred, answer in zip(preds, answers):
            # Use pre-compiled regex patterns for efficiency
            pred = TAG_CLEANUP_PATTERN.sub("", pred)
            answer = HTML_TAG_PATTERN.sub("", answer, count=1)
            answer = answer.replace(self.model.decoder.tokenizer.eos_token, "")
            
            # Optimize edit distance calculation for short strings
            if len(pred) < 50 and len(answer) < 50:
                # For short strings, use simple character-by-character comparison
                score = self._fast_edit_distance(pred, answer)
            else:
                # For longer strings, use the full edit distance algorithm
                score = edit_distance(pred, answer) / max(len(pred), len(answer))
            
            scores.append(score)

            if self.config.get("verbose", False) and len(scores) == 1:
                # Only log a brief summary for verbose mode
                pred_summary = pred[:50] + "..." if len(pred) > 50 else pred
                answer_summary = answer[:50] + "..." if len(answer) > 50 else answer
                self._logger.info(f"Sample - Pred: {pred_summary} | Answer: {answer_summary} | ED: {score:.4f}")

        self.validation_step_outputs[dataloader_idx].append(scores)

        return scores

    def _fast_edit_distance(self, s1: str, s2: str) -> float:
        """Optimized edit distance for short strings using dynamic programming"""
        if len(s1) == 0:
            return len(s2)
        if len(s2) == 0:
            return len(s1)
        
        # Early exit for identical strings
        if s1 == s2:
            return 0.0
            
        # Use numpy for faster computation with optimized memory layout
        m, n = len(s1), len(s2)
        matrix = np.zeros((m + 1, n + 1), dtype=np.int32)
        
        # Initialize first row and column
        matrix[0, :] = np.arange(n + 1)
        matrix[:, 0] = np.arange(m + 1)
        
        # Fill the matrix with optimized loops
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    matrix[i, j] = matrix[i-1, j-1]
                else:
                    matrix[i, j] = min(matrix[i-1, j], matrix[i, j-1], matrix[i-1, j-1]) + 1
                    
        return matrix[m, n] / max(m, n)

    def on_validation_epoch_end(self):
        assert len(self.validation_step_outputs) == self.num_of_loaders
        cnt = [0] * self.num_of_loaders
        total_metric = [0] * self.num_of_loaders
        val_metric = [0] * self.num_of_loaders
        
        current_epoch = self.current_epoch
        self._logger.info(f"📊 Epoch {current_epoch} | Computing validation metrics...")
        
        for i, results in enumerate(self.validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self._logger.info(f"  Dataset {i}: {cnt[i]:3d} samples | Metric: {val_metric[i]:.4f}")
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
            
        overall_metric = np.sum(total_metric) / np.sum(cnt)
        total_samples = np.sum(cnt)
        self._logger.info(f"✅ Epoch {current_epoch} | Overall: {overall_metric:.4f} ({total_samples} samples)")
        self._logger.info("─" * 60)
        self.log_dict({"val_metric": overall_metric}, sync_dist=True)

    def configure_optimizers(self):
        self._logger.info("Configuring optimizers and schedulers...")

        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )
            self._logger.info(f"Max iterations based on epochs: {max_iter:.0f}")

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps
            self._logger.info(f"Max iterations after considering max_steps: {max_iter:.0f}")

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        self._logger.info(f"Optimizer: Adam with learning rate {self.config.lr}")
        
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        self._logger.info(f"Scheduler: Cosine with {self.config.warmup_steps} warmup steps")
        
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
        self._logger.info(f"Saving model checkpoint to: {save_path}")
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)
        self._logger.info("Model checkpoint saved successfully")


class DonutDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._logger = get_logger()
        
        self._logger.info("Initializing DonutDataPLModule...")
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)
        
        self._logger.info(f"Train batch sizes: {self.train_batch_sizes}")
        self._logger.info(f"Validation batch sizes: {self.val_batch_sizes}")
        self._logger.info(f"Number of workers: {self.config.num_workers}")
        self._logger.info("Data module initialization completed")

    def train_dataloader(self):
        self._logger.info("Setting up training dataloaders...")
        loaders = list()
        for i, (train_dataset, batch_size) in enumerate(zip(self.train_datasets, self.train_batch_sizes)):
            self._logger.info(f"Creating training dataloader {i+1}/{len(self.train_datasets)} with batch size {batch_size}")
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
            self._logger.info(f"Training dataloader {i+1} created with {len(train_dataset)} samples")
        self._logger.info(f"All training dataloaders created: {len(loaders)} loaders")
        return loaders

    def val_dataloader(self):
        self._logger.info("Setting up validation dataloaders...")
        loaders = list()
        for i, (val_dataset, batch_size) in enumerate(zip(self.val_datasets, self.val_batch_sizes)):
            self._logger.info(f"Creating validation dataloader {i+1}/{len(self.val_datasets)} with batch size {batch_size}")
            loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                pin_memory=True,
                shuffle=False,
            )
            loaders.append(loader)
            self._logger.info(f"Validation dataloader {i+1} created with {len(val_dataset)} samples")
        self._logger.info(f"All validation dataloaders created: {len(loaders)} loaders")
        return loaders

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
