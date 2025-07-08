"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from io import BytesIO
from os.path import basename
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config

from donut import DonutDataset
from donut.util import PerformanceMonitor, setup_logging
from lightning_module import DonutDataPLModule, DonutModelPLModule





def setup_logging(config):
    """Setup comprehensive logging for training"""
    log_dir = Path(config.result_path) / config.exp_name / config.exp_version
    return setup_logging(log_dir, 'donut_training')


@rank_zero_only
def log_system_info(logger):
    """Log system information for reproducibility"""
    logger.info("=" * 60)
    logger.info("🚀 DONUT TRAINING STARTED")
    logger.info("=" * 60)
    logger.info(f"🐍 Python: {sys.version.split()[0]}")
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"⚡ Lightning: {pl.__version__}")
    logger.info(f"🎮 CUDA: {'✅ Available' if torch.cuda.is_available() else '❌ Not Available'}")
    if torch.cuda.is_available():
        logger.info(f"  Version: {torch.version.cuda}")
        logger.info(f"  GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            # Log GPU memory info
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"      Memory: {memory_total:.1f}GB")
    logger.info(f"💻 Platform: {sys.platform}")
    logger.info(f"📁 Working Dir: {os.getcwd()}")
    logger.info("=" * 60)


@rank_zero_only
def log_config_summary(logger, config):
    """Log a summary of the configuration"""
    logger.info("📋 CONFIGURATION SUMMARY")
    logger.info("─" * 40)
    
    # Log key training parameters
    key_params = [
        ('exp_name', 'Experiment'),
        ('exp_version', 'Version'),
        ('seed', 'Seed'),
        ('max_epochs', 'Max Epochs'),
        ('max_steps', 'Max Steps'),
        ('lr', 'Learning Rate'),
        ('warmup_steps', 'Warmup Steps'),
        ('gradient_clip_val', 'Gradient Clip'),
        ('train_batch_sizes', 'Train Batch Size'),
        ('val_batch_sizes', 'Val Batch Size'),
        ('num_workers', 'Workers'),
        ('input_size', 'Input Size'),
        ('max_length', 'Max Length')
    ]
    
    for param, label in key_params:
        if hasattr(config, param):
            value = getattr(config, param)
            logger.info(f"  {label:15}: {value}")
    
    # Log dataset information
    logger.info(f"  {'Datasets':15}: {len(config.dataset_name_or_paths)}")
    for i, dataset_path in enumerate(config.dataset_name_or_paths):
        dataset_name = os.path.basename(dataset_path)
        logger.info(f"    Dataset {i}: {dataset_name}")
    
    # Performance recommendations
    logger.info("─" * 40)
    logger.info("💡 PERFORMANCE RECOMMENDATIONS:")
    if config.num_workers < 4:
        logger.info("  ⚠️  Consider increasing num_workers for better data loading")
    if torch.cuda.is_available() and config.train_batch_sizes[0] < 4:
        logger.info("  ⚠️  Consider increasing batch size for better GPU utilization")
    logger.info("─" * 40)


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {"model." + key: value for key, value in state_dict.items()}
        return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


@rank_zero_only
def save_config_file(config, path, logger):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    logger.info(f"Saving config to: {save_path}")
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
    logger.info("Config saved successfully")


class ProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, config, logger):
        super().__init__()
        self.enable = True
        self.config = config
        self.logger = logger

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items


def set_seed(seed, logger):
    logger.info(f"Setting random seed to: {seed}")
    pytorch_lightning_version = int(pl.__version__[0])
    if pytorch_lightning_version < 2:
        pl.utilities.seed.seed_everything(seed, workers=True)
    else:
        import lightning_fabric
        lightning_fabric.utilities.seed.seed_everything(seed, workers=True)
    logger.info("Random seed set successfully")


def train(config):
    # Setup logging first
    logger = setup_logging(config)
    
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor("donut_training", logger)
    
    try:
        # Log system information
        log_system_info(logger)
        log_config_summary(logger, config)
        
        # Set seed
        set_seed(config.get("seed", 42), logger)
        
        logger.info("🔧 Initializing model and data modules...")
        model_module = DonutModelPLModule(config)
        data_module = DonutDataPLModule(config)
        logger.info("✅ Model and data modules initialized successfully")

        # add datasets to data_module
        logger.info("📚 Setting up datasets...")
        datasets = {"train": [], "validation": []}
        for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
            task_name = os.path.basename(dataset_name_or_path)  # e.g., cord-v2, docvqa, rvlcdip, ...
            logger.info(f"  Processing dataset {i+1}/{len(config.dataset_name_or_paths)}: {task_name}")
            
            # add categorical special tokens (optional)
            if task_name == "rvlcdip":
                logger.info("    Adding RVL-CDIP special tokens")
                model_module.model.decoder.add_special_tokens([
                    "<advertisement/>", "<budget/>", "<email/>", "<file_folder/>", 
                    "<form/>", "<handwritten/>", "<invoice/>", "<letter/>", 
                    "<memo/>", "<news_article/>", "<presentation/>", "<questionnaire/>", 
                    "<resume/>", "<scientific_publication/>", "<scientific_report/>", "<specification/>"
                ])
            if task_name == "docvqa":
                logger.info("    Adding DocVQA special tokens")
                model_module.model.decoder.add_special_tokens(["<yes/>", "<no/>"])
                
            for split in ["train", "validation"]:
                logger.info(f"    Creating {split} dataset for {task_name}")
                datasets[split].append(
                    DonutDataset(
                        dataset_name_or_path=dataset_name_or_path,
                        donut_model=model_module.model,
                        max_length=config.max_length,
                        split=split,
                        task_start_token=config.task_start_tokens[i]
                        if config.get("task_start_tokens", None)
                        else f"<s_{task_name}>",
                        prompt_end_token="<s_answer>" if "docvqa" in dataset_name_or_path else f"<s_{task_name}>",
                        sort_json_key=config.sort_json_key,
                    )
                )
                logger.info(f"    ✅ {split.capitalize()} dataset created for {task_name}")
                
        data_module.train_datasets = datasets["train"]
        data_module.val_datasets = datasets["validation"]
        logger.info(f"✅ Dataset setup complete. Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}")

        # Setup logging and callbacks
        logger.info("📊 Setting up TensorBoard logger...")
        tensorboard_logger = TensorBoardLogger(
            save_dir=config.result_path,
            name=config.exp_name,
            version=config.exp_version,
            default_hp_metric=False,
        )

        logger.info("⚙️  Setting up callbacks...")
        lr_callback = LearningRateMonitor(logging_interval="step")

        checkpoint_callback = ModelCheckpoint(
            monitor="val_metric",
            dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
            filename="artifacts",
            save_top_k=1,
            save_last=False,
            mode="min",
        )

        bar = ProgressBar(config, logger)

        custom_ckpt = CustomCheckpointIO()
        
        # Setup trainer
        logger.info("🏃 Configuring PyTorch Lightning trainer...")
        trainer = pl.Trainer(
            num_nodes=config.get("num_nodes", 1),
            devices=torch.cuda.device_count(),
            strategy="ddp",
            accelerator="gpu",
            plugins=custom_ckpt,
            max_epochs=config.max_epochs,
            max_steps=config.max_steps,
            val_check_interval=config.val_check_interval,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            gradient_clip_val=config.gradient_clip_val,
            precision=16,
            num_sanity_val_steps=0,
            logger=tensorboard_logger,
            callbacks=[lr_callback, checkpoint_callback, bar],
        )
        logger.info("✅ Trainer configured successfully")

        # Start training
        logger.info("🎯 Starting training...")
        logger.info("─" * 60)
        
        # Start performance monitoring
        perf_monitor.start_training()
        
        trainer.fit(model_module, data_module, ckpt_path=config.get("resume_from_checkpoint_path", None))
        
        # End performance monitoring
        perf_monitor.end_training()
        
        logger.info("─" * 60)
        logger.info("🎉 Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_version else args.exp_version

    # Setup basic logging for config saving
    basic_logger = setup_logging(config)
    save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version, basic_logger)
    
    train(config)
