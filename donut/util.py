"""
Utility functions for Donut model training and inference
"""
import logging
import time
import torch
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path


class PerformanceMonitor:
    """Centralized performance monitoring for Donut training and inference"""
    
    def __init__(self, name: str = "donut", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger('donut')
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.start_time = None
        self.epoch_times = []
        self.batch_times = []
        self.gpu_memory_usage = []
        self.epoch_start_time = None
        
    def start_training(self):
        """Start overall training timing"""
        self.start_time = time.time()
        self.logger.info(f"⏱️  {self.name} performance monitoring started")
        
    def start_epoch(self):
        """Start timing an epoch"""
        self.epoch_start_time = time.time()
        
    def end_epoch(self, epoch: int) -> float:
        """End timing an epoch and return duration"""
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            self.epoch_times.append(duration)
            self.epoch_start_time = None
            
            avg_epoch_time = np.mean(self.epoch_times)
            self.logger.info(f"⏱️  Epoch {epoch} took {duration:.2f}s (avg: {avg_epoch_time:.2f}s)")
            return duration
        return 0
        
    def record_batch(self, batch_time: float, gpu_memory: Optional[float] = None):
        """Record batch timing and GPU memory"""
        self.batch_times.append(batch_time)
        if gpu_memory is not None:
            self.gpu_memory_usage.append(gpu_memory)
            
    def log_memory_usage(self):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            self.logger.info(f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
    def get_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        stats = {
            "total_batches": len(self.batch_times),
            "total_epochs": len(self.epoch_times),
        }
        
        if self.batch_times:
            stats.update({
                "avg_batch_time": np.mean(self.batch_times),
                "min_batch_time": np.min(self.batch_times),
                "max_batch_time": np.max(self.batch_times),
            })
            
        if self.epoch_times:
            stats.update({
                "avg_epoch_time": np.mean(self.epoch_times),
                "min_epoch_time": np.min(self.epoch_times),
                "max_epoch_time": np.max(self.epoch_times),
            })
            
        if self.gpu_memory_usage:
            stats.update({
                "avg_gpu_memory": np.mean(self.gpu_memory_usage),
                "max_gpu_memory": np.max(self.gpu_memory_usage),
            })
            
        if self.start_time:
            stats["total_training_time"] = time.time() - self.start_time
            
        return stats
        
    def end_training(self):
        """End training and log final statistics"""
        if self.start_time:
            total_time = time.time() - self.start_time
            stats = self.get_stats()
            
            self.logger.info(f"⏱️  {self.name} training completed in {total_time:.2f}s ({total_time/3600:.2f}h)")
            
            if stats.get("avg_epoch_time"):
                self.logger.info(f"📊 Average epoch time: {stats['avg_epoch_time']:.2f}s")
                self.logger.info(f"📊 Fastest epoch: {stats['min_epoch_time']:.2f}s")
                self.logger.info(f"📊 Slowest epoch: {stats['max_epoch_time']:.2f}s")
                
            if stats.get("avg_batch_time"):
                self.logger.info(f"📊 Average batch time: {stats['avg_batch_time']:.3f}s")


def optimize_batch_processing(batch, logger: Optional[logging.Logger] = None):
    """
    Optimized batch processing for single or multiple dataloaders
    
    Args:
        batch: Either a tuple (single dataloader) or list of tuples (multiple dataloaders)
        logger: Optional logger for debugging
        
    Returns:
        tuple: (image_tensors, decoder_input_ids, decoder_labels)
    """
    if isinstance(batch, list):
        # Multiple dataloaders - concatenate efficiently
        all_image_tensors = []
        all_decoder_input_ids = []
        all_decoder_labels = []
        
        for batch_data in batch:
            all_image_tensors.append(batch_data[0])
            all_decoder_input_ids.append(batch_data[1][:, :-1])
            all_decoder_labels.append(batch_data[2][:, 1:])
        
        # Use torch.cat for efficient concatenation
        image_tensors = torch.cat(all_image_tensors, dim=0)
        decoder_input_ids = torch.cat(all_decoder_input_ids, dim=0)
        decoder_labels = torch.cat(all_decoder_labels, dim=0)
        
        if logger:
            logger.debug(f"Multi-dataloader batch shapes - images: {image_tensors.shape}, "
                        f"input_ids: {decoder_input_ids.shape}, labels: {decoder_labels.shape}")
    else:
        # Single dataloader - direct assignment
        image_tensors = batch[0]
        decoder_input_ids = batch[1][:, :-1]
        decoder_labels = batch[2][:, 1:]
        
        if logger:
            logger.debug(f"Single dataloader batch shapes - images: {image_tensors.shape}, "
                        f"input_ids: {decoder_input_ids.shape}, labels: {decoder_labels.shape}")
    
    return image_tensors, decoder_input_ids, decoder_labels


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory information in GB"""
    if not torch.cuda.is_available():
        return {}
        
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,
        "reserved": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
    }


def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_logging(log_dir: Path, name: str = "donut") -> logging.Logger:
    """Setup comprehensive logging for Donut operations"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / f'{name}.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler for critical errors
    error_handler = logging.FileHandler(log_dir / f'{name}_errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    return logger


def validate_config(config: Dict) -> bool:
    """Validate configuration parameters"""
    required_fields = ['input_size', 'max_length', 'train_batch_sizes', 'val_batch_sizes']
    
    for field in required_fields:
        if field not in config:
            logging.error(f"Missing required config field: {field}")
            return False
    
    # Validate input size
    if not isinstance(config['input_size'], list) or len(config['input_size']) != 2:
        logging.error("input_size must be a list of 2 integers [height, width]")
        return False
    
    # Validate batch sizes
    if not isinstance(config['train_batch_sizes'], list) or len(config['train_batch_sizes']) == 0:
        logging.error("train_batch_sizes must be a non-empty list")
        return False
    
    return True


def get_optimal_batch_size(gpu_memory_gb: float, model_size_mb: float) -> int:
    """Calculate optimal batch size based on available GPU memory"""
    # Rough estimation: reserve 20% for overhead, rest for model and data
    available_memory = gpu_memory_gb * 0.8
    model_memory_gb = model_size_mb / 1024
    
    # Estimate memory per sample (rough approximation)
    memory_per_sample_gb = 0.5  # This varies based on input size and model
    
    optimal_batch_size = int((available_memory - model_memory_gb) / memory_per_sample_gb)
    return max(1, min(optimal_batch_size, 32))  # Clamp between 1 and 32