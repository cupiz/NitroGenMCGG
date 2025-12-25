"""
NitroGen Fine-Tuning Utilities
==============================
Logging, checkpointing, and helper functions.
"""

import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} (avg: {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class CheckpointManager:
    """
    Manages checkpoint saving and loading with timed auto-save.
    Critical for Paperspace 6-hour timeout resilience.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        save_interval_minutes: int = 30,
        keep_last_n: int = 3
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval_minutes = save_interval_minutes
        self.keep_last_n = keep_last_n
        self.last_save_time = time.time()
        self.logger = logging.getLogger(__name__)
    
    def should_save(self) -> bool:
        """Check if enough time has passed to save."""
        elapsed = (time.time() - self.last_save_time) / 60
        return elapsed >= self.save_interval_minutes
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        loss: float,
        best_loss: float,
        is_best: bool = False,
        extra_info: Dict = None
    ) -> Path:
        """
        Save training checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            scheduler: LR scheduler state
            epoch: Current epoch
            step: Global step
            loss: Current loss
            best_loss: Best loss so far
            is_best: Whether this is the best model
            extra_info: Additional metadata
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'best_loss': best_loss,
            'timestamp': datetime.now().isoformat(),
            'extra_info': extra_info or {}
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        self.logger.info(f"Saved checkpoint to {latest_path}")
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        torch.save(checkpoint, epoch_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with loss: {loss:.6f}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        self.last_save_time = time.time()
        return latest_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old epoch checkpoints, keeping only the last N."""
        epoch_checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for ckpt in epoch_checkpoints[self.keep_last_n:]:
            ckpt.unlink()
            self.logger.debug(f"Removed old checkpoint: {ckpt}")
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to restore state
            scheduler: Scheduler to restore state
            checkpoint_path: Specific checkpoint path (default: latest)
        
        Returns:
            Dictionary with epoch, step, and other metadata
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        
        if not checkpoint_path.exists():
            self.logger.warning(f"No checkpoint found at {checkpoint_path}")
            return {'epoch': 0, 'step': 0, 'best_loss': float('inf')}
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(
            f"Resumed from epoch {checkpoint['epoch']}, "
            f"step {checkpoint['step']}, loss {checkpoint['loss']:.6f}"
        )
        
        return {
            'epoch': checkpoint['epoch'],
            'step': checkpoint['step'],
            'best_loss': checkpoint.get('best_loss', float('inf')),
            'timestamp': checkpoint.get('timestamp'),
            'extra_info': checkpoint.get('extra_info', {})
        }
    
    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists."""
        return (self.checkpoint_dir / 'checkpoint_latest.pth').exists()


def setup_logging(
    log_dir: Path,
    experiment_name: str,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging to file and console.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name for log file
        level: Logging level
    
    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging to {log_file}")
    
    return logger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dictionary with total, trainable, and frozen parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_pct': 100 * trainable / total if total > 0 else 0
    }


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'total': 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        'allocated': round(allocated, 2),
        'reserved': round(reserved, 2),
        'total': round(total, 2),
        'free': round(total - reserved, 2)
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class ProgressBar:
    """Wrapper around tqdm for training progress."""
    
    def __init__(
        self,
        total: int,
        desc: str = "Training",
        unit: str = "batch"
    ):
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            leave=True
        )
        self.metrics = {}
    
    def update(self, n: int = 1, **metrics):
        """Update progress bar with optional metrics."""
        self.metrics.update(metrics)
        
        # Format metrics for display
        postfix = {k: f"{v:.4f}" if isinstance(v, float) else v 
                   for k, v in self.metrics.items()}
        self.pbar.set_postfix(postfix)
        self.pbar.update(n)
    
    def close(self):
        self.pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test AverageMeter
    meter = AverageMeter("loss")
    for i in range(10):
        meter.update(0.5 - i * 0.01)
    print(f"AverageMeter: {meter}")
    
    # Test GPU memory
    print(f"GPU Memory: {get_gpu_memory_usage()}")
    
    # Test time formatting
    print(f"Time format: {format_time(7265)}")
    
    print("Utilities OK!")
