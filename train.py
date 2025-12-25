"""
NitroGen Fine-Tuning Training Script
=====================================
Robust training loop with checkpointing for Paperspace 6-hour timeout.

Features:
- Auto-resume from checkpoint
- Timed saves every 30 minutes
- Mixed precision training (AMP)
- Learning rate scheduling with warmup
- Gradient clipping
- VRAM-optimized for A4000 (16GB)
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Local imports
from config import get_config, Config
from model import create_model, NitroGenCoordinateModel
from dataset import create_dataloaders, MagicChessDataset
from utils import (
    setup_logging,
    set_seed,
    CheckpointManager,
    AverageMeter,
    ProgressBar,
    count_parameters,
    get_gpu_memory_usage,
    format_time
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer class for NitroGen coordinate regression.
    
    Handles:
    - Training loop with validation
    - Automatic checkpoint saving/resuming
    - Mixed precision training
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        config: Config,
        model: NitroGenCoordinateModel,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function - MSE for coordinate regression
        self.criterion = nn.MSELoss()
        
        # Optimizer - AdamW with weight decay
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Checkpoint manager - saves to temp, syncs to persistent
        self.checkpoint_manager = CheckpointManager(
            temp_dir=config.checkpoint.temp_checkpoint_dir,
            persistent_dir=config.checkpoint.persistent_dir,
            save_interval_minutes=config.checkpoint.save_interval_minutes,
            keep_last_n=config.checkpoint.keep_last_n,
            sync_to_persistent=config.checkpoint.sync_to_persistent
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_start_time = None
    
    def _create_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with linear warmup and cosine decay."""
        warmup_steps = self.config.training.warmup_epochs * len(self.train_loader)
        total_steps = self.config.training.num_epochs * len(self.train_loader)
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / max(1, warmup_steps)
            else:
                # Cosine decay
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return max(
                    self.config.training.min_lr / self.config.training.learning_rate,
                    0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
                )
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def resume_from_checkpoint(self) -> bool:
        """
        Resume training from checkpoint if available.
        
        Returns:
            True if resumed, False if starting fresh
        """
        if not self.config.checkpoint.auto_resume:
            return False
        
        if not self.checkpoint_manager.has_checkpoint():
            logger.info("No checkpoint found. Starting fresh training.")
            return False
        
        try:
            state = self.checkpoint_manager.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler
            )
            
            self.current_epoch = state['epoch'] + 1  # Start from next epoch
            self.global_step = state['step']
            self.best_loss = state['best_loss']
            
            logger.info(f"Resumed from epoch {self.current_epoch}, best loss: {self.best_loss:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh training.")
            return False
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        # Keep vision encoder in eval mode if frozen
        if self.model.frozen:
            self.model.vision_encoder.eval()
        
        loss_meter = AverageMeter('loss')
        
        with ProgressBar(
            total=len(self.train_loader),
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}"
        ) as pbar:
            
            for batch_idx, (images, coords) in enumerate(self.train_loader):
                # Move to device
                images = images.to(self.device, non_blocking=True)
                coords = coords.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                self.optimizer.zero_grad()
                
                if self.config.training.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, coords)
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config.training.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.grad_clip_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, coords)
                    
                    loss.backward()
                    
                    if self.config.training.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.grad_clip_norm
                        )
                    
                    self.optimizer.step()
                
                # Update scheduler
                self.scheduler.step()
                
                # Update metrics
                loss_meter.update(loss.item(), images.size(0))
                self.global_step += 1
                
                # Update progress bar
                pbar.update(
                    1,
                    loss=loss_meter.avg,
                    lr=self.optimizer.param_groups[0]['lr']
                )
                
                # Timed checkpoint save (every 30 minutes)
                if self.checkpoint_manager.should_save():
                    self._save_checkpoint(
                        loss=loss_meter.avg,
                        is_best=False,
                        reason="timed"
                    )
        
        return {'train_loss': loss_meter.avg}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter('val_loss')
        
        for images, coords in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            coords = coords.to(self.device, non_blocking=True)
            
            if self.config.training.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, coords)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, coords)
            
            loss_meter.update(loss.item(), images.size(0))
        
        return {'val_loss': loss_meter.avg}
    
    def _save_checkpoint(
        self,
        loss: float,
        is_best: bool,
        reason: str = "epoch"
    ):
        """Save training checkpoint."""
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            loss=loss,
            best_loss=self.best_loss,
            is_best=is_best,
            extra_info={
                'reason': reason,
                'config': {
                    'learning_rate': self.config.training.learning_rate,
                    'batch_size': self.config.training.batch_size
                }
            }
        )
    
    def train(self):
        """
        Main training loop.
        
        Runs for num_epochs, with:
        - Validation every val_frequency epochs
        - Checkpoint saving at end of each epoch
        - Timed checkpoint saves every 30 minutes
        """
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)
        
        # Log configuration
        param_info = count_parameters(self.model)
        logger.info(f"Model parameters:")
        logger.info(f"  Total: {param_info['total']:,}")
        logger.info(f"  Trainable: {param_info['trainable']:,} ({param_info['trainable_pct']:.1f}%)")
        logger.info(f"  Frozen: {param_info['frozen']:,}")
        
        logger.info(f"\nTraining config:")
        logger.info(f"  Epochs: {self.config.training.num_epochs}")
        logger.info(f"  Batch size: {self.config.training.batch_size}")
        logger.info(f"  Learning rate: {self.config.training.learning_rate}")
        logger.info(f"  Device: {self.device}")
        
        gpu_mem = get_gpu_memory_usage()
        logger.info(f"\nGPU Memory: {gpu_mem['allocated']:.1f}GB / {gpu_mem['total']:.1f}GB")
        
        # Resume from checkpoint if available
        resumed = self.resume_from_checkpoint()
        
        self.training_start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if (epoch + 1) % self.config.training.val_frequency == 0:
                val_metrics = self.validate()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            val_loss = val_metrics.get('val_loss', train_metrics['train_loss'])
            logger.info(
                f"Epoch {epoch + 1}/{self.config.training.num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Time: {format_time(epoch_time)}"
            )
            
            # Check if best model
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                logger.info(f"New best model! Loss: {self.best_loss:.6f}")
            
            # Save checkpoint at end of epoch
            if self.config.checkpoint.save_every_epoch:
                self._save_checkpoint(
                    loss=val_loss,
                    is_best=is_best,
                    reason="epoch_end"
                )
            
            # Estimate remaining time
            elapsed = time.time() - self.training_start_time
            epochs_done = epoch - (self.current_epoch - 1 if resumed else 0) + 1
            eta = (elapsed / epochs_done) * (self.config.training.num_epochs - epoch - 1)
            logger.info(f"ETA: {format_time(eta)}")
        
        # Final save
        self._save_checkpoint(
            loss=self.best_loss,
            is_best=False,
            reason="training_complete"
        )
        
        total_time = time.time() - self.training_start_time
        logger.info("=" * 60)
        logger.info(f"Training Complete!")
        logger.info(f"Total time: {format_time(total_time)}")
        logger.info(f"Best loss: {self.best_loss:.6f}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train NitroGen for coordinate regression")
    
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/frames",
        help="Directory containing training frames"
    )
    
    parser.add_argument(
        "--labels-file",
        type=str,
        default="data/labels.json",
        help="Path to labels JSON file"
    )
    
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained ng.pt weights"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/storage/checkpoints",
        help="Directory for checkpoints"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8 for 16GB VRAM)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training"
    )
    
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=0,
        help="Number of transformer blocks to unfreeze (0 = freeze all)"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(
        pretrained_path=args.pretrained,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    config.checkpoint.checkpoint_dir = Path(args.checkpoint_dir)
    config.training.use_amp = not args.no_amp
    config.data.frames_dir = Path(args.frames_dir)
    config.data.labels_file = Path(args.labels_file)
    
    # Setup logging
    setup_logging(config.log_dir, config.experiment_name)
    
    # Set random seed
    set_seed(config.seed)
    
    logger.info("NitroGen Coordinate Training")
    logger.info(f"Frames: {args.frames_dir}")
    logger.info(f"Labels: {args.labels_file}")
    
    # Check if data exists
    if not Path(args.frames_dir).exists():
        logger.error(f"Frames directory not found: {args.frames_dir}")
        logger.info("Run preprocess_video.py first to extract frames.")
        sys.exit(1)
    
    if not Path(args.labels_file).exists():
        logger.error(f"Labels file not found: {args.labels_file}")
        logger.info("Run label_tool.py first to annotate coordinates.")
        sys.exit(1)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        frames_dir=str(config.data.frames_dir),
        labels_file=str(config.data.labels_file),
        batch_size=config.training.batch_size,
        val_split=config.training.val_split,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        seed=config.seed
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = create_model(
        pretrained_path=args.pretrained,
        freeze_encoder=True,
        image_size=config.model.image_size,
        vision_embed_dim=config.model.vision_hidden_dim,
        vision_num_layers=config.model.vision_num_layers,
        vision_num_heads=config.model.vision_num_heads,
        head_hidden_dims=config.model.head_hidden_dims,
        head_dropout=config.model.head_dropout
    )
    
    # Optionally unfreeze some layers
    if args.unfreeze_blocks > 0:
        model.unfreeze_last_n_blocks(args.unfreeze_blocks)
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
