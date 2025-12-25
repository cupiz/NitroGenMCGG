"""
NitroGen Fine-Tuning Configuration
===================================
All hyperparameters and paths for training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Input settings (NitroGen standard)
    image_size: int = 256
    in_channels: int = 3
    
    # Vision encoder (SigLip2-based)
    vision_hidden_dim: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    vision_patch_size: int = 16
    
    # Coordinate head settings
    head_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 64])
    head_dropout: float = 0.1
    output_dim: int = 2  # (x, y) coordinates
    
    # Pretrained weights
    pretrained_path: Optional[str] = None  # Path to ng.pt


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Basic training
    batch_size: int = 8  # Optimized for 16GB VRAM (A4000)
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Learning rate schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Gradient settings
    grad_clip_norm: float = 1.0
    accumulation_steps: int = 1  # Gradient accumulation
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Validation
    val_split: float = 0.1
    val_frequency: int = 1  # Validate every N epochs


@dataclass
class CheckpointConfig:
    """Checkpointing configuration for 6-hour timeout resilience."""
    # Paths
    checkpoint_dir: Path = Path("/storage/checkpoints")
    best_model_name: str = "best_model.pth"
    latest_checkpoint_name: str = "checkpoint_latest.pth"
    
    # Timing (critical for Paperspace 6-hour limit)
    save_interval_minutes: int = 30  # Save every 30 minutes
    save_every_epoch: bool = True
    keep_last_n: int = 3  # Keep last N checkpoints
    
    # Resume
    auto_resume: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Paths (using root folder to avoid Paperspace storage limits)
    video_dir: Path = Path("/video")  # Video files go here
    data_dir: Path = Path("/data")
    frames_dir: Path = Path("/data/frames")
    labels_file: Path = Path("/data/labels.json")
    
    # Video preprocessing
    extract_fps: int = 10  # Frames per second to extract
    
    # Augmentations
    use_augmentation: bool = False  # Disabled for coordinate regression
    
    # Normalization (ImageNet stats for SigLip2)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class Config:
    """Master configuration combining all settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_dir: Path = Path("logs")
    experiment_name: str = "nitrogen_magic_chess"
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.checkpoint.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data.video_dir.mkdir(parents=True, exist_ok=True)
        self.data.data_dir.mkdir(parents=True, exist_ok=True)
        self.data.frames_dir.mkdir(parents=True, exist_ok=True)


def get_config(**overrides) -> Config:
    """Get configuration with optional overrides."""
    config = Config()
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.checkpoint, key):
            setattr(config.checkpoint, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
    
    return config


if __name__ == "__main__":
    # Print default configuration
    config = get_config()
    print("=" * 50)
    print("NitroGen Fine-Tuning Configuration")
    print("=" * 50)
    print(f"\nModel Config:")
    print(f"  Image size: {config.model.image_size}x{config.model.image_size}")
    print(f"  Vision hidden dim: {config.model.vision_hidden_dim}")
    print(f"  Output dim: {config.model.output_dim}")
    print(f"\nTraining Config:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"\nCheckpoint Config:")
    print(f"  Save interval: {config.checkpoint.save_interval_minutes} minutes")
    print(f"  Checkpoint dir: {config.checkpoint.checkpoint_dir}")
    print(f"\nDevice: {config.device}")
