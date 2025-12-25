"""
Magic Chess Dataset Loader
===========================
PyTorch Dataset for loading extracted frames and coordinate labels.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class MagicChessDataset(Dataset):
    """
    Dataset for Magic Chess coordinate training.
    
    Loads extracted video frames and their corresponding click coordinates.
    
    Expected label format (JSON):
    [
        {"frame": "frame_000001.png", "x": 0.45, "y": 0.67, "click": true},
        {"frame": "frame_000002.png", "x": 0.12, "y": 0.89, "click": true},
        ...
    ]
    
    Frames without click events can be included with click=false for 
    learning when NOT to click.
    """
    
    def __init__(
        self,
        frames_dir: str,
        labels_file: str,
        image_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        augment: bool = False,
        filter_clicks_only: bool = True
    ):
        """
        Args:
            frames_dir: Directory containing extracted frame images
            labels_file: Path to JSON file with coordinate labels
            image_size: Target image size (default: 256 for NitroGen)
            mean: Normalization mean (ImageNet defaults)
            std: Normalization std (ImageNet defaults)
            augment: Whether to apply data augmentation
            filter_clicks_only: If True, only include frames with click=true
        """
        self.frames_dir = Path(frames_dir)
        self.labels_file = Path(labels_file)
        self.image_size = image_size
        self.filter_clicks_only = filter_clicks_only
        
        # Load labels
        self.samples = self._load_labels()
        
        # Setup transforms
        self.transform = self._build_transform(mean, std, augment)
        
        logger.info(f"Loaded {len(self.samples)} samples from {labels_file}")
    
    def _load_labels(self) -> List[Dict]:
        """Load and validate labels from JSON file."""
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
        
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, dict) and 'frames' in data:
            labels = data['frames']
        elif isinstance(data, list):
            labels = data
        else:
            raise ValueError(f"Invalid labels format in {self.labels_file}")
        
        # Filter and validate
        valid_samples = []
        for item in labels:
            frame_name = item.get('frame') or item.get('filename')
            x = item.get('x')
            y = item.get('y')
            click = item.get('click', True)
            
            # Skip if missing required fields
            if frame_name is None or x is None or y is None:
                continue
            
            # Skip non-click events if filtering
            if self.filter_clicks_only and not click:
                continue
            
            # Validate coordinate range
            if not (0 <= x <= 1 and 0 <= y <= 1):
                logger.warning(f"Skipping {frame_name}: coordinates out of range ({x}, {y})")
                continue
            
            # Check if frame exists
            frame_path = self.frames_dir / frame_name
            if not frame_path.exists():
                logger.warning(f"Frame not found: {frame_path}")
                continue
            
            valid_samples.append({
                'frame': frame_name,
                'x': float(x),
                'y': float(y),
                'click': bool(click)
            })
        
        if len(valid_samples) == 0:
            logger.warning("No valid samples found! Check your labels file.")
        
        return valid_samples
    
    def _build_transform(
        self,
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
        augment: bool
    ) -> transforms.Compose:
        """Build image transform pipeline."""
        transform_list = [
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        
        if augment:
            # Note: For coordinate regression, we need to be careful with augmentations
            # that change spatial relationships. Avoid horizontal flip, rotation, etc.
            # Only use augmentations that don't affect coordinates.
            augment_transforms = [
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                ),
            ]
            transform_list = augment_transforms + transform_list
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            image: Tensor of shape (3, H, W)
            coords: Tensor of shape (2,) containing (x, y) in [0, 1]
        """
        sample = self.samples[idx]
        
        # Load image
        frame_path = self.frames_dir / sample['frame']
        image = Image.open(frame_path).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        # Create coordinate tensor
        coords = torch.tensor([sample['x'], sample['y']], dtype=torch.float32)
        
        return image, coords
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample (for debugging/visualization)."""
        return self.samples[idx]


class MagicChessDatasetWithMetadata(MagicChessDataset):
    """Extended dataset that also returns frame metadata."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        image, coords = super().__getitem__(idx)
        metadata = self.get_sample_info(idx)
        return image, coords, metadata


def create_dataloaders(
    frames_dir: str,
    labels_file: str,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        frames_dir: Directory with frame images
        labels_file: Path to labels JSON
        batch_size: Batch size
        val_split: Fraction for validation (0.1 = 10%)
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for split
        **dataset_kwargs: Additional dataset arguments
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = MagicChessDataset(
        frames_dir=frames_dir,
        labels_file=labels_file,
        **dataset_kwargs
    )
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Use seeded random split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    logger.info(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def create_dummy_dataset(
    output_dir: str,
    num_samples: int = 100,
    image_size: int = 256
) -> Tuple[str, str]:
    """
    Create a dummy dataset for testing.
    
    Args:
        output_dir: Directory to create dataset in
        num_samples: Number of samples to generate
        image_size: Image size
    
    Returns:
        Tuple of (frames_dir, labels_file)
    """
    import random
    
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    labels = []
    
    for i in range(num_samples):
        # Generate random image
        img_array = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        
        # Add a "target" marker at random position
        x = random.random()
        y = random.random()
        
        # Draw marker on image
        cx, cy = int(x * image_size), int(y * image_size)
        cv2_available = True
        try:
            import cv2
            cv2.circle(img_array, (cx, cy), 10, (255, 0, 0), -1)
        except ImportError:
            cv2_available = False
        
        # Save image
        frame_name = f"frame_{i:06d}.png"
        img = Image.fromarray(img_array)
        img.save(frames_dir / frame_name)
        
        # Add label
        labels.append({
            "frame": frame_name,
            "x": round(x, 4),
            "y": round(y, 4),
            "click": True
        })
    
    # Save labels
    labels_file = output_dir / "labels.json"
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)
    
    logger.info(f"Created dummy dataset with {num_samples} samples")
    logger.info(f"  Frames: {frames_dir}")
    logger.info(f"  Labels: {labels_file}")
    
    return str(frames_dir), str(labels_file)


if __name__ == "__main__":
    # Test dataset
    print("=" * 60)
    print("Testing MagicChessDataset")
    print("=" * 60)
    
    # Create dummy dataset
    dummy_dir = "data/dummy_test"
    frames_dir, labels_file = create_dummy_dataset(dummy_dir, num_samples=50)
    
    # Create dataset
    dataset = MagicChessDataset(
        frames_dir=frames_dir,
        labels_file=labels_file
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test sample loading
    image, coords = dataset[0]
    print(f"Sample image shape: {image.shape}")
    print(f"Sample coords: {coords}")
    print(f"Coords range: x={coords[0]:.4f}, y={coords[1]:.4f}")
    
    # Test dataloader
    train_loader, val_loader = create_dataloaders(
        frames_dir=frames_dir,
        labels_file=labels_file,
        batch_size=8,
        val_split=0.2,
        num_workers=0  # Use 0 for testing
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test batch
    for images, coords in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch coords shape: {coords.shape}")
        break
    
    print("\n✓ Dataset tests passed!")
