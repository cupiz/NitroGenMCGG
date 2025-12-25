"""
NitroGen Model Architecture for Coordinate Regression
======================================================
Modified NitroGen model that predicts mouse coordinates instead of gamepad actions.

Architecture:
- Vision Encoder: SigLip2-based Vision Transformer (FROZEN)
- Coordinate Head: MLP that outputs (x, y) normalized coordinates (TRAINABLE)
"""

import math
import logging
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings (SigLip2 style)."""
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding via convolution
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        # CLS token (optional, used for classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        
        # Flatten: (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block."""
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionEncoder(nn.Module):
    """
    SigLip2-style Vision Transformer encoder.
    This will be FROZEN during fine-tuning to preserve pretrained features.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Features (B, embed_dim) from CLS token
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # Return CLS token features
        return x[:, 0]


class CoordinateHead(nn.Module):
    """
    Coordinate regression head.
    This is the ONLY trainable part during fine-tuning.
    
    Outputs normalized (x, y) coordinates in range [0, 1].
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] = [512, 256, 64],
        dropout: float = 0.1,
        output_dim: int = 2
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Vision encoder features (B, embed_dim)
        
        Returns:
            Coordinates (B, 2) in range [0, 1]
        """
        x = self.mlp(x)
        # Sigmoid to ensure output in [0, 1]
        x = torch.sigmoid(x)
        return x


class NitroGenCoordinateModel(nn.Module):
    """
    NitroGen model modified for coordinate regression.
    
    Architecture:
    - Vision Encoder (SigLip2-style ViT): FROZEN
    - Coordinate Head (MLP): TRAINABLE
    
    The vision encoder weights are loaded from ng.pt pretrained checkpoint.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        vision_embed_dim: int = 768,
        vision_num_layers: int = 12,
        vision_num_heads: int = 12,
        head_hidden_dims: List[int] = [512, 256, 64],
        head_dropout: float = 0.1,
        output_dim: int = 2
    ):
        super().__init__()
        
        # Vision encoder (will be frozen)
        self.vision_encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vision_embed_dim,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads
        )
        
        # Coordinate regression head (trainable)
        self.coordinate_head = CoordinateHead(
            input_dim=vision_embed_dim,
            hidden_dims=head_hidden_dims,
            dropout=head_dropout,
            output_dim=output_dim
        )
        
        self.frozen = False
    
    def freeze_vision_encoder(self):
        """
        Freeze the vision encoder to preserve pretrained features.
        This saves VRAM and speeds up training significantly.
        
        Why freeze:
        1. The vision encoder already learned powerful features from 40,000 hours of gameplay
        2. Fine-tuning only the head prevents catastrophic forgetting
        3. Reduces VRAM usage by ~70% (no gradients for encoder)
        4. Faster training (fewer parameters to update)
        """
        logger.info("Freezing vision encoder parameters...")
        
        for name, param in self.vision_encoder.named_parameters():
            param.requires_grad = False
        
        # Put encoder in eval mode (affects dropout, batchnorm)
        self.vision_encoder.eval()
        self.frozen = True
        
        logger.info(f"Vision encoder frozen. Only coordinate_head is trainable.")
    
    def unfreeze_last_n_blocks(self, n: int = 2):
        """
        Optionally unfreeze the last N transformer blocks for deeper fine-tuning.
        Use this if you have more VRAM and want better adaptation.
        """
        logger.info(f"Unfreezing last {n} transformer blocks...")
        
        total_blocks = len(self.vision_encoder.blocks)
        for i, block in enumerate(self.vision_encoder.blocks):
            if i >= total_blocks - n:
                for param in block.parameters():
                    param.requires_grad = True
                logger.info(f"  Block {i} unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, 256, 256)
        
        Returns:
            Coordinates (B, 2) with values in [0, 1]
        """
        # Extract features from vision encoder
        if self.frozen:
            with torch.no_grad():
                features = self.vision_encoder(x)
        else:
            features = self.vision_encoder(x)
        
        # Predict coordinates
        coords = self.coordinate_head(features)
        
        return coords
    
    def predict(self, x: torch.Tensor) -> Tuple[float, float]:
        """
        Predict coordinates for a single image.
        
        Args:
            x: Single image tensor (1, 3, 256, 256) or (3, 256, 256)
        
        Returns:
            Tuple of (x, y) coordinates
        """
        self.eval()
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            coords = self.forward(x)
        
        return coords[0, 0].item(), coords[0, 1].item()


def load_pretrained_weights(
    model: NitroGenCoordinateModel,
    checkpoint_path: str,
    strict: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Load pretrained NitroGen weights from ng.pt.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to ng.pt file
        strict: If False, allows missing/unexpected keys (needed for head swap)
    
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    logger.info(f"Loading pretrained weights from {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Try to match keys between pretrained and our model
    model_state = model.state_dict()
    matched_state = {}
    missing_keys = []
    unexpected_keys = list(state_dict.keys())
    
    for key in model_state.keys():
        if key in state_dict:
            matched_state[key] = state_dict[key]
            unexpected_keys.remove(key)
        else:
            # Try to find matching key with different prefix
            for pretrained_key in state_dict.keys():
                if pretrained_key.endswith(key) or key.endswith(pretrained_key):
                    matched_state[key] = state_dict[pretrained_key]
                    if pretrained_key in unexpected_keys:
                        unexpected_keys.remove(pretrained_key)
                    break
            else:
                missing_keys.append(key)
    
    # Load matched weights
    model.load_state_dict(matched_state, strict=False)
    
    logger.info(f"Loaded {len(matched_state)} weight tensors")
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        logger.debug(f"Missing: {missing_keys[:10]}...")
    if unexpected_keys:
        logger.debug(f"Unexpected: {unexpected_keys[:10]}...")
    
    return missing_keys, unexpected_keys


def create_model(
    pretrained_path: Optional[str] = None,
    freeze_encoder: bool = True,
    **kwargs
) -> NitroGenCoordinateModel:
    """
    Factory function to create NitroGen coordinate model.
    
    Args:
        pretrained_path: Path to ng.pt (optional)
        freeze_encoder: Whether to freeze vision encoder
        **kwargs: Model configuration overrides
    
    Returns:
        Configured model instance
    """
    model = NitroGenCoordinateModel(**kwargs)
    
    # Load pretrained weights if available
    if pretrained_path and Path(pretrained_path).exists():
        load_pretrained_weights(model, pretrained_path)
    else:
        logger.warning("No pretrained weights loaded. Training from scratch.")
    
    # Freeze encoder for efficient fine-tuning
    if freeze_encoder:
        model.freeze_vision_encoder()
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    print("=" * 60)
    print("Testing NitroGen Coordinate Model")
    print("=" * 60)
    
    # Create model
    model = create_model(freeze_encoder=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Trainable ratio: {100*trainable_params/total_params:.2f}%")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, 256, 256)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Sample output: {output[0].tolist()}")
    
    # Verify output is in [0, 1]
    assert output.min() >= 0 and output.max() <= 1, "Output should be in [0, 1]"
    assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"
    
    print("\n✓ All tests passed!")
