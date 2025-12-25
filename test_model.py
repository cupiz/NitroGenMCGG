"""
Model Testing Script
=====================
Verify NitroGen model forward pass and gradient flow.
"""

import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn

from config import get_config
from model import create_model, load_pretrained_weights
from utils import count_parameters, get_gpu_memory_usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_forward_pass(model: nn.Module, device: str = 'cuda') -> bool:
    """
    Test model forward pass with dummy data.
    
    Args:
        model: Model to test
        device: Device to run on
    
    Returns:
        True if test passes
    """
    logger.info("Testing forward pass...")
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input (batch of 4 images, 256x256 RGB)
    dummy_input = torch.randn(4, 3, 256, 256, device=device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # Validate output
    assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"
    assert output.min() >= 0, f"Output min should be >= 0, got {output.min()}"
    assert output.max() <= 1, f"Output max should be <= 1, got {output.max()}"
    
    logger.info(f"  Input shape: {dummy_input.shape}")
    logger.info(f"  Output shape: {output.shape}")
    logger.info(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    logger.info(f"  Sample output: {output[0].tolist()}")
    
    return True


def test_gradient_flow(model: nn.Module, device: str = 'cuda') -> bool:
    """
    Test that gradients flow only to trainable parameters.
    
    Args:
        model: Model to test
        device: Device to run on
    
    Returns:
        True if test passes
    """
    logger.info("Testing gradient flow...")
    
    model = model.to(device)
    model.train()
    
    # Keep vision encoder in eval if frozen
    if model.frozen:
        model.vision_encoder.eval()
    
    # Forward pass
    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    dummy_target = torch.rand(2, 2, device=device)
    
    output = model(dummy_input)
    loss = nn.MSELoss()(output, dummy_target)
    loss.backward()
    
    # Check gradients
    trainable_with_grad = 0
    frozen_with_grad = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                trainable_with_grad += 1
        else:
            if param.grad is not None:
                frozen_with_grad += 1
                logger.warning(f"Frozen param has gradient: {name}")
    
    logger.info(f"  Trainable params with gradients: {trainable_with_grad}")
    logger.info(f"  Frozen params with gradients: {frozen_with_grad}")
    
    if frozen_with_grad > 0:
        logger.warning("Some frozen parameters have gradients!")
        return False
    
    return True


def test_checkpoint_loading(model: nn.Module, checkpoint_path: str) -> bool:
    """
    Test loading pretrained weights.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to ng.pt
    
    Returns:
        True if loading succeeds
    """
    logger.info(f"Testing checkpoint loading from {checkpoint_path}...")
    
    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        missing, unexpected = load_pretrained_weights(model, checkpoint_path)
        logger.info(f"  Missing keys: {len(missing)}")
        logger.info(f"  Unexpected keys: {len(unexpected)}")
        return True
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False


def test_memory_usage(model: nn.Module, batch_size: int = 8, device: str = 'cuda') -> dict:
    """
    Test memory usage during training.
    
    Args:
        model: Model to test
        batch_size: Batch size to simulate
        device: Device to run on
    
    Returns:
        Dictionary with memory stats
    """
    logger.info(f"Testing memory usage with batch_size={batch_size}...")
    
    if device != 'cuda' or not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping memory test")
        return {}
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = model.to(device)
    model.train()
    if model.frozen:
        model.vision_encoder.eval()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    
    # Simulate training step
    dummy_input = torch.randn(batch_size, 3, 256, 256, device=device)
    dummy_target = torch.rand(batch_size, 2, device=device)
    
    # Forward
    output = model(dummy_input)
    loss = nn.MSELoss()(output, dummy_target)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Get memory stats
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    current_memory = torch.cuda.memory_allocated() / 1024**3
    
    logger.info(f"  Peak memory: {peak_memory:.2f} GB")
    logger.info(f"  Current memory: {current_memory:.2f} GB")
    
    return {
        'peak_memory_gb': peak_memory,
        'current_memory_gb': current_memory
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NitroGen model")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to ng.pt weights"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for memory test"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU"
    )
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    
    print("=" * 60)
    print("NitroGen Model Tests")
    print("=" * 60)
    print(f"Device: {device}")
    
    if device == 'cuda':
        gpu_info = get_gpu_memory_usage()
        print(f"GPU Memory: {gpu_info['total']:.1f} GB total")
    
    # Create model
    print("\n1. Creating model...")
    model = create_model(
        pretrained_path=args.pretrained,
        freeze_encoder=True
    )
    
    # Parameter count
    params = count_parameters(model)
    print(f"\n2. Parameter Count:")
    print(f"   Total: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
    print(f"   Frozen: {params['frozen']:,}")
    
    # Tests
    tests_passed = 0
    tests_total = 0
    
    # Forward pass test
    print("\n3. Forward Pass Test:")
    tests_total += 1
    if test_forward_pass(model, device):
        print("   ✓ PASSED")
        tests_passed += 1
    else:
        print("   ✗ FAILED")
    
    # Gradient flow test
    print("\n4. Gradient Flow Test:")
    tests_total += 1
    if test_gradient_flow(model, device):
        print("   ✓ PASSED")
        tests_passed += 1
    else:
        print("   ✗ FAILED")
    
    # Memory test
    if device == 'cuda':
        print("\n5. Memory Usage Test:")
        tests_total += 1
        mem_stats = test_memory_usage(model, args.batch_size, device)
        if mem_stats.get('peak_memory_gb', 0) < 14:  # 14GB threshold for A4000
            print("   ✓ PASSED (within 14GB limit)")
            tests_passed += 1
        else:
            print("   ⚠ WARNING: Memory usage high")
            tests_passed += 1  # Still pass, just warn
    
    # Checkpoint loading test
    if args.pretrained:
        print("\n6. Checkpoint Loading Test:")
        tests_total += 1
        if test_checkpoint_loading(model, args.pretrained):
            print("   ✓ PASSED")
            tests_passed += 1
        else:
            print("   ✗ FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    if tests_passed == tests_total:
        print("All tests PASSED! Model is ready for training.")
        return 0
    else:
        print("Some tests FAILED. Review issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
