"""
Realtime Inference for Magic Chess
===================================
Capture screen, predict click coordinates, and optionally auto-click.
"""

import time
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import numpy as np
from PIL import Image

# Screen capture
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    print("Install mss for screen capture: pip install mss")

# Mouse control (optional)
try:
    import pyautogui
    HAS_PYAUTOGUI = True
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
except ImportError:
    HAS_PYAUTOGUI = False

from model import create_model
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameAgent:
    """
    Realtime game agent that captures screen and predicts click positions.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        screen_region: Optional[Tuple[int, int, int, int]] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model_path: Path to trained checkpoint (best_model.pth)
            device: cuda or cpu
            screen_region: (left, top, width, height) or None for full screen
            confidence_threshold: Minimum confidence to execute click
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.screen_region = screen_region
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Screen capture
        if HAS_MSS:
            self.sct = mss.mss()
        
        logger.info(f"Agent initialized on {self.device}")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {model_path}")
        
        model = create_model(freeze_encoder=True)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    def capture_screen(self) -> Image.Image:
        """Capture current screen."""
        if not HAS_MSS:
            raise RuntimeError("mss not installed. Run: pip install mss")
        
        if self.screen_region:
            monitor = {
                "left": self.screen_region[0],
                "top": self.screen_region[1],
                "width": self.screen_region[2],
                "height": self.screen_region[3]
            }
        else:
            monitor = self.sct.monitors[1]  # Primary monitor
        
        screenshot = self.sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        return img
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[float, float]:
        """
        Predict click coordinates from image.
        
        Args:
            image: PIL Image
        
        Returns:
            (x, y) normalized coordinates in [0, 1]
        """
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        coords = self.model(img_tensor)
        
        x, y = coords[0].cpu().numpy()
        return float(x), float(y)
    
    def predict_screen_coords(self) -> Tuple[int, int]:
        """
        Capture screen and return absolute pixel coordinates.
        
        Returns:
            (x, y) absolute screen coordinates
        """
        # Capture
        image = self.capture_screen()
        
        # Predict normalized coords
        x_norm, y_norm = self.predict(image)
        
        # Convert to absolute coords
        if self.screen_region:
            abs_x = self.screen_region[0] + int(x_norm * self.screen_region[2])
            abs_y = self.screen_region[1] + int(y_norm * self.screen_region[3])
        else:
            screen_w, screen_h = image.size
            abs_x = int(x_norm * screen_w)
            abs_y = int(y_norm * screen_h)
        
        return abs_x, abs_y
    
    def click(self, x: int, y: int):
        """Execute mouse click at position."""
        if not HAS_PYAUTOGUI:
            logger.warning("pyautogui not installed. Cannot click.")
            return
        
        pyautogui.click(x, y)
        logger.debug(f"Clicked at ({x}, {y})")
    
    def run_loop(
        self,
        fps: float = 2,
        auto_click: bool = False,
        duration: Optional[float] = None
    ):
        """
        Run inference loop.
        
        Args:
            fps: Predictions per second
            auto_click: Whether to auto-click predicted positions
            duration: Run for N seconds (None = run forever)
        """
        logger.info(f"Starting inference loop at {fps} FPS")
        logger.info(f"Auto-click: {auto_click}")
        logger.info("Press Ctrl+C to stop")
        
        if auto_click and HAS_PYAUTOGUI:
            logger.warning("FAILSAFE: Move mouse to top-left corner to abort!")
        
        interval = 1.0 / fps
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                loop_start = time.time()
                
                # Predict
                x, y = self.predict_screen_coords()
                
                # Log
                logger.info(f"Frame {frame_count}: Predicted click at ({x}, {y})")
                
                # Click if enabled
                if auto_click:
                    self.click(x, y)
                
                frame_count += 1
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Maintain FPS
                elapsed = time.time() - loop_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                    
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        
        total_time = time.time() - start_time
        logger.info(f"Processed {frame_count} frames in {total_time:.1f}s ({frame_count/total_time:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Run NitroGen agent on game")
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best_model.pth",
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=2,
        help="Predictions per second (default: 2)"
    )
    
    parser.add_argument(
        "--auto-click",
        action="store_true",
        help="Enable auto-clicking (use with caution!)"
    )
    
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Screen region to capture (default: full screen)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for N seconds (default: run forever)"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAS_MSS:
        print("ERROR: mss required for screen capture")
        print("Install: pip install mss")
        return
    
    if args.auto_click and not HAS_PYAUTOGUI:
        print("ERROR: pyautogui required for auto-click")
        print("Install: pip install pyautogui")
        return
    
    # Create agent
    agent = GameAgent(
        model_path=args.model,
        device="cpu" if args.cpu else "cuda",
        screen_region=tuple(args.region) if args.region else None
    )
    
    # Run
    agent.run_loop(
        fps=args.fps,
        auto_click=args.auto_click,
        duration=args.duration
    )


if __name__ == "__main__":
    main()
