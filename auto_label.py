"""
Auto-Label: Mouse Cursor Detection from Video
==============================================
Automatically detect mouse cursor position in gameplay videos
and generate labels.json for training.

Supports:
- Standard Windows/Mac mouse cursors (template matching)
- Touch indicators (colored circles)
- Custom cursor detection
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MouseDetector:
    """
    Detect mouse cursor position in video frames.
    
    Uses multiple detection methods:
    1. Template matching (for standard cursors)
    2. Color-based detection (for touch indicators/circles)
    3. Motion-based detection (for moving cursor)
    """
    
    def __init__(
        self,
        method: str = "auto",
        cursor_color: Tuple[int, int, int] = None,
        color_tolerance: int = 30,
        min_area: int = 50,
        max_area: int = 5000
    ):
        """
        Args:
            method: Detection method - "template", "color", "motion", or "auto"
            cursor_color: BGR color of cursor/indicator (for color method)
            color_tolerance: Color matching tolerance
            min_area: Minimum contour area for detection
            max_area: Maximum contour area for detection
        """
        self.method = method
        self.cursor_color = cursor_color
        self.color_tolerance = color_tolerance
        self.min_area = min_area
        self.max_area = max_area
        
        # Previous frame for motion detection
        self.prev_frame = None
        self.prev_position = None
        
        # Default cursor templates (will be created on demand)
        self.templates = []
    
    def _create_cursor_templates(self):
        """Create common cursor shape templates."""
        templates = []
        
        # Arrow cursor (pointing upper-left)
        arrow = np.zeros((24, 24), dtype=np.uint8)
        pts = np.array([[0, 0], [0, 18], [5, 14], [8, 22], [12, 20], [9, 13], [16, 13]], np.int32)
        cv2.fillPoly(arrow, [pts], 255)
        templates.append(("arrow", arrow))
        
        # Simple pointer (small triangle)
        pointer = np.zeros((16, 16), dtype=np.uint8)
        pts = np.array([[0, 0], [0, 12], [8, 8]], np.int32)
        cv2.fillPoly(pointer, [pts], 255)
        templates.append(("pointer", pointer))
        
        # Circle cursor (common for touch)
        circle = np.zeros((32, 32), dtype=np.uint8)
        cv2.circle(circle, (16, 16), 12, 255, 2)
        templates.append(("circle", circle))
        
        # Filled circle (touch indicator)
        filled_circle = np.zeros((32, 32), dtype=np.uint8)
        cv2.circle(filled_circle, (16, 16), 10, 255, -1)
        templates.append(("filled_circle", filled_circle))
        
        self.templates = templates
    
    def detect_by_template(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect cursor using template matching."""
        if not self.templates:
            self._create_cursor_templates()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_match = None
        best_val = 0.5  # Minimum threshold
        
        for name, template in self.templates:
            # Try multiple scales
            for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
                scaled = cv2.resize(template, None, fx=scale, fy=scale)
                if scaled.shape[0] > gray.shape[0] or scaled.shape[1] > gray.shape[1]:
                    continue
                
                result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_val:
                    best_val = max_val
                    # Cursor tip is at top-left for arrow
                    best_match = (max_loc[0], max_loc[1])
        
        return best_match
    
    def detect_by_color(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect cursor/indicator by color."""
        if self.cursor_color is None:
            # Try common cursor colors: red, yellow, white
            colors_to_try = [
                (0, 0, 255),    # Red
                (0, 255, 255),  # Yellow
                (255, 255, 255), # White
                (0, 255, 0),    # Green
                (255, 0, 255),  # Magenta
            ]
            
            for color in colors_to_try:
                result = self._find_color(frame, color)
                if result:
                    return result
            return None
        else:
            return self._find_color(frame, self.cursor_color)
    
    def _find_color(
        self,
        frame: np.ndarray,
        target_color: Tuple[int, int, int]
    ) -> Optional[Tuple[float, float]]:
        """Find regions matching target color."""
        # Convert to HSV for better color matching
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target_hsv = cv2.cvtColor(
            np.uint8([[target_color]]), cv2.COLOR_BGR2HSV
        )[0][0]
        
        # Create color range
        lower = np.array([
            max(0, target_hsv[0] - self.color_tolerance),
            max(0, target_hsv[1] - 50),
            max(0, target_hsv[2] - 50)
        ])
        upper = np.array([
            min(179, target_hsv[0] + self.color_tolerance),
            min(255, target_hsv[1] + 50),
            min(255, target_hsv[2] + 50)
        ])
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find best contour (by area and circularity)
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            score = circularity * (area / self.max_area)
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return (cx, cy)
        
        return None
    
    def detect_by_motion(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect cursor by motion (moving object)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return None
        
        # Frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.prev_frame = gray
        
        # Find centroid of motion
        if contours:
            all_points = np.vstack(contours)
            M = cv2.moments(all_points)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"] 
                return (cx, cy)
        
        return None
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect mouse cursor position in frame.
        
        Returns:
            (x, y) pixel coordinates or None if not detected
        """
        if self.method == "template":
            return self.detect_by_template(frame)
        elif self.method == "color":
            return self.detect_by_color(frame)
        elif self.method == "motion":
            return self.detect_by_motion(frame)
        else:  # auto
            # Try methods in order of reliability
            result = self.detect_by_color(frame)
            if result:
                return result
            
            result = self.detect_by_template(frame)
            if result:
                return result
            
            return self.detect_by_motion(frame)


def process_video(
    video_path: str,
    output_dir: str,
    fps: int = 5,
    detector: MouseDetector = None,
    save_frames: bool = True,
    target_size: Tuple[int, int] = (256, 256),
    visualize: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Process video and extract frames with mouse positions.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for frames and labels
        fps: Frames per second to extract
        detector: MouseDetector instance
        save_frames: Whether to save frame images
        target_size: Target frame size
        visualize: Show detection visualization
    
    Returns:
        Tuple of (labels_list, stats_dict)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    
    if detector is None:
        detector = MouseDetector(method="auto")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {video_path.name}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {video_fps:.2f}, Extracting at: {fps} FPS")
    logger.info(f"  Total frames: {total_frames}")
    
    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))
    expected_frames = total_frames // frame_interval
    
    labels = []
    stats = {
        "total_extracted": 0,
        "detected": 0,
        "not_detected": 0
    }
    
    frame_count = 0
    extracted_count = 0
    
    pbar = tqdm(total=expected_frames, desc="Processing", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Detect mouse position
            position = detector.detect(frame)
            
            # Generate frame filename
            frame_name = f"frame_{extracted_count:06d}.png"
            
            if position:
                # Normalize coordinates to [0, 1]
                x_norm = position[0] / width
                y_norm = position[1] / height
                
                labels.append({
                    "frame": frame_name,
                    "x": round(x_norm, 4),
                    "y": round(y_norm, 4),
                    "click": True,
                    "raw_x": int(position[0]),
                    "raw_y": int(position[1])
                })
                stats["detected"] += 1
                
                # Visualize detection
                if visualize:
                    vis_frame = frame.copy()
                    cv2.circle(vis_frame, (int(position[0]), int(position[1])), 10, (0, 255, 0), 2)
                    cv2.imshow("Detection", cv2.resize(vis_frame, (640, 480)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                # No detection - skip or add with null coords
                stats["not_detected"] += 1
            
            # Save frame (resized)
            if save_frames and position:  # Only save frames with detected cursor
                frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(frames_dir / frame_name), frame_resized)
            
            extracted_count += 1
            stats["total_extracted"] = extracted_count
            pbar.update(1)
        
        frame_count += 1
    
    pbar.close()
    cap.release()
    
    if visualize:
        cv2.destroyAllWindows()
    
    # Save labels
    labels_file = output_dir / "labels.json"
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    
    logger.info(f"\nResults:")
    logger.info(f"  Total extracted: {stats['total_extracted']}")
    logger.info(f"  Cursor detected: {stats['detected']} ({100*stats['detected']/max(1,stats['total_extracted']):.1f}%)")
    logger.info(f"  Not detected: {stats['not_detected']}")
    logger.info(f"  Labels saved to: {labels_file}")
    
    return labels, stats


def analyze_cursor_color(video_path: str, num_samples: int = 10):
    """
    Analyze video to find common cursor colors.
    Helps user identify the right color for detection.
    """
    logger.info(f"Analyzing cursor colors in {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    print("\nClick on the cursor/touch indicator in the popup windows.")
    print("Press 'q' to skip a frame, 's' to save the color.")
    
    colors_found = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Display frame
        display = cv2.resize(frame, (800, 600))
        
        clicked_color = [None]
        
        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Get color at click position
                scale_x = frame.shape[1] / 800
                scale_y = frame.shape[0] / 600
                orig_x = int(x * scale_x)
                orig_y = int(y * scale_y)
                color = frame[orig_y, orig_x].tolist()
                clicked_color[0] = color
                print(f"  Selected color (BGR): {color}")
        
        cv2.imshow("Click on cursor", display)
        cv2.setMouseCallback("Click on cursor", on_click)
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and clicked_color[0]:
                colors_found.append(clicked_color[0])
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if colors_found:
        # Average the colors
        avg_color = np.mean(colors_found, axis=0).astype(int).tolist()
        print(f"\nRecommended cursor color (BGR): {avg_color}")
        print(f"Use: --cursor-color {avg_color[0]} {avg_color[1]} {avg_color[2]}")
    
    return colors_found


def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect mouse cursor in gameplay videos"
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data",
        help="Output directory for frames and labels"
    )
    
    parser.add_argument(
        "--fps", "-f",
        type=int,
        default=5,
        help="Frames per second to extract (default: 5)"
    )
    
    parser.add_argument(
        "--method", "-m",
        choices=["auto", "color", "template", "motion"],
        default="auto",
        help="Detection method (default: auto)"
    )
    
    parser.add_argument(
        "--cursor-color",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        help="Cursor color in BGR format (e.g., 0 0 255 for red)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show detection visualization"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze video to find cursor color (interactive)"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Output frame size (default: 256)"
    )
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_cursor_color(args.video)
        return
    
    # Create detector
    cursor_color = tuple(args.cursor_color) if args.cursor_color else None
    detector = MouseDetector(
        method=args.method,
        cursor_color=cursor_color
    )
    
    # Process video
    labels, stats = process_video(
        video_path=args.video,
        output_dir=args.output_dir,
        fps=args.fps,
        detector=detector,
        target_size=(args.size, args.size),
        visualize=args.visualize
    )
    
    print(f"\n✓ Done! Generated {len(labels)} labeled frames.")
    print(f"  Frames: {args.output_dir}/frames/")
    print(f"  Labels: {args.output_dir}/labels.json")


if __name__ == "__main__":
    main()
