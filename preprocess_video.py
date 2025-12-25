"""
Video Preprocessing for NitroGen Training
==========================================
Extract frames from MP4 gameplay videos for training.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: int = 10,
    target_size: Tuple[int, int] = (256, 256),
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    prefix: str = "frame"
) -> List[dict]:
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to input MP4 video
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 10)
        target_size: Resize frames to this size (default: 256x256 for NitroGen)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        prefix: Filename prefix for frames
    
    Returns:
        List of frame metadata dictionaries
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {video_path.name}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {video_fps:.2f}")
    logger.info(f"  Duration: {duration:.2f}s ({total_frames} frames)")
    logger.info(f"  Extracting at: {fps} FPS")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1
        logger.warning(f"Requested FPS ({fps}) > video FPS ({video_fps}), extracting all frames")
    
    # Calculate frame range
    start_frame = 0
    end_frame = total_frames
    
    if start_time is not None:
        start_frame = int(start_time * video_fps)
    if end_time is not None:
        end_frame = int(end_time * video_fps)
    
    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(start_frame, min(end_frame, total_frames))
    
    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames
    frames_metadata = []
    frame_count = 0
    extracted_count = 0
    
    expected_frames = (end_frame - start_frame) // frame_interval
    pbar = tqdm(total=expected_frames, desc="Extracting frames", unit="frame")
    
    while True:
        ret, frame = cap.read()
        current_frame = start_frame + frame_count
        
        if not ret or current_frame >= end_frame:
            break
        
        # Only save at specified interval
        if frame_count % frame_interval == 0:
            # Resize to target size
            frame_resized = cv2.resize(
                frame, target_size, 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Generate filename
            filename = f"{prefix}_{extracted_count:06d}.png"
            filepath = output_dir / filename
            
            # Save frame
            cv2.imwrite(str(filepath), frame_resized)
            
            # Record metadata
            timestamp = current_frame / video_fps
            frames_metadata.append({
                "frame": filename,
                "index": extracted_count,
                "timestamp": round(timestamp, 3),
                "original_frame": current_frame,
                "x": None,  # To be filled during labeling
                "y": None,  # To be filled during labeling
                "click": None  # Whether mouse was clicked
            })
            
            extracted_count += 1
            pbar.update(1)
        
        frame_count += 1
    
    pbar.close()
    cap.release()
    
    logger.info(f"Extracted {extracted_count} frames to {output_dir}")
    
    return frames_metadata


def save_metadata(
    frames_metadata: List[dict],
    output_path: str,
    video_info: dict = None
):
    """
    Save frame metadata to JSON file.
    
    Args:
        frames_metadata: List of frame metadata
        output_path: Path to save JSON file
        video_info: Optional video information
    """
    output_path = Path(output_path)
    
    metadata = {
        "created": datetime.now().isoformat(),
        "total_frames": len(frames_metadata),
        "video_info": video_info or {},
        "frames": frames_metadata
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved metadata to {output_path}")


def process_video(
    video_path: str,
    output_dir: str,
    fps: int = 10,
    target_size: Tuple[int, int] = (256, 256)
) -> str:
    """
    Process a video file: extract frames and create metadata.
    
    Args:
        video_path: Path to input video
        output_dir: Base output directory
        fps: Frames per second to extract
        target_size: Target frame size
    
    Returns:
        Path to metadata JSON file
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    # Create output directories
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames
    video_name = video_path.stem
    frames_metadata = extract_frames(
        video_path=str(video_path),
        output_dir=str(frames_dir),
        fps=fps,
        target_size=target_size,
        prefix=video_name
    )
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    video_info = {
        "filename": video_path.name,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "extraction_fps": fps,
        "target_size": list(target_size)
    }
    cap.release()
    
    # Save metadata
    metadata_path = output_dir / f"{video_name}_metadata.json"
    save_metadata(frames_metadata, str(metadata_path), video_info)
    
    # Also create an empty labels file template
    labels_path = output_dir / f"{video_name}_labels.json"
    if not labels_path.exists():
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        logger.info(f"Created empty labels template at {labels_path}")
    
    return str(metadata_path)


def process_multiple_videos(
    video_dir: str,
    output_dir: str,
    fps: int = 10,
    extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mkv', '.mov')
):
    """
    Process multiple video files in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Base output directory
        fps: Frames per second to extract
        extensions: Video file extensions to process
    """
    video_dir = Path(video_dir)
    
    # Find all video files
    video_files = []
    for ext in extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        logger.warning(f"No video files found in {video_dir}")
        return
    
    logger.info(f"Found {len(video_files)} video files")
    
    for video_path in video_files:
        logger.info(f"\nProcessing: {video_path.name}")
        try:
            process_video(
                video_path=str(video_path),
                output_dir=output_dir,
                fps=fps
            )
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from gameplay videos for NitroGen training"
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--video-dir", "-d",
        type=str,
        help="Directory containing multiple video files"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data",
        help="Output directory for frames and metadata (default: data)"
    )
    
    parser.add_argument(
        "--fps", "-f",
        type=int,
        default=10,
        help="Frames per second to extract (default: 10)"
    )
    
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=256,
        help="Target frame size (default: 256 for NitroGen)"
    )
    
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    
    if args.video:
        # Process single video
        process_video(
            video_path=args.video,
            output_dir=args.output_dir,
            fps=args.fps,
            target_size=target_size
        )
    elif args.video_dir:
        # Process multiple videos
        process_multiple_videos(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            fps=args.fps
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python preprocess_video.py --video gameplay.mp4 --output-dir data --fps 10")
        print("  python preprocess_video.py --video-dir videos/ --output-dir data")


if __name__ == "__main__":
    main()
