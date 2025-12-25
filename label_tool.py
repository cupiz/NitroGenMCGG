"""
Coordinate Labeling Tool
=========================
Simple GUI tool to annotate click coordinates on extracted frames.
Click on frames to record the (x, y) coordinates for training.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from PIL import Image, ImageTk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("Warning: tkinter not available. Install it for GUI labeling.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelingTool:
    """
    GUI tool for labeling click coordinates on game frames.
    
    Features:
    - Click on frame to mark coordinates
    - Navigate with arrow keys or buttons
    - Auto-save progress
    - Skip frames without clicks
    - Review and edit existing labels
    """
    
    def __init__(
        self,
        frames_dir: str,
        labels_file: str,
        metadata_file: Optional[str] = None,
        display_size: int = 512
    ):
        if not HAS_TKINTER:
            raise RuntimeError("tkinter is required for the labeling tool")
        
        self.frames_dir = Path(frames_dir)
        self.labels_file = Path(labels_file)
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.display_size = display_size
        
        # Load frame list
        self.frames = self._load_frames()
        self.current_idx = 0
        
        # Load existing labels
        self.labels = self._load_labels()
        
        # Track modifications
        self.modified = False
        self.last_save_time = datetime.now()
        
        # Setup GUI
        self._setup_gui()
    
    def _load_frames(self) -> List[str]:
        """Load list of frame files."""
        extensions = ('.png', '.jpg', '.jpeg')
        frames = []
        
        for ext in extensions:
            frames.extend(self.frames_dir.glob(f"*{ext}"))
        
        frames = sorted([f.name for f in frames])
        logger.info(f"Found {len(frames)} frames in {self.frames_dir}")
        
        return frames
    
    def _load_labels(self) -> Dict[str, Dict]:
        """Load existing labels from file."""
        labels = {}
        
        if self.labels_file.exists():
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    frame = item.get('frame') or item.get('filename')
                    if frame:
                        labels[frame] = item
            
            logger.info(f"Loaded {len(labels)} existing labels")
        
        return labels
    
    def _save_labels(self):
        """Save labels to file."""
        # Convert to list format
        labels_list = list(self.labels.values())
        
        # Sort by frame name
        labels_list.sort(key=lambda x: x.get('frame', ''))
        
        # Save
        with open(self.labels_file, 'w', encoding='utf-8') as f:
            json.dump(labels_list, f, indent=2, ensure_ascii=False)
        
        self.modified = False
        self.last_save_time = datetime.now()
        logger.info(f"Saved {len(labels_list)} labels to {self.labels_file}")
    
    def _setup_gui(self):
        """Setup the tkinter GUI."""
        self.root = tk.Tk()
        self.root.title("NitroGen - Coordinate Labeling Tool")
        self.root.configure(bg='#2b2b2b')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style
        style = ttk.Style()
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', padding=5)
        
        # Image canvas
        self.canvas = tk.Canvas(
            main_frame,
            width=self.display_size,
            height=self.display_size,
            bg='#1e1e1e',
            highlightthickness=0
        )
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self._on_click)
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_label = ttk.Label(
            info_frame,
            text="Click on the frame to mark coordinates",
            font=('Arial', 11)
        )
        self.info_label.pack(side=tk.LEFT)
        
        self.progress_label = ttk.Label(
            info_frame,
            text="0 / 0",
            font=('Arial', 11)
        )
        self.progress_label.pack(side=tk.RIGHT)
        
        # Coordinate display
        coord_frame = ttk.Frame(main_frame)
        coord_frame.pack(fill=tk.X, pady=5)
        
        self.coord_label = ttk.Label(
            coord_frame,
            text="Coordinates: Not set",
            font=('Arial', 12, 'bold')
        )
        self.coord_label.pack()
        
        # Navigation buttons
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(nav_frame, text="◀ Prev (A)", command=self._prev_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next (D) ▶", command=self._next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Skip (S)", command=self._skip_frame).pack(side=tk.LEFT, padx=20)
        ttk.Button(nav_frame, text="Clear (C)", command=self._clear_label).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Save (Ctrl+S)", command=self._save_labels).pack(side=tk.RIGHT, padx=2)
        
        # Jump to frame
        jump_frame = ttk.Frame(main_frame)
        jump_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(jump_frame, text="Go to frame:").pack(side=tk.LEFT)
        self.jump_entry = ttk.Entry(jump_frame, width=10)
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(jump_frame, text="Go", command=self._jump_to_frame).pack(side=tk.LEFT)
        
        # Status bar
        self.status_label = ttk.Label(
            main_frame,
            text="Ready. Use A/D keys to navigate, click to label.",
            font=('Arial', 10)
        )
        self.status_label.pack(fill=tk.X, pady=5)
        
        # Key bindings
        self.root.bind('<Left>', lambda e: self._prev_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('a', lambda e: self._prev_frame())
        self.root.bind('d', lambda e: self._next_frame())
        self.root.bind('s', lambda e: self._skip_frame())
        self.root.bind('c', lambda e: self._clear_label())
        self.root.bind('<Control-s>', lambda e: self._save_labels())
        self.root.bind('<Escape>', lambda e: self._on_close())
        
        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Load first frame
        self._load_frame()
    
    def _load_frame(self):
        """Load and display current frame."""
        if not self.frames:
            self.info_label.config(text="No frames found!")
            return
        
        frame_name = self.frames[self.current_idx]
        frame_path = self.frames_dir / frame_name
        
        # Load image
        image = Image.open(frame_path)
        self.original_size = image.size
        
        # Resize for display
        image = image.resize(
            (self.display_size, self.display_size),
            Image.Resampling.LANCZOS
        )
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Draw existing label if present
        if frame_name in self.labels:
            label = self.labels[frame_name]
            x, y = label['x'], label['y']
            self._draw_marker(x * self.display_size, y * self.display_size)
            self.coord_label.config(text=f"Coordinates: ({x:.4f}, {y:.4f})")
        else:
            self.coord_label.config(text="Coordinates: Not set")
        
        # Update info
        labeled_count = len(self.labels)
        self.progress_label.config(
            text=f"{self.current_idx + 1} / {len(self.frames)} ({labeled_count} labeled)"
        )
        self.info_label.config(text=f"Frame: {frame_name}")
    
    def _draw_marker(self, x: float, y: float, color: str = 'red'):
        """Draw a marker at the specified position."""
        r = 8  # Marker radius
        
        # Crosshair
        self.canvas.create_line(x - r*2, y, x + r*2, y, fill=color, width=2, tags="marker")
        self.canvas.create_line(x, y - r*2, x, y + r*2, fill=color, width=2, tags="marker")
        
        # Circle
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline=color, width=2, tags="marker"
        )
    
    def _on_click(self, event):
        """Handle click on canvas to set coordinates."""
        if not self.frames:
            return
        
        # Get normalized coordinates
        x = event.x / self.display_size
        y = event.y / self.display_size
        
        # Clamp to [0, 1]
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        
        # Update label
        frame_name = self.frames[self.current_idx]
        self.labels[frame_name] = {
            "frame": frame_name,
            "x": round(x, 4),
            "y": round(y, 4),
            "click": True
        }
        
        self.modified = True
        
        # Redraw
        self.canvas.delete("marker")
        self._draw_marker(event.x, event.y)
        
        self.coord_label.config(text=f"Coordinates: ({x:.4f}, {y:.4f})")
        self.status_label.config(text=f"Marked at ({x:.4f}, {y:.4f})")
        
        # Auto-save every 50 labels
        if len(self.labels) % 50 == 0:
            self._save_labels()
            self.status_label.config(text="Auto-saved!")
    
    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_frame()
    
    def _next_frame(self):
        """Go to next frame."""
        if self.current_idx < len(self.frames) - 1:
            self.current_idx += 1
            self._load_frame()
    
    def _skip_frame(self):
        """Skip current frame (mark as no-click) and go to next."""
        frame_name = self.frames[self.current_idx]
        self.labels[frame_name] = {
            "frame": frame_name,
            "x": 0.5,
            "y": 0.5,
            "click": False
        }
        self.modified = True
        self.status_label.config(text=f"Skipped: {frame_name}")
        self._next_frame()
    
    def _clear_label(self):
        """Clear label for current frame."""
        frame_name = self.frames[self.current_idx]
        if frame_name in self.labels:
            del self.labels[frame_name]
            self.modified = True
            self.canvas.delete("marker")
            self.coord_label.config(text="Coordinates: Not set")
            self.status_label.config(text="Label cleared")
    
    def _jump_to_frame(self):
        """Jump to a specific frame number."""
        try:
            frame_num = int(self.jump_entry.get()) - 1
            if 0 <= frame_num < len(self.frames):
                self.current_idx = frame_num
                self._load_frame()
            else:
                messagebox.showwarning("Invalid", f"Frame number must be 1-{len(self.frames)}")
        except ValueError:
            messagebox.showwarning("Invalid", "Please enter a valid number")
    
    def _on_close(self):
        """Handle window close."""
        if self.modified:
            result = messagebox.askyesnocancel(
                "Save Changes",
                "You have unsaved changes. Save before closing?"
            )
            if result is True:
                self._save_labels()
            elif result is None:
                return  # Cancel close
        
        self.root.destroy()
    
    def run(self):
        """Start the labeling tool."""
        logger.info("Starting labeling tool...")
        logger.info("Controls:")
        logger.info("  - Click on frame to mark coordinates")
        logger.info("  - A/Left Arrow: Previous frame")
        logger.info("  - D/Right Arrow: Next frame")
        logger.info("  - S: Skip frame (no click)")
        logger.info("  - C: Clear current label")
        logger.info("  - Ctrl+S: Save labels")
        
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Coordinate labeling tool for NitroGen training"
    )
    
    parser.add_argument(
        "--frames-dir", "-f",
        type=str,
        default="data/frames",
        help="Directory containing extracted frames"
    )
    
    parser.add_argument(
        "--labels-file", "-l",
        type=str,
        default="data/labels.json",
        help="Path to save/load labels JSON"
    )
    
    parser.add_argument(
        "--metadata-file", "-m",
        type=str,
        default=None,
        help="Optional metadata file from video extraction"
    )
    
    parser.add_argument(
        "--display-size", "-s",
        type=int,
        default=512,
        help="Display size for frames (default: 512)"
    )
    
    args = parser.parse_args()
    
    if not HAS_TKINTER:
        print("ERROR: tkinter is required for the labeling tool.")
        print("Install it with your system package manager:")
        print("  - Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  - Windows: Usually included with Python")
        print("  - macOS: Usually included with Python")
        return
    
    # Check frames directory
    if not Path(args.frames_dir).exists():
        print(f"ERROR: Frames directory not found: {args.frames_dir}")
        print("Run preprocess_video.py first to extract frames.")
        return
    
    # Create labels file if not exists
    labels_path = Path(args.labels_file)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    if not labels_path.exists():
        with open(labels_path, 'w') as f:
            json.dump([], f)
    
    # Start labeling tool
    tool = LabelingTool(
        frames_dir=args.frames_dir,
        labels_file=args.labels_file,
        metadata_file=args.metadata_file,
        display_size=args.display_size
    )
    tool.run()


if __name__ == "__main__":
    main()
