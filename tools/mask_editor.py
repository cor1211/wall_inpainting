"""
Interactive Mask Editor for Dataset Refinement.

Simple GUI tool to review and edit segmentation masks:
- View image-mask pairs
- Draw to add/remove regions
- Navigate between samples
- Export refined masks

Usage:
    python tools/mask_editor.py --input dataset/masks/ --images dataset/images/
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import argparse


class MaskEditor:
    """
    Simple mask editor with drawing capabilities.
    
    Features:
    - Overlay mask on image
    - Draw to add (left click) or remove (right click)
    - Adjustable brush size
    - Undo/redo
    - Navigate between images
    """
    
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        output_dir: Optional[Path] = None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir) if output_dir else mask_dir
        
        # Find all mask files
        self.mask_files = sorted(list(self.mask_dir.glob("*.png")))
        self.current_index = 0
        
        # Drawing state
        self.brush_size = 20
        self.drawing = False
        self.draw_mode = "add"  # "add" or "remove"
        
        # History for undo
        self.history: List[np.ndarray] = []
        self.history_index = -1
        self.max_history = 20
        
        # Current data
        self.current_image: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.display_size = (800, 600)
        
        # Build UI
        self._build_ui()
        
        # Load first image
        if self.mask_files:
            self._load_current()
    
    def _build_ui(self):
        """Build the GUI."""
        self.root = tk.Tk()
        self.root.title("Mask Editor - Dataset Refinement")
        self.root.geometry("1200x800")
        
        # Main layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Canvas
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(
            left_frame,
            width=self.display_size[0],
            height=self.display_size[1],
            bg="gray20",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)
        
        # Right panel - Controls
        right_frame = ttk.Frame(main_frame, width=200)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # File info
        info_frame = ttk.LabelFrame(right_frame, text="Current File")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_label = ttk.Label(info_frame, text="No file loaded", wraplength=180)
        self.file_label.pack(pady=5)
        
        self.index_label = ttk.Label(info_frame, text="0 / 0")
        self.index_label.pack(pady=5)
        
        # Navigation
        nav_frame = ttk.LabelFrame(right_frame, text="Navigation")
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        nav_buttons = ttk.Frame(nav_frame)
        nav_buttons.pack(pady=5)
        
        ttk.Button(nav_buttons, text="◄ Prev", command=self._prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_buttons, text="Next ►", command=self._next_image).pack(side=tk.LEFT, padx=2)
        
        # Brush settings
        brush_frame = ttk.LabelFrame(right_frame, text="Brush")
        brush_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(brush_frame, text="Size:").pack(pady=(5, 0))
        self.brush_slider = ttk.Scale(
            brush_frame,
            from_=5,
            to=100,
            orient=tk.HORIZONTAL,
            command=self._on_brush_change,
        )
        self.brush_slider.pack(fill=tk.X, padx=5, pady=5)
        
        self.brush_label = ttk.Label(brush_frame, text=f"{self.brush_size}px")
        self.brush_label.pack()
        
        # Set initial value (triggers callback, so label must exist)
        self.brush_slider.set(self.brush_size)
        
        # Draw mode
        mode_frame = ttk.LabelFrame(right_frame, text="Mode")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="add")
        ttk.Radiobutton(
            mode_frame, text="Add (Left Click)", 
            variable=self.mode_var, value="add"
        ).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Remove (Right Click)", 
            variable=self.mode_var, value="remove"
        ).pack(anchor=tk.W, padx=5)
        
        # View options
        view_frame = ttk.LabelFrame(right_frame, text="View")
        view_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            view_frame, text="Show Overlay",
            variable=self.overlay_var, command=self._update_display
        ).pack(anchor=tk.W, padx=5)
        
        self.opacity_label = ttk.Label(view_frame, text="Opacity:")
        self.opacity_label.pack(pady=(5, 0))
        
        self.opacity_slider = ttk.Scale(
            view_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            command=lambda v: self._update_display(),
        )
        self.opacity_slider.set(0.5)
        self.opacity_slider.pack(fill=tk.X, padx=5, pady=5)
        
        # Actions
        action_frame = ttk.LabelFrame(right_frame, text="Actions")
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Undo (Ctrl+Z)", command=self._undo).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(action_frame, text="Redo (Ctrl+Y)", command=self._redo).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(action_frame, text="Reset", command=self._reset_mask).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(action_frame, text="Clear Mask", command=self._clear_mask).pack(fill=tk.X, padx=5, pady=2)
        
        # Save
        save_frame = ttk.Frame(right_frame)
        save_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            save_frame, text="💾 Save Mask",
            command=self._save_mask
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            save_frame, text="Save & Next",
            command=self._save_and_next
        ).pack(fill=tk.X, pady=2)
        
        # Keyboard shortcuts
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Control-y>", lambda e: self._redo())
        self.root.bind("<Control-s>", lambda e: self._save_mask())
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<bracketleft>", lambda e: self._change_brush(-5))
        self.root.bind("<bracketright>", lambda e: self._change_brush(5))
        
        # Rapid review shortcuts
        self.root.bind("<space>", lambda e: self._save_and_next())
        self.root.bind("<Delete>", lambda e: self._clear_mask())
        self.root.bind("r", lambda e: self._reset_mask())
        self.root.bind("<Escape>", lambda e: self.root.quit())
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready (Space=Save&Next, Del=Clear, R=Reset)")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _load_current(self):
        """Load current image and mask."""
        if not self.mask_files or self.current_index >= len(self.mask_files):
            return
        
        mask_path = self.mask_files[self.current_index]
        
        # Find corresponding image
        image_path = self.image_dir / mask_path.name
        if not image_path.exists():
            # Try jpg
            image_path = self.image_dir / f"{mask_path.stem}.jpg"
        
        if not image_path.exists():
            self.status_var.set(f"Image not found for {mask_path.name}")
            return
        
        # Load
        self.current_image = np.array(Image.open(image_path).convert("RGB"))
        self.current_mask = np.array(Image.open(mask_path).convert("L"))
        
        # Reset history
        self.history = [self.current_mask.copy()]
        self.history_index = 0
        
        # Update display
        self.file_label.config(text=mask_path.name)
        self.index_label.config(text=f"{self.current_index + 1} / {len(self.mask_files)}")
        self._update_display()
        
        self.status_var.set(f"Loaded: {mask_path.name}")
    
    def _update_display(self):
        """Update canvas display."""
        if self.current_image is None:
            return
        
        # Create display image
        display = self.current_image.copy()
        
        if self.overlay_var.get() and self.current_mask is not None:
            # Create colored overlay
            opacity = self.opacity_slider.get()
            mask_rgb = np.zeros_like(display)
            mask_rgb[:, :, 1] = self.current_mask  # Green channel
            
            # Blend
            mask_bool = self.current_mask > 127
            display[mask_bool] = (
                display[mask_bool] * (1 - opacity) +
                mask_rgb[mask_bool] * opacity
            ).astype(np.uint8)
        
        # Resize for display
        h, w = display.shape[:2]
        scale = min(self.display_size[0] / w, self.display_size[1] / h)
        new_size = (int(w * scale), int(h * scale))
        
        self.display_scale = scale
        self.display_offset = (0, 0)
        
        # Convert to PhotoImage
        img = Image.fromarray(display)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
    def _canvas_to_image(self, x: int, y: int) -> Tuple[int, int]:
        """Convert canvas coordinates to image coordinates."""
        img_x = int(x / self.display_scale)
        img_y = int(y / self.display_scale)
        return img_x, img_y
    
    def _draw_at(self, x: int, y: int, add: bool = True):
        """Draw on mask at position."""
        if self.current_mask is None:
            return
        
        img_x, img_y = self._canvas_to_image(x, y)
        
        # Draw circle on mask
        h, w = self.current_mask.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((x_coords - img_x) ** 2 + (y_coords - img_y) ** 2)
        
        if add:
            self.current_mask[dist <= self.brush_size / 2] = 255
        else:
            self.current_mask[dist <= self.brush_size / 2] = 0
        
        self._update_display()
    
    def _save_to_history(self):
        """Save current mask to history."""
        # Truncate future history
        self.history = self.history[:self.history_index + 1]
        
        # Add current state
        self.history.append(self.current_mask.copy())
        self.history_index = len(self.history) - 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.history_index = len(self.history) - 1
    
    def _on_left_click(self, event):
        self.drawing = True
        self.draw_mode = "add"
        self._save_to_history()
        self._draw_at(event.x, event.y, add=True)
    
    def _on_left_drag(self, event):
        if self.drawing:
            self._draw_at(event.x, event.y, add=True)
    
    def _on_right_click(self, event):
        self.drawing = True
        self.draw_mode = "remove"
        self._save_to_history()
        self._draw_at(event.x, event.y, add=False)
    
    def _on_right_drag(self, event):
        if self.drawing:
            self._draw_at(event.x, event.y, add=False)
    
    def _on_release(self, event):
        self.drawing = False
    
    def _on_brush_change(self, value):
        self.brush_size = int(float(value))
        self.brush_label.config(text=f"{self.brush_size}px")
    
    def _change_brush(self, delta: int):
        new_size = max(5, min(100, self.brush_size + delta))
        self.brush_size = new_size
        self.brush_slider.set(new_size)
        self.brush_label.config(text=f"{self.brush_size}px")
    
    def _undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_mask = self.history[self.history_index].copy()
            self._update_display()
            self.status_var.set("Undo")
    
    def _redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_mask = self.history[self.history_index].copy()
            self._update_display()
            self.status_var.set("Redo")
    
    def _reset_mask(self):
        """Reset to original mask."""
        if self.history:
            self.current_mask = self.history[0].copy()
            self._update_display()
            self.status_var.set("Reset to original")
    
    def _clear_mask(self):
        """Clear entire mask."""
        self._save_to_history()
        self.current_mask = np.zeros_like(self.current_mask)
        self._update_display()
        self.status_var.set("Mask cleared")
    
    def _save_mask(self):
        """Save current mask."""
        if self.current_mask is None:
            return
        
        mask_path = self.mask_files[self.current_index]
        output_path = self.output_dir / mask_path.name
        
        Image.fromarray(self.current_mask).save(output_path)
        self.status_var.set(f"Saved: {output_path.name}")
    
    def _save_and_next(self):
        """Save mask and go to next."""
        self._save_mask()
        self._next_image()
    
    def _prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current()
    
    def _next_image(self):
        if self.current_index < len(self.mask_files) - 1:
            self.current_index += 1
            self._load_current()
    
    def run(self):
        """Start the editor."""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Interactive mask editor")
    parser.add_argument("--masks", "-m", type=str, required=True, help="Mask directory")
    parser.add_argument("--images", "-i", type=str, required=True, help="Image directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory (default: same as masks)")
    
    args = parser.parse_args()
    
    editor = MaskEditor(
        image_dir=Path(args.images),
        mask_dir=Path(args.masks),
        output_dir=Path(args.output) if args.output else None,
    )
    
    editor.run()


if __name__ == "__main__":
    main()
