#!/usr/bin/env python3
"""
Dual Board Manual Annotation Tool with Green Tape Detection

This tool helps manually annotate snap circuit images that contain two circuit boards
side by side. It includes detection of green tape coverage to filter images where
hands are blocking the components.

Features:
- Dual board annotation support with split-screen view
- Green tape coverage detection
- Data augmentation pipeline (rotations: 90°, 180°, 270°, color variations)
- YOLO format annotation export
- Keyboard shortcuts for efficient annotation
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math
import colorsys
from dataclasses import dataclass
import time


@dataclass
class Annotation:
    """Single component annotation"""
    class_id: int
    class_name: str
    center_x: float  # Normalized [0-1]
    center_y: float  # Normalized [0-1]
    width: float     # Normalized [0-1]
    height: float    # Normalized [0-1]
    side: str        # 'left' or 'right'
    
    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.center_x:.6f} {self.center_y:.6f} {self.width:.6f} {self.height:.6f}"


@dataclass
class GreenTapeStatus:
    """Green tape detection results"""
    left_side_covered: bool
    right_side_covered: bool
    coverage_percentage: float
    should_skip: bool


class DualBoardAnnotator:
    """Advanced dual board annotation tool"""
    
    def __init__(self, classes_file: str):
        """Initialize the annotator"""
        # Load classes
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Current state
        self.current_class = 0
        self.current_side = "left"  # Which side we're annotating
        self.split_ratio = 0.5      # Where to split the image
        self.drawing = False
        self.start_point = None
        self.annotations = []
        
        # Image data
        self.original_image = None
        self.display_image = None
        self.image_path = None
        self.height = 0
        self.width = 0
        
        # Green tape detection parameters
        self.green_lower = np.array([40, 50, 50])    # HSV lower bound for green
        self.green_upper = np.array([80, 255, 255])  # HSV upper bound for green
        self.tape_coverage_threshold = 0.02          # Minimum tape area ratio to be considered "covered"
        
        print("🎯 Dual Board Annotator Initialized")
        print(f"📋 Classes ({len(self.classes)}): {', '.join(self.classes[:5])}{'...' if len(self.classes) > 5 else ''}")
        print(f"🎮 Current class: {self.classes[self.current_class]}")
        print(f"🔄 Current side: {self.current_side}")
        self._print_controls()
    
    def _print_controls(self):
        """Print control instructions"""
        print("\n🎮 CONTROLS:")
        print("  Mouse:")
        print("    • Click + drag: Draw bounding box")
        print("    • Right click: Delete last annotation")
        print("  Keyboard:")
        print("    • Numbers 0-9: Change class")
        print("    • TAB: Switch between left/right side")
        print("    • SPACE: Toggle green tape detection")
        print("    • 's': Save annotations and continue")
        print("    • 'r': Reset current image")
        print("    • 'q': Quit")
        print("    • 'h': Show this help")
        print("    • 'v': Toggle side visibility")
    
    def detect_green_tape_coverage(self, image: np.ndarray) -> GreenTapeStatus:
        """
        Detect green tape coverage to determine if hands are blocking components
        
        Args:
            image: Input image
            
        Returns:
            GreenTapeStatus with coverage information
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for green areas (tape)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Split image to analyze each side
        split_x = int(self.width * self.split_ratio)
        left_mask = green_mask[:, :split_x]
        right_mask = green_mask[:, split_x:]
        
        # Calculate coverage for each side
        left_total_pixels = left_mask.shape[0] * left_mask.shape[1]
        right_total_pixels = right_mask.shape[0] * right_mask.shape[1]
        
        left_green_pixels = np.sum(left_mask > 0)
        right_green_pixels = np.sum(right_mask > 0)
        
        left_coverage = left_green_pixels / left_total_pixels
        right_coverage = right_green_pixels / right_total_pixels
        
        # Determine if each side is covered (hand blocking tape)
        left_covered = left_coverage < self.tape_coverage_threshold
        right_covered = right_coverage < self.tape_coverage_threshold
        
        # Calculate overall coverage
        total_green_pixels = left_green_pixels + right_green_pixels
        total_pixels = left_total_pixels + right_total_pixels
        overall_coverage = total_green_pixels / total_pixels
        
        # Determine if we should skip this image
        should_skip = left_covered or right_covered
        
        return GreenTapeStatus(
            left_side_covered=left_covered,
            right_side_covered=right_covered,
            coverage_percentage=overall_coverage * 100,
            should_skip=should_skip
        )
    
    def visualize_green_tape_detection(self, image: np.ndarray, tape_status: GreenTapeStatus) -> np.ndarray:
        """Add green tape detection visualization to image"""
        overlay = image.copy()
        
        # Create green mask visualization
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Highlight detected green areas
        green_areas = cv2.bitwise_and(image, image, mask=green_mask)
        overlay = cv2.addWeighted(overlay, 0.7, green_areas, 0.3, 0)
        
        # Add status text
        split_x = int(self.width * self.split_ratio)
        
        # Left side status
        left_color = (0, 0, 255) if tape_status.left_side_covered else (0, 255, 0)  # Red if covered, Green if visible
        left_text = "BLOCKED" if tape_status.left_side_covered else "CLEAR"
        cv2.putText(overlay, f"LEFT: {left_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
        
        # Right side status  
        right_color = (0, 0, 255) if tape_status.right_side_covered else (0, 255, 0)
        right_text = "BLOCKED" if tape_status.right_side_covered else "CLEAR"
        cv2.putText(overlay, f"RIGHT: {right_text}", (split_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)
        
        # Overall status
        overall_color = (0, 0, 255) if tape_status.should_skip else (0, 255, 0)
        overall_text = "SKIP IMAGE" if tape_status.should_skip else "PROCESS IMAGE"
        cv2.putText(overlay, f"STATUS: {overall_text}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, overall_color, 2)
        
        return overlay
    
    def load_image(self, image_path: str) -> bool:
        """Load image for annotation"""
        self.image_path = Path(image_path)
        
        try:
            # Try standard loading first
            self.original_image = cv2.imread(str(image_path))
            
            if self.original_image is None:
                # Try loading with raw bytes (handles special characters in filenames)
                with open(image_path, 'rb') as f:
                    file_bytes = f.read()
                img_array = np.frombuffer(file_bytes, np.uint8)
                self.original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if self.original_image is None:
                print(f"❌ Could not load image: {image_path}")
                return False
            
            self.height, self.width = self.original_image.shape[:2]
            self.display_image = self.original_image.copy()
            self.annotations = []
            
            # Detect green tape status
            tape_status = self.detect_green_tape_coverage(self.original_image)
            
            print(f"\n📸 Loaded: {self.image_path.name}")
            print(f"   📐 Size: {self.width}x{self.height}")
            print(f"   🎭 Green tape status:")
            print(f"     • Left side: {'🚫 BLOCKED' if tape_status.left_side_covered else '✅ CLEAR'}")
            print(f"     • Right side: {'🚫 BLOCKED' if tape_status.right_side_covered else '✅ CLEAR'}")
            print(f"     • Coverage: {tape_status.coverage_percentage:.1f}%")
            print(f"     • Recommendation: {'⏭️  SKIP' if tape_status.should_skip else '✅ ANNOTATE'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return False
    
    def get_side_bounds(self, side: str) -> Tuple[int, int]:
        """Get pixel bounds for a side"""
        split_x = int(self.width * self.split_ratio)
        
        if side == "left":
            return (0, split_x)
        else:  # right
            return (split_x, self.width)
    
    def normalize_coordinates(self, x: int, y: int, side: str) -> Tuple[float, float]:
        """Convert pixel coordinates to normalized coordinates for a side"""
        x_start, x_end = self.get_side_bounds(side)
        side_width = x_end - x_start
        
        # Normalize to side coordinates
        norm_x = (x - x_start) / side_width
        norm_y = y / self.height
        
        return (norm_x, norm_y)
    
    def denormalize_coordinates(self, norm_x: float, norm_y: float, side: str) -> Tuple[int, int]:
        """Convert normalized coordinates back to pixel coordinates"""
        x_start, x_end = self.get_side_bounds(side)
        side_width = x_end - x_start
        
        x = int(x_start + norm_x * side_width)
        y = int(norm_y * self.height)
        
        return (x, y)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Determine which side we're on
            split_x = int(self.width * self.split_ratio)
            self.current_side = "left" if x < split_x else "right"
            
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_image = self.display_image.copy()
                
                # Draw current rectangle
                cv2.rectangle(temp_image, self.start_point, (x, y), (0, 255, 0), 2)
                
                # Show class label
                label_text = f"{self.classes[self.current_class]} ({self.current_side})"
                cv2.putText(temp_image, label_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add split line and annotations
                temp_image = self._draw_annotations(temp_image)
                temp_image = self._draw_split_line(temp_image)
                temp_image = self._draw_ui_overlay(temp_image)
                
                cv2.imshow('Dual Board Annotator', temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self._add_annotation(self.start_point, (x, y))
                self._update_display()
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last annotation
            if self.annotations:
                removed = self.annotations.pop()
                print(f"🗑️  Removed: {removed.class_name} from {removed.side} side")
                self._update_display()
    
    def _add_annotation(self, start_point: Tuple[int, int], end_point: Tuple[int, int]):
        """Add annotation from drawn rectangle"""
        x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
        
        # Calculate normalized coordinates relative to the side
        center_x_px = (x1 + x2) / 2
        center_y_px = (y1 + y2) / 2
        width_px = x2 - x1
        height_px = y2 - y1
        
        # Normalize coordinates for the current side
        center_x_norm, center_y_norm = self.normalize_coordinates(center_x_px, center_y_px, self.current_side)
        
        side_bounds = self.get_side_bounds(self.current_side)
        side_width = side_bounds[1] - side_bounds[0]
        
        width_norm = width_px / side_width
        height_norm = height_px / self.height
        
        annotation = Annotation(
            class_id=self.current_class,
            class_name=self.classes[self.current_class],
            center_x=center_x_norm,
            center_y=center_y_norm,
            width=width_norm,
            height=height_norm,
            side=self.current_side
        )
        
        self.annotations.append(annotation)
        print(f"✅ Added: {annotation.class_name} on {annotation.side} side "
              f"({center_x_norm:.3f}, {center_y_norm:.3f})")
    
    def _draw_split_line(self, image: np.ndarray) -> np.ndarray:
        """Draw the split line between left and right sides"""
        split_x = int(self.width * self.split_ratio)
        cv2.line(image, (split_x, 0), (split_x, self.height), (255, 255, 255), 2)
        
        # Add side labels
        cv2.putText(image, "LEFT", (10, self.height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(image, "RIGHT", (split_x + 10, self.height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return image
    
    def _draw_annotations(self, image: np.ndarray) -> np.ndarray:
        """Draw existing annotations on the image"""
        for ann in self.annotations:
            # Convert back to pixel coordinates
            center_x, center_y = self.denormalize_coordinates(ann.center_x, ann.center_y, ann.side)
            
            side_bounds = self.get_side_bounds(ann.side)
            side_width = side_bounds[1] - side_bounds[0]
            
            width_px = int(ann.width * side_width)
            height_px = int(ann.height * self.height)
            
            x1 = center_x - width_px // 2
            y1 = center_y - height_px // 2
            x2 = center_x + width_px // 2
            y2 = center_y + height_px // 2
            
            # Choose color based on side
            color = (0, 255, 255) if ann.side == "left" else (255, 0, 255)  # Yellow for left, Magenta for right
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{ann.class_name}"
            cv2.putText(image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def _draw_ui_overlay(self, image: np.ndarray) -> np.ndarray:
        """Draw UI information overlay"""
        # Current class info
        class_text = f"Class: {self.classes[self.current_class]} ({self.current_class})"
        cv2.putText(image, class_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Current side info
        side_color = (0, 255, 255) if self.current_side == "left" else (255, 0, 255)
        side_text = f"Side: {self.current_side.upper()}"
        cv2.putText(image, side_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, side_color, 2)
        
        # Annotation count
        left_count = len([a for a in self.annotations if a.side == "left"])
        right_count = len([a for a in self.annotations if a.side == "right"])
        count_text = f"Annotations: L:{left_count} R:{right_count}"
        cv2.putText(image, count_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def _update_display(self):
        """Update the display image"""
        self.display_image = self.original_image.copy()
        self.display_image = self._draw_annotations(self.display_image)
        self.display_image = self._draw_split_line(self.display_image)
        self.display_image = self._draw_ui_overlay(self.display_image)
        cv2.imshow('Dual Board Annotator', self.display_image)
    
    def save_annotations(self, output_dir: str) -> Tuple[str, str]:
        """
        Save annotations in YOLO format for both sides
        
        Returns:
            Tuple of (left_label_file, right_label_file)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create base filename
        base_name = self.image_path.stem
        
        # Group annotations by side
        left_annotations = [a for a in self.annotations if a.side == "left"]
        right_annotations = [a for a in self.annotations if a.side == "right"]
        
        # Save left side annotations
        left_file = output_path / f"{base_name}_left.txt"
        with open(left_file, 'w') as f:
            for ann in left_annotations:
                f.write(ann.to_yolo_line() + "\n")
        
        # Save right side annotations  
        right_file = output_path / f"{base_name}_right.txt"
        with open(right_file, 'w') as f:
            for ann in right_annotations:
                f.write(ann.to_yolo_line() + "\n")
        
        print(f"💾 Saved annotations:")
        print(f"   • Left side: {left_file} ({len(left_annotations)} annotations)")
        print(f"   • Right side: {right_file} ({len(right_annotations)} annotations)")
        
        return str(left_file), str(right_file)
    
    def annotate_image(self, image_path: str, output_dir: str = "annotations") -> bool:
        """
        Annotate a single image
        
        Args:
            image_path: Path to image file
            output_dir: Output directory for annotations
            
        Returns:
            True if annotations were saved, False if skipped
        """
        if not self.load_image(image_path):
            return False
        
        # Check green tape status
        tape_status = self.detect_green_tape_coverage(self.original_image)
        
        if tape_status.should_skip:
            print(f"⏭️  Skipping image due to green tape coverage (hands detected)")
            response = input("   Continue anyway? (y/n): ").lower()
            if response != 'y':
                return False
        
        # Setup window and callbacks
        cv2.namedWindow('Dual Board Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Dual Board Annotator', self.mouse_callback)
        
        self._update_display()
        
        print(f"\n🎯 Annotating: {self.image_path.name}")
        print("   Press 'h' for help, 's' to save, 'q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                return False
            elif key == ord('s'):
                self.save_annotations(output_dir)
                return True
            elif key == ord('r'):
                # Reset annotations
                self.annotations = []
                self._update_display()
                print("🔄 Reset annotations")
            elif key == ord('h'):
                self._print_controls()
            elif key == ord('\t'):  # Tab key
                # Switch sides
                self.current_side = "right" if self.current_side == "left" else "left"
                print(f"🔄 Switched to {self.current_side} side")
                self._update_display()
            elif key == ord(' '):  # Space key
                # Toggle green tape visualization
                tape_status = self.detect_green_tape_coverage(self.original_image)
                vis_image = self.visualize_green_tape_detection(self.original_image, tape_status)
                cv2.imshow('Green Tape Detection', vis_image)
                print("🟢 Green tape detection overlay displayed")
            elif key == ord('v'):
                # Toggle side visibility (dim one side)
                pass  # TODO: Implement if needed
            elif ord('0') <= key <= ord('9'):
                # Change class
                class_id = key - ord('0')
                if class_id < len(self.classes):
                    self.current_class = class_id
                    print(f"📝 Changed class to: {self.classes[self.current_class]}")
                    self._update_display()
    
    def batch_annotate(self, images_directory: str, output_dir: str = "annotations"):
        """
        Annotate multiple images in batch
        
        Args:
            images_directory: Directory containing images
            output_dir: Output directory for annotations
        """
        images_path = Path(images_directory)
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        
        # Collect all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_path.glob(ext))
        
        if not image_files:
            print(f"❌ No images found in {images_directory}")
            return
        
        print(f"🎯 DUAL BOARD BATCH ANNOTATION")
        print(f"📁 Directory: {images_directory}")
        print(f"📸 Found {len(image_files)} images")
        print(f"💾 Output: {output_dir}")
        print(f"🎮 Classes: {', '.join(self.classes)}")
        
        annotated_count = 0
        skipped_count = 0
        
        for i, image_path in enumerate(image_files):
            print(f"\n{'='*60}")
            print(f"📸 Image {i+1}/{len(image_files)}: {image_path.name}")
            
            if self.annotate_image(str(image_path), output_dir):
                annotated_count += 1
                print("✅ Annotations saved")
            else:
                skipped_count += 1
                print("⏭️  Skipped")
                
                # Ask if user wants to continue
                if i < len(image_files) - 1:  # Not the last image
                    response = input("   Continue to next image? (y/n): ").lower()
                    if response != 'y':
                        break
        
        cv2.destroyAllWindows()
        
        print(f"\n🎉 BATCH ANNOTATION COMPLETE")
        print(f"   • Annotated: {annotated_count}")
        print(f"   • Skipped: {skipped_count}")
        print(f"   • Total: {len(image_files)}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Board Annotation Tool")
    parser.add_argument("--images", "-i", required=True, help="Images directory or single image file")
    parser.add_argument("--classes", "-c", default="classes.txt", help="Classes file")
    parser.add_argument("--output", "-o", default="annotations", help="Output directory")
    parser.add_argument("--single", action="store_true", help="Annotate single image instead of batch")
    
    args = parser.parse_args()
    
    # Initialize annotator
    annotator = DualBoardAnnotator(args.classes)
    
    if args.single:
        # Single image annotation
        annotator.annotate_image(args.images, args.output)
    else:
        # Batch annotation
        annotator.batch_annotate(args.images, args.output)


if __name__ == "__main__":
    main()
