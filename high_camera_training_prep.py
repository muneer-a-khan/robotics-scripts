#!/usr/bin/env python3
"""
High Camera Training Data Preparation

This script helps prepare training data for a high-up camera setup by:
1. Collecting images at lower resolutions
2. Creating zoomed-out training samples
3. Augmenting existing data for high-up camera scenarios
4. Preparing dataset structure for retraining

The goal is to train the model to detect components from a greater distance
with lower resolution, suitable for a high-mounted camera monitoring two circuit boards.
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import shutil
import argparse
from ultralytics import YOLO

from config import YOLO_CONFIG, COMPONENT_CLASSES


class HighCameraTrainingPrep:
    """Prepare training data for high-up camera setup."""
    
    def __init__(self, output_dir: str = "data/high_camera_training"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        # Create directory structure
        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        self.component_classes = COMPONENT_CLASSES
        self.class_to_id = {cls: i for i, cls in enumerate(self.component_classes)}
        
        print(f"High camera training data preparation initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Component classes: {len(self.component_classes)}")
    
    def collect_live_data(self, camera_id: int = 0, duration_minutes: int = 10):
        """
        Collect live training data from the high-up camera.
        
        Args:
            camera_id: Camera device ID
            duration_minutes: How long to collect data (minutes)
        """
        print(f"Starting live data collection for {duration_minutes} minutes...")
        print("Position circuit components on both sides of the camera view")
        print("Press SPACE to capture training images, 'q' to quit early")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties for high resolution capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        capture_count = 0
        
        try:
            while time.time() < end_time:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Show current frame
                display_frame = frame.copy()
                
                # Add collection info overlay
                remaining_time = int(end_time - time.time())
                cv2.putText(display_frame, f"Collecting data: {remaining_time}s remaining", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Captured: {capture_count} images", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press SPACE to capture, 'q' to quit", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # Draw split line to show dual board area
                height, width = frame.shape[:2]
                cv2.line(display_frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)
                cv2.putText(display_frame, "LEFT BOARD", (50, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "RIGHT BOARD", (width//2 + 50, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow("High Camera Data Collection", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space key to capture
                    # Save high-resolution image
                    timestamp = int(time.time() * 1000)
                    image_path = self.images_dir / "train" / f"high_camera_{timestamp}.jpg"
                    cv2.imwrite(str(image_path), frame)
                    
                    # Create empty label file (to be annotated later)
                    label_path = self.labels_dir / "train" / f"high_camera_{timestamp}.txt"
                    label_path.touch()
                    
                    capture_count += 1
                    print(f"Captured image {capture_count}: {image_path.name}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"Data collection complete. Captured {capture_count} images.")
        print(f"Images saved to: {self.images_dir / 'train'}")
        print("Next step: Annotate the images using a tool like labelImg or Roboflow")
    
    def create_resolution_variants(self, source_dir: str, target_resolutions: List[Tuple[int, int]]):
        """
        Create lower resolution variants of existing training data.
        
        Args:
            source_dir: Directory containing original training images
            target_resolutions: List of (width, height) tuples for different resolutions
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Source directory not found: {source_dir}")
            return
        
        print(f"Creating resolution variants from {source_dir}")
        print(f"Target resolutions: {target_resolutions}")
        
        # Process each split (train/val/test)
        for split in ["train", "val"]:
            source_images = source_path / "images" / split
            source_labels = source_path / "labels" / split
            
            if not source_images.exists():
                continue
            
            image_files = list(source_images.glob("*.jpg")) + list(source_images.glob("*.png"))
            print(f"Processing {len(image_files)} images in {split} split")
            
            for img_path in image_files:
                # Load original image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Load corresponding label if exists
                label_path = source_labels / f"{img_path.stem}.txt"
                labels = []
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        labels = f.readlines()
                
                # Create variants for each resolution
                for width, height in target_resolutions:
                    # Resize image
                    resized = cv2.resize(image, (width, height))
                    
                    # Save resized image
                    variant_name = f"{img_path.stem}_res{width}x{height}.jpg"
                    variant_img_path = self.images_dir / split / variant_name
                    cv2.imwrite(str(variant_img_path), resized)
                    
                    # Copy labels (YOLO format is resolution-independent)
                    variant_label_path = self.labels_dir / split / f"{img_path.stem}_res{width}x{height}.txt"
                    if labels:
                        with open(variant_label_path, 'w') as f:
                            f.writelines(labels)
                    else:
                        variant_label_path.touch()
        
        print("Resolution variants created successfully")
    
    def create_zoomed_out_variants(self, source_dir: str, zoom_factors: List[float]):
        """
        Create zoomed-out variants by adding padding/background to simulate distance.
        
        Args:
            source_dir: Directory containing original training images
            zoom_factors: List of zoom factors (e.g., [0.5, 0.3] for 50% and 30% of original size)
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Source directory not found: {source_dir}")
            return
        
        print(f"Creating zoomed-out variants from {source_dir}")
        print(f"Zoom factors: {zoom_factors}")
        
        for split in ["train", "val"]:
            source_images = source_path / "images" / split
            source_labels = source_path / "labels" / split
            
            if not source_images.exists():
                continue
            
            image_files = list(source_images.glob("*.jpg")) + list(source_images.glob("*.png"))
            
            for img_path in image_files:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                orig_height, orig_width = image.shape[:2]
                
                # Load labels
                label_path = source_labels / f"{img_path.stem}.txt"
                labels = []
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                labels.append([int(parts[0]), float(parts[1]), float(parts[2]), 
                                             float(parts[3]), float(parts[4])])
                
                for zoom_factor in zoom_factors:
                    # Create larger canvas (simulating distance)
                    canvas_width = int(orig_width / zoom_factor)
                    canvas_height = int(orig_height / zoom_factor)
                    
                    # Create background canvas (gray/table color)
                    canvas = np.full((canvas_height, canvas_width, 3), (128, 128, 128), dtype=np.uint8)
                    
                    # Calculate position to center the original image
                    x_offset = (canvas_width - orig_width) // 2
                    y_offset = (canvas_height - orig_height) // 2
                    
                    # Place original image on canvas
                    canvas[y_offset:y_offset+orig_height, x_offset:x_offset+orig_width] = image
                    
                    # Save zoomed-out image
                    zoom_name = f"{img_path.stem}_zoom{int(zoom_factor*100)}.jpg"
                    zoom_img_path = self.images_dir / split / zoom_name
                    cv2.imwrite(str(zoom_img_path), canvas)
                    
                    # Adjust labels for new canvas size
                    zoom_label_path = self.labels_dir / split / f"{img_path.stem}_zoom{int(zoom_factor*100)}.txt"
                    with open(zoom_label_path, 'w') as f:
                        for class_id, x_center, y_center, width, height in labels:
                            # Convert from original image coordinates to canvas coordinates
                            new_x_center = (x_center * orig_width + x_offset) / canvas_width
                            new_y_center = (y_center * orig_height + y_offset) / canvas_height
                            new_width = (width * orig_width) / canvas_width
                            new_height = (height * orig_height) / canvas_height
                            
                            f.write(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} "
                                   f"{new_width:.6f} {new_height:.6f}\n")
        
        print("Zoomed-out variants created successfully")
    
    def create_dual_board_layout(self, source_dir: str, num_combinations: int = 100):
        """
        Create training images with dual circuit board layout (left and right).
        
        Args:
            source_dir: Directory containing individual circuit images
            num_combinations: Number of dual-board combinations to create
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Source directory not found: {source_dir}")
            return
        
        print(f"Creating dual board layout training data")
        print(f"Combinations to create: {num_combinations}")
        
        # Get available images
        source_images = source_path / "images" / "train"
        source_labels = source_path / "labels" / "train"
        
        if not source_images.exists():
            print("No source images found")
            return
        
        image_files = list(source_images.glob("*.jpg")) + list(source_images.glob("*.png"))
        
        if len(image_files) < 2:
            print("Need at least 2 source images to create dual layouts")
            return
        
        for i in range(num_combinations):
            # Randomly select two different images
            import random
            left_img_path = random.choice(image_files)
            right_img_path = random.choice([f for f in image_files if f != left_img_path])
            
            # Load images
            left_img = cv2.imread(str(left_img_path))
            right_img = cv2.imread(str(right_img_path))
            
            if left_img is None or right_img is None:
                continue
            
            # Resize images to consistent size
            target_height = 540  # Half of 1080p
            target_width = 960   # Half of 1920p
            
            left_resized = cv2.resize(left_img, (target_width, target_height))
            right_resized = cv2.resize(right_img, (target_width, target_height))
            
            # Create dual layout
            dual_image = np.hstack([left_resized, right_resized])
            
            # Save dual image
            dual_name = f"dual_{i:04d}_{left_img_path.stem}_{right_img_path.stem}.jpg"
            dual_img_path = self.images_dir / "train" / dual_name
            cv2.imwrite(str(dual_img_path), dual_image)
            
            # Combine labels
            dual_label_path = self.labels_dir / "train" / f"dual_{i:04d}_{left_img_path.stem}_{right_img_path.stem}.txt"
            
            with open(dual_label_path, 'w') as f:
                # Process left side labels
                left_label_path = source_labels / f"{left_img_path.stem}.txt"
                if left_label_path.exists():
                    with open(left_label_path, 'r') as left_f:
                        for line in left_f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id, x_center, y_center, width, height = parts
                                # Adjust coordinates for left half
                                new_x_center = float(x_center) * 0.5  # Scale to left half
                                f.write(f"{class_id} {new_x_center:.6f} {y_center} {float(width)*0.5:.6f} {height}\n")
                
                # Process right side labels
                right_label_path = source_labels / f"{right_img_path.stem}.txt"
                if right_label_path.exists():
                    with open(right_label_path, 'r') as right_f:
                        for line in right_f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id, x_center, y_center, width, height = parts
                                # Adjust coordinates for right half
                                new_x_center = float(x_center) * 0.5 + 0.5  # Scale and shift to right half
                                f.write(f"{class_id} {new_x_center:.6f} {y_center} {float(width)*0.5:.6f} {height}\n")
        
        print(f"Created {num_combinations} dual board training images")
    
    def create_data_yaml(self):
        """Create data.yaml file for YOLOv8 training."""
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.component_classes),
            'names': self.component_classes
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"Created data.yaml at {yaml_path}")
        return yaml_path
    
    def train_high_camera_model(self, epochs: int = 100, img_size: int = 640):
        """
        Train a new model optimized for high camera setup.
        
        Args:
            epochs: Number of training epochs
            img_size: Input image size for training
        """
        # Create data.yaml
        data_yaml = self.create_data_yaml()
        
        # Initialize model
        model = YOLO('yolov8n.pt')  # Start with smaller model for high-distance detection
        
        # Train model
        print(f"Starting training for high camera model...")
        print(f"Epochs: {epochs}, Image size: {img_size}")
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            device='cpu',  # Force CPU training
            project='high_camera_training',
            name='high_camera_model',
            patience=20,
            save=True,
            verbose=True,
            # Optimizations for high-distance detection
            conf=0.15,  # Lower confidence threshold
            iou=0.4,    # NMS IoU threshold
            augment=True,  # Enable augmentation
            mosaic=1.0,    # Mosaic augmentation
            mixup=0.1,     # Mixup augmentation
        )
        
        print("Training completed!")
        print(f"Best model saved to: {results.save_dir}")
        return results


def main():
    """Main function for high camera training preparation."""
    parser = argparse.ArgumentParser(description="High Camera Training Data Preparation")
    parser.add_argument("--mode", choices=["collect", "variants", "zoom", "dual", "train", "all"],
                       default="collect", help="Operation mode")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--duration", type=int, default=10, help="Data collection duration (minutes)")
    parser.add_argument("--source", type=str, help="Source directory for processing existing data")
    parser.add_argument("--output", type=str, default="data/high_camera_training", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    
    args = parser.parse_args()
    
    # Initialize preparation system
    prep = HighCameraTrainingPrep(args.output)
    
    if args.mode == "collect":
        prep.collect_live_data(args.camera, args.duration)
    
    elif args.mode == "variants":
        if not args.source:
            print("--source required for variants mode")
            return
        resolutions = [(640, 640), (416, 416), (320, 320)]
        prep.create_resolution_variants(args.source, resolutions)
    
    elif args.mode == "zoom":
        if not args.source:
            print("--source required for zoom mode")
            return
        zoom_factors = [0.7, 0.5, 0.3]
        prep.create_zoomed_out_variants(args.source, zoom_factors)
    
    elif args.mode == "dual":
        if not args.source:
            print("--source required for dual mode")
            return
        prep.create_dual_board_layout(args.source, 100)
    
    elif args.mode == "train":
        prep.train_high_camera_model(args.epochs)
    
    elif args.mode == "all":
        if args.source:
            print("Running complete preparation pipeline...")
            resolutions = [(640, 640), (416, 416)]
            zoom_factors = [0.7, 0.5]
            
            prep.create_resolution_variants(args.source, resolutions)
            prep.create_zoomed_out_variants(args.source, zoom_factors)
            prep.create_dual_board_layout(args.source, 50)
            prep.train_high_camera_model(args.epochs)
        else:
            print("--source required for 'all' mode")
    
    else:
        print("Invalid mode selected")


if __name__ == "__main__":
    main()
