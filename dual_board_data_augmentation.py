#!/usr/bin/env python3
"""
Dual Board Data Augmentation Pipeline

This module provides comprehensive data augmentation for dual board circuit images.
It handles:
- Image rotations (90°, 180°, 270°)
- Color variations (HSV adjustments)
- Brightness and contrast adjustments
- YOLO annotation transformation for all augmentations
- Proper handling of dual board setup (left/right annotations)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import random
import math
from dataclasses import dataclass
import colorsys


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""
    # Rotation augmentations
    enable_rotations: bool = True
    rotation_angles: List[int] = None  # Will default to [90, 180, 270]
    
    # Color augmentations
    enable_color_variations: bool = True
    hue_shift_range: Tuple[int, int] = (-20, 20)         # HSV hue shift
    saturation_range: Tuple[float, float] = (0.8, 1.2)  # Saturation multiplier
    value_range: Tuple[float, float] = (0.8, 1.2)       # Brightness multiplier
    
    # Brightness/Contrast
    brightness_range: Tuple[int, int] = (-30, 30)       # Brightness offset
    contrast_range: Tuple[float, float] = (0.8, 1.2)    # Contrast multiplier
    
    # Other augmentations
    enable_noise: bool = True
    noise_intensity: int = 10
    
    # Output settings
    preserve_original: bool = True  # Keep original images in output
    max_augmentations_per_image: int = 12  # Total augmented versions per image
    
    def __post_init__(self):
        if self.rotation_angles is None:
            self.rotation_angles = [90, 180, 270]


@dataclass 
class YOLOAnnotation:
    """YOLO format annotation"""
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float
    
    @classmethod
    def from_line(cls, line: str):
        parts = line.strip().split()
        return cls(
            class_id=int(parts[0]),
            center_x=float(parts[1]),
            center_y=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4])
        )
    
    def to_line(self) -> str:
        return f"{self.class_id} {self.center_x:.6f} {self.center_y:.6f} {self.width:.6f} {self.height:.6f}"
    
    def copy(self):
        return YOLOAnnotation(self.class_id, self.center_x, self.center_y, self.width, self.height)


class DualBoardAugmentationPipeline:
    """Advanced data augmentation pipeline for dual board images"""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """Initialize the augmentation pipeline"""
        self.config = config or AugmentationConfig()
        self.classes_file = None
        self.classes = []
        
        print("🎨 Dual Board Data Augmentation Pipeline Initialized")
        print(f"   • Rotations: {self.config.enable_rotations} ({self.config.rotation_angles})")
        print(f"   • Color variations: {self.config.enable_color_variations}")
        print(f"   • Brightness/Contrast: {self.config.brightness_range}, {self.config.contrast_range}")
        print(f"   • Max augmentations per image: {self.config.max_augmentations_per_image}")
    
    def load_classes(self, classes_file: str):
        """Load class names from file"""
        self.classes_file = classes_file
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        print(f"📋 Loaded {len(self.classes)} classes from {classes_file}")
    
    def load_image_and_annotations(self, image_path: str, left_annotation_path: str, 
                                 right_annotation_path: str) -> Tuple[np.ndarray, List[YOLOAnnotation], List[YOLOAnnotation]]:
        """
        Load image and its dual annotations
        
        Returns:
            Tuple of (image, left_annotations, right_annotations)
        """
        # Load image with special character handling
        try:
            image = cv2.imread(image_path)
            if image is None:
                with open(image_path, 'rb') as f:
                    file_bytes = f.read()
                img_array = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None, [], []
        
        # Load left annotations
        left_annotations = []
        if Path(left_annotation_path).exists():
            with open(left_annotation_path, 'r') as f:
                for line in f:
                    if line.strip():
                        left_annotations.append(YOLOAnnotation.from_line(line))
        
        # Load right annotations
        right_annotations = []
        if Path(right_annotation_path).exists():
            with open(right_annotation_path, 'r') as f:
                for line in f:
                    if line.strip():
                        right_annotations.append(YOLOAnnotation.from_line(line))
        
        return image, left_annotations, right_annotations
    
    def rotate_image_90(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by 90-degree increments"""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image
    
    def transform_annotations_rotation(self, annotations: List[YOLOAnnotation], 
                                     angle: int) -> List[YOLOAnnotation]:
        """Transform YOLO annotations for rotation"""
        transformed = []
        
        for ann in annotations:
            new_ann = ann.copy()
            
            if angle == 90:
                # 90° clockwise: (x,y) -> (1-y, x)
                new_ann.center_x = 1.0 - ann.center_y
                new_ann.center_y = ann.center_x
                # Swap width and height
                new_ann.width, new_ann.height = ann.height, ann.width
                
            elif angle == 180:
                # 180°: (x,y) -> (1-x, 1-y)
                new_ann.center_x = 1.0 - ann.center_x
                new_ann.center_y = 1.0 - ann.center_y
                # Width and height stay the same
                
            elif angle == 270:
                # 270° clockwise: (x,y) -> (y, 1-x)
                new_ann.center_x = ann.center_y
                new_ann.center_y = 1.0 - ann.center_x
                # Swap width and height
                new_ann.width, new_ann.height = ann.height, ann.width
            
            transformed.append(new_ann)
        
        return transformed
    
    def adjust_color_hsv(self, image: np.ndarray, hue_shift: int = 0, 
                        sat_mult: float = 1.0, val_mult: float = 1.0) -> np.ndarray:
        """Adjust image colors in HSV space"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust hue (with wrapping)
        hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180
        
        # Adjust saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_mult, 0, 255)
        
        # Adjust value (brightness)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * val_mult, 0, 255)
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result
    
    def adjust_brightness_contrast(self, image: np.ndarray, brightness: int = 0, 
                                 contrast: float = 1.0) -> np.ndarray:
        """Adjust brightness and contrast"""
        # Apply contrast and brightness
        result = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return result
    
    def add_noise(self, image: np.ndarray, intensity: int = 10) -> np.ndarray:
        """Add gaussian noise to image"""
        noise = np.random.normal(0, intensity, image.shape).astype(np.int16)
        result = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return result
    
    def generate_augmented_versions(self, image: np.ndarray, 
                                  left_annotations: List[YOLOAnnotation],
                                  right_annotations: List[YOLOAnnotation]) -> List[Tuple[np.ndarray, List[YOLOAnnotation], List[YOLOAnnotation], str]]:
        """
        Generate all augmented versions of an image
        
        Returns:
            List of (augmented_image, left_annotations, right_annotations, suffix) tuples
        """
        augmented_versions = []
        
        # Original (if preserving)
        if self.config.preserve_original:
            augmented_versions.append((image, left_annotations, right_annotations, "original"))
        
        # Rotation augmentations
        if self.config.enable_rotations:
            for angle in self.config.rotation_angles:
                rotated_image = self.rotate_image_90(image, angle)
                rotated_left = self.transform_annotations_rotation(left_annotations, angle)
                rotated_right = self.transform_annotations_rotation(right_annotations, angle)
                augmented_versions.append((rotated_image, rotated_left, rotated_right, f"rot{angle}"))
        
        # Color variations
        if self.config.enable_color_variations:
            # Random color variations
            for i in range(3):  # Generate 3 color variants
                hue_shift = random.randint(*self.config.hue_shift_range)
                sat_mult = random.uniform(*self.config.saturation_range)
                val_mult = random.uniform(*self.config.value_range)
                
                color_image = self.adjust_color_hsv(image, hue_shift, sat_mult, val_mult)
                augmented_versions.append((color_image, left_annotations, right_annotations, f"color{i+1}"))
        
        # Brightness/Contrast variations
        for i in range(2):  # Generate 2 brightness variants
            brightness = random.randint(*self.config.brightness_range)
            contrast = random.uniform(*self.config.contrast_range)
            
            bright_image = self.adjust_brightness_contrast(image, brightness, contrast)
            augmented_versions.append((bright_image, left_annotations, right_annotations, f"bright{i+1}"))
        
        # Noise variations
        if self.config.enable_noise:
            for i in range(2):  # Generate 2 noise variants  
                noise_image = self.add_noise(image, self.config.noise_intensity)
                augmented_versions.append((noise_image, left_annotations, right_annotations, f"noise{i+1}"))
        
        # Combined variations (rotation + color)
        if self.config.enable_rotations and self.config.enable_color_variations:
            # Pick one rotation and one color variation
            angle = random.choice(self.config.rotation_angles)
            hue_shift = random.randint(*self.config.hue_shift_range)
            sat_mult = random.uniform(*self.config.saturation_range)
            
            combined_image = self.rotate_image_90(image, angle)
            combined_image = self.adjust_color_hsv(combined_image, hue_shift, sat_mult)
            
            combined_left = self.transform_annotations_rotation(left_annotations, angle)
            combined_right = self.transform_annotations_rotation(right_annotations, angle)
            
            augmented_versions.append((combined_image, combined_left, combined_right, f"comb_rot{angle}_col"))
        
        # Limit to max augmentations
        if len(augmented_versions) > self.config.max_augmentations_per_image:
            # Keep original and randomly sample the rest
            original_kept = augmented_versions[:1] if self.config.preserve_original else []
            others = augmented_versions[1:] if self.config.preserve_original else augmented_versions
            
            random.shuffle(others)
            keep_count = self.config.max_augmentations_per_image - len(original_kept)
            augmented_versions = original_kept + others[:keep_count]
        
        return augmented_versions
    
    def save_augmented_image_and_annotations(self, image: np.ndarray, 
                                           left_annotations: List[YOLOAnnotation],
                                           right_annotations: List[YOLOAnnotation],
                                           base_name: str, suffix: str,
                                           output_images_dir: str, output_labels_dir: str):
        """Save augmented image and its annotations"""
        output_images_path = Path(output_images_dir)
        output_labels_path = Path(output_labels_dir)
        
        # Create output directories
        output_images_path.mkdir(parents=True, exist_ok=True)
        output_labels_path.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image_filename = f"{base_name}_{suffix}.jpg"
        image_path = output_images_path / image_filename
        cv2.imwrite(str(image_path), image)
        
        # Save left annotations
        left_label_filename = f"{base_name}_{suffix}_left.txt"
        left_label_path = output_labels_path / left_label_filename
        with open(left_label_path, 'w') as f:
            for ann in left_annotations:
                f.write(ann.to_line() + "\n")
        
        # Save right annotations
        right_label_filename = f"{base_name}_{suffix}_right.txt"
        right_label_path = output_labels_path / right_label_filename
        with open(right_label_path, 'w') as f:
            for ann in right_annotations:
                f.write(ann.to_line() + "\n")
        
        return str(image_path), str(left_label_path), str(right_label_path)
    
    def create_augmented_dataset(self, input_images_dir: str, input_annotations_dir: str,
                               output_dataset_dir: str) -> str:
        """
        Create full augmented dataset from annotated images
        
        Args:
            input_images_dir: Directory containing original images
            input_annotations_dir: Directory containing YOLO annotations (left/right pairs)
            output_dataset_dir: Output directory for augmented dataset
            
        Returns:
            Path to data.yaml file
        """
        input_images_path = Path(input_images_dir)
        input_annotations_path = Path(input_annotations_dir)
        output_path = Path(output_dataset_dir)
        
        print(f"🎨 Creating Augmented Dual Board Dataset")
        print(f"   📁 Input images: {input_images_dir}")
        print(f"   📋 Input annotations: {input_annotations_dir}")
        print(f"   💾 Output: {output_dataset_dir}")
        
        # Create dataset structure
        train_images_dir = output_path / "images" / "train"
        val_images_dir = output_path / "images" / "val"
        test_images_dir = output_path / "images" / "test"
        train_labels_dir = output_path / "labels" / "train"
        val_labels_dir = output_path / "labels" / "val"
        test_labels_dir = output_path / "labels" / "test"
        
        for dir_path in [train_images_dir, val_images_dir, test_images_dir,
                        train_labels_dir, val_labels_dir, test_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Find all annotated image pairs
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(input_images_path.glob(ext))
        
        print(f"📸 Found {len(image_files)} images to augment")
        
        total_augmented = 0
        successful_images = 0
        
        for i, image_path in enumerate(image_files):
            print(f"\n🎨 Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            # Look for corresponding annotation files
            base_name = image_path.stem
            left_annotation_path = input_annotations_path / f"{base_name}_left.txt"
            right_annotation_path = input_annotations_path / f"{base_name}_right.txt"
            
            if not (left_annotation_path.exists() or right_annotation_path.exists()):
                print(f"   ⚠️  No annotations found, skipping")
                continue
            
            # Load image and annotations
            image, left_annotations, right_annotations = self.load_image_and_annotations(
                str(image_path), str(left_annotation_path), str(right_annotation_path)
            )
            
            if image is None:
                print(f"   ❌ Could not load image")
                continue
            
            # Generate augmented versions
            augmented_versions = self.generate_augmented_versions(image, left_annotations, right_annotations)
            
            print(f"   🎯 Generated {len(augmented_versions)} augmented versions")
            
            # Split into train/val/test (80%/15%/5%)
            random.shuffle(augmented_versions)
            n_total = len(augmented_versions)
            n_train = int(0.8 * n_total)
            n_val = int(0.15 * n_total)
            
            train_versions = augmented_versions[:n_train]
            val_versions = augmented_versions[n_train:n_train + n_val]
            test_versions = augmented_versions[n_train + n_val:]
            
            # Save training set
            for j, (aug_image, aug_left, aug_right, suffix) in enumerate(train_versions):
                self.save_augmented_image_and_annotations(
                    aug_image, aug_left, aug_right,
                    f"{base_name}_{j:02d}", suffix,
                    str(train_images_dir), str(train_labels_dir)
                )
            
            # Save validation set
            for j, (aug_image, aug_left, aug_right, suffix) in enumerate(val_versions):
                self.save_augmented_image_and_annotations(
                    aug_image, aug_left, aug_right,
                    f"{base_name}_val_{j:02d}", suffix,
                    str(val_images_dir), str(val_labels_dir)
                )
            
            # Save test set
            for j, (aug_image, aug_left, aug_right, suffix) in enumerate(test_versions):
                self.save_augmented_image_and_annotations(
                    aug_image, aug_left, aug_right,
                    f"{base_name}_test_{j:02d}", suffix,
                    str(test_images_dir), str(test_labels_dir)
                )
            
            total_augmented += len(augmented_versions)
            successful_images += 1
            
            print(f"   ✅ Saved: {len(train_versions)} train, {len(val_versions)} val, {len(test_versions)} test")
        
        # Create data.yaml
        data_yaml_path = output_path / "data.yaml"
        data_yaml_content = f"""# Dual Board Snap Circuit Augmented Dataset
path: {output_path.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')
test: images/test    # test images (relative to 'path')

# Number of classes
nc: {len(self.classes)}

# Class names
names:
"""
        
        for i, class_name in enumerate(self.classes):
            data_yaml_content += f"  {i}: {class_name}\n"
        
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            f.write(data_yaml_content)
        
        print(f"\n🎉 AUGMENTED DATASET CREATED")
        print(f"   ✅ Successfully processed: {successful_images}/{len(image_files)} images")
        print(f"   🎨 Total augmented images: {total_augmented}")
        print(f"   📄 Data config: {data_yaml_path}")
        
        # Count final dataset
        train_count = len(list(train_images_dir.glob('*.jpg')))
        val_count = len(list(val_images_dir.glob('*.jpg')))
        test_count = len(list(test_images_dir.glob('*.jpg')))
        
        print(f"   📊 Final dataset split:")
        print(f"     • Training: {train_count} images")
        print(f"     • Validation: {val_count} images")
        print(f"     • Test: {test_count} images")
        
        return str(data_yaml_path)


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Board Data Augmentation Pipeline")
    parser.add_argument("--images", "-i", required=True, help="Input images directory")
    parser.add_argument("--annotations", "-a", required=True, help="Input annotations directory")
    parser.add_argument("--output", "-o", required=True, help="Output dataset directory")
    parser.add_argument("--classes", "-c", default="classes.txt", help="Classes file")
    parser.add_argument("--max-aug", type=int, default=12, help="Max augmentations per image")
    parser.add_argument("--no-rotations", action="store_true", help="Disable rotations")
    parser.add_argument("--no-color", action="store_true", help="Disable color variations")
    
    args = parser.parse_args()
    
    # Create configuration
    config = AugmentationConfig(
        enable_rotations=not args.no_rotations,
        enable_color_variations=not args.no_color,
        max_augmentations_per_image=args.max_aug
    )
    
    # Initialize pipeline
    pipeline = DualBoardAugmentationPipeline(config)
    pipeline.load_classes(args.classes)
    
    # Create augmented dataset
    data_yaml_path = pipeline.create_augmented_dataset(
        args.images, args.annotations, args.output
    )
    
    print(f"\n🚀 Ready for training with: {data_yaml_path}")


if __name__ == "__main__":
    main()
