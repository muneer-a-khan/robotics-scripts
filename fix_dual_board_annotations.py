#!/usr/bin/env python3
"""
Fix Dual Board Annotations

This script merges the separate left and right annotation files into single 
annotation files that YOLO can use for training.
"""

import os
from pathlib import Path
from typing import List, Tuple


def convert_right_coordinates(annotations: List[str], split_ratio: float = 0.5) -> List[str]:
    """Convert right side coordinates to full image coordinates"""
    converted = []
    
    for line in annotations:
        if line.strip():
            parts = line.strip().split()
            class_id = parts[0]
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert right side coordinates to full image
            # Right side starts at split_ratio (0.5) and goes to 1.0
            # So we need to map [0,1] -> [split_ratio, 1]
            full_center_x = split_ratio + (center_x * (1.0 - split_ratio))
            full_width = width * (1.0 - split_ratio)
            
            converted.append(f"{class_id} {full_center_x:.6f} {center_y:.6f} {full_width:.6f} {height:.6f}")
    
    return converted


def convert_left_coordinates(annotations: List[str], split_ratio: float = 0.5) -> List[str]:
    """Convert left side coordinates to full image coordinates"""
    converted = []
    
    for line in annotations:
        if line.strip():
            parts = line.strip().split()
            class_id = parts[0]
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert left side coordinates to full image
            # Left side coordinates are already in [0, split_ratio] range
            full_center_x = center_x * split_ratio
            full_width = width * split_ratio
            
            converted.append(f"{class_id} {full_center_x:.6f} {center_y:.6f} {full_width:.6f} {height:.6f}")
    
    return converted


def merge_annotations(dataset_dir: str):
    """Merge left and right annotations into single files"""
    dataset_path = Path(dataset_dir)
    
    for split in ['train', 'val']:
        labels_dir = dataset_path / "labels" / split
        images_dir = dataset_path / "images" / split
        
        if not labels_dir.exists():
            print(f"⚠️  Labels directory not found: {labels_dir}")
            continue
        
        print(f"📝 Processing {split} annotations...")
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(images_dir.glob(f"*{ext}"))
        
        processed = 0
        for image_file in image_files:
            base_name = image_file.stem
            
            left_file = labels_dir / f"{base_name}_left.txt"
            right_file = labels_dir / f"{base_name}_right.txt"
            merged_file = labels_dir / f"{base_name}.txt"
            
            if left_file.exists() or right_file.exists():
                merged_annotations = []
                
                # Process left side annotations
                if left_file.exists():
                    with open(left_file, 'r') as f:
                        left_annotations = f.readlines()
                    converted_left = convert_left_coordinates(left_annotations)
                    merged_annotations.extend(converted_left)
                
                # Process right side annotations  
                if right_file.exists():
                    with open(right_file, 'r') as f:
                        right_annotations = f.readlines()
                    converted_right = convert_right_coordinates(right_annotations)
                    merged_annotations.extend(converted_right)
                
                # Write merged annotations
                if merged_annotations:
                    with open(merged_file, 'w') as f:
                        for annotation in merged_annotations:
                            f.write(annotation + '\n')
                    
                    processed += 1
                    print(f"   ✅ Merged {base_name}: {len(merged_annotations)} annotations")
        
        print(f"📊 {split}: Processed {processed} image annotation sets")


def validate_merged_annotations(dataset_dir: str):
    """Validate the merged annotations"""
    dataset_path = Path(dataset_dir)
    
    for split in ['train', 'val']:
        labels_dir = dataset_path / "labels" / split
        images_dir = dataset_path / "images" / split
        
        if not labels_dir.exists():
            continue
        
        print(f"🔍 Validating {split} annotations...")
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg"))
        
        missing_labels = 0
        total_annotations = 0
        
        for image_file in image_files:
            base_name = image_file.stem
            label_file = labels_dir / f"{base_name}.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    annotations = [line.strip() for line in f if line.strip()]
                total_annotations += len(annotations)
            else:
                missing_labels += 1
                print(f"   ⚠️  Missing: {base_name}.txt")
        
        print(f"   📸 Images: {len(image_files)}")
        print(f"   📝 Total annotations: {total_annotations}")
        print(f"   ❌ Missing labels: {missing_labels}")


def main():
    """Main function"""
    dataset_dir = "photos_training_dataset"
    
    print("🔧 FIXING DUAL BOARD ANNOTATIONS")
    print("=" * 60)
    print(f"📂 Dataset: {dataset_dir}")
    print()
    
    # Merge annotations
    merge_annotations(dataset_dir)
    
    print()
    print("🔍 VALIDATION")
    print("=" * 30)
    
    # Validate merged annotations
    validate_merged_annotations(dataset_dir)
    
    print()
    print("✅ ANNOTATION FIX COMPLETE!")
    print("🚀 Ready for training!")


if __name__ == "__main__":
    main()
