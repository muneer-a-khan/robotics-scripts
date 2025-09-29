#!/usr/bin/env python3
"""
Create Training Dataset from Photos Annotations

This script processes the annotated photos and creates a proper training dataset
for the dual board system with the updated 15 classes.
"""

import os
import shutil
from pathlib import Path
import random
from typing import List, Tuple
import yaml


def create_photos_training_dataset():
    """Create training dataset from photos and annotations"""
    print("🎯 CREATING PHOTOS TRAINING DATASET")
    print("=" * 60)
    
    # Set up paths
    photos_dir = Path("photos")
    annotations_dir = Path("photos_annotations")
    dataset_dir = Path("photos_training_dataset")
    
    # Create dataset structure
    dataset_dir.mkdir(exist_ok=True)
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created dataset directory: {dataset_dir}")
    
    # Get all photo files
    photo_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    photo_files = []
    
    for file in photos_dir.iterdir():
        if file.suffix in photo_extensions:
            photo_files.append(file)
    
    print(f"📸 Found {len(photo_files)} photos")
    
    # Check for corresponding annotations
    valid_photos = []
    for photo_file in photo_files:
        left_annotation = annotations_dir / f"{photo_file.stem}_left.txt"
        right_annotation = annotations_dir / f"{photo_file.stem}_right.txt"
        
        if left_annotation.exists() and right_annotation.exists():
            valid_photos.append(photo_file)
        else:
            print(f"⚠️  Missing annotations for {photo_file.name}")
    
    print(f"✅ Valid annotated photos: {len(valid_photos)}")
    
    if not valid_photos:
        print("❌ No valid annotated photos found!")
        return False
    
    # Split into train/val (80/20)
    random.shuffle(valid_photos)
    split_idx = int(len(valid_photos) * 0.8)
    train_photos = valid_photos[:split_idx]
    val_photos = valid_photos[split_idx:]
    
    print(f"📊 Training photos: {len(train_photos)}")
    print(f"📊 Validation photos: {len(val_photos)}")
    
    # Process training photos
    print("\n📝 Processing training photos...")
    processed_train = process_photos(train_photos, photos_dir, annotations_dir, 
                                   dataset_dir / "images" / "train",
                                   dataset_dir / "labels" / "train")
    
    # Process validation photos
    print("\n📝 Processing validation photos...")
    processed_val = process_photos(val_photos, photos_dir, annotations_dir,
                                 dataset_dir / "images" / "val", 
                                 dataset_dir / "labels" / "val")
    
    # Create data.yaml file
    create_data_yaml(dataset_dir)
    
    print(f"\n🎉 DATASET CREATION COMPLETE!")
    print(f"   📂 Dataset location: {dataset_dir}")
    print(f"   🚂 Training images: {processed_train}")
    print(f"   🧪 Validation images: {processed_val}")
    print(f"   📝 Total annotations: {(processed_train + processed_val) * 2} (left + right sides)")
    
    return True


def process_photos(photo_files: List[Path], photos_dir: Path, annotations_dir: Path,
                  images_output_dir: Path, labels_output_dir: Path) -> int:
    """Process a list of photos and their annotations"""
    processed_count = 0
    
    for photo_file in photo_files:
        # Copy the original photo
        shutil.copy2(photos_dir / photo_file.name, images_output_dir / photo_file.name)
        
        # Process left side annotation
        left_annotation = annotations_dir / f"{photo_file.stem}_left.txt"
        if left_annotation.exists():
            left_output = labels_output_dir / f"{photo_file.stem}_left.txt"
            shutil.copy2(left_annotation, left_output)
        
        # Process right side annotation
        right_annotation = annotations_dir / f"{photo_file.stem}_right.txt"
        if right_annotation.exists():
            right_output = labels_output_dir / f"{photo_file.stem}_right.txt"
            shutil.copy2(right_annotation, right_output)
        
        processed_count += 1
    
    return processed_count


def create_data_yaml(dataset_dir: Path):
    """Create data.yaml file for the dataset"""
    
    # Updated 15 classes with Green tape
    classes = [
        "Wire",
        "Battery Holder", 
        "LED_1 (Yellow)",
        "LED_2 (Red)",
        "Resistor",
        "Lamp",
        "Photoresistor",
        "U_1 blue music circuit",
        "U_2 red alarm circuit", 
        "U_3 green space war circuit",
        "Speaker",
        "Slide switch",
        "Press switch",
        "Whistle chip",
        "Green tape"
    ]
    
    data_config = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    yaml_file = dataset_dir / "data.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"📋 Created data.yaml with {len(classes)} classes")
    print(f"   File: {yaml_file}")


def validate_dataset(dataset_dir: Path):
    """Validate the created dataset"""
    print("\n🔍 VALIDATING DATASET...")
    
    train_images = list((dataset_dir / "images" / "train").glob("*.jpg"))
    val_images = list((dataset_dir / "images" / "val").glob("*.jpg"))
    train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
    val_labels = list((dataset_dir / "labels" / "val").glob("*.txt"))
    
    print(f"📸 Train images: {len(train_images)}")
    print(f"📸 Val images: {len(val_images)}")
    print(f"📝 Train labels: {len(train_labels)}")
    print(f"📝 Val labels: {len(val_labels)}")
    
    # Check for missing labels
    missing_labels = 0
    for img in train_images + val_images:
        # Get the correct label directory (train or val)
        split_name = img.parent.name  # 'train' or 'val'
        expected_left = dataset_dir / "labels" / split_name / f"{img.stem}_left.txt"
        expected_right = dataset_dir / "labels" / split_name / f"{img.stem}_right.txt"
        
        if not expected_left.exists():
            print(f"⚠️  Missing left label: {expected_left}")
            missing_labels += 1
        if not expected_right.exists():
            print(f"⚠️  Missing right label: {expected_right}")
            missing_labels += 1
    
    if missing_labels == 0:
        print("✅ All images have corresponding labels")
    else:
        print(f"⚠️  {missing_labels} missing labels found")
    
    return missing_labels == 0


if __name__ == "__main__":
    success = create_photos_training_dataset()
    
    if success:
        dataset_dir = Path("photos_training_dataset")
        validate_dataset(dataset_dir)
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Train model: python train_dual_board_model.py --data {dataset_dir}/data.yaml")
        print(f"   2. Test model: python test_dual_model.py")
        print(f"   3. Run live system: python dual_board_live_system.py")
    else:
        print(f"\n❌ Dataset creation failed!")
