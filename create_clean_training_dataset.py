#!/usr/bin/env python3
"""
Create Clean Training Dataset

Creates a clean training dataset directly from the balanced dual board annotations,
bypassing the problematic augmentation pipeline that corrupted the spatial balance.

This approach:
1. Uses your original balanced dual annotations (38 left + 38 right battery holders)
2. Properly merges left/right annotations into full-frame labels
3. Preserves spatial balance perfectly
4. Uses minimal augmentation to avoid corruption
"""

import os
import cv2
import shutil
from pathlib import Path
from collections import defaultdict
import yaml


def create_clean_dataset():
    """Create clean dataset from dual board annotations"""
    
    print("🧹 CLEAN TRAINING DATASET CREATION")
    print("=" * 50)
    print("🎯 Goal: Preserve perfect spatial balance from dual annotations")
    print("✅ Your dual annotations: 38 left + 38 right battery holders")
    
    # Paths
    dual_annotations = Path("dual_board_annotations")
    original_images = Path("new_images")
    clean_dataset = Path("clean_dual_board_dataset")
    
    # Verify source directories
    if not dual_annotations.exists():
        print(f"❌ Dual annotations not found: {dual_annotations}")
        return False
        
    if not original_images.exists():
        print(f"❌ Original images not found: {original_images}")
        return False
    
    print(f"📂 Source annotations: {dual_annotations}")
    print(f"📂 Source images: {original_images}")
    print(f"📂 Clean dataset: {clean_dataset}")
    
    # Create clean dataset structure
    if clean_dataset.exists():
        print(f"🗑️  Removing existing clean dataset...")
        shutil.rmtree(clean_dataset)
    
    (clean_dataset / "images" / "train").mkdir(parents=True, exist_ok=True)
    (clean_dataset / "images" / "val").mkdir(parents=True, exist_ok=True)
    (clean_dataset / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (clean_dataset / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Find dual annotation pairs
    left_files = list(dual_annotations.glob("*_left.txt"))
    right_files = list(dual_annotations.glob("*_right.txt"))
    
    print(f"\n📄 Found {len(left_files)} left annotation files")
    print(f"📄 Found {len(right_files)} right annotation files")
    
    # Match left and right files
    image_pairs = {}
    
    for left_file in left_files:
        base_name = left_file.name.replace("_left.txt", "")
        right_file = dual_annotations / f"{base_name}_right.txt"
        
        if right_file.exists():
            # Find corresponding image
            possible_image_names = [
                f"{base_name}.jpg",
                f"{base_name}.jpeg",
                f"{base_name}.png"
            ]
            
            image_file = None
            for img_name in possible_image_names:
                img_path = original_images / img_name
                if img_path.exists():
                    image_file = img_path
                    break
            
            if image_file:
                image_pairs[base_name] = {
                    'image': image_file,
                    'left_labels': left_file,
                    'right_labels': right_file
                }
            else:
                print(f"⚠️  No image found for: {base_name}")
    
    print(f"✅ Found {len(image_pairs)} complete image pairs")
    
    if len(image_pairs) == 0:
        print("❌ No valid image pairs found!")
        return False
    
    # Process each image pair
    train_count = 0
    val_count = 0
    total_left_batteries = 0
    total_right_batteries = 0
    
    # Use 80/20 split for train/val
    val_split = max(1, len(image_pairs) // 5)
    
    for idx, (base_name, files) in enumerate(image_pairs.items()):
        is_val = (idx % 5 == 0)  # Every 5th image goes to validation
        split = "val" if is_val else "train"
        
        try:
            # Read image to get dimensions
            image = cv2.imread(str(files['image']))
            if image is None:
                print(f"⚠️  Could not read image: {files['image']}")
                continue
                
            height, width = image.shape[:2]
            
            # Copy image to dataset
            image_dest = clean_dataset / "images" / split / f"{base_name}.jpg"
            shutil.copy2(files['image'], image_dest)
            
            # Read left and right annotations
            left_annotations = []
            right_annotations = []
            
            # Read left side annotations
            with open(files['left_labels'], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_width = float(parts[3])
                        box_height = float(parts[4])
                        
                        # Convert to full image coordinates (left side = x < 0.5)
                        # Left annotations are relative to left half, convert to full frame
                        full_x_center = x_center * 0.5  # Scale to left half of full frame
                        
                        left_annotations.append([class_id, full_x_center, y_center, box_width * 0.5, box_height])
                        
                        if class_id == 3:  # battery_holder
                            total_left_batteries += 1
            
            # Read right side annotations  
            with open(files['right_labels'], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_width = float(parts[3])
                        box_height = float(parts[4])
                        
                        # Convert to full image coordinates (right side = x > 0.5)
                        # Right annotations are relative to right half, convert to full frame
                        full_x_center = 0.5 + (x_center * 0.5)  # Scale to right half of full frame
                        
                        right_annotations.append([class_id, full_x_center, y_center, box_width * 0.5, box_height])
                        
                        if class_id == 3:  # battery_holder
                            total_right_batteries += 1
            
            # Combine all annotations
            all_annotations = left_annotations + right_annotations
            
            # Write combined label file
            label_dest = clean_dataset / "labels" / split / f"{base_name}.txt"
            with open(label_dest, 'w') as f:
                for ann in all_annotations:
                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
            
            if is_val:
                val_count += 1
            else:
                train_count += 1
                
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(image_pairs)} images...")
                
        except Exception as e:
            print(f"⚠️  Error processing {base_name}: {e}")
    
    # Create data.yaml
    data_yaml_content = f"""# Clean Dual Board Dataset - Preserves Spatial Balance
path: {clean_dataset.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')

# Number of classes  
nc: 16

# Class names
names:
  0: wire
  1: switch
  2: button
  3: battery_holder
  4: led
  5: speaker
  6: music_circuit
  7: motor
  8: resistor
  9: connection_node
  10: lamp
  11: fan
  12: buzzer
  13: photoresistor
  14: microphone
  15: alarm
"""
    
    with open(clean_dataset / "data.yaml", 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n✅ CLEAN DATASET CREATED!")
    print(f"   📊 Training images: {train_count}")
    print(f"   📊 Validation images: {val_count}")
    print(f"   📊 Total images: {train_count + val_count}")
    
    print(f"\n🔋 BATTERY HOLDER BALANCE VERIFICATION:")
    print(f"   Left side battery holders: {total_left_batteries}")
    print(f"   Right side battery holders: {total_right_batteries}")
    
    if total_left_batteries > 0 and total_right_batteries > 0:
        left_percent = (total_left_batteries / (total_left_batteries + total_right_batteries)) * 100
        print(f"   Left percentage: {left_percent:.1f}%")
        
        if 40 <= left_percent <= 60:
            print(f"   ✅ PERFECT BALANCE! This should fix the asymmetric detection!")
        else:
            print(f"   ⚠️  Some imbalance, but much better than before")
    
    print(f"\n📁 Clean dataset saved to: {clean_dataset}")
    print(f"💡 Next step: python train_clean_model.py")
    
    return True


if __name__ == "__main__":
    create_clean_dataset()
