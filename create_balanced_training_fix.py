#!/usr/bin/env python3
"""
Create Balanced Training Fix

Creates horizontally flipped copies of training images to rebalance the spatial distribution.
This fixes the augmentation bias without re-annotating everything.
"""

import cv2
import shutil
from pathlib import Path
from collections import defaultdict


def create_balanced_fix():
    """Create flipped copies to rebalance training data"""
    
    print("🔧 TRAINING DATA REBALANCING FIX")
    print("=" * 50)
    
    # Paths
    train_images = Path("dual_board_augmented_dataset/images/train")
    train_labels = Path("dual_board_augmented_dataset/labels/train") 
    
    if not train_images.exists() or not train_labels.exists():
        print("❌ Training directories not found!")
        return
    
    # Find images and labels
    image_files = list(train_images.glob("*.jpg"))
    label_files = list(train_labels.glob("*.txt"))
    
    print(f"📂 Found {len(image_files)} training images")
    print(f"📄 Found {len(label_files)} training labels")
    
    # Analyze current bias
    print("\n🔍 Analyzing current spatial bias...")
    left_batteries = 0
    right_batteries = 0
    total_annotations = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    
                    total_annotations += 1
                    
                    # Battery holder class (assuming it's class 3 based on previous analysis)
                    if class_id == 3:  # battery_holder
                        if x_center < 0.5:
                            left_batteries += 1
                        else:
                            right_batteries += 1
                            
        except Exception as e:
            print(f"⚠️  Error reading {label_file.name}: {e}")
    
    print(f"\n📊 CURRENT BATTERY HOLDER DISTRIBUTION:")
    print(f"   Left side: {left_batteries}")
    print(f"   Right side: {right_batteries}")
    
    if left_batteries == 0:
        print(f"   🚨 CRITICAL: No left battery holders!")
        fix_needed = True
    elif right_batteries / (left_batteries + right_batteries) > 0.7:
        print(f"   ⚠️  Heavy right bias detected")
        fix_needed = True
    else:
        print(f"   ✅ Distribution looks reasonable")
        fix_needed = False
    
    if not fix_needed:
        print("\n💡 Training data appears balanced. No fix needed.")
        return
    
    print(f"\n🔧 CREATING BALANCED FIX...")
    
    # Create backup
    backup_dir = Path("dual_board_augmented_dataset_backup")
    if not backup_dir.exists():
        print("📦 Creating backup...")
        shutil.copytree("dual_board_augmented_dataset", backup_dir)
        print("✅ Backup created")
    
    # Create flipped copies of images with right-heavy battery holders
    flipped_count = 0
    processed_files = 0
    
    for label_file in label_files:
        try:
            # Read annotations
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Check if this file has right-heavy battery holders
            right_battery_count = 0
            left_battery_count = 0
            
            annotations = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    
                    if class_id == 3:  # battery_holder
                        if x_center < 0.5:
                            left_battery_count += 1
                        else:
                            right_battery_count += 1
                    
                    annotations.append(parts)
            
            # If this file contributes to right bias, create flipped version
            if right_battery_count > left_battery_count:
                # Find corresponding image
                image_name = label_file.stem + ".jpg"
                image_path = train_images / image_name
                
                if image_path.exists():
                    # Read and flip image
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        flipped_img = cv2.flip(img, 1)  # Horizontal flip
                        
                        # Create flipped filenames
                        flipped_image_name = f"{label_file.stem}_flipped.jpg"
                        flipped_label_name = f"{label_file.stem}_flipped.txt"
                        
                        flipped_image_path = train_images / flipped_image_name
                        flipped_label_path = train_labels / flipped_label_name
                        
                        # Save flipped image
                        cv2.imwrite(str(flipped_image_path), flipped_img)
                        
                        # Create flipped annotations
                        flipped_annotations = []
                        for parts in annotations:
                            if len(parts) >= 5:
                                class_id = parts[0]
                                x_center = 1.0 - float(parts[1])  # Flip x coordinate
                                y_center = parts[2]
                                width = parts[3] 
                                height = parts[4]
                                
                                flipped_line = f"{class_id} {x_center:.6f} {y_center} {width} {height}\n"
                                flipped_annotations.append(flipped_line)
                        
                        # Save flipped labels
                        with open(flipped_label_path, 'w') as f:
                            f.writelines(flipped_annotations)
                        
                        flipped_count += 1
                
                processed_files += 1
                if processed_files % 50 == 0:
                    print(f"   Processed {processed_files}/{len(label_files)} files...")
                    
        except Exception as e:
            print(f"⚠️  Error processing {label_file.name}: {e}")
    
    print(f"\n✅ REBALANCING COMPLETE!")
    print(f"   📸 Created {flipped_count} flipped image pairs")
    print(f"   📂 Total training images now: {len(list(train_images.glob('*.jpg')))}")
    print(f"   📄 Total training labels now: {len(list(train_labels.glob('*.txt')))}")
    
    # Verify the fix
    print(f"\n🔍 VERIFYING REBALANCED DATA...")
    new_left_batteries = 0
    new_right_batteries = 0
    
    for label_file in train_labels.glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    
                    if class_id == 3:  # battery_holder
                        if x_center < 0.5:
                            new_left_batteries += 1
                        else:
                            new_right_batteries += 1
                            
        except Exception as e:
            continue
    
    print(f"\n📊 NEW BATTERY HOLDER DISTRIBUTION:")
    print(f"   Left side: {new_left_batteries}")
    print(f"   Right side: {new_right_batteries}")
    
    if new_left_batteries > 0:
        total_new = new_left_batteries + new_right_batteries
        left_percent = (new_left_batteries / total_new) * 100
        print(f"   Left percentage: {left_percent:.1f}%")
        
        if 30 <= left_percent <= 70:
            print(f"   ✅ Much better balance achieved!")
        else:
            print(f"   ⚠️  Some imbalance remains but improved")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. ✅ Training data is now rebalanced")
    print(f"   2. 🔄 Retrain your model with this balanced dataset")
    print(f"   3. 🧪 Test the new model for symmetric detection")
    print(f"   4. 📦 Original data backed up to: dual_board_augmented_dataset_backup")
    

if __name__ == "__main__":
    create_balanced_fix()
