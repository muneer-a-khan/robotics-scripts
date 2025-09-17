#!/usr/bin/env python3
"""
Fix Dataset Path for Different Computers

Updates the data.yaml file to use the correct path for the current computer.
"""

import os
from pathlib import Path
import sys


def fix_dataset_path():
    """Fix the dataset path in data.yaml for the current computer"""
    
    print("🔧 DATASET PATH FIX")
    print("=" * 30)
    
    # Find the data.yaml file
    data_yaml = Path("dual_board_augmented_dataset/data.yaml")
    
    if not data_yaml.exists():
        print("❌ data.yaml not found!")
        print(f"   Looking for: {data_yaml.absolute()}")
        return False
    
    # Get current working directory
    current_dir = Path.cwd()
    dataset_dir = current_dir / "dual_board_augmented_dataset"
    
    print(f"📂 Current directory: {current_dir}")
    print(f"📊 Dataset directory: {dataset_dir}")
    
    # Check if dataset exists
    if not dataset_dir.exists():
        print("❌ Dataset directory doesn't exist!")
        print(f"   Expected: {dataset_dir}")
        
        # Check if it exists elsewhere
        possible_locations = [
            current_dir / "dual_board_augmented_dataset",
            current_dir.parent / "dual_board_augmented_dataset", 
            Path.home() / "dual_board_augmented_dataset",
        ]
        
        print("\n🔍 Checking possible locations:")
        for location in possible_locations:
            if location.exists():
                print(f"   ✅ Found at: {location}")
                dataset_dir = location
                break
            else:
                print(f"   ❌ Not at: {location}")
        
        if not dataset_dir.exists():
            print("\n💡 You may need to:")
            print("   1. Copy dual_board_augmented_dataset to this computer")
            print("   2. Or run the rebalancing script again")
            return False
    
    # Read current data.yaml
    with open(data_yaml, 'r') as f:
        content = f.read()
    
    print(f"\n📄 Current data.yaml content:")
    print(content)
    
    # Update the path
    new_content = f"""# Dual Board Snap Circuit Augmented Dataset
path: {dataset_dir}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')
test: images/test    # test images (relative to 'path')

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
    
    # Write updated data.yaml
    with open(data_yaml, 'w') as f:
        f.write(new_content)
    
    print(f"\n✅ Updated data.yaml with correct path:")
    print(f"   New path: {dataset_dir}")
    
    # Verify the directories exist
    train_dir = dataset_dir / "images" / "train"
    val_dir = dataset_dir / "images" / "val" 
    labels_train = dataset_dir / "labels" / "train"
    labels_val = dataset_dir / "labels" / "val"
    
    print(f"\n🔍 Verifying dataset structure:")
    
    dirs_to_check = [
        ("Train images", train_dir),
        ("Val images", val_dir),
        ("Train labels", labels_train),
        ("Val labels", labels_val)
    ]
    
    all_good = True
    for name, dir_path in dirs_to_check:
        if dir_path.exists():
            count = len(list(dir_path.glob("*")))
            print(f"   ✅ {name}: {count} files")
        else:
            print(f"   ❌ {name}: Missing")
            all_good = False
    
    if all_good:
        print(f"\n🎉 Dataset is ready! You can now run:")
        print(f"   python train_dual_board_model.py --data dual_board_augmented_dataset/data.yaml --name rebalanced_model")
    else:
        print(f"\n⚠️  Some directories are missing. You may need to:")
        print(f"   1. Copy the complete dual_board_augmented_dataset folder")
        print(f"   2. Or run python create_balanced_training_fix.py")
    
    return all_good


if __name__ == "__main__":
    fix_dataset_path()
