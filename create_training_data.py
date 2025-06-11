"""
Script to create training data from the Snap Circuit image.
Based on visual analysis of the uploaded image.
"""

import os
from pathlib import Path


def create_snap_circuit_training_data():
    """Create training annotation for the Snap Circuit image."""
    
    print("🔧 CREATING SNAP CIRCUIT TRAINING DATA")
    print("="*50)
    
    # Based on visual analysis of the uploaded image, create manual annotations
    # The image shows:
    # 1. Battery holder at the top center
    # 2. Green multi-connection component on the left 
    # 3. Blue component on the right
    # 4. Small red component (LED) in the center
    
    # Create YOLO format annotations (class_id x_center y_center width height)
    # All coordinates are normalized (0-1)
    
    annotations = []
    
    # Battery holder (top center) - class_id 3 for battery_holder
    annotations.append("3 0.500 0.250 0.280 0.200")
    
    # Green component (left side) - class_id 1 for switch
    annotations.append("1 0.225 0.555 0.120 0.220")
    
    # Blue component (right side) - class_id 1 for switch  
    annotations.append("1 0.775 0.555 0.120 0.220")
    
    # Red LED component (center) - class_id 4 for led
    annotations.append("4 0.500 0.650 0.060 0.060")
    
    # Additional snap connection points - class_id 9 for connection_node
    # These are the metallic hexagonal connection points visible on the board
    snap_points = [
        "9 0.300 0.400 0.040 0.040",  # Left side snap points
        "9 0.400 0.400 0.040 0.040",
        "9 0.500 0.400 0.040 0.040",  # Center snap points
        "9 0.600 0.400 0.040 0.040",
        "9 0.700 0.400 0.040 0.040",  # Right side snap points
        "9 0.350 0.500 0.040 0.040",  # Lower row
        "9 0.450 0.500 0.040 0.040",
        "9 0.550 0.500 0.040 0.040",
        "9 0.650 0.500 0.040 0.040",
    ]
    
    annotations.extend(snap_points)
    
    # Save annotation file
    annotation_file = "snap_circuit_training.txt"
    with open(annotation_file, 'w') as f:
        f.write('\n'.join(annotations))
    
    print(f"✅ Created training annotation: {annotation_file}")
    print(f"📊 Annotation contains {len(annotations)} objects:")
    
    # Component mapping for display
    class_names = [
        "wire", "switch", "button", "battery_holder", "led", "speaker",
        "music_circuit", "motor", "resistor", "connection_node", "lamp",
        "fan", "buzzer", "photoresistor", "microphone", "alarm"
    ]
    
    component_counts = {}
    for ann in annotations:
        class_id = int(ann.split()[0])
        class_name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
        component_counts[class_name] = component_counts.get(class_name, 0) + 1
    
    for component, count in component_counts.items():
        print(f"   • {count}x {component}")
    
    return annotation_file


def create_dataset_structure():
    """Create the proper dataset structure for YOLOv8 training."""
    
    print(f"\n📁 CREATING DATASET STRUCTURE")
    print("="*30)
    
    # Create directories
    dirs_to_create = [
        "data/training/images/train",
        "data/training/images/val", 
        "data/training/images/test",
        "data/training/labels/train",
        "data/training/labels/val",
        "data/training/labels/test"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    # Create data.yaml configuration
    yaml_content = """# Snap Circuit Dataset Configuration
path: data/training  # dataset root dir
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
    
    yaml_file = "data/training/data.yaml"
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ Created: {yaml_file}")
    
    return yaml_file


def create_sample_images():
    """Create instructions for adding the training image."""
    
    print(f"\n🖼️  IMAGE SETUP INSTRUCTIONS")
    print("="*30)
    
    print("To complete the training setup:")
    print("1. Save your Snap Circuit image as: 'snap_circuit_image.jpg'")
    print("2. Copy it to: data/training/images/train/snap_circuit_image.jpg")
    print("3. Copy the annotation to: data/training/labels/train/snap_circuit_image.txt")
    print("4. For validation, copy the same files to the 'val' directories")
    
    # Create a simple script to help with file copying
    copy_script = """# Copy training files
import shutil
from pathlib import Path

# Assuming you save your image as 'snap_circuit_image.jpg'
if Path('snap_circuit_image.jpg').exists():
    # Copy to train directory
    shutil.copy('snap_circuit_image.jpg', 'data/training/images/train/')
    shutil.copy('snap_circuit_training.txt', 'data/training/labels/train/snap_circuit_image.txt')
    
    # Copy to val directory for validation
    shutil.copy('snap_circuit_image.jpg', 'data/training/images/val/')
    shutil.copy('snap_circuit_training.txt', 'data/training/labels/val/snap_circuit_image.txt')
    
    print("✅ Training files copied successfully!")
else:
    print("❌ Please save your image as 'snap_circuit_image.jpg' first")
"""
    
    with open('copy_training_files.py', 'w', encoding='utf-8') as f:
        f.write(copy_script)
    
    print("5. Run: python copy_training_files.py")
    print("6. Then train with: python train_model.py")


def create_training_script():
    """Create a simple training script."""
    
    print(f"\n🏋️  CREATING TRAINING SCRIPT")
    print("="*30)
    
    training_script = """#!/usr/bin/env python3
\"\"\"
Simple script to train YOLOv8 on Snap Circuit data.
\"\"\"

from models.component_detector import ComponentDetector
import os

def train_snap_circuit_model():
    print("🚀 Starting Snap Circuit Model Training...")
    
    # Check if dataset exists
    data_yaml = "data/training/data.yaml"
    if not os.path.exists(data_yaml):
        print("❌ Dataset not found. Run create_training_data.py first!")
        return
    
    # Initialize detector
    detector = ComponentDetector()
    
    # Train the model
    print(f"Training on dataset: {data_yaml}")
    results = detector.train(
        data_yaml_path=data_yaml,
        epochs=50,        # Start with fewer epochs for testing
        imgsz=640,
        batch=1,          # Small batch size for single image
        lr0=0.01,
        patience=20
    )
    
    print("✅ Training complete!")
    print("Model saved to: models/weights/snap_circuit_yolov8.pt")
    
    return results

if __name__ == "__main__":
    train_snap_circuit_model()
"""
    
    with open('train_model.py', 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    print("✅ Created: train_model.py")


def main():
    """Main function to set up everything for training."""
    
    print("🤖 SNAP CIRCUIT TRAINING DATA SETUP")
    print("="*60)
    
    # Step 1: Create training annotation
    annotation_file = create_snap_circuit_training_data()
    
    # Step 2: Create dataset structure
    yaml_file = create_dataset_structure()
    
    # Step 3: Create setup instructions
    create_sample_images()
    
    # Step 4: Create training script
    create_training_script()
    
    print(f"\n🎯 NEXT STEPS:")
    print("="*20)
    print("1. Save your Snap Circuit image as 'snap_circuit_image.jpg'")
    print("2. Run: python copy_training_files.py")
    print("3. Run: python train_model.py")
    print("4. Test with: python main.py --mode image --input snap_circuit_image.jpg")
    
    print(f"\n📚 FILES CREATED:")
    print(f"   • {annotation_file} - Training annotations")
    print(f"   • {yaml_file} - Dataset configuration")
    print(f"   • copy_training_files.py - File copying helper")
    print(f"   • train_model.py - Training script")
    
    print(f"\n✨ Ready to train your Snap Circuit detector!")


if __name__ == "__main__":
    main() 