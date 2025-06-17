#!/usr/bin/env python3
"""
Create training dataset with individual component images.
This will help improve model accuracy by training with diverse component views.
"""

import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import os


def create_component_directory_structure():
    """Create directory structure for individual component images."""
    print("📁 Creating directory structure for individual components...")
    
    # Create main directory
    components_dir = Path("data/individual_components")
    components_dir.mkdir(parents=True, exist_ok=True)
    
    # Component classes from config
    component_classes = [
        "wire", "switch", "button", "battery_holder", "led", "speaker",
        "music_circuit", "motor", "resistor", "connection_node", "lamp",
        "fan", "buzzer", "photoresistor", "microphone", "alarm"
    ]
    
    # Create subdirectories for each component
    for component in component_classes:
        component_dir = components_dir / component
        component_dir.mkdir(exist_ok=True)
        print(f"✅ Created directory: {component_dir}")
    
    # Create README with instructions
    readme_content = """# Individual Component Images

## Instructions for Taking Photos:

1. **Lighting**: Use bright, even lighting. Avoid harsh shadows.
2. **Background**: Use a plain white or light-colored background.
3. **Angle**: Take photos from multiple angles:
   - Top-down view (main)
   - 30-degree angle
   - Side view
4. **Distance**: Fill 60-80% of the frame with the component.
5. **Focus**: Ensure the component is in sharp focus.
6. **Quantity**: Take 5-10 images per component for best results.

## Directory Structure:
Each component should have its own folder with multiple images:
- battery_holder/: 5-10 images of battery holders
- led/: 5-10 images of LEDs (different colors)
- switch/: 5-10 images of switches
- etc.

## File Naming:
Use descriptive names like:
- battery_holder_01.jpg
- battery_holder_02.jpg
- led_red_01.jpg
- led_blue_01.jpg
"""
    
    with open(components_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Created README with instructions")
    return components_dir


def process_individual_component_images(components_dir: Path):
    """Process individual component images and create training annotations."""
    print("🔍 Processing individual component images...")
    
    # Component class mapping
    class_mapping = {
        "wire": 0, "switch": 1, "button": 2, "battery_holder": 3,
        "led": 4, "speaker": 5, "music_circuit": 6, "motor": 7,
        "resistor": 8, "connection_node": 9, "lamp": 10, "fan": 11,
        "buzzer": 12, "photoresistor": 13, "microphone": 14, "alarm": 15
    }
    
    processed_images = []
    
    for component_name in class_mapping.keys():
        component_dir = components_dir / component_name
        
        if not component_dir.exists():
            continue
        
        # Find all image files in component directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(component_dir.glob(ext))
        
        if not image_files:
            print(f"⚠️  No images found in {component_dir}")
            continue
        
        print(f"📸 Processing {len(image_files)} images for {component_name}")
        
        for img_path in image_files:
            try:
                # Load and validate image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"   ❌ Could not load {img_path}")
                    continue
                
                height, width = image.shape[:2]
                
                # Create annotation for full-frame component
                # Assume component fills 60-80% of frame, centered
                class_id = class_mapping[component_name]
                x_center = 0.5
                y_center = 0.5
                bbox_width = 0.7  # 70% of image width
                bbox_height = 0.7  # 70% of image height
                
                annotation = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}"
                
                processed_images.append({
                    'image_path': str(img_path),
                    'annotation': annotation,
                    'component': component_name,
                    'width': width,
                    'height': height
                })
                
            except Exception as e:
                print(f"   ❌ Error processing {img_path}: {e}")
    
    print(f"✅ Processed {len(processed_images)} individual component images")
    return processed_images


def create_augmented_dataset(processed_images: List[Dict]):
    """Create augmented dataset by combining individual components with original images."""
    print("🎯 Creating augmented training dataset...")
    
    # Create new dataset directory
    augmented_dir = Path("data/augmented_training")
    
    dirs_to_create = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    for dir_name in dirs_to_create:
        (augmented_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Copy original training data if it exists
    original_train_dir = Path("data/training")
    if original_train_dir.exists():
        print("📋 Copying original training data...")
        
        # Copy original images and labels
        for split in ["train", "val"]:
            orig_img_dir = original_train_dir / "images" / split
            orig_lbl_dir = original_train_dir / "labels" / split
            
            if orig_img_dir.exists():
                for img_file in orig_img_dir.glob("*"):
                    shutil.copy2(img_file, augmented_dir / "images" / split)
            
            if orig_lbl_dir.exists():
                for lbl_file in orig_lbl_dir.glob("*"):
                    shutil.copy2(lbl_file, augmented_dir / "labels" / split)
    
    # Add individual component images
    print("📸 Adding individual component images...")
    
    # Split images: 70% train, 20% val, 10% test
    np.random.shuffle(processed_images)
    n_train = int(0.7 * len(processed_images))
    n_val = int(0.2 * len(processed_images))
    
    splits = {
        'train': processed_images[:n_train],
        'val': processed_images[n_train:n_train + n_val],
        'test': processed_images[n_train + n_val:]
    }
    
    for split_name, images in splits.items():
        print(f"   Adding {len(images)} images to {split_name} split")
        
        for i, img_data in enumerate(images):
            # Create unique filename
            component = img_data['component']
            filename = f"{component}_{i:03d}.jpg"
            
            # Copy image
            src_path = img_data['image_path']
            dst_img_path = augmented_dir / "images" / split_name / filename
            shutil.copy2(src_path, dst_img_path)
            
            # Create annotation file
            annotation_filename = filename.replace('.jpg', '.txt')
            dst_lbl_path = augmented_dir / "labels" / split_name / annotation_filename
            
            with open(dst_lbl_path, 'w') as f:
                f.write(img_data['annotation'])
    
    # Create updated data.yaml
    create_augmented_data_yaml(augmented_dir, len(processed_images))
    
    print(f"✅ Created augmented dataset with {len(processed_images)} additional images")
    return augmented_dir


def create_augmented_data_yaml(dataset_dir: Path, num_individual_images: int):
    """Create data.yaml for augmented dataset."""
    yaml_content = f"""# Augmented Snap Circuit Dataset Configuration
path: {dataset_dir}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')
test: images/test    # test images (relative to 'path')

# Number of classes
nc: 16

# Dataset info
# Original circuit board images + {num_individual_images} individual component images

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
    
    yaml_file = dataset_dir / "data.yaml"
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ Created augmented data.yaml: {yaml_file}")


def create_improved_training_script():
    """Create an improved training script with better parameters."""
    script_content = '''#!/usr/bin/env python3
"""
Improved training script for Snap Circuit component detection.
Uses augmented dataset with individual component images.
"""

import os
from pathlib import Path
from models.component_detector import ComponentDetector


def train_improved_model():
    """Train model with improved parameters and augmented dataset."""
    print("🚀 Starting Improved Snap Circuit Model Training...")
    print("="*50)
    
    # Check for augmented dataset
    augmented_data_yaml = "data/augmented_training/data.yaml"
    
    if not os.path.exists(augmented_data_yaml):
        print("❌ Augmented dataset not found!")
        print("   Run: python create_individual_component_dataset.py first")
        return
    
    # Initialize detector
    detector = ComponentDetector()
    
    # Improved training parameters
    training_params = {
        "data": augmented_data_yaml,
        "epochs": 200,          # More epochs for better learning
        "imgsz": 640,           # Standard YOLO image size
        "batch": 8,             # Larger batch size if memory allows
        "lr0": 0.001,           # Lower learning rate for fine-tuning
        "patience": 50,         # More patience for convergence
        "device": "cpu",        # Use "cuda" if GPU available
        "optimizer": "AdamW",   # Better optimizer
        "cos_lr": True,         # Cosine learning rate schedule
        "augment": True,        # Enable data augmentation
        "mixup": 0.1,           # Mixup augmentation
        "mosaic": 0.5,          # Mosaic augmentation
        "copy_paste": 0.1,      # Copy-paste augmentation
        "save_period": 10,      # Save every 10 epochs
        "val": True,            # Enable validation
        "plots": True,          # Generate training plots
        "verbose": True         # Verbose output
    }
    
    print("📊 Training Parameters:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    # Train the model
    print("\\n🏋️ Starting training...")
    results = detector.train(**training_params)
    
    print("✅ Training complete!")
    print(f"📈 Results saved to: {results.save_dir}")
    
    # Validate the model
    print("\\n📊 Running validation...")
    val_results = detector.validate(augmented_data_yaml)
    
    print("✅ Validation complete!")
    print(f"📈 mAP50: {val_results.box.map50:.3f}")
    print(f"📈 mAP50-95: {val_results.box.map:.3f}")
    
    return results


if __name__ == "__main__":
    train_improved_model()
'''
    
    with open("train_improved_model.py", "w", encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Created improved training script: train_improved_model.py")


def main():
    """Main function to set up individual component training."""
    print("🎯 SNAP CIRCUIT INDIVIDUAL COMPONENT TRAINING SETUP")
    print("="*60)
    
    # Step 1: Create directory structure
    components_dir = create_component_directory_structure()
    
    print("\\n" + "="*60)
    print("📋 NEXT STEPS:")
    print("="*60)
    print("1. Take 5-10 photos of each component type")
    print("2. Save them in the appropriate directories:")
    print(f"   {components_dir}/battery_holder/")
    print(f"   {components_dir}/led/")
    print(f"   {components_dir}/switch/")
    print("   etc.")
    print("\\n3. Run this script again to process the images:")
    print("   python create_individual_component_dataset.py --process")
    print("\\n4. Train the improved model:")
    print("   python train_improved_model.py")
    
    # Check if --process flag is provided
    import sys
    if "--process" in sys.argv:
        print("\\n🔄 Processing individual component images...")
        processed_images = process_individual_component_images(components_dir)
        
        if processed_images:
            augmented_dir = create_augmented_dataset(processed_images)
            create_improved_training_script()
            
            print("\\n✅ Dataset preparation complete!")
            print(f"📁 Augmented dataset created at: {augmented_dir}")
            print("\\n🏋️ Ready to train improved model:")
            print("   python train_improved_model.py")
        else:
            print("\\n⚠️  No component images found. Please add images first.")


if __name__ == "__main__":
    main() 