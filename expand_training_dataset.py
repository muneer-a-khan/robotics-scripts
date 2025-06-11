#!/usr/bin/env python3
"""
Expand Training Dataset with Additional Snap Circuit Images
"""

import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from models.component_detector import ComponentDetector
from vision.connection_detector import ConnectionDetector


def analyze_new_images():
    """Analyze all new images and provide insights."""
    print("🔍 ANALYZING NEW SNAP CIRCUIT IMAGES")
    print("="*50)
    
    images_dir = Path("images")
    image_files = []
    
    # Find all image files
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(images_dir.glob(ext))
    
    if not image_files:
        print("❌ No images found in images/ directory")
        return []
    
    print(f"📸 Found {len(image_files)} images:")
    
    analyzed_images = []
    
    for img_path in image_files:
        print(f"\n📄 Analyzing: {img_path.name}")
        
        # Load image
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"   ❌ Could not load image")
                continue
                
            height, width = image.shape[:2]
            print(f"   📐 Dimensions: {width}x{height}")
            print(f"   💾 Size: {img_path.stat().st_size / 1024:.1f} KB")
            
            # Analyze image characteristics
            analysis = analyze_image_characteristics(image)
            analysis['filename'] = img_path.name
            analysis['path'] = str(img_path)
            analysis['width'] = width
            analysis['height'] = height
            
            print(f"   🎨 Brightness: {analysis['brightness']:.1f}")
            print(f"   📊 Contrast: {analysis['contrast']:.1f}")
            print(f"   🔍 Estimated components: {analysis['estimated_components']}")
            print(f"   ⚡ Board detected: {'✅' if analysis['board_detected'] else '❌'}")
            
            analyzed_images.append(analysis)
            
        except Exception as e:
            print(f"   ❌ Error analyzing image: {e}")
    
    return analyzed_images


def analyze_image_characteristics(image):
    """Analyze image characteristics to help with annotation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Basic image metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Detect potential circuit board (look for rectangular/hexagonal patterns)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for large rectangular regions (potential boards)
    board_detected = False
    large_contours = [c for c in contours if cv2.contourArea(c) > 10000]
    if large_contours:
        board_detected = True
    
    # Estimate number of components based on color regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Count colored regions (potential components)
    component_colors = 0
    
    # Red regions (LEDs, etc.)
    red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    if np.sum(red_mask) > 1000:
        component_colors += 1
    
    # Green regions
    green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
    if np.sum(green_mask) > 1000:
        component_colors += 1
    
    # Blue regions  
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    if np.sum(blue_mask) > 1000:
        component_colors += 1
    
    estimated_components = max(2, component_colors * 2)  # Rough estimate
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'board_detected': board_detected,
        'estimated_components': estimated_components,
        'large_contours': len(large_contours)
    }


def auto_detect_components_in_images(analyzed_images):
    """Use current model to auto-detect components and suggest annotations."""
    print(f"\n🤖 AUTO-DETECTING COMPONENTS IN NEW IMAGES")
    print("="*45)
    
    # Initialize detector with current model
    try:
        detector = ComponentDetector()
        print("✅ Loaded current trained model")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        print("   Using base YOLOv8 model instead")
        detector = ComponentDetector()
    
    auto_annotations = {}
    
    for img_data in analyzed_images:
        print(f"\n🔍 Processing: {img_data['filename']}")
        
        try:
            # Load image
            image = cv2.imread(img_data['path'])
            
            # Detect components
            detections = detector.detect(image)
            
            print(f"   🔧 Found {len(detections)} components:")
            
            annotations = []
            
            for i, detection in enumerate(detections):
                print(f"      {i+1}. {detection.label} (confidence: {detection.confidence:.2f})")
                
                # Convert to YOLO format (normalized coordinates)
                img_h, img_w = image.shape[:2]
                bbox = detection.bbox
                
                x_center = (bbox.x1 + bbox.x2) / 2 / img_w
                y_center = (bbox.y1 + bbox.y2) / 2 / img_h
                width = (bbox.x2 - bbox.x1) / img_w
                height = (bbox.y2 - bbox.y1) / img_h
                
                # Get class ID
                class_mapping = {
                    "wire": 0, "switch": 1, "button": 2, "battery_holder": 3,
                    "led": 4, "speaker": 5, "music_circuit": 6, "motor": 7,
                    "resistor": 8, "connection_node": 9, "lamp": 10, "fan": 11,
                    "buzzer": 12, "photoresistor": 13, "microphone": 14, "alarm": 15
                }
                
                class_id = class_mapping.get(detection.label, 0)
                
                # Create YOLO annotation line
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                annotations.append(yolo_line)
            
            auto_annotations[img_data['filename']] = annotations
            
        except Exception as e:
            print(f"   ❌ Error processing image: {e}")
            auto_annotations[img_data['filename']] = []
    
    return auto_annotations


def create_expanded_dataset(analyzed_images, auto_annotations):
    """Create expanded dataset with new images."""
    print(f"\n📚 CREATING EXPANDED TRAINING DATASET")
    print("="*40)
    
    # Create new dataset directories
    new_dataset_dir = Path("data/expanded_training")
    
    dirs_to_create = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    for dir_name in dirs_to_create:
        (new_dataset_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Copy original training data
    print("📋 Copying original training data...")
    
    original_data_dir = Path("data/training")
    
    # Copy original image and label
    if Path("snap_circuit_image.jpg").exists():
        shutil.copy("snap_circuit_image.jpg", 
                   new_dataset_dir / "images/train/snap_circuit_image.jpg")
        shutil.copy("snap_circuit_training.txt",
                   new_dataset_dir / "labels/train/snap_circuit_image.txt")
        print("   ✅ Original training image copied")
    
    # Process new images
    print(f"\n📸 Adding {len(analyzed_images)} new images...")
    
    train_count = 0
    val_count = 0
    
    for i, img_data in enumerate(analyzed_images):
        # Determine split (80% train, 20% val)
        is_validation = (i % 5 == 0)  # Every 5th image goes to validation
        
        split_dir = "val" if is_validation else "train"
        if is_validation:
            val_count += 1
        else:
            train_count += 1
        
        # Clean filename for dataset
        original_name = Path(img_data['filename'])
        clean_name = f"circuit_{i+1:02d}{original_name.suffix}"
        
        # Copy image
        src_path = Path(img_data['path'])
        dst_image_path = new_dataset_dir / "images" / split_dir / clean_name
        shutil.copy(str(src_path), str(dst_image_path))
        
        # Create annotation file
        annotation_filename = clean_name.replace(original_name.suffix, '.txt')
        dst_label_path = new_dataset_dir / "labels" / split_dir / annotation_filename
        
        annotations = auto_annotations.get(img_data['filename'], [])
        
        if annotations:
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(annotations))
            print(f"   ✅ {clean_name} -> {split_dir} ({len(annotations)} annotations)")
        else:
            # Create empty annotation file
            dst_label_path.touch()
            print(f"   ⚠️  {clean_name} -> {split_dir} (no annotations - needs manual review)")
    
    print(f"\n📊 Dataset split:")
    print(f"   • Training: {train_count + 1} images (including original)")
    print(f"   • Validation: {val_count} images")
    
    # Create new data.yaml
    yaml_content = f"""# Expanded Snap Circuit Dataset Configuration
path: {new_dataset_dir.absolute()}  # dataset root dir
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
    
    yaml_path = new_dataset_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ Created: {yaml_path}")
    
    return str(yaml_path)


def train_expanded_model(data_yaml_path):
    """Train model on expanded dataset."""
    print(f"\n🏋️  TRAINING EXPANDED MODEL")
    print("="*30)
    
    try:
        detector = ComponentDetector()
        
        print("🚀 Starting training on expanded dataset...")
        print(f"📁 Dataset: {data_yaml_path}")
        print("⏱️  This may take 10-20 minutes depending on your hardware...")
        
        results = detector.train(
            data_yaml_path=data_yaml_path,
            epochs=100,       # More epochs for better learning
            imgsz=640,
            batch=2,          # Slightly larger batch size
            lr0=0.01,
            patience=30,      # More patience for convergence
            device='cpu',
            name='expanded_snap_circuit_model'
        )
        
        print("✅ Training completed!")
        print("📁 Model saved to: models/weights/")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


def main():
    """Main function to expand and retrain the dataset."""
    print("🤖 EXPANDING SNAP CIRCUIT TRAINING DATASET")
    print("="*60)
    
    # Step 1: Analyze new images
    analyzed_images = analyze_new_images()
    
    if not analyzed_images:
        print("❌ No images found to process")
        return
    
    # Step 2: Auto-detect components
    auto_annotations = auto_detect_components_in_images(analyzed_images)
    
    # Step 3: Create expanded dataset
    data_yaml_path = create_expanded_dataset(analyzed_images, auto_annotations)
    
    # Step 4: Train expanded model
    print(f"\n🎯 READY TO TRAIN EXPANDED MODEL")
    print("="*35)
    print("The expanded dataset has been created with:")
    print(f"   • Original perfectly annotated image")
    print(f"   • {len(analyzed_images)} new images with auto-generated annotations")
    print(f"   • Mix of training and validation data")
    print()
    
    response = input("🤖 Start training now? (y/n): ").lower().strip()
    
    if response == 'y' or response == 'yes':
        results = train_expanded_model(data_yaml_path)
        
        if results:
            print(f"\n🎉 SUCCESS!")
            print("="*15)
            print("✅ Expanded model training completed")
            print("✅ Model now trained on multiple Snap Circuit images")
            print("✅ Should handle various angles and lighting conditions better")
            print()
            print("🧪 Test the improved model with:")
            print("   python main.py --mode image --input images/[any_image]")
            print("   python main.py --mode camera")
        else:
            print("❌ Training failed. Check the error messages above.")
    else:
        print("⏸️  Training skipped. You can train later with:")
        print(f"   python -c \"from models.component_detector import ComponentDetector; d=ComponentDetector(); d.train('{data_yaml_path}', epochs=100, device='cpu', name='expanded_model')\"")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   • {data_yaml_path} - Expanded dataset configuration")
    print(f"   • data/expanded_training/ - Full expanded dataset")
    print(f"   • Auto-generated annotations for all new images")


if __name__ == "__main__":
    main() 