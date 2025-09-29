#!/usr/bin/env python3
"""
Test Photos Model

Test the trained dual board model on some sample images
to verify it's working correctly with the new classes.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import random


def test_model():
    """Test the trained photos model"""
    print("🧪 TESTING PHOTOS MODEL")
    print("=" * 60)
    
    # Find the trained model
    model_path = Path("dual_board_training/photos_model_fixed/weights/best.pt")
    
    if not model_path.exists():
        # Try alternative paths
        alt_paths = [
            Path("dual_board_training/photos_model_fixed/weights/last.pt"),
            Path("dual_board_training/photos_model/weights/best.pt"),
            Path("dual_board_training/photos_model/weights/last.pt")
        ]
        
        model_path = None
        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                break
        
        if not model_path:
            print("❌ No trained model found!")
            print("   Expected locations:")
            print("   • dual_board_training/photos_model_fixed/weights/best.pt")
            print("   • dual_board_training/photos_model_fixed/weights/last.pt")
            return
    
    print(f"🤖 Model: {model_path}")
    
    # Load model
    try:
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {model.names}")
        print(f"   Number of classes: {len(model.names)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Find test images
    photos_dir = Path("photos")
    test_images = list(photos_dir.glob("*.jpg"))
    
    if not test_images:
        print("❌ No test images found in photos directory")
        return
    
    # Select random test images
    num_test = min(5, len(test_images))
    test_images = random.sample(test_images, num_test)
    
    print(f"🖼️  Testing on {num_test} images...")
    
    # Create output directory
    output_dir = Path("photos_model_test_results")
    output_dir.mkdir(exist_ok=True)
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n📸 Testing image {i}/{num_test}: {img_path.name}")
        
        try:
            # Run inference
            results = model(str(img_path), conf=0.3, iou=0.5)
            
            if results and len(results) > 0:
                result = results[0]
                
                # Count detections
                if result.boxes is not None:
                    num_detections = len(result.boxes)
                    print(f"   🎯 Detections: {num_detections}")
                    
                    # Show detection classes
                    if num_detections > 0:
                        classes_detected = {}
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            if class_name not in classes_detected:
                                classes_detected[class_name] = []
                            classes_detected[class_name].append(confidence)
                        
                        print(f"   📋 Detected classes:")
                        for class_name, confidences in classes_detected.items():
                            avg_conf = sum(confidences) / len(confidences)
                            print(f"     • {class_name}: {len(confidences)} detections (avg conf: {avg_conf:.2f})")
                
                # Save annotated image
                annotated_img = result.plot()
                output_path = output_dir / f"test_{i}_{img_path.name}"
                cv2.imwrite(str(output_path), annotated_img)
                print(f"   💾 Saved: {output_path}")
                
            else:
                print(f"   ⚠️  No detections found")
        
        except Exception as e:
            print(f"   ❌ Error processing image: {e}")
    
    print(f"\n🎉 TESTING COMPLETE!")
    print(f"📂 Results saved to: {output_dir}")
    print(f"🔍 Review the annotated images to check detection quality")


if __name__ == "__main__":
    test_model()
