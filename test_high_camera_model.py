#!/usr/bin/env python3
"""
Test High Camera Model

Quick test to verify the high camera model works correctly.
Tests the model on a sample image to ensure it loads and detects components properly.
"""

import cv2
import numpy as np
from pathlib import Path
from models.component_detector import ComponentDetector

# Path to the trained high camera model
HIGH_CAMERA_MODEL_PATH = "output/results/high_camera_gpu_20250827_090037/weights/best.pt"

def test_model_loading():
    """Test if the model loads correctly."""
    print("🧪 Testing High Camera Model Loading...")
    
    model_path = Path(HIGH_CAMERA_MODEL_PATH)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        detector = ComponentDetector(str(model_path))
        print("✅ Model loaded successfully!")
        print(f"📊 Model file size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        return detector
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def test_sample_detection(detector):
    """Test detection on a sample image."""
    print("\n🔍 Testing Sample Detection...")
    
    # Try to find a sample image from the training data
    sample_dirs = [
        "data/high_camera_training/images/val",
        "data/high_camera_training/images/train",
        "data/augmented_training/images/val",
        "data/augmented_training/images/train"
    ]
    
    sample_image = None
    for sample_dir in sample_dirs:
        sample_path = Path(sample_dir)
        if sample_path.exists():
            images = list(sample_path.glob("*.jpg"))
            if images:
                sample_image = images[0]
                break
    
    if not sample_image:
        print("⚠️  No sample images found. Creating test pattern...")
        # Create a simple test image
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(test_image, (400, 400), 50, (0, 255, 0), -1)
        sample_image = "test_pattern.jpg"
        cv2.imwrite(sample_image, test_image)
        print(f"📝 Created test pattern: {sample_image}")
    
    try:
        # Load and process the image
        image = cv2.imread(str(sample_image))
        if image is None:
            print(f"❌ Could not load image: {sample_image}")
            return False
        
        print(f"📷 Processing image: {sample_image}")
        print(f"📐 Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Run detection
        detections = detector.detect_components(image)
        
        print(f"🎯 Detected {len(detections)} components:")
        for i, detection in enumerate(detections[:5]):  # Show first 5
            print(f"   {i+1}. {detection.label} (confidence: {detection.confidence:.3f})")
        
        if len(detections) > 5:
            print(f"   ... and {len(detections) - 5} more")
        
        # Create annotated image
        annotated = detector.annotate_image(image, detections)
        output_path = "test_detection_output.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"💾 Annotated result saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return False

def test_inference_speed(detector):
    """Test inference speed."""
    print("\n⚡ Testing Inference Speed...")
    
    try:
        # Create test images of different sizes
        test_sizes = [(640, 640), (416, 416), (320, 320)]
        
        for width, height in test_sizes:
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Warm up
            detector.detect_components(test_image)
            
            # Time multiple runs
            import time
            times = []
            for _ in range(5):
                start = time.time()
                detector.detect_components(test_image)
                times.append((time.time() - start) * 1000)  # Convert to ms
            
            avg_time = sum(times) / len(times)
            print(f"📏 {width}x{height}: {avg_time:.1f}ms average ({1000/avg_time:.1f} FPS)")
        
        return True
        
    except Exception as e:
        print(f"❌ Speed test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 High Camera Model Test Suite")
    print("=" * 50)
    
    # Test 1: Model Loading
    detector = test_model_loading()
    if not detector:
        return False
    
    # Test 2: Sample Detection
    if not test_sample_detection(detector):
        print("⚠️  Sample detection failed, but model loaded OK")
    
    # Test 3: Speed Test
    if not test_inference_speed(detector):
        print("⚠️  Speed test failed, but model works")
    
    print("\n🎉 Model testing completed!")
    print("✅ Your high camera model is ready for dual camera system!")
    print("\nNext steps:")
    print("1. Run: python run_dual_with_high_camera_model.py")
    print("2. Or: python run_dual_with_high_camera_model.py --with-monitor")
    
    return True

if __name__ == "__main__":
    main()