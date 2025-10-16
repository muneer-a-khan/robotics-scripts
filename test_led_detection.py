#!/usr/bin/env python3
"""
Test LED orientation detection
"""

import cv2
from ultralytics import YOLO
from led_orientation_detector import LEDOrientationDetector

# Load model
model = YOLO("dual_board_training/photos_model_fixed/weights/best.pt")

# Load image
image = cv2.imread("photos/photo_with_plus.jpg")
print(f"Image size: {image.shape}")

# Run detection
results = model(image, conf=0.6, iou=0.5)
result = results[0]

# Find LEDs
print("\nDetected components:")
for i, box in enumerate(result.boxes):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    
    print(f"{i}: {class_name} - bbox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) - conf: {confidence:.2f}")
    
    if 'LED_2 (Red)' in class_name:
        # Crop and save LED region
        led_crop = image[int(y1):int(y2), int(x1):int(x2)]
        crop_filename = f"led_crop_{i}.jpg"
        cv2.imwrite(crop_filename, led_crop)
        print(f"  Saved LED crop: {crop_filename} (size: {led_crop.shape})")
        
        # Test orientation detection
        detector = LEDOrientationDetector()
        result = detector.detect_orientation(image, [x1, y1, x2, y2], debug=True)
        print(f"  Orientation result: {result}")

