#!/usr/bin/env python3
"""
Precision-Focused Dual Board Test

Use higher confidence thresholds to reduce false positives and phantom detections.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def test_with_higher_precision():
    """Test with higher confidence to reduce false positives"""
    
    print("🎯 PRECISION-FOCUSED DUAL BOARD TEST")
    print("=" * 60)
    print("📱 Model: models/weights/dual_board_dual_board_1758049257.pt")
    print("🔧 Strategy: Higher confidence to reduce false positives")
    print("⚡ Should show only real components!")
    
    # Load model
    model_path = "models/weights/dual_board_dual_board_1758049257.pt"
    print(f"🔧 Loading model: {model_path}")
    
    model = YOLO(model_path)
    
    # Try different confidence levels
    confidence_levels = [0.3, 0.5, 0.7]
    
    print(f"✅ Model loaded")
    
    # Start camera
    print("📹 Starting camera 0...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📐 Camera resolution: {width}x{height}")
    
    current_confidence = 0
    split_x = width // 2
    
    print(f"\n🚀 PRECISION TEST")
    print("=" * 50)
    print(f"🎮 Controls:")
    print(f"   • '1': confidence=0.3 (low)")
    print(f"   • '2': confidence=0.5 (medium)") 
    print(f"   • '3': confidence=0.7 (high)")
    print(f"   • 'd': Print detailed detection info")
    print(f"   • 'q': Quit")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading from camera")
                break
                
            frame_count += 1
            
            # Set confidence based on current level
            if current_confidence < len(confidence_levels):
                conf = confidence_levels[current_confidence]
                model.conf = conf
                model.iou = 0.45
            
            # Run detection on full frame
            results = model(frame, verbose=False)
            
            # Create display frame
            display_frame = frame.copy()
            
            # Draw split line
            cv2.line(display_frame, (split_x, 0), (split_x, height), (0, 255, 255), 3)
            
            # Process detections
            detections = results[0].boxes
            left_components = []
            right_components = []
            
            if detections is not None and len(detections) > 0:
                for box in detections:
                    # Get box info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name
                    class_name = model.names[class_id]
                    
                    # Calculate center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Determine side
                    side = "left" if center_x < split_x else "right"
                    
                    # Store detection
                    detection_info = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'box': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'side': side
                    }
                    
                    if side == "left":
                        left_components.append(detection_info)
                    else:
                        right_components.append(detection_info)
                    
                    # Draw bounding box
                    color = (0, 255, 0) if side == "left" else (0, 0, 255)
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.3f}"
                    cv2.putText(display_frame, label, (int(x1), int(y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw counts
            total_detections = len(left_components) + len(right_components)
            conf_text = f"Conf: {model.conf:.2f}" if hasattr(model, 'conf') else "Conf: N/A"
            
            cv2.putText(display_frame, f"Total: {total_detections} | {conf_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Left: {len(left_components)} | Right: {len(right_components)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Expected counts
            cv2.putText(display_frame, f"Expected: 1 battery + 3 wires per side", 
                       (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Precision-Focused Dual Board Test", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_confidence = 0
                print(f"\n📊 Switched to confidence=0.3 (low)")
            elif key == ord('2'):
                current_confidence = 1
                print(f"\n📊 Switched to confidence=0.5 (medium)")
            elif key == ord('3'):
                current_confidence = 2
                print(f"\n📊 Switched to confidence=0.7 (high)")
            elif key == ord('d'):
                print(f"\n📊 DETECTION DETAILS (Frame {frame_count}, Conf={model.conf:.2f}):")
                print(f"   Total detected: {total_detections}")
                
                if len(left_components) > 0:
                    print(f"   Left side ({len(left_components)} components):")
                    for comp in left_components:
                        print(f"      • {comp['class_name']}: {comp['confidence']:.3f} at ({int(comp['center'][0])},{int(comp['center'][1])})")
                
                if len(right_components) > 0:
                    print(f"   Right side ({len(right_components)} components):")
                    for comp in right_components:
                        print(f"      • {comp['class_name']}: {comp['confidence']:.3f} at ({int(comp['center'][0])},{int(comp['center'][1])})")
                
                # Analysis
                left_batteries = sum(1 for c in left_components if c['class_name'] == 'battery_holder')
                right_batteries = sum(1 for c in right_components if c['class_name'] == 'battery_holder')
                left_wires = sum(1 for c in left_components if c['class_name'] == 'wire')
                right_wires = sum(1 for c in right_components if c['class_name'] == 'wire')
                
                print(f"\n🔍 COMPONENT ANALYSIS:")
                print(f"   Battery holders: {left_batteries + right_batteries} (expected: 2)")
                print(f"      • Left: {left_batteries} (expected: 1)")
                print(f"      • Right: {right_batteries} (expected: 1)")
                print(f"   Wires: {left_wires + right_wires} (expected: 6)")
                print(f"      • Left: {left_wires} (expected: 3)")
                print(f"      • Right: {right_wires} (expected: 3)")
                
                if total_detections > 8:
                    print(f"   ⚠️  {total_detections - 8} excess detections (possible false positives)")
                elif total_detections < 8:
                    print(f"   ⚠️  {8 - total_detections} missing detections")
                else:
                    print(f"   ✅ Correct total count!")
                
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    test_with_higher_precision()
