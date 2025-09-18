#!/usr/bin/env python3
"""
Test Rebalanced Model

Test the newly trained rebalanced dual board model for symmetric detection.
This should detect battery holders on BOTH left and right sides correctly.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time


def test_rebalanced_model():
    """Test the rebalanced model for symmetric detection"""
    
    print("🎯 REBALANCED MODEL TEST")
    print("=" * 50)
    print("📱 Model: models/weights/dual_board_rebalanced_model.pt")
    print("🎯 Testing for SYMMETRIC detection!")
    print("✅ Expected: 1 battery holder + 3 wires on EACH side")
    
    # Load the rebalanced model
    model_path = "models/weights/dual_board_rebalanced_model.pt"
    model_file = Path(model_path)
    
    if not model_file.exists():
        print(f"❌ Model file not found: {model_path}")
        print(f"   Looking for: {model_file.absolute()}")
        return
    
    print(f"🔧 Loading rebalanced model: {model_path}")
    
    try:
        model = YOLO(model_path)
        
        # Set detection parameters
        model.conf = 0.25  # Medium confidence to start
        model.iou = 0.45
        
        print(f"✅ Rebalanced model loaded successfully!")
        print(f"   Confidence: {model.conf}")
        print(f"   IoU: {model.iou}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Start camera
    print("📹 Starting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📐 Camera resolution: {width}x{height}")
    
    split_x = width // 2
    frame_count = 0
    
    print(f"\n🚀 REBALANCED MODEL SYMMETRIC TEST")
    print("=" * 50)
    print(f"🎮 Controls:")
    print(f"   • '1': Lower confidence (0.15)")
    print(f"   • '2': Medium confidence (0.25)")
    print(f"   • '3': Higher confidence (0.4)")
    print(f"   • 'd': Print detailed detection analysis")
    print(f"   • 'q': Quit")
    print(f"\n🎯 TESTING HYPOTHESIS:")
    print(f"   • Rebalanced training (55.8% left battery holders)")
    print(f"   • Should now detect LEFT battery holders!")
    print(f"   • Expected: 1 battery + 3 wires per side")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading from camera")
                break
                
            frame_count += 1
            
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
                    
                    # Draw bounding box with different colors for each side
                    color = (0, 255, 0) if side == "left" else (0, 0, 255)
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.3f}"
                    cv2.putText(display_frame, label, (int(x1), int(y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Analyze battery holders specifically
            left_batteries = sum(1 for c in left_components if c['class_name'] == 'battery_holder')
            right_batteries = sum(1 for c in right_components if c['class_name'] == 'battery_holder')
            left_wires = sum(1 for c in left_components if c['class_name'] == 'wire')
            right_wires = sum(1 for c in right_components if c['class_name'] == 'wire')
            
            # Draw analysis
            total_detections = len(left_components) + len(right_components)
            
            # Status display
            cv2.putText(display_frame, f"REBALANCED MODEL TEST", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Total: {total_detections} | Conf: {model.conf:.2f}", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Left side analysis
            cv2.putText(display_frame, f"LEFT: {left_batteries} batteries, {left_wires} wires", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Right side analysis  
            cv2.putText(display_frame, f"RIGHT: {right_batteries} batteries, {right_wires} wires", 
                       (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Success/failure indicator
            success_left = (left_batteries >= 1 and left_wires >= 2)
            success_right = (right_batteries >= 1 and right_wires >= 2)
            
            if success_left and success_right:
                status = "✅ SYMMETRIC DETECTION WORKING!"
                status_color = (0, 255, 0)
            elif left_batteries > 0:
                status = "🟡 LEFT BATTERY DETECTED! (Progress!)"  
                status_color = (0, 255, 255)
            else:
                status = "❌ Still missing left battery"
                status_color = (0, 0, 255)
            
            cv2.putText(display_frame, status, 
                       (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Expected vs actual
            cv2.putText(display_frame, f"Expected: 1+3 per side | Actual: L({left_batteries}+{left_wires}) R({right_batteries}+{right_wires})", 
                       (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Rebalanced Model Test - Symmetric Detection", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                model.conf = 0.15
                print(f"\n📊 Switched to confidence=0.15 (lower)")
            elif key == ord('2'):
                model.conf = 0.25
                print(f"\n📊 Switched to confidence=0.25 (medium)")
            elif key == ord('3'):
                model.conf = 0.40
                print(f"\n📊 Switched to confidence=0.40 (higher)")
            elif key == ord('d'):
                print(f"\n📊 REBALANCED MODEL ANALYSIS (Frame {frame_count}, Conf={model.conf:.2f}):")
                print(f"   Total detected: {total_detections}")
                
                print(f"\n🔋 BATTERY HOLDER ANALYSIS:")
                print(f"   Left side: {left_batteries} (expected: 1)")
                print(f"   Right side: {right_batteries} (expected: 1)")
                
                if left_batteries > 0:
                    print(f"   🎉 SUCCESS! Left battery holder detected!")
                    print(f"   📈 Rebalanced training worked!")
                else:
                    print(f"   ⚠️  Still missing left battery holder")
                
                print(f"\n🔌 WIRE ANALYSIS:")
                print(f"   Left side: {left_wires} (expected: 3)")
                print(f"   Right side: {right_wires} (expected: 3)")
                
                if len(left_components) > 0:
                    print(f"\n📍 LEFT SIDE DETECTIONS:")
                    for comp in left_components:
                        print(f"      • {comp['class_name']}: {comp['confidence']:.3f}")
                
                if len(right_components) > 0:
                    print(f"\n📍 RIGHT SIDE DETECTIONS:")
                    for comp in right_components:
                        print(f"      • {comp['class_name']}: {comp['confidence']:.3f}")
                
                # Overall assessment
                print(f"\n🎯 REBALANCED MODEL ASSESSMENT:")
                if left_batteries >= 1:
                    print(f"   ✅ MAJOR IMPROVEMENT: Left battery detection working!")
                    if success_left and success_right:
                        print(f"   🏆 COMPLETE SUCCESS: Symmetric detection achieved!")
                    else:
                        print(f"   📈 Good progress, fine-tuning may help")
                else:
                    print(f"   ⚠️  Left battery still not detected")
                    print(f"   🤔 May need more training or different approach")
                
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    test_rebalanced_model()
