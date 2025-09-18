#!/usr/bin/env python3
"""
Test Clean Model

Test the clean dual board model trained without spatial augmentations
to verify it achieves symmetric detection of battery holders on both sides.

This should be the FINAL test - the clean model was trained on perfectly
balanced data (50% left, 50% right battery holders) with no spatial corruption.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time


def test_clean_model():
    """Test the clean model for symmetric detection - the moment of truth!"""
    
    print("🧹 CLEAN MODEL TEST - MOMENT OF TRUTH!")
    print("=" * 60)
    print("📱 Model: models/weights/clean_dual_board_model.pt")
    print("🎯 TESTING: Symmetric battery holder detection")
    print("✅ Expected: Perfect 50/50 left-right balance")
    print("🚫 Training: NO spatial augmentations that corrupt balance")
    
    # Load the clean model
    model_path = "models/weights/clean_dual_board_model.pt"
    model_file = Path(model_path)
    
    if not model_file.exists():
        print(f"❌ Clean model not found: {model_path}")
        print(f"   Looking for: {model_file.absolute()}")
        return
    
    print(f"🔧 Loading clean model: {model_path}")
    
    try:
        model = YOLO(model_path)
        
        # Set detection parameters for optimal detection
        model.conf = 0.3  # Start with medium confidence
        model.iou = 0.45
        
        print(f"✅ Clean model loaded successfully!")
        print(f"   Model trained on: 38 left + 38 right battery holders")
        print(f"   Confidence: {model.conf}")
        print(f"   IoU: {model.iou}")
        
    except Exception as e:
        print(f"❌ Error loading clean model: {e}")
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
    success_frames = 0
    total_test_frames = 0
    
    print(f"\n🚀 CLEAN MODEL SYMMETRIC DETECTION TEST")
    print("=" * 60)
    print(f"🎮 Controls:")
    print(f"   • '1': Lower confidence (0.2)")
    print(f"   • '2': Medium confidence (0.3)")  
    print(f"   • '3': Higher confidence (0.5)")
    print(f"   • 'd': Print detailed analysis")
    print(f"   • 's': Show success statistics")
    print(f"   • 'q': Quit")
    print(f"\n🎯 SUCCESS CRITERIA:")
    print(f"   • LEFT battery holder: Detected ✅")
    print(f"   • RIGHT battery holder: Detected ✅")
    print(f"   • Symmetric detection achieved!")
    
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
            cv2.line(display_frame, (split_x, 0), (split_x, height), (0, 255, 255), 4)
            
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
                    
                    # Draw bounding box with vibrant colors
                    color = (0, 255, 0) if side == "left" else (0, 100, 255)  # Green for left, Orange for right
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    
                    # Draw label with background
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_frame, (int(x1), int(y1-25)), (int(x1+label_size[0]+10), int(y1)), color, -1)
                    cv2.putText(display_frame, label, (int(x1+5), int(y1-8)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Analyze battery holders specifically
            left_batteries = sum(1 for c in left_components if c['class_name'] == 'battery_holder')
            right_batteries = sum(1 for c in right_components if c['class_name'] == 'battery_holder')
            left_wires = sum(1 for c in left_components if c['class_name'] == 'wire')
            right_wires = sum(1 for c in right_components if c['class_name'] == 'wire')
            
            # Check for success (symmetric detection)
            has_left_battery = left_batteries >= 1
            has_right_battery = right_batteries >= 1
            is_symmetric = has_left_battery and has_right_battery
            
            # Track success rate
            if frame_count % 10 == 0:  # Sample every 10 frames
                total_test_frames += 1
                if is_symmetric:
                    success_frames += 1
            
            # Status display with large, clear text
            cv2.putText(display_frame, f"CLEAN MODEL TEST", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Detection counts
            total_detections = len(left_components) + len(right_components)
            cv2.putText(display_frame, f"Total: {total_detections} | Conf: {model.conf:.2f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Left side analysis
            left_status = f"LEFT: {left_batteries} batteries, {left_wires} wires"
            left_color = (0, 255, 0) if has_left_battery else (0, 0, 255)
            cv2.putText(display_frame, left_status, 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
            
            # Right side analysis  
            right_status = f"RIGHT: {right_batteries} batteries, {right_wires} wires"
            right_color = (0, 255, 0) if has_right_battery else (0, 0, 255)
            cv2.putText(display_frame, right_status, 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
            
            # Success indicator - BIG and prominent
            if is_symmetric:
                status = "🏆 SYMMETRIC DETECTION SUCCESS! 🏆"
                status_color = (0, 255, 0)  # Bright green
                # Add celebratory border
                cv2.rectangle(display_frame, (5, 5), (width-5, height-5), (0, 255, 0), 8)
            elif has_left_battery:
                status = "🎉 LEFT BATTERY DETECTED! (Major progress!)"
                status_color = (0, 255, 255)  # Yellow
            else:
                status = "⚠️ Still missing left battery"
                status_color = (0, 0, 255)  # Red
            
            cv2.putText(display_frame, status[:50], 
                       (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Success rate
            if total_test_frames > 0:
                success_rate = (success_frames / total_test_frames) * 100
                cv2.putText(display_frame, f"Success Rate: {success_rate:.1f}% ({success_frames}/{total_test_frames})", 
                           (10, height-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Expected vs actual (top right)
            expected_text = f"Expected: 1+3 per side"
            actual_text = f"Actual: L({left_batteries}+{left_wires}) R({right_batteries}+{right_wires})"
            cv2.putText(display_frame, expected_text, 
                       (width-350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, actual_text, 
                       (width-350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Clean Model Test - Symmetric Detection", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                model.conf = 0.2
                print(f"\n📊 Switched to confidence=0.2 (lower)")
            elif key == ord('2'):
                model.conf = 0.3
                print(f"\n📊 Switched to confidence=0.3 (medium)")
            elif key == ord('3'):
                model.conf = 0.5
                print(f"\n📊 Switched to confidence=0.5 (higher)")
            elif key == ord('s'):
                print(f"\n📊 SUCCESS STATISTICS:")
                print(f"   Frames tested: {total_test_frames}")
                print(f"   Symmetric detections: {success_frames}")
                print(f"   Success rate: {success_rate:.1f}%" if total_test_frames > 0 else "   Success rate: N/A")
            elif key == ord('d'):
                print(f"\n📊 CLEAN MODEL DETAILED ANALYSIS (Frame {frame_count}):")
                print(f"   Model: Clean (no spatial augmentation)")
                print(f"   Confidence: {model.conf:.2f}")
                print(f"   Total detected: {total_detections}")
                
                print(f"\n🔋 BATTERY HOLDER ANALYSIS:")
                print(f"   Left side: {left_batteries} (expected: 1)")
                print(f"   Right side: {right_batteries} (expected: 1)")
                
                if has_left_battery and has_right_battery:
                    print(f"   🏆 COMPLETE SUCCESS! Both sides detected!")
                    print(f"   ✅ Clean training approach WORKED!")
                elif has_left_battery:
                    print(f"   🎉 MAJOR BREAKTHROUGH! Left battery finally detected!")
                    print(f"   📈 Clean model is working!")
                elif has_right_battery:
                    print(f"   ⚠️  Only right battery detected (same as before)")
                else:
                    print(f"   ❌ No battery holders detected at this confidence")
                
                print(f"\n🔌 WIRE ANALYSIS:")
                print(f"   Left side: {left_wires}")
                print(f"   Right side: {right_wires}")
                
                if len(left_components) > 0:
                    print(f"\n📍 LEFT SIDE DETECTIONS:")
                    for comp in sorted(left_components, key=lambda x: x['confidence'], reverse=True):
                        print(f"      • {comp['class_name']}: {comp['confidence']:.3f}")
                else:
                    print(f"\n📍 LEFT SIDE: No detections")
                
                if len(right_components) > 0:
                    print(f"\n📍 RIGHT SIDE DETECTIONS:")
                    for comp in sorted(right_components, key=lambda x: x['confidence'], reverse=True):
                        print(f"      • {comp['class_name']}: {comp['confidence']:.3f}")
                else:
                    print(f"\n📍 RIGHT SIDE: No detections")
                
                # Final assessment
                print(f"\n🎯 CLEAN MODEL VERDICT:")
                if is_symmetric:
                    print(f"   🏆 COMPLETE SUCCESS: Symmetric detection achieved!")
                    print(f"   🎉 Clean training approach solved the problem!")
                    print(f"   ✅ No more asymmetric bias!")
                elif has_left_battery:
                    print(f"   🚀 MAJOR BREAKTHROUGH: Left battery detection working!")
                    print(f"   📈 Huge improvement over previous models!")
                    print(f"   🔧 May need minor tuning for perfect symmetry")
                else:
                    print(f"   🤔 Left battery still not detected")
                    print(f"   📊 Model may need different approach or more data")
                
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final summary
        print(f"\n📊 FINAL TEST SUMMARY:")
        print(f"   Model: Clean (no spatial augmentation)")
        print(f"   Total frames: {frame_count}")
        if total_test_frames > 0:
            final_success_rate = (success_frames / total_test_frames) * 100
            print(f"   Symmetric detection rate: {final_success_rate:.1f}%")
            
            if final_success_rate >= 80:
                print(f"   🏆 EXCELLENT! Clean model achieved symmetric detection!")
            elif final_success_rate >= 50:
                print(f"   ✅ GOOD! Major improvement achieved!")
            elif final_success_rate >= 20:
                print(f"   📈 PROGRESS! Some symmetric detection happening!")
            else:
                print(f"   ⚠️  Limited success - may need further work")
        
        print("✅ Cleanup complete")


if __name__ == "__main__":
    test_clean_model()
