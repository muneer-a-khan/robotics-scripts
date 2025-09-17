#!/usr/bin/env python3
"""
Direct Model Debug Script

Tests your model directly on camera frames to see what it's actually detecting,
bypassing all the confidence filtering in the dual board system.
"""

import cv2
import time
from ultralytics import YOLO
from pathlib import Path


def main():
    model_path = "models/weights/dual_board_dual_board_1758049257.pt"
    
    print("🔍 DIRECT MODEL DEBUG TEST")
    print("=" * 50)
    print(f"📱 Model: {model_path}")
    print(f"🎯 Testing model directly without any filtering")
    print()
    
    # Load model directly
    print("📥 Loading model...")
    model = YOLO(model_path)
    
    # Set very low confidence
    model.conf = 0.01  # 1% confidence threshold
    model.iou = 0.45   # IoU threshold
    
    print(f"✅ Model loaded with confidence={model.conf}, iou={model.iou}")
    print()
    
    # Start camera
    print("📹 Starting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera!")
        return 1
    
    print("✅ Camera started!")
    print()
    print("🎮 Controls:")
    print("   • 'q': Quit")
    print("   • 's': Save current frame and detections")
    print("   • 'r': Run detection on current frame")
    print()
    print("👀 Look for bounding boxes around detected objects!")
    print("   Even very low confidence detections will be shown.")
    print()
    
    frame_count = 0
    last_detection_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            current_time = time.time()
            
            # Run detection every 3 seconds or when 'r' is pressed
            key = cv2.waitKey(1) & 0xFF
            should_detect = (current_time - last_detection_time) > 3.0
            
            if should_detect or key == ord('r'):
                print(f"🔄 Running detection on frame {frame_count}...")
                
                # Run YOLO detection
                results = model(frame, verbose=False)
                
                # Print all detections (even very low confidence)
                total_detections = 0
                for r in results:
                    boxes = r.boxes
                    if boxes is not None and len(boxes) > 0:
                        print(f"   Found {len(boxes)} detections:")
                        for i, box in enumerate(boxes):
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            class_name = model.names[cls_id]
                            print(f"      {i+1}. {class_name}: {conf:.3f} confidence")
                            total_detections += 1
                
                if total_detections == 0:
                    print("   ❌ No detections found at all!")
                    print("   This suggests:")
                    print("      • No circuit components in camera view")
                    print("      • Poor lighting or focus")
                    print("      • Model needs retraining")
                else:
                    print(f"   ✅ Total detections: {total_detections}")
                
                # Annotate and display frame
                annotated_frame = results[0].plot()
                cv2.imshow('Direct Model Debug', annotated_frame)
                
                frame_count += 1
                last_detection_time = current_time
                print()
            
            else:
                # Show raw frame
                cv2.imshow('Direct Model Debug', frame)
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f"debug_frame_{int(time.time())}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"💾 Saved frame to {save_path}")
                
                # Run detection and save results
                results = model(frame, verbose=False)
                annotated = results[0].plot()
                annotated_path = f"debug_annotated_{int(time.time())}.jpg"
                cv2.imwrite(annotated_path, annotated)
                print(f"💾 Saved annotated frame to {annotated_path}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")
    
    return 0


if __name__ == "__main__":
    exit(main())
