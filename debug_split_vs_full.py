#!/usr/bin/env python3
"""
Debug Split vs Full Frame Detection

Compare detection results on full frame vs split frames to see what's going wrong.
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO


def main():
    model_path = "models/weights/dual_board_dual_board_1758049257.pt"
    
    print("🔍 SPLIT vs FULL FRAME DEBUG")
    print("=" * 60)
    
    # Load model
    model = YOLO(model_path)
    model.conf = 0.01
    model.iou = 0.45
    
    print(f"✅ Model loaded: {model_path}")
    
    # Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed")
        return
    
    print("📹 Camera started")
    print("\n🎮 Controls:")
    print("   • 'q': Quit")
    print("   • 'r': Run detection comparison")
    print("   • '1': Show full frame results")
    print("   • '2': Show left split results") 
    print("   • '3': Show right split results")
    print()
    
    display_mode = "full"  # "full", "left", "right"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('1'):
                display_mode = "full"
                print("🔄 Switched to full frame view")
            elif key == ord('2'):
                display_mode = "left"
                print("🔄 Switched to left split view")
            elif key == ord('3'):
                display_mode = "right"
                print("🔄 Switched to right split view")
            elif key == ord('r'):
                print("\n🔍 DETECTION COMPARISON")
                print("-" * 40)
                
                # 1. Full frame detection
                print("1️⃣ FULL FRAME DETECTION:")
                full_results = model(frame, verbose=False)
                full_count = 0
                full_detections = []
                
                for r in full_results:
                    if r.boxes is not None:
                        boxes = r.boxes
                        full_count = len(boxes)
                        print(f"   Found {full_count} components:")
                        for i, box in enumerate(boxes):
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            class_name = model.names[cls_id]
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            print(f"      {i+1}. {class_name}: {conf:.3f} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
                            full_detections.append((class_name, conf, x1, y1, x2, y2))
                
                # 2. Split frame detection
                height, width = frame.shape[:2]
                split_x = int(width * 0.5)
                
                left_frame = frame[:, :split_x]
                right_frame = frame[:, split_x:]
                
                print(f"\n2️⃣ LEFT SPLIT DETECTION (width: {left_frame.shape[1]}):")
                left_results = model(left_frame, verbose=False)
                left_count = 0
                for r in left_results:
                    if r.boxes is not None:
                        left_count = len(r.boxes)
                        print(f"   Found {left_count} components:")
                        for i, box in enumerate(r.boxes):
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            class_name = model.names[cls_id]
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            print(f"      {i+1}. {class_name}: {conf:.3f} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
                
                print(f"\n3️⃣ RIGHT SPLIT DETECTION (width: {right_frame.shape[1]}):")
                right_results = model(right_frame, verbose=False)
                right_count = 0
                for r in right_results:
                    if r.boxes is not None:
                        right_count = len(r.boxes)
                        print(f"   Found {right_count} components:")
                        for i, box in enumerate(r.boxes):
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            class_name = model.names[cls_id]
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            print(f"      {i+1}. {class_name}: {conf:.3f} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
                
                print(f"\n📊 SUMMARY:")
                print(f"   Full frame: {full_count} components")
                print(f"   Left split: {left_count} components")  
                print(f"   Right split: {right_count} components")
                print(f"   Split total: {left_count + right_count} components")
                
                if full_count > (left_count + right_count):
                    print(f"   ❌ DETECTION LOSS: {full_count - (left_count + right_count)} components lost in splitting!")
                    print(f"   🔍 Components near split line may be getting cut off")
                elif full_count < (left_count + right_count):
                    print(f"   ⚠️  Split detections higher than full frame - possible duplicates")
                else:
                    print(f"   ✅ Detection counts match!")
                
                print()
            
            # Display based on mode
            if display_mode == "full":
                full_results = model(frame, verbose=False)
                if len(full_results) > 0:
                    display_frame = full_results[0].plot()
                    cv2.putText(display_frame, f"FULL FRAME", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"FULL FRAME (no detections)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            elif display_mode == "left":
                height, width = frame.shape[:2]
                split_x = int(width * 0.5)
                left_frame = frame[:, :split_x]
                
                left_results = model(left_frame, verbose=False)
                if len(left_results) > 0:
                    display_frame = left_results[0].plot()
                    cv2.putText(display_frame, f"LEFT SPLIT", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    display_frame = left_frame.copy()
                    cv2.putText(display_frame, f"LEFT SPLIT (no detections)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            elif display_mode == "right":
                height, width = frame.shape[:2]
                split_x = int(width * 0.5)
                right_frame = frame[:, split_x:]
                
                right_results = model(right_frame, verbose=False)
                if len(right_results) > 0:
                    display_frame = right_results[0].plot()
                    cv2.putText(display_frame, f"RIGHT SPLIT", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    display_frame = right_frame.copy()
                    cv2.putText(display_frame, f"RIGHT SPLIT (no detections)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Detection Debug', display_frame)
    
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
