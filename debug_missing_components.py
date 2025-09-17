#!/usr/bin/env python3
"""
Debug Missing Components

Find out why we're only detecting 7 components instead of expected 8,
and why they're not distributed evenly between left/right sides.
"""

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    model_path = "models/weights/dual_board_dual_board_1758049257.pt"
    
    print("🔍 MISSING COMPONENTS DEBUG")
    print("=" * 60)
    print("🎯 Expected: 8 components (4 per side)")
    print("   • Left: 1 battery_holder + 3 wires")
    print("   • Right: 1 battery_holder + 3 wires")
    print()
    
    # Load model with different settings to find missing components
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed")
        return
    
    print("🎮 Controls:")
    print("   • 'q': Quit")
    print("   • '1': Ultra-low confidence (0.001)")
    print("   • '2': Low IoU threshold (0.1)")
    print("   • '3': Both ultra-low conf + low IoU")
    print("   • '4': Normal settings (0.01 conf, 0.45 IoU)")
    print("   • 'd': Detailed analysis")
    print()
    
    # Test different settings
    settings = {
        '1': {'conf': 0.001, 'iou': 0.45, 'name': 'Ultra-low confidence'},
        '2': {'conf': 0.01, 'iou': 0.1, 'name': 'Low IoU threshold'},  
        '3': {'conf': 0.001, 'iou': 0.1, 'name': 'Ultra-low conf + low IoU'},
        '4': {'conf': 0.01, 'iou': 0.45, 'name': 'Normal settings'}
    }
    
    current_setting = '4'  # Start with normal
    model.conf = settings[current_setting]['conf']
    model.iou = settings[current_setting]['iou']
    
    print(f"🔧 Starting with: {settings[current_setting]['name']}")
    print(f"   Confidence: {model.conf}, IoU: {model.iou}")
    print()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle setting changes
            if key in ['1', '2', '3', '4']:
                current_setting = chr(key)
                model.conf = settings[current_setting]['conf']
                model.iou = settings[current_setting]['iou']
                print(f"🔄 Switched to: {settings[current_setting]['name']}")
                print(f"   Confidence: {model.conf}, IoU: {model.iou}")
                continue
            elif key == ord('q'):
                break
            elif key == ord('d'):
                print(f"\n📊 DETAILED ANALYSIS ({settings[current_setting]['name']}):")
                print("-" * 50)
                
                # Run detection
                results = model(frame, verbose=False)
                
                height, width = frame.shape[:2]
                split_x = width // 2
                
                all_components = []
                left_components = []
                right_components = []
                
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    print(f"Total detections: {len(boxes)}")
                    
                    for i, box in enumerate(boxes):
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        component_info = {
                            'name': class_name,
                            'conf': conf,
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'side': 'left' if center_x < split_x else 'right'
                        }
                        
                        all_components.append(component_info)
                        
                        if center_x < split_x:
                            left_components.append(component_info)
                        else:
                            right_components.append(component_info)
                        
                        print(f"   {i+1}. {class_name}: {conf:.4f} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) center=({center_x:.0f},{center_y:.0f}) [{component_info['side']}]")
                
                # Analyze by component type
                print(f"\n🔍 COMPONENT ANALYSIS:")
                battery_holders = [c for c in all_components if 'battery' in c['name']]
                wires = [c for c in all_components if 'wire' in c['name']]
                
                print(f"   Battery holders: {len(battery_holders)}")
                for bh in battery_holders:
                    print(f"      • {bh['name']}: {bh['conf']:.4f} [{bh['side']}] at ({bh['center'][0]:.0f},{bh['center'][1]:.0f})")
                
                print(f"   Wires: {len(wires)}")
                for w in wires:
                    print(f"      • {w['name']}: {w['conf']:.4f} [{w['side']}] at ({w['center'][0]:.0f},{w['center'][1]:.0f})")
                
                # Side distribution
                print(f"\n📍 SIDE DISTRIBUTION:")
                left_bh = len([c for c in left_components if 'battery' in c['name']])
                left_w = len([c for c in left_components if 'wire' in c['name']])
                right_bh = len([c for c in right_components if 'battery' in c['name']])
                right_w = len([c for c in right_components if 'wire' in c['name']])
                
                print(f"   Left side: {left_bh} battery_holders, {left_w} wires")
                print(f"   Right side: {right_bh} battery_holders, {right_w} wires")
                
                # Check if we have expected distribution
                expected_left = (left_bh == 1 and left_w == 3)
                expected_right = (right_bh == 1 and right_w == 3)
                
                if expected_left and expected_right:
                    print(f"   ✅ PERFECT! Expected distribution achieved")
                else:
                    print(f"   ❌ MISMATCH! Expected 1+3 on each side")
                    if len(all_components) < 8:
                        print(f"   🔍 Missing {8 - len(all_components)} components total")
                    if left_bh == 0:
                        print(f"   🔍 Missing battery_holder on left side")
                    if right_bh == 0:
                        print(f"   🔍 Missing battery_holder on right side")
                    if left_w < 3:
                        print(f"   🔍 Missing {3 - left_w} wires on left side")
                    if right_w < 3:
                        print(f"   🔍 Missing {3 - right_w} wires on right side")
                
                print()
                continue
            
            # Display with current settings
            results = model(frame, verbose=False)
            if len(results) > 0:
                display_frame = results[0].plot()
            else:
                display_frame = frame.copy()
            
            # Add split line and info
            height, width = display_frame.shape[:2]
            split_x = width // 2
            cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 2)
            
            # Add current settings info
            info_text = f"{settings[current_setting]['name']} (conf:{model.conf}, iou:{model.iou})"
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Missing Components Debug', display_frame)
    
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
