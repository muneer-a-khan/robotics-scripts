#!/usr/bin/env python3
"""
Fixed Dual Board Test - Full Frame Detection, Split Display

Keeps detection on full frame (where it works), but splits the display.
Also checks for resolution mismatches causing wrong bounding boxes.
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO


class FixedDualBoardTester:
    """Dual board tester that detects on full frame but displays split"""
    
    def __init__(self, model_path: str, split_ratio: float = 0.5):
        self.model_path = model_path
        self.split_ratio = split_ratio
        
        print(f"🔧 Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model.conf = 0.01  # Very low like successful test
        self.model.iou = 0.45
        
        print(f"✅ Model loaded with confidence={self.model.conf}")
        
        self.cap = None
        
    def start_camera(self, camera_id: int = 0) -> bool:
        """Start camera and check resolution"""
        print(f"📹 Starting camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {camera_id}")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Check actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📐 Camera resolution: {actual_width}x{actual_height}")
        
        if actual_width != 1920 or actual_height != 1080:
            print(f"⚠️  Resolution mismatch! Requested 1920x1080, got {actual_width}x{actual_height}")
            print(f"   This might explain wrong bounding box positions!")
        else:
            print(f"✅ Resolution matches expected 1920x1080")
        
        return True
    
    def detect_full_frame_split_display(self, frame: np.ndarray):
        """
        Detect on FULL frame but categorize detections by left/right for display
        """
        # Run detection on FULL FRAME (where it works!)
        results = self.model(frame, verbose=False)
        
        # Get frame dimensions for splitting logic
        height, width = frame.shape[:2]
        split_x = int(width * self.split_ratio)
        
        left_components = []
        right_components = []
        total_count = 0
        
        # Process results and categorize by side
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            total_count = len(boxes)
            
            for box in boxes:
                # Get bounding box center
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                # Categorize by which side the CENTER is on
                if center_x < split_x:
                    left_components.append((class_name, conf, x1, y1, x2, y2))
                else:
                    right_components.append((class_name, conf, x1, y1, x2, y2))
        
        return results, left_components, right_components, total_count
    
    def create_dual_display(self, frame: np.ndarray, results, left_components, right_components):
        """Create dual display with full frame detection results"""
        
        # Start with full frame detection overlay
        if len(results) > 0:
            annotated_full = results[0].plot()
        else:
            annotated_full = frame.copy()
        
        height, width = annotated_full.shape[:2]
        split_x = int(width * self.split_ratio)
        
        # Add split line
        cv2.line(annotated_full, (split_x, 0), (split_x, height), (255, 255, 255), 3)
        
        # Add side labels and component counts
        cv2.putText(annotated_full, f"LEFT: {len(left_components)} components", 
                   (10, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(annotated_full, f"RIGHT: {len(right_components)} components", 
                   (split_x + 10, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add total count
        cv2.putText(annotated_full, f"TOTAL: {len(left_components) + len(right_components)} components", 
                   (width//2 - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return annotated_full
    
    def run_fixed_test(self, camera_id: int = 0):
        """Run the fixed dual board test"""
        if not self.start_camera(camera_id):
            return
        
        print("\n🚀 FIXED DUAL BOARD TEST")
        print("=" * 50)
        print("🎯 Strategy:")
        print("   • Detect on FULL FRAME (where detection works perfectly)")
        print("   • Categorize components by left/right based on position")
        print("   • Display with split line and side counts")
        print("   • Should show all 7 components with correct positions!")
        print()
        print("🎮 Controls:")
        print("   • 'q': Quit")
        print("   • 'd': Print detailed detection info")
        print()
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                
                # Detect on full frame, categorize for display
                results, left_components, right_components, total_count = self.detect_full_frame_split_display(frame)
                
                # Create display
                display_frame = self.create_dual_display(frame, results, left_components, right_components)
                
                # Add frame counter
                cv2.putText(display_frame, f"Frame: {frame_count}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Fixed Dual Board Test', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    print(f"\n📊 DETECTION DETAILS (Frame {frame_count}):")
                    print(f"   Total detected: {total_count}")
                    print(f"   Left side ({len(left_components)} components):")
                    for comp in left_components:
                        name, conf, x1, y1, x2, y2 = comp
                        print(f"      • {name}: {conf:.3f} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
                    print(f"   Right side ({len(right_components)} components):")
                    for comp in right_components:
                        name, conf, x1, y1, x2, y2 = comp
                        print(f"      • {name}: {conf:.3f} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
                    print()
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Cleanup complete")


def main():
    """Main entry point"""
    model_path = "models/weights/dual_board_dual_board_1758049257.pt"
    
    print("🎯 FIXED DUAL BOARD DETECTION TEST")
    print("=" * 60)
    print(f"📱 Model: {model_path}")
    print("🔧 Strategy: Full frame detection + split display")
    print("⚡ Should detect all components correctly now!")
    print()
    
    # Create and run tester
    tester = FixedDualBoardTester(model_path)
    tester.run_fixed_test()
    
    return 0


if __name__ == "__main__":
    exit(main())
