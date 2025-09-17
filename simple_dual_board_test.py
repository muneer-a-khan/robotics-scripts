#!/usr/bin/env python3
"""
Simple Dual Board Detection Test - NO GRAPHS

Just tests basic component detection with bounding boxes on dual board setup.
Bypasses all graph generation and complex pipeline.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from models.component_detector import ComponentDetector


class SimpleDualBoardTester:
    """Simple dual board tester that just shows bounding boxes"""
    
    def __init__(self, model_path: str, split_ratio: float = 0.5):
        self.model_path = model_path
        self.split_ratio = split_ratio
        
        # Initialize detector EXACTLY like the successful direct test
        print(f"🔧 Loading model: {model_path}")
        print("   Using DIRECT MODEL LOADING (bypassing ComponentDetector)")
        
        # Load model directly like the successful debug test
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.conf = 0.01  # Very low threshold like direct test
        self.model.iou = 0.45
        
        print(f"✅ Model loaded directly with confidence={self.model.conf}, iou={self.model.iou}")
        print("   This matches the successful direct debug test!")
            
        print("✅ Model loaded for simple testing!")
        
        self.cap = None
        
    def start_camera(self, camera_id: int = 0) -> bool:
        """Start camera"""
        print(f"📹 Starting camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {camera_id}")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("✅ Camera started!")
        return True
    
    def split_frame(self, frame: np.ndarray):
        """Split frame into left and right"""
        height, width = frame.shape[:2]
        split_x = int(width * self.split_ratio)
        
        left_frame = frame[:, :split_x]
        right_frame = frame[:, split_x:]
        
        return left_frame, right_frame, split_x
    
    def detect_and_annotate(self, frame: np.ndarray, side: str):
        """Detect components using direct YOLO and return annotated frame"""
        # Run YOLO detection DIRECTLY like successful debug test
        results = self.model(frame, verbose=False)
        
        # Count detections for display
        component_count = 0
        for r in results:
            if r.boxes is not None:
                component_count += len(r.boxes)
        
        # Use YOLO's built-in plotting (same as successful debug test)
        if len(results) > 0:
            annotated = results[0].plot()
        else:
            annotated = frame.copy()
        
        return annotated, component_count
    
    def run_simple_test(self, camera_id: int = 0):
        """Run the simple dual board test"""
        if not self.start_camera(camera_id):
            return
        
        print("\n🚀 CONTINUOUS DUAL BOARD DETECTION")
        print("=" * 50)
        print("🎯 This will show:")
        print("   • Live camera feed split in the middle")
        print("   • Bounding boxes around detected components")
        print("   • Component names and confidence scores")
        print("   • CONTINUOUS detection - every frame processed!")
        print("   • Optimized for close camera positioning")
        print()
        print("🎮 Controls:")
        print("   • 'q': Quit")
        print("   • ESC: Quit")
        print("   • CONTINUOUS DETECTION - No delays!")
        print()
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # CONTINUOUS DETECTION - Process every single frame
                frame_count += 1
                
                # Split frame
                left_frame, right_frame, split_x = self.split_frame(frame)
                
                # Detect on both sides
                left_annotated, left_count = self.detect_and_annotate(left_frame, "left")
                right_annotated, right_count = self.detect_and_annotate(right_frame, "right")
                
                # Combine back into full frame
                display_frame = frame.copy()
                display_frame[:, :split_x] = left_annotated
                display_frame[:, split_x:] = right_annotated
                
                # Add split line
                cv2.line(display_frame, (split_x, 0), (split_x, frame.shape[0]), (255, 255, 255), 3)
                
                # Add text overlay with frame counter
                height = frame.shape[0]
                cv2.putText(display_frame, f"LEFT: {left_count} components", (10, height - 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(display_frame, f"RIGHT: {right_count} components", (split_x + 10, height - 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Continuous Dual Board Detection', display_frame)
                
                # Handle keys (minimal delay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Cleanup complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple dual board detection test")
    parser.add_argument("--model", "-m", 
                       default="models/weights/dual_board_dual_board_1758049257.pt",
                       help="Path to your model")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--split", "-s", type=float, default=0.5, help="Split ratio")
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return 1
    
    print("🎯 SIMPLE DUAL BOARD DETECTION TEST")
    print("=" * 60)
    print(f"📱 Model: {Path(args.model).name}")
    print(f"📹 Camera: {args.camera}")
    print(f"✂️  Split: {args.split} ({args.split*100:.0f}% left)")
    print("🎯 Mode: BOUNDING BOXES ONLY (no graphs)")
    print()
    
    # Create and run tester
    tester = SimpleDualBoardTester(args.model, args.split)
    tester.run_simple_test(args.camera)
    
    return 0


if __name__ == "__main__":
    exit(main())
