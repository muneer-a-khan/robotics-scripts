#!/usr/bin/env python3
"""
Board Integrated Detection - Uses your existing board detection system
Integrates with your existing board coordinate mapping for maximum accuracy
"""

import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Import dual board visualizer
try:
    from dual_board_visualizer import DualBoardVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False

class BoardIntegratedSystem:
    def __init__(self):
        self.show_board_viz = False
        self.visualizer = None
        self.fig = None
        self.matrixcoor_to_realcoor = None
        self.board_bounds = None
        
        if VISUALIZER_AVAILABLE:
            self.visualizer = DualBoardVisualizer(cell_size=25)
    
    def toggle_board_visualization(self):
        """Toggle board visualization on/off"""
        if not VISUALIZER_AVAILABLE:
            print("⚠️ Board visualizer not available")
            return
            
        self.show_board_viz = not self.show_board_viz
        
        if self.show_board_viz:
            plt.ion()
            print("🎨 Board visualization enabled")
        else:
            if self.fig:
                plt.close(self.fig)
                self.fig = None
            print("🎨 Board visualization disabled")
    
    def detect_board_and_setup_coordinates(self, frame):
        """
        Detect the board and set up coordinate mapping
        This integrates with your existing board detection system
        """
        try:
            # Basic board detection using color masking (simplified version)
            # In your full system, this would use draws_pegs_on_rotated_board()
            
            # Convert to HSV for board detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Look for the board (assuming dark board on light background)
            # This is a simplified version - your full system is more sophisticated
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            
            # Find largest contour (board)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 10000:  # Minimum area threshold
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Store board bounds
                    self.board_bounds = (x, y, x + w, y + h)
                    
                    # Create simplified coordinate mapping
                    self.create_coordinate_mapping(x, y, w, h)
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Board detection error: {e}")
            return False
    
    def create_coordinate_mapping(self, board_x, board_y, board_w, board_h):
        """Create coordinate mapping from board bounds to grid"""
        self.matrixcoor_to_realcoor = {}
        
        # Create grid mapping (13 rows, 15 cols)
        for row in range(13):
            for col in range(15):
                # Map grid position to pixel coordinates
                pixel_x = board_x + (col / 14) * board_w
                pixel_y = board_y + (row / 12) * board_h
                
                self.matrixcoor_to_realcoor[(row, col)] = (int(pixel_x), int(pixel_y))
    
    def convert_detections_to_board_grid(self, left_boxes, right_boxes, model_names, frame_width, frame_height):
        """Convert YOLO detections to board grid coordinates using board detection"""
        if self.matrixcoor_to_realcoor is None:
            # Fallback to simple coordinate mapping
            return self.convert_detections_simple(left_boxes, right_boxes, model_names, frame_width, frame_height)
        
        left_detections = {}
        right_detections = {}
        
        def pixel_to_board_grid(pixel_x, pixel_y):
            """Find closest grid position to pixel coordinates"""
            best_distance = float('inf')
            best_grid = (0, 0)
            
            for (row, col), (grid_x, grid_y) in self.matrixcoor_to_realcoor.items():
                distance = np.sqrt((pixel_x - grid_x)**2 + (pixel_y - grid_y)**2)
                if distance < best_distance:
                    best_distance = distance
                    best_grid = (row, col)
            
            return best_grid
        
        # Process left side detections
        for i, box in left_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = model_names[class_id]
            confidence = float(box.conf[0])
            
            # Get center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Convert to board grid
            grid_row, grid_col = pixel_to_board_grid(center_x, center_y)
            
            if class_name not in left_detections:
                left_detections[class_name] = []
            
            left_detections[class_name].append({
                'x1': grid_row, 'y1': grid_col,
                'x2': grid_row + 1, 'y2': grid_col + 1,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'grid_pos': (grid_row, grid_col)
            })
        
        # Process right side detections
        for i, box in right_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = model_names[class_id]
            confidence = float(box.conf[0])
            
            # Get center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Convert to board grid
            grid_row, grid_col = pixel_to_board_grid(center_x, center_y)
            
            if class_name not in right_detections:
                right_detections[class_name] = []
            
            right_detections[class_name].append({
                'x1': grid_row, 'y1': grid_col,
                'x2': grid_row + 1, 'y2': grid_col + 1,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'grid_pos': (grid_row, grid_col)
            })
        
        return left_detections, right_detections
    
    def convert_detections_simple(self, left_boxes, right_boxes, model_names, frame_width, frame_height):
        """Fallback simple coordinate mapping"""
        left_detections = {}
        right_detections = {}
        split_x = frame_width // 2
        
        # Process left side detections
        for i, box in left_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = model_names[class_id]
            confidence = float(box.conf[0])
            
            # Simple grid mapping for left side
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            grid_x = int((center_x / split_x) * 6)  # Left side: 0-6
            grid_y = int((center_y / frame_height) * 14)
            grid_x = max(0, min(grid_x, 6))
            grid_y = max(0, min(grid_y, 14))
            
            if class_name not in left_detections:
                left_detections[class_name] = []
            
            left_detections[class_name].append({
                'x1': grid_x, 'y1': grid_y,
                'x2': grid_x + 1, 'y2': grid_y + 1,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
        
        # Process right side detections
        for i, box in right_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = model_names[class_id]
            confidence = float(box.conf[0])
            
            # Simple grid mapping for right side
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            grid_x = 7 + int(((center_x - split_x) / split_x) * 5)  # Right side: 7-12
            grid_y = int((center_y / frame_height) * 14)
            grid_x = max(7, min(grid_x, 12))
            grid_y = max(0, min(grid_y, 14))
            
            if class_name not in right_detections:
                right_detections[class_name] = []
            
            right_detections[class_name].append({
                'x1': grid_x, 'y1': grid_y,
                'x2': grid_x + 1, 'y2': grid_y + 1,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
        
        return left_detections, right_detections
    
    def update_board_visualization(self, left_boxes, right_boxes, model_names, frame_width, frame_height):
        """Update board visualization with accurate grid positions"""
        if not self.show_board_viz or not VISUALIZER_AVAILABLE:
            return
            
        try:
            # Convert to board grid coordinates
            left_detections, right_detections = self.convert_detections_to_board_grid(
                left_boxes, right_boxes, model_names, frame_width, frame_height)
            
            # Show grid positions in terminal for debugging
            if left_detections or right_detections:
                print(f"   📍 Grid Positions:")
                for comp_type, detections in left_detections.items():
                    for det in detections:
                        print(f"     LEFT {comp_type}: row={det['x1']}, col={det['y1']}")
                for comp_type, detections in right_detections.items():
                    for det in detections:
                        print(f"     RIGHT {comp_type}: row={det['x1']}, col={det['y1']}")
            
            # Close previous figure if exists
            if self.fig:
                plt.close(self.fig)
            
            # Create new visualization
            self.fig = self.visualizer.create_dual_board_visualization(left_detections, right_detections)
            plt.pause(0.01)
            
        except Exception as e:
            print(f"⚠️ Board visualization error: {e}")
    
    def draw_board_overlay(self, frame):
        """Draw board detection overlay on frame"""
        if self.board_bounds:
            x1, y1, x2, y2 = self.board_bounds
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "DETECTED BOARD", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

def main():
    """Board integrated detection system"""
    print("🎥 BOARD INTEGRATED DETECTION SYSTEM")
    print("=" * 45)
    print("📋 Integrates with board detection for accurate positioning")
    print("🎯 Components show at their actual board grid positions")
    
    # Initialize system
    system = BoardIntegratedSystem()
    
    # Load model
    model_path = Path("dual_board_training/photos_model_fixed/weights/best.pt")
    
    if not model_path.exists():
        alt_paths = [
            Path("dual_board_training/photos_model_fixed/weights/last.pt"),
            Path("dual_board_training/photos_model/weights/best.pt"),
            Path("dual_board_training/photos_model/weights/last.pt")
        ]
        
        model_path = None
        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                break
        
        if not model_path:
            print("❌ No trained model found!")
            return
    
    print(f"🤖 Model: {model_path}")
    
    try:
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("🎥 Camera opened successfully!")
    print("📋 Controls:")
    print("   • 'q': Quit")
    print("   • 'b': Toggle board visualization")
    print("   • 'd': Detect board (setup coordinate mapping)")
    print("   • 'r': Reset board detection")
    print("   • SPACE: Force detection")
    
    frame_count = 0
    last_detection_time = 0
    detection_interval = 2.0
    last_annotated_frame = None
    split_ratio = 0.5
    board_detected = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            current_time = time.time()
            should_detect = (current_time - last_detection_time) >= detection_interval
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            force_detection = (key == ord(' '))
            
            if key == ord('d'):
                print("🔍 Detecting board...")
                board_detected = system.detect_board_and_setup_coordinates(frame)
                if board_detected:
                    print("✅ Board detected and coordinate mapping created")
                else:
                    print("❌ Board detection failed")
            elif key == ord('r'):
                print("🔄 Resetting board detection...")
                system.matrixcoor_to_realcoor = None
                system.board_bounds = None
                board_detected = False
            elif key == ord('b') and VISUALIZER_AVAILABLE:
                system.toggle_board_visualization()
            
            # Detect board automatically on first frame
            if frame_count == 0:
                print("🔍 Auto-detecting board on first frame...")
                board_detected = system.detect_board_and_setup_coordinates(frame)
                if board_detected:
                    print("✅ Board detected automatically")
            
            if should_detect or force_detection:
                print(f"\n🔄 Processing frame {frame_count}...")
                
                # Process FULL frame
                results = model(frame, conf=0.6, iou=0.5)
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None:
                        height, width = frame.shape[:2]
                        split_x = int(width * split_ratio)
                        
                        # Split detections by position
                        left_boxes = []
                        right_boxes = []
                        
                        for i, box in enumerate(result.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            
                            if center_x < split_x:
                                left_boxes.append((i, box))
                            else:
                                right_boxes.append((i, box))
                        
                        print(f"   🎯 LEFT: {len(left_boxes)}, RIGHT: {len(right_boxes)}")
                        
                        if board_detected:
                            print("   🎯 Using board coordinate mapping")
                        else:
                            print("   ⚠️ Using simple coordinate mapping (press 'd' to detect board)")
                        
                        # Update board visualization with accurate positions
                        system.update_board_visualization(
                            left_boxes, right_boxes, model.names, width, height)
                        
                        # Create annotated frame
                        display_frame = result.plot()
                        cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                        cv2.putText(display_frame, "LEFT", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(display_frame, "RIGHT", (split_x + 10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Draw board overlay if detected
                        if board_detected:
                            display_frame = system.draw_board_overlay(display_frame)
                        
                        last_annotated_frame = display_frame.copy()
                
                last_detection_time = current_time
                frame_count += 1
            
            # Show frame
            if last_annotated_frame is not None:
                display_frame = last_annotated_frame
            else:
                height, width = frame.shape[:2]
                split_x = int(width * split_ratio)
                display_frame = frame.copy()
                cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                cv2.putText(display_frame, "LEFT", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "RIGHT", (split_x + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if board_detected:
                    display_frame = system.draw_board_overlay(display_frame)
            
            cv2.imshow('Board Integrated Detection', display_frame)
            
            # Handle other keys
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if system.fig:
            plt.close(system.fig)
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
