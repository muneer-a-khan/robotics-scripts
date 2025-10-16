#!/usr/bin/env python3
"""
Integrated Circuit Building System - Simplified

Combines live detection and board visualization for circuit building activities.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt

# Import our components
try:
    from dual_board_visualizer import DualBoardVisualizer, convert_detections_to_7x5_grid
    from circuit_graph_analyzer import CircuitGraphAnalyzer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False

class IntegratedCircuitSystem:
    def __init__(self):
        self.model = None
        self.visualizer = None
        self.graph_analyzer = None
        
        # System state
        self.show_board_viz = False
        self.processing_paused = False
        
        # Visualization
        self.fig = None
        self.ax1 = None
        self.ax2 = None 
        
        if VISUALIZER_AVAILABLE:
            self.visualizer = DualBoardVisualizer(cell_size=60)  # Larger cells for better visibility
            self.graph_analyzer = CircuitGraphAnalyzer(connection_threshold=50.0)
    
    def load_model(self):
        """Load YOLO model"""
        model_path = Path("dual_board_training/photos_model_fixed/weights/best.pt")
        
        if not model_path.exists():
            alt_paths = [
                Path("dual_board_training/photos_model_fixed/weights/last.pt"),
                Path("dual_board_training/photos_model/weights/best.pt")
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break
        
        if not model_path or not model_path.exists():
            print("❌ No trained model found!")
            return False
            
        try:
            self.model = YOLO(str(model_path))
            print(f"✅ Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def analyze_circuit_graph(self, left_boxes, right_boxes, frame_width, frame_height, frame=None):
        """Analyze circuit connectivity and save results"""
        if not VISUALIZER_AVAILABLE or not self.graph_analyzer:
            print("⚠️ Graph analyzer not available")
            return
        
        try:
            # Perform circuit analysis (with frame for LED orientation detection)
            graph_data = self.graph_analyzer.analyze_circuit(
                left_boxes, right_boxes, self.model.names, frame_width, frame_height, frame=frame
            )
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save to JSON
            json_filepath = f"circuit_graph_{timestamp}.json"
            self.graph_analyzer.save_to_json(json_filepath)
            
            # Save summary to text
            txt_filepath = f"circuit_summary_{timestamp}.txt"
            self.graph_analyzer.save_summary_to_text(txt_filepath)
            
            # Print summary to console
            print("\n" + "="*60)
            print(self.graph_analyzer.get_connection_summary())
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Error analyzing circuit graph: {e}")
    
    def update_board_visualization(self, left_detections, right_detections):
        """Update board visualization"""
        if not self.show_board_viz or not VISUALIZER_AVAILABLE:
            return
            
        try:
            # Create figure only once
            if self.fig is None:
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
                plt.ion()  # Enable interactive mode
            
            # Clear and redraw
            self.ax1.clear()
            self.ax2.clear()
            
            self.visualizer._draw_single_board(self.ax1, left_detections, "LEFT BOARD")
            self.visualizer._draw_single_board(self.ax2, right_detections, "RIGHT BOARD")
            self.fig.suptitle("Dual Board Visualization", fontsize=16, fontweight='bold')
            
            # Update display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
    
    def run_system(self):
        """Run the integrated circuit system"""
        if not self.load_model():
            return
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("\n🎯 SIMPLIFIED DUAL BOARD DETECTION SYSTEM")
        print("=" * 45)
        print("\n💡 Instructions:")
        print("   • Live detection with dual board splitting")
        print("   • 🟢 Green tape uncovered on BOTH sides = Processing active (board visualization updates)")
        print("   • 🔴 Green tape covered on EITHER side = Processing paused (no updates)")
        
        print("\n📋 Controls:")
        print("   • 'b': Toggle board visualization")
        print("   • 'g': Analyze circuit graph and save results")
        print("   • 's': Save current frame") 
        print("   • 'q': Quit")
        print()
        
        split_ratio = 0.5
        frame_count = 0
        
        # Initialize detection variables
        left_boxes = []
        right_boxes = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                key = cv2.waitKey(1) & 0xFF
                
                # Handle controls
                if key == ord('b') and VISUALIZER_AVAILABLE:
                    self.show_board_viz = not self.show_board_viz
                    if self.show_board_viz:
                        plt.ion()
                        print("🎨 Board visualization enabled")
                    else:
                        if self.fig:
                            plt.close(self.fig)
                            self.fig = None
                            self.ax1 = None
                            self.ax2 = None
                        print("🎨 Board visualization disabled")
                
                elif key == ord('g'):
                    # Analyze circuit graph with current detections
                    if left_boxes or right_boxes:
                        print("🔍 Analyzing circuit connectivity...")
                        self.analyze_circuit_graph(left_boxes, right_boxes, width, height, frame=frame)
                    else:
                        print("⚠️ No detections available for graph analysis - ensure components are visible on camera")
                
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"circuit_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Frame saved: {filename}")
                
                elif key == ord('q'):
                    break
                
                # Run detection continuously
                results = self.model(frame, conf=0.6, iou=0.5)
                
                if results and len(results) > 0:
                    result = results[0]
                    display_frame = result.plot()
                    
                    if result.boxes is not None:
                        height, width = frame.shape[:2]
                        split_x = int(width * split_ratio)
                        
                        # Check for green tape coverage on both sides
                        left_green_tape_boxes = []
                        right_green_tape_boxes = []
                        left_other_boxes = []
                        right_other_boxes = []
                        left_boxes = []
                        right_boxes = []
                        
                        # Separate detections by side and type
                        for i, box in enumerate(result.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            
                            # Categorize by left/right for processing
                            if center_x < split_x:
                                left_boxes.append((i, box))
                                if class_name == "Green tape":
                                    left_green_tape_boxes.append((x1, y1, x2, y2))
                                else:
                                    left_other_boxes.append((x1, y1, x2, y2))
                            else:
                                right_boxes.append((i, box))
                                if class_name == "Green tape":
                                    right_green_tape_boxes.append((x1, y1, x2, y2))
                                else:
                                    right_other_boxes.append((x1, y1, x2, y2))
                        
                        # Check if green tape is covered on LEFT side
                        left_tape_covered = False
                        for tape_box in left_green_tape_boxes:
                            tape_x1, tape_y1, tape_x2, tape_y2 = tape_box
                            for other_box in left_other_boxes:
                                other_x1, other_y1, other_x2, other_y2 = other_box
                                # Check for bounding box overlap
                                if (tape_x1 < other_x2 and tape_x2 > other_x1 and 
                                    tape_y1 < other_y2 and tape_y2 > other_y1):
                                    left_tape_covered = True
                                    break
                            if left_tape_covered:
                                break
                        
                        # Check if green tape is covered on RIGHT side
                        right_tape_covered = False  
                        for tape_box in right_green_tape_boxes:
                            tape_x1, tape_y1, tape_x2, tape_y2 = tape_box
                            for other_box in right_other_boxes:
                                other_x1, other_y1, other_x2, other_y2 = other_box
                                # Check for bounding box overlap
                                if (tape_x1 < other_x2 and tape_x2 > other_x1 and 
                                    tape_y1 < other_y2 and tape_y2 > other_y1):
                                    right_tape_covered = True
                                    break
                            if right_tape_covered:
                                break
                        
                        # Update processing status based on green tape coverage on either side
                        total_left_tapes = len(left_green_tape_boxes)
                        total_right_tapes = len(right_green_tape_boxes)
                        
                        if total_left_tapes == 0 and total_right_tapes == 0:
                            # No green tape detected on either side
                            if not self.processing_paused:
                                print("🔴 No green tape detected on either side - PAUSING processing")
                                self.processing_paused = True
                        elif (total_left_tapes > 0 and left_tape_covered) or (total_right_tapes > 0 and right_tape_covered):
                            # Green tape is covered on at least one side
                            if not self.processing_paused:
                                covered_sides = []
                                if total_left_tapes > 0 and left_tape_covered:
                                    covered_sides.append("LEFT")
                                if total_right_tapes > 0 and right_tape_covered:
                                    covered_sides.append("RIGHT")
                                print(f"🔴 Green tape covered on {' and '.join(covered_sides)} side(s) - PAUSING processing")
                                self.processing_paused = True
                        else:
                            # Green tape is uncovered on both sides (where it exists)
                            if self.processing_paused:
                                print("🟢 Green tape uncovered on both sides - RESUMING processing")
                                self.processing_paused = False
                        
                        # Only process if not paused (green tape visible)
                        if not self.processing_paused:
                            # Convert detections to 7x5 grid coordinates
                            left_detections, right_detections = convert_detections_to_7x5_grid(
                                left_boxes, right_boxes, self.model.names, width, height)
                            
                            # Update visualization
                            self.update_board_visualization(left_detections, right_detections)
                else:
                    display_frame = frame.copy()
                
                frame_count += 1
                
                # Add overlay information
                height, width = frame.shape[:2]
                split_x = int(width * split_ratio)
                cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                cv2.putText(display_frame, "LEFT", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "RIGHT", (split_x + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add processing status
                y_offset = 60
                if self.processing_paused:
                    status_text = "PROCESSING PAUSED - Green tape covered on either side"
                    status_color = (0, 0, 255)  # Red
                else:
                    status_text = "PROCESSING ACTIVE - Green tape uncovered on both sides"
                    status_color = (0, 255, 0)  # Green
                
                cv2.putText(display_frame, status_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                
                cv2.imshow('Dual Board Detection System', display_frame)
        
        except KeyboardInterrupt:
            print("\n🛑 System interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.fig:
                plt.close(self.fig)
                plt.close('all')  # Close any remaining matplotlib windows
            print("✅ System shutdown complete")

def main():
    """Main function"""
    print("🎯 SIMPLIFIED DUAL BOARD DETECTION SYSTEM")
    print("=" * 40)
    print()
    print("Features:")
    print("• Live component detection with YOLO")
    print("• Dual board splitting (left/right)")
    print("• Real-time board visualization")  
    print("• Dual-side green tape coverage detection")
    print()
    
    system = IntegratedCircuitSystem()
    system.run_system()

if __name__ == "__main__":
    main()
