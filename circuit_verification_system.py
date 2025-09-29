#!/usr/bin/env python3
"""
Circuit Verification System

This system compares the current detected circuit against ground truth circuits
and provides feedback on completion status and suggested next steps.
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Import our visualizer
try:
    from dual_board_visualizer import DualBoardVisualizer, convert_detections_with_positions
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False

class CircuitVerificationSystem:
    def __init__(self):
        self.model = None
        self.visualizer = None
        self.ground_truths = {}
        self.current_target_circuit = None
        self.last_verification_time = 0
        self.verification_interval = 5.0  # Check every 5 seconds
        self.completion_status = False
        
        if VISUALIZER_AVAILABLE:
            self.visualizer = DualBoardVisualizer(cell_size=25)
        
        self.load_ground_truths()
    
    def load_model(self):
        """Load YOLO model"""
        model_path = Path("dual_board_training/photos_model_fixed/weights/best.pt")
        
        if not model_path.exists():
            alt_paths = [
                Path("dual_board_training/photos_model_fixed/weights/last.pt"),
                Path("dual_board_training/photos_model/weights/best.pt")
            ]
            
            model_path = None
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break
        
        if not model_path:
            print("❌ No trained model found!")
            return False
            
        try:
            self.model = YOLO(str(model_path))
            print(f"✅ Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_ground_truths(self):
        """Load ground truth data"""
        gt_file = Path("ground_truths.json")
        if gt_file.exists():
            try:
                with open(gt_file, 'r') as f:
                    self.ground_truths = json.load(f)
                print(f"📥 Loaded {len(self.ground_truths)} ground truth circuits")
            except Exception as e:
                print(f"❌ Error loading ground truths: {e}")
                self.ground_truths = {}
        else:
            print("❌ No ground truths found! Please run ground_truth_capture.py first")
    
    def select_target_circuit(self, circuit_id):
        """Select which circuit the user is trying to build"""
        if str(circuit_id) not in self.ground_truths:
            print(f"❌ Circuit {circuit_id} not found in ground truths")
            return False
        
        self.current_target_circuit = circuit_id
        self.completion_status = False
        
        gt = self.ground_truths[str(circuit_id)]
        print(f"\n🎯 Target Circuit Selected: {circuit_id}")
        print(f"   Name: {gt['name']}")
        print(f"   Difficulty: {gt['difficulty']}")
        print(f"   Description: {gt['description']}")
        print(f"   Target Components: {gt['total_components']}")
        
        return True
    
    def compare_components(self, current_left, current_right, target_left, target_right):
        """Compare current detections with target ground truth"""
        # Count components in current detection
        current_components = defaultdict(int)
        for comp_type, detections in current_left.items():
            current_components[comp_type] += len(detections)
        for comp_type, detections in current_right.items():
            current_components[comp_type] += len(detections)
        
        # Count components in target (duplicating LEFT ground truth to both sides)
        # Since ground truths were captured on LEFT board only, we allow participants 
        # to build on either board or both boards
        target_components = defaultdict(int)
        
        # Ground truth components (originally captured on LEFT board only)
        ground_truth_components = defaultdict(int)
        for comp_type, detections in target_left.items():
            ground_truth_components[comp_type] += len(detections)
        # target_right should be empty since we only capture on left, but check anyway
        for comp_type, detections in target_right.items():
            ground_truth_components[comp_type] += len(detections)
        
        # Allow building the same circuit on either side - just match component counts
        target_components = ground_truth_components.copy()
        
        # Calculate match statistics
        total_matches = 0
        total_target = sum(target_components.values())
        missing_components = {}
        extra_components = {}
        
        # Check each target component
        for comp_type, target_count in target_components.items():
            current_count = current_components.get(comp_type, 0)
            matches = min(current_count, target_count)
            total_matches += matches
            
            if current_count < target_count:
                missing_components[comp_type] = target_count - current_count
            elif current_count > target_count:
                extra_components[comp_type] = current_count - target_count
        
        # Check for unexpected components
        for comp_type, current_count in current_components.items():
            if comp_type not in target_components and current_count > 0:
                extra_components[comp_type] = current_count
        
        # Calculate match percentage
        match_percentage = (total_matches / total_target * 100) if total_target > 0 else 0
        
        return {
            'match_percentage': match_percentage,
            'total_matches': total_matches,
            'total_target': total_target,
            'missing_components': missing_components,
            'extra_components': extra_components,
            'is_complete': match_percentage >= 95 and len(missing_components) == 0
        }
    
    def suggest_next_step(self, comparison_result):
        """Suggest the next step based on comparison results"""
        suggestions = []
        
        if comparison_result['missing_components']:
            suggestions.append("🔧 MISSING COMPONENTS:")
            for comp_type, count in comparison_result['missing_components'].items():
                if count == 1:
                    suggestions.append(f"   • Add {count} {comp_type}")
                else:
                    suggestions.append(f"   • Add {count} {comp_type}s")
        
        if comparison_result['extra_components']:
            suggestions.append("⚠️ EXTRA COMPONENTS:")
            for comp_type, count in comparison_result['extra_components'].items():
                if count == 1:
                    suggestions.append(f"   • Remove {count} {comp_type}")
                else:
                    suggestions.append(f"   • Remove {count} {comp_type}s")
        
        if not suggestions:
            suggestions.append("✅ Circuit looks complete! Great job!")
        
        return suggestions
    
    def verify_current_circuit(self, current_left, current_right):
        """Verify current circuit against target"""
        if not self.current_target_circuit:
            return None, []
        
        gt = self.ground_truths[str(self.current_target_circuit)]
        target_left = gt['left_detections']
        target_right = gt['right_detections']
        
        # Compare circuits
        comparison = self.compare_components(current_left, current_right, target_left, target_right)
        
        # Generate suggestions
        suggestions = self.suggest_next_step(comparison)
        
        # Check if circuit is complete
        if comparison['is_complete'] and not self.completion_status:
            self.completion_status = True
            print(f"\n🎉 CIRCUIT COMPLETED! 🎉")
            print(f"Congratulations! You successfully built Circuit {self.current_target_circuit}")
            print(f"Match: {comparison['match_percentage']:.1f}%")
        
        return comparison, suggestions
    
    def create_comparison_visualization(self, current_left, current_right):
        """Create side-by-side comparison visualization"""
        if not self.current_target_circuit or not VISUALIZER_AVAILABLE:
            return None
        
        gt = self.ground_truths[str(self.current_target_circuit)]
        target_left = gt['left_detections']
        target_right = gt['right_detections']
        
        # Create comparison figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Draw current state
        self.visualizer._draw_single_board(ax1, current_left, "CURRENT - LEFT")
        self.visualizer._draw_single_board(ax2, current_right, "CURRENT - RIGHT")
        
        # Draw target state
        self.visualizer._draw_single_board(ax3, target_left, "TARGET - LEFT")
        self.visualizer._draw_single_board(ax4, target_right, "TARGET - RIGHT")
        
        # Add title
        fig.suptitle(f"Circuit Verification - Target: {gt['name']}", 
                     fontsize=16, fontweight='bold')
        
        return fig
    
    def run_verification_session(self):
        """Run the circuit verification session"""
        if not self.load_model():
            return
        
        if not self.ground_truths:
            print("❌ No ground truths available. Please run ground_truth_capture.py first")
            return
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("\n🔍 CIRCUIT VERIFICATION SYSTEM")
        print("=" * 40)
        print("📋 Available Circuits:")
        for circuit_id, gt in self.ground_truths.items():
            print(f"   {circuit_id}: {gt['name']} ({gt['difficulty']}) - {gt['total_components']} components")
        
        print("\n📋 Controls:")
        print("   • Number keys (1-6): Select target circuit to build")
        print("   • 'v': Show comparison visualization")
        print("   • 'n': Select new circuit (reset completion status)")
        print("   • 'h': Show help and suggestions")
        print("   • 'q': Quit")
        print()
        
        split_ratio = 0.5
        last_detection_time = 0
        detection_interval = 1.0
        show_suggestions = True
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                current_time = time.time()
                key = cv2.waitKey(1) & 0xFF
                
                # Handle circuit selection
                if key >= ord('1') and key <= ord('6'):
                    circuit_id = key - ord('0')
                    if self.select_target_circuit(circuit_id):
                        show_suggestions = True
                
                # Handle other controls
                elif key == ord('v') and VISUALIZER_AVAILABLE and self.current_target_circuit:
                    # Show comparison visualization (will be updated with next detection)
                    print("🎨 Comparison visualization will update on next detection")
                
                elif key == ord('n'):
                    print("\n🔄 Select new target circuit (press number 1-6)")
                    self.current_target_circuit = None
                    self.completion_status = False
                
                elif key == ord('h'):
                    show_suggestions = True
                
                elif key == ord('q'):
                    break
                
                # Periodic detection and verification
                should_verify = (self.current_target_circuit and 
                               current_time - self.last_verification_time >= self.verification_interval)
                
                if current_time - last_detection_time >= detection_interval:
                    # Run detection
                    results = self.model(frame, conf=0.6, iou=0.5)
                    
                    if results and len(results) > 0:
                        result = results[0]
                        display_frame = result.plot()
                        
                        if result.boxes is not None and self.current_target_circuit:
                            height, width = frame.shape[:2]
                            split_x = int(width * split_ratio)
                            
                            # Split detections
                            left_boxes = []
                            right_boxes = []
                            
                            for i, box in enumerate(result.boxes):
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                center_x = (x1 + x2) / 2
                                
                                if center_x < split_x:
                                    left_boxes.append((i, box))
                                else:
                                    right_boxes.append((i, box))
                            
                            # Convert to our format
                            current_left, current_right = convert_detections_with_positions(
                                left_boxes, right_boxes, self.model.names, width, height)
                            
                            # Verify circuit every 5 seconds
                            if should_verify:
                                comparison, suggestions = self.verify_current_circuit(current_left, current_right)
                                
                                if comparison:
                                    print(f"\n🔍 Circuit Verification - {datetime.now().strftime('%H:%M:%S')}")
                                    print(f"   Match: {comparison['match_percentage']:.1f}% ({comparison['total_matches']}/{comparison['total_target']} components)")
                                    
                                    if show_suggestions and suggestions:
                                        print("   Suggestions:")
                                        for suggestion in suggestions:
                                            print(f"   {suggestion}")
                                    
                                    # Create comparison visualization
                                    if VISUALIZER_AVAILABLE and key == ord('v'):
                                        fig = self.create_comparison_visualization(current_left, current_right)
                                        if fig:
                                            plt.show(block=False)
                                
                                self.last_verification_time = current_time
                                show_suggestions = False  # Only show once per verification
                    
                    else:
                        display_frame = frame.copy()
                    
                    last_detection_time = current_time
                else:
                    display_frame = frame.copy()
                
                # Add overlay information
                height, width = frame.shape[:2]
                split_x = int(width * split_ratio)
                cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                cv2.putText(display_frame, "LEFT", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "RIGHT", (split_x + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add current target info
                if self.current_target_circuit:
                    gt = self.ground_truths[str(self.current_target_circuit)]
                    status = "COMPLETED ✅" if self.completion_status else "IN PROGRESS"
                    info_text = f"Target: Circuit {self.current_target_circuit} - {status}"
                    cv2.putText(display_frame, info_text, (10, height - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.completion_status else (0, 255, 255), 2)
                    
                    target_text = f"{gt['name']} ({gt['difficulty']})"
                    cv2.putText(display_frame, target_text, (10, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "Press 1-6 to select target circuit", (10, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('Circuit Verification System', display_frame)
        
        except KeyboardInterrupt:
            print("\n🛑 Verification session interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if VISUALIZER_AVAILABLE:
                plt.close('all')
            print("✅ Verification session complete")

def main():
    """Main function"""
    print("🔍 CIRCUIT VERIFICATION SYSTEM")
    print("=" * 35)
    print()
    print("This system compares your current circuit against reference circuits")
    print("and provides feedback on completion and next steps.")
    print()
    
    verifier = CircuitVerificationSystem()
    verifier.run_verification_session()

if __name__ == "__main__":
    main()
