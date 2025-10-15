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
        
        # Timed capture state
        self.capture_mode = False
        self.capture_duration = 120  # 2 minutes
        self.capture_interval = 2  # Every 2 seconds
        self.capture_start_time = None
        self.last_capture_time = None
        self.captured_frames = []
        self.last_detection_data = None
        
        # Output directory
        self.output_dir = Path("modified_system_output")
        self.output_dir.mkdir(exist_ok=True)
        
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

    def check_circuit_completion_by_side(self, graph_data):
        """
        Check circuit completion separately for left and right sides
        """
        # Separate nodes by side
        left_nodes = {nid: ndata for nid, ndata in graph_data['nodes'].items() if ndata['board_side'] == 'left'}
        right_nodes = {nid: ndata for nid, ndata in graph_data['nodes'].items() if ndata['board_side'] == 'right'}
        
        # Separate connections by side (only connections within the same side)
        left_connections = []
        right_connections = []
        
        for conn in graph_data['connections']:
            comp1_id = conn['component1']
            comp2_id = conn['component2']
            
            # Check if both components are on the same side
            if comp1_id in left_nodes and comp2_id in left_nodes:
                left_connections.append(conn)
            elif comp1_id in right_nodes and comp2_id in right_nodes:
                right_connections.append(conn)
        
        # Analyze left side
        left_result = self._analyze_single_side_completion(left_nodes, left_connections, 'LEFT')
        
        # Analyze right side
        right_result = self._analyze_single_side_completion(right_nodes, right_connections, 'RIGHT')
        
        return {
            'left_side': left_result,
            'right_side': right_result
        }
    
    def _analyze_single_side_completion(self, nodes, connections, side_name):
        """Analyze circuit completion for a single side"""
        if not nodes:
            return {
                'is_complete': False,
                'reason': f'No components detected on {side_name} side',
                'status': 'OPEN',
                'side': side_name,
                'total_components': 0,
                'connected_components': 0
            }
        
        # Find battery holder on this side
        battery_id = None
        for node_id, node_data in nodes.items():
            if 'Battery Holder' in node_data['type']:
                battery_id = node_id
                break
        
        if not battery_id:
            return {
                'is_complete': False,
                'reason': f'No battery holder detected on {side_name} side',
                'status': 'OPEN',
                'side': side_name,
                'total_components': len(nodes),
                'connected_components': 0
            }
        
        # Build adjacency list from connections on this side
        connections_map = {}
        for conn in connections:
            comp1_id = conn['component1']
            comp2_id = conn['component2']
            
            if comp1_id not in connections_map:
                connections_map[comp1_id] = []
            if comp2_id not in connections_map:
                connections_map[comp2_id] = []
            
            connections_map[comp1_id].append(comp2_id)
            connections_map[comp2_id].append(comp1_id)
        
        # Check if battery has at least 2 connections (needed for a closed loop)
        battery_connections = len(connections_map.get(battery_id, []))
        
        if battery_connections < 2:
            return {
                'is_complete': False,
                'reason': f'Battery holder has {battery_connections} connection(s) on {side_name} side (need at least 2 for a circuit)',
                'status': 'OPEN',
                'side': side_name,
                'total_components': len(nodes),
                'connected_components': 1,
                'battery_connections': battery_connections
            }
        
        # Find all components connected to the battery using BFS
        visited = set()
        queue = [battery_id]
        visited.add(battery_id)
        
        while queue:
            current = queue.pop(0)
            for neighbor in connections_map.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        connected_components = len(visited)
        total_components = len(nodes)
        
        # Check if we have a closed loop (battery connects back to itself through multiple paths)
        has_loop = battery_connections >= 2
        
        if has_loop and connected_components >= 3:  # Battery + at least 2 other components
            return {
                'is_complete': True,
                'reason': f'Circuit forms a closed loop with {connected_components} connected components on {side_name} side',
                'status': 'CLOSED',
                'side': side_name,
                'connected_components': connected_components,
                'total_components': total_components,
                'battery_connections': battery_connections
            }
        else:
            return {
                'is_complete': False,
                'reason': f'Circuit does not form a closed loop on {side_name} side (battery connections: {battery_connections}, connected components: {connected_components})',
                'status': 'OPEN',
                'side': side_name,
                'connected_components': connected_components,
                'total_components': total_components,
                'battery_connections': battery_connections
            }
    
    def analyze_circuit_graph(self, left_boxes, right_boxes, frame_width, frame_height, save_files=True, timestamp=None):
        """Analyze circuit connectivity and save results"""
        if not VISUALIZER_AVAILABLE or not self.graph_analyzer:
            print("⚠️ Graph analyzer not available")
            return None
        
        # Check if there are any detections
        if not left_boxes and not right_boxes:
            print("⚠️ No component detections found - skipping graph analysis")
            return None
        
        try:
            print(f"   Analyzing {len(left_boxes)} left components and {len(right_boxes)} right components...")
            
            # Perform circuit analysis
            graph_data = self.graph_analyzer.analyze_circuit(
                left_boxes, right_boxes, self.model.names, frame_width, frame_height
            )
            
            if not graph_data or 'nodes' not in graph_data:
                print("⚠️ Graph analysis returned no data")
                return None
            
            # Check circuit completion by side
            completion_status = self.check_circuit_completion_by_side(graph_data)
            graph_data['circuit_completion'] = completion_status
            
            # Update the analyzer's graph_data with completion status
            self.graph_analyzer.graph_data['circuit_completion'] = completion_status
            
            if save_files:
                # Generate timestamp for filenames if not provided
                if timestamp is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save to JSON in output directory
                json_filepath = self.output_dir / f"circuit_graph_{timestamp}.json"
                print(f"   Saving graph to: {json_filepath}")
                self.graph_analyzer.save_to_json(str(json_filepath))
                
                # Save summary to text in output directory (with side separation)
                txt_filepath = self.output_dir / f"circuit_summary_{timestamp}.txt"
                print(f"   Saving summary to: {txt_filepath}")
                self._save_summary_with_sides(str(txt_filepath), graph_data, completion_status)
                
                # Print summary to console
                print("\n" + "="*60)
                print(self._get_connection_summary_with_sides(graph_data, completion_status))
                print("="*60 + "\n")
            
            return graph_data
            
        except Exception as e:
            print(f"❌ Error analyzing circuit graph: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_connection_summary_with_sides(self, graph_data, completion_status):
        """Generate connection summary separated by side"""
        lines = []
        lines.append("=== CIRCUIT CONNECTIVITY ANALYSIS (BY SIDE) ===")
        
        # Separate nodes by side
        left_nodes = {nid: ndata for nid, ndata in graph_data['nodes'].items() if ndata['board_side'] == 'left'}
        right_nodes = {nid: ndata for nid, ndata in graph_data['nodes'].items() if ndata['board_side'] == 'right'}
        
        # Separate connections by side
        left_connections = [c for c in graph_data['connections'] 
                           if graph_data['nodes'][c['component1']]['board_side'] == 'left']
        right_connections = [c for c in graph_data['connections'] 
                            if graph_data['nodes'][c['component1']]['board_side'] == 'right']
        
        lines.append(f"\nLEFT SIDE: {len(left_nodes)} components, {len(left_connections)} connections")
        lines.append(f"RIGHT SIDE: {len(right_nodes)} components, {len(right_connections)} connections")
        lines.append(f"TOTAL: {len(graph_data['nodes'])} components, {len(graph_data['connections'])} connections")
        
        # Left side circuit status
        lines.append("\n" + "="*60)
        lines.append("LEFT SIDE CIRCUIT STATUS:")
        left_status = completion_status['left_side']
        lines.append(f"  Status: {left_status['status']}")
        lines.append(f"  Reason: {left_status['reason']}")
        lines.append(f"  Components: {left_status.get('connected_components', 0)}/{left_status.get('total_components', 0)} connected")
        
        # Right side circuit status
        lines.append("\n" + "="*60)
        lines.append("RIGHT SIDE CIRCUIT STATUS:")
        right_status = completion_status['right_side']
        lines.append(f"  Status: {right_status['status']}")
        lines.append(f"  Reason: {right_status['reason']}")
        lines.append(f"  Components: {right_status.get('connected_components', 0)}/{right_status.get('total_components', 0)} connected")
        
        lines.append("\n" + "="*60)
        
        # Use the existing summary from graph analyzer
        lines.append("\n" + self.graph_analyzer.get_connection_summary())
        
        return "\n".join(lines)
    
    def _save_summary_with_sides(self, filepath, graph_data, completion_status):
        """Save summary with side separation to text file"""
        with open(filepath, 'w') as f:
            f.write(self._get_connection_summary_with_sides(graph_data, completion_status))
        print(f"✅ Circuit summary saved to: {filepath}")
    
    def analyze_saved_final_detection(self, image_path, timestamp=None):
        """
        Analyze a saved final detection image and create graph files
        Uses the same simple analysis as pressing 'g' during live capture
        """
        import cv2
        
        print(f"\n🔍 Analyzing saved image: {image_path}")
        
        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return False
        
        height, width = image.shape[:2]
        
        # Run detection on the image
        results = self.model(image, conf=0.6, iou=0.5)
        if not results or len(results) == 0:
            print("❌ No detections found in image")
            return False
        
        result = results[0]
        if result.boxes is None:
            print("❌ No bounding boxes detected")
            return False
        
        # Split boxes by side (same as live system)
        split_x = int(width * 0.5)
        left_boxes = []
        right_boxes = []
        
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            
            if center_x < split_x:
                left_boxes.append((i, box))
            else:
                right_boxes.append((i, box))
        
        print(f"   Found {len(left_boxes)} left components and {len(right_boxes)} right components")
        
        if not left_boxes and not right_boxes:
            print("⚠️ No component detections to analyze")
            return False
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Run simple graph analysis (same as pressing 'g')
        print("🔍 Analyzing circuit connectivity...")
        
        try:
            # Use the graph analyzer directly
            graph_data = self.graph_analyzer.analyze_circuit(
                left_boxes, right_boxes, self.model.names, width, height
            )
            
            # Check circuit completion by side
            completion_status = self.check_circuit_completion_by_side(graph_data)
            graph_data['circuit_completion'] = completion_status
            self.graph_analyzer.graph_data['circuit_completion'] = completion_status
            
            # Save to JSON in output directory
            json_filepath = self.output_dir / f"circuit_graph_{timestamp}.json"
            print(f"   Saving graph to: {json_filepath}")
            self.graph_analyzer.save_to_json(str(json_filepath))
            
            # Save summary to text in output directory (with side separation)
            txt_filepath = self.output_dir / f"circuit_summary_{timestamp}.txt"
            print(f"   Saving summary to: {txt_filepath}")
            self._save_summary_with_sides(str(txt_filepath), graph_data, completion_status)
            
            # Print summary to console
            print("\n" + "="*60)
            print(self._get_connection_summary_with_sides(graph_data, completion_status))
            print("="*60 + "\n")
            
            print("✅ Graph analysis complete!")
            return True
            
        except Exception as e:
            print(f"❌ Graph analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
        """Run the integrated circuit system with timed capture mode"""
        if not self.load_model():
            return
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("\n🎯 TIMED CIRCUIT CAPTURE SYSTEM")
        print("=" * 50)
        print("\n💡 Workflow:")
        print("   • 2-minute timer starts automatically")
        print("   • Frame captured every 2 seconds with bounding boxes")
        print("   • At end: 3-second delay, then final analysis")
        print("   • Final output: Board visualization + Circuit JSON")
        
        print("\n📋 Manual Controls (optional):")
        print("   • 'c': Start timed capture mode manually")
        print("   • 'b': Toggle board visualization")
        print("   • 'g': Analyze circuit graph and save results")
        print("   • 's': Save current frame") 
        print("   • 'q': Quit")
        print("\n⏱️  Starting 2-minute capture in 3 seconds...")
        time.sleep(3)
        
        # Auto-start capture mode
        self.capture_mode = True
        self.capture_start_time = time.time()
        self.last_capture_time = time.time()
        print(f"🟢 CAPTURE MODE STARTED - {self.capture_duration} seconds remaining")
        print()
        
        split_ratio = 0.5
        frame_count = 0
        
        # Initialize detection variables
        left_boxes = []
        right_boxes = []
        current_frame_with_boxes = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                key = cv2.waitKey(1) & 0xFF
                
                # Check if capture mode timer has expired
                if self.capture_mode:
                    elapsed_time = current_time - self.capture_start_time
                    remaining_time = self.capture_duration - elapsed_time
                    
                    if elapsed_time >= self.capture_duration:
                        print(f"\n⏱️  2-minute timer complete!")
                        print("⏸️  Waiting 3 seconds before final analysis...")
                        time.sleep(3)
                        
                        # Perform final analysis on last captured detection
                        if self.last_detection_data:
                            print("\n🔍 Performing final circuit analysis...")
                            
                            # Extract data from last detection
                            last_left_boxes = self.last_detection_data['left_boxes']
                            last_right_boxes = self.last_detection_data['right_boxes']
                            last_frame = self.last_detection_data['frame']
                            last_frame_with_boxes = self.last_detection_data['frame_with_boxes']
                            width = self.last_detection_data['width']
                            height = self.last_detection_data['height']
                            
                            # Debug: show component counts
                            if 'component_count' in self.last_detection_data:
                                left_count, right_count = self.last_detection_data['component_count']
                                print(f"   Components detected: {left_count} left, {right_count} right (excluding green tape)")
                            print(f"   Total detections: {len(last_left_boxes)} left boxes, {len(last_right_boxes)} right boxes")
                            
                            # Save final frame with bounding boxes
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            final_frame_path = self.output_dir / f"final_detection_{timestamp}.jpg"
                            cv2.imwrite(str(final_frame_path), last_frame_with_boxes)
                            print(f"📸 Final detection frame saved: {final_frame_path}")
                            
                            # Generate board visualization
                            left_detections, right_detections = convert_detections_to_7x5_grid(
                                last_left_boxes, last_right_boxes, self.model.names, width, height)
                            
                            if VISUALIZER_AVAILABLE:
                                print("🎨 Generating board visualization...")
                                fig = self.visualizer.create_dual_board_visualization(left_detections, right_detections)
                                viz_path = self.output_dir / f"board_visualization_{timestamp}.png"
                                fig.savefig(str(viz_path), dpi=150, bbox_inches='tight')
                                print(f"📊 Board visualization saved: {viz_path}")
                                plt.close(fig)
                            
                            # Generate circuit graph JSON with completion status
                            print("🔍 Analyzing circuit connectivity and completion...")
                            graph_data = self.analyze_circuit_graph(last_left_boxes, last_right_boxes, 
                                                                   width, height, save_files=True, timestamp=timestamp)
                            
                            # If graph analysis failed, try to re-analyze from the saved image
                            if not graph_data:
                                print("\n⚠️ Initial graph analysis failed - attempting to re-analyze from saved image...")
                                try:
                                    # Re-run detection on the saved final frame
                                    results = self.model(last_frame, conf=0.6, iou=0.5)
                                    if results and len(results) > 0:
                                        result = results[0]
                                        if result.boxes is not None:
                                            # Re-split boxes by side
                                            split_x = int(width * 0.5)
                                            retry_left_boxes = []
                                            retry_right_boxes = []
                                            
                                            for i, box in enumerate(result.boxes):
                                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                                center_x = (x1 + x2) / 2
                                                
                                                if center_x < split_x:
                                                    retry_left_boxes.append((i, box))
                                                else:
                                                    retry_right_boxes.append((i, box))
                                            
                                            # Try graph analysis again
                                            graph_data = self.analyze_circuit_graph(retry_left_boxes, retry_right_boxes,
                                                                                   width, height, save_files=True, timestamp=timestamp)
                                except Exception as e:
                                    print(f"❌ Re-analysis also failed: {e}")
                            
                            if graph_data:
                                print(f"\n✅ Analysis complete! Total frames captured: {len(self.captured_frames)}")
                            else:
                                print(f"\n⚠️ Graph analysis could not be completed, but {len(self.captured_frames)} frames were captured")
                            
                        else:
                            print("⚠️ No detection data available for final analysis")
                        
                        # End capture mode and exit
                        self.capture_mode = False
                        print("\n🏁 Timed capture complete. Shutting down system...")
                        break  # Exit the main loop
                
                # Handle controls
                if key == ord('c') and not self.capture_mode:
                    # Manually start capture mode
                    self.capture_mode = True
                    self.capture_start_time = time.time()
                    self.last_capture_time = time.time()
                    self.captured_frames = []
                    self.last_detection_data = None
                    print(f"🟢 CAPTURE MODE STARTED - {self.capture_duration} seconds")
                
                elif key == ord('b') and VISUALIZER_AVAILABLE:
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
                        self.analyze_circuit_graph(left_boxes, right_boxes, width, height, save_files=True)
                    else:
                        print("⚠️ No detections available for graph analysis - ensure components are visible on camera")
                
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = self.output_dir / f"circuit_frame_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
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
                        
                        # In capture mode, save frame every 2 seconds
                        if self.capture_mode:
                            time_since_last_capture = current_time - self.last_capture_time
                            
                            if time_since_last_capture >= self.capture_interval:
                                # Save frame with bounding boxes in output directory
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                capture_filename = self.output_dir / f"capture_{len(self.captured_frames)+1:03d}_{timestamp}.jpg"
                                cv2.imwrite(str(capture_filename), display_frame)
                                self.captured_frames.append(str(capture_filename))
                                
                                elapsed = current_time - self.capture_start_time
                                remaining = max(0, self.capture_duration - elapsed)
                                print(f"📸 Captured frame {len(self.captured_frames)} - {remaining:.0f}s remaining")
                                
                                self.last_capture_time = current_time
                        
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
                        
                        # Store detection data for final analysis (always in capture mode)
                        if self.capture_mode:
                            # Count non-green-tape components for debugging
                            left_non_tape = sum(1 for i, box in left_boxes 
                                              if self.model.names[int(box.cls[0])] != "Green tape")
                            right_non_tape = sum(1 for i, box in right_boxes 
                                               if self.model.names[int(box.cls[0])] != "Green tape")
                            
                            self.last_detection_data = {
                                'left_boxes': left_boxes,
                                'right_boxes': right_boxes,
                                'frame': frame.copy(),
                                'frame_with_boxes': display_frame.copy(),
                                'width': width,
                                'height': height,
                                'component_count': (left_non_tape, right_non_tape)
                            }
                        
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
                
                # Add capture mode status
                y_offset = 60
                if self.capture_mode:
                    elapsed = time.time() - self.capture_start_time
                    remaining = max(0, self.capture_duration - elapsed)
                    timer_text = f"CAPTURE MODE: {remaining:.0f}s remaining | Frames: {len(self.captured_frames)}"
                    cv2.putText(display_frame, timer_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    y_offset += 35
                
                # Add processing status
                if self.processing_paused:
                    status_text = "PROCESSING PAUSED - Green tape covered on either side"
                    status_color = (0, 0, 255)  # Red
                else:
                    status_text = "PROCESSING ACTIVE - Green tape uncovered on both sides"
                    status_color = (0, 255, 0)  # Green
                
                cv2.putText(display_frame, status_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                
                cv2.imshow('Timed Circuit Capture System', display_frame)
        
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
    print("🎯 TIMED CIRCUIT CAPTURE SYSTEM")
    print("=" * 50)
    print()
    print("Features:")
    print("• Automatic 2-minute timed capture workflow")
    print("• Captures frame every 2 seconds with bounding boxes")
    print("• Final board visualization and circuit analysis")
    print("• Circuit completion detection (open/closed)")
    print("• Structured JSON output with connectivity data")
    print()
    
    system = IntegratedCircuitSystem()
    system.run_system()

if __name__ == "__main__":
    import sys
    
    # Check if user wants to analyze a saved image
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-image":
        if len(sys.argv) < 3:
            print("Usage: python integrated_circuit_system_modified.py --analyze-image <path_to_final_detection.jpg>")
            print("\nExample:")
            print("  python integrated_circuit_system_modified.py --analyze-image modified_system_output/final_detection_20251015_171442.jpg")
            sys.exit(1)
        
        image_path = sys.argv[2]
        
        # Extract timestamp from filename if possible
        import re
        timestamp_match = re.search(r'(\d{8}_\d{6})', image_path)
        timestamp = timestamp_match.group(1) if timestamp_match else None
        
        print("🔍 ANALYZING SAVED DETECTION IMAGE")
        print("=" * 50)
        
        system = IntegratedCircuitSystem()
        if system.load_model():
            system.analyze_saved_final_detection(image_path, timestamp=timestamp)
        
    else:
        # Normal mode - run timed capture system
        main()
