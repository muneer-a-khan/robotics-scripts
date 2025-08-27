#!/usr/bin/env python3
"""
Dual Circuit Board Camera System

This system splits a single camera feed into two halves (left and right) and processes
each half as a separate circuit board. This allows monitoring two circuit boards
simultaneously with one camera positioned high above.

Features:
- Single camera input split into left/right regions
- Separate detection and processing for each circuit board
- Independent graph generation and validation for each side
- Dual live visualizations
- Optimized for high-up camera positioning
"""

import cv2
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import threading
from concurrent.futures import ThreadPoolExecutor

from config import VIDEO_CONFIG, OUTPUT_CONFIG, YOLO_CONFIG
from models.component_detector import ComponentDetector
from vision.connection_detector import ConnectionDetector
from circuit.graph_builder import CircuitGraphBuilder
from data_structures import DetectionResult
from graph_output_converter import DetectionToGraphConverter
from live_circuit_visualizer import create_live_visualization
from circuit_validator import CircuitValidator
from enhanced_orientation_detector import EnhancedOrientationDetector


class DualCircuitBoardSystem:
    """
    System for processing two circuit boards from a single camera feed.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 save_outputs: bool = True,
                 display_results: bool = True,
                 enable_validation: bool = False,
                 split_ratio: float = 0.5):
        """
        Initialize the dual circuit board system.
        
        Args:
            model_path: Path to trained YOLOv8 model
            save_outputs: Whether to save detection outputs
            display_results: Whether to display real-time results
            enable_validation: Whether to enable circuit validation
            split_ratio: Ratio for splitting camera feed (0.5 = equal halves)
        """
        self.save_outputs = save_outputs
        self.display_results = display_results
        self.enable_validation = enable_validation
        self.split_ratio = split_ratio
        
        # Initialize pipeline components for both sides
        print("Initializing Dual Circuit Board Vision System...")
        
        # Shared components (can be used for both sides)
        self.component_detector = ComponentDetector(model_path)
        self.connection_detector = ConnectionDetector()
        self.graph_builder = CircuitGraphBuilder()
        self.graph_converter = DetectionToGraphConverter()
        
        # Initialize validation components if enabled
        if self.enable_validation:
            print("Initializing circuit validation system...")
            self.circuit_validator = CircuitValidator()
            self.orientation_detector = EnhancedOrientationDetector()
        else:
            self.circuit_validator = None
            self.orientation_detector = None
        
        # Video capture setup
        self.cap = None
        self.frame_count = {'left': 0, 'right': 0}
        
        # Output directories
        if self.save_outputs:
            self.output_dir = Path("output")
            self.frames_dir = self.output_dir / "frames"
            self.data_dir = self.output_dir / "data"
            
            # Create separate directories for left and right
            self.left_frames_dir = self.frames_dir / "left"
            self.right_frames_dir = self.frames_dir / "right"
            self.left_data_dir = self.data_dir / "left"
            self.right_data_dir = self.data_dir / "right"
            
            for dir_path in [self.output_dir, self.frames_dir, self.data_dir,
                           self.left_frames_dir, self.right_frames_dir,
                           self.left_data_dir, self.right_data_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        print("Dual circuit board system initialized successfully!")
        print(f"Camera feed will be split at ratio: {split_ratio}")
    
    def start_camera(self, camera_id: Optional[int] = None) -> bool:
        """
        Start the camera capture.
        
        Args:
            camera_id: Camera device ID (uses config default if None)
            
        Returns:
            True if camera started successfully
        """
        camera_id = camera_id or VIDEO_CONFIG["camera_id"]
        
        print(f"Starting camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return False
        
        # Set camera properties for high-up positioning
        resolution = VIDEO_CONFIG["resolution"]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, VIDEO_CONFIG["fps"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, VIDEO_CONFIG["buffer_size"])
        
        print("Camera started successfully!")
        return True
    
    def split_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the camera frame into left and right halves.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Tuple of (left_frame, right_frame)
        """
        height, width = frame.shape[:2]
        split_x = int(width * self.split_ratio)
        
        left_frame = frame[:, :split_x]
        right_frame = frame[:, split_x:]
        
        return left_frame, right_frame
    
    def process_side(self, image: np.ndarray, side: str, frame_count: int) -> DetectionResult:
        """
        Process one side of the split frame through the complete pipeline.
        
        Args:
            image: Input image for one side (BGR format)
            side: Either 'left' or 'right'
            frame_count: Frame number for this side
            
        Returns:
            DetectionResult containing the complete analysis
        """
        start_time = time.time()
        
        try:
            # Step 1: Detect components
            components = self.component_detector.detect(image)
            
            # Step 2: Detect connections
            connections = self.connection_detector.detect_connections(image, components)
            
            # Step 3: Build circuit graph
            connection_graph = self.graph_builder.build_graph(
                components, connections, start_time, frame_count
            )
            
            # Step 4: Validate circuit (if validation is enabled)
            validation_result = None
            if self.enable_validation and self.circuit_validator:
                # Perform enhanced orientation detection
                orientation_results = self.orientation_detector.validate_all_orientations(
                    components, image
                )
                
                # Auto-detect best matching reference design
                reference_design = self.circuit_validator.find_matching_reference_design(
                    connection_graph
                )
                
                # Validate circuit
                validation_result = self.circuit_validator.validate_circuit(
                    connection_graph, reference_design, confidence_threshold=0.75
                )
                
                # Add orientation issues to validation result
                orientation_issues = self.orientation_detector.get_orientation_issues(
                    orientation_results
                )
                validation_result["issues"].extend(orientation_issues)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = DetectionResult(
                connection_graph=connection_graph,
                raw_detections=[comp.to_dict() for comp in components],
                processing_time=processing_time,
                validation_result=validation_result
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error processing {side} side: {e}")
            
            # Return empty result with error
            from data_structures import ConnectionGraph, CircuitState
            empty_graph = ConnectionGraph(
                components=[],
                edges=[],
                state=CircuitState(is_circuit_closed=False, power_on=False),
                timestamp=start_time
            )
            
            return DetectionResult(
                connection_graph=empty_graph,
                raw_detections=[],
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def annotate_dual_frame(self, frame: np.ndarray, 
                           left_result: DetectionResult, 
                           right_result: DetectionResult) -> np.ndarray:
        """
        Annotate the dual frame with detection results from both sides.
        
        Args:
            frame: Original full frame
            left_result: Detection result for left side
            right_result: Detection result for right side
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        height, width = frame.shape[:2]
        split_x = int(width * self.split_ratio)
        
        # Annotate left side
        left_frame = annotated[:, :split_x]
        left_annotated = self.component_detector.annotate_image(
            left_frame, left_result.connection_graph.components
        )
        left_annotated = self.connection_detector.visualize_connections(
            left_annotated,
            left_result.connection_graph.components,
            left_result.connection_graph.edges
        )
        annotated[:, :split_x] = left_annotated
        
        # Annotate right side
        right_frame = annotated[:, split_x:]
        right_annotated = self.component_detector.annotate_image(
            right_frame, right_result.connection_graph.components
        )
        right_annotated = self.connection_detector.visualize_connections(
            right_annotated,
            right_result.connection_graph.components,
            right_result.connection_graph.edges
        )
        annotated[:, split_x:] = right_annotated
        
        # Add split line
        cv2.line(annotated, (split_x, 0), (split_x, height), (255, 255, 255), 2)
        
        # Add text overlays for both sides
        self._add_side_overlay(annotated, left_result, "LEFT", 10, 30)
        self._add_side_overlay(annotated, right_result, "RIGHT", split_x + 10, 30)
        
        return annotated
    
    def _add_side_overlay(self, image: np.ndarray, result: DetectionResult, 
                         side: str, x_offset: int, y_offset: int):
        """Add text overlay for one side."""
        state = result.connection_graph.state
        text_lines = [
            f"{side} BOARD:",
            f"Components: {len(result.connection_graph.components)}",
            f"Connections: {len(result.connection_graph.edges)}",
            f"Circuit: {'Closed' if state.is_circuit_closed else 'Open'}",
            f"Power: {'On' if state.power_on else 'Off'}",
            f"Time: {result.processing_time:.3f}s"
        ]
        
        # Add validation information if available
        if result.validation_result:
            validation = result.validation_result
            overall_result = validation.get("overall_result", "unknown")
            score = validation.get("summary", {}).get("score", 0)
            text_lines.extend([
                f"Status: {overall_result.title()}",
                f"Score: {score}%"
            ])
        
        # Draw background
        text_height = 25
        background_height = len(text_lines) * text_height + 10
        cv2.rectangle(image, (x_offset - 5, y_offset - 20), 
                     (x_offset + 250, y_offset + background_height), (0, 0, 0), -1)
        
        # Draw text
        for i, line in enumerate(text_lines):
            y_pos = y_offset + i * text_height
            cv2.putText(image, line, (x_offset, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save_dual_results(self, frame: np.ndarray, 
                         left_result: DetectionResult, 
                         right_result: DetectionResult) -> None:
        """
        Save detection results for both sides.
        
        Args:
            frame: Original full frame
            left_result: Detection result for left side
            right_result: Detection result for right side
        """
        if not self.save_outputs:
            return
        
        timestamp = int(time.time() * 1000)
        
        # Split frame for individual side processing
        left_frame, right_frame = self.split_frame(frame)
        
        # Save results for each side
        self._save_side_results(left_frame, left_result, "left", timestamp)
        self._save_side_results(right_frame, right_result, "right", timestamp)
        
        # Save full annotated frame
        if OUTPUT_CONFIG["save_annotated_frames"]:
            annotated = self.annotate_dual_frame(frame, left_result, right_result)
            frame_path = self.frames_dir / f"dual_frame_{timestamp}.jpg"
            cv2.imwrite(str(frame_path), annotated)
    
    def _save_side_results(self, image: np.ndarray, result: DetectionResult, 
                          side: str, timestamp: int) -> None:
        """Save results for one side."""
        frame_count = self.frame_count[side]
        
        # Determine directories
        if side == "left":
            frames_dir = self.left_frames_dir
            data_dir = self.left_data_dir
        else:
            frames_dir = self.right_frames_dir
            data_dir = self.right_data_dir
        
        # Save annotated frame
        if OUTPUT_CONFIG["save_annotated_frames"]:
            annotated = self.component_detector.annotate_image(
                image, result.connection_graph.components
            )
            frame_path = frames_dir / f"frame_{side}_{timestamp}_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated)
        
        # Save detection data
        if OUTPUT_CONFIG["save_detection_data"]:
            # Save traditional format
            data_path = data_dir / f"detection_{side}_{timestamp}_{frame_count:06d}.json"
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            # Save graph format
            graph_path = data_dir / f"graph_{side}_{timestamp}_{frame_count:06d}.json"
            circuit_graph = self.graph_converter.convert_detection_result(result)
            with open(graph_path, 'w', encoding='utf-8') as f:
                f.write(circuit_graph.to_json())
            
            # Generate live circuit visualization
            try:
                graph_data = json.loads(circuit_graph.to_json())
                visualization_path = create_live_visualization(
                    graph_data, 
                    timestamp=timestamp, 
                    output_dir=str(self.output_dir),
                    validation_data=result.validation_result
                )
                
                if visualization_path:
                    # Rename to include side information
                    side_viz_path = self.output_dir / f"live_circuit_visual_{side}_{timestamp}.png"
                    import shutil
                    shutil.move(visualization_path, side_viz_path)
                    
                    # Also save as "latest" for each side
                    latest_path = self.output_dir / f"latest_circuit_visual_{side}.png"
                    shutil.copy2(side_viz_path, latest_path)
                    
            except Exception as e:
                print(f"Warning: Could not generate live visualization for {side} side: {e}")
    
    def run_dual_real_time(self, camera_id: Optional[int] = None) -> None:
        """
        Run real-time detection on dual circuit boards.
        
        Args:
            camera_id: Camera device ID
        """
        if not self.start_camera(camera_id):
            return
        
        processing_interval = VIDEO_CONFIG.get("processing_interval", 3.0)
        print(f"Starting dual circuit board detection (processing every {processing_interval}s)...")
        print("🔄 Live circuit visualizations will be generated for both sides")
        print("📁 Left side saved to: output/data/left/ and output/latest_circuit_visual_left.png")
        print("📁 Right side saved to: output/data/right/ and output/latest_circuit_visual_right.png")
        
        if self.enable_validation:
            print("✅ Circuit validation is ENABLED for both sides")
        else:
            print("⚠️ Circuit validation is DISABLED - use --validate to enable")
        
        print("Press 'q' to quit, 's' to save current frame, 'p' to pause")
        
        paused = False
        last_process_time = 0
        last_frame = None
        last_left_result = None
        last_right_result = None
        
        try:
            while True:
                current_time = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame at intervals
                should_process = (current_time - last_process_time) >= processing_interval
                
                if not paused and should_process:
                    # Split frame
                    left_frame, right_frame = self.split_frame(frame)
                    
                    # Process both sides in parallel
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        left_future = executor.submit(
                            self.process_side, left_frame, "left", self.frame_count["left"]
                        )
                        right_future = executor.submit(
                            self.process_side, right_frame, "right", self.frame_count["right"]
                        )
                        
                        left_result = left_future.result()
                        right_result = right_future.result()
                    
                    # Save results
                    if self.save_outputs:
                        self.save_dual_results(frame, left_result, right_result)
                    
                    # Update counters
                    self.frame_count["left"] += 1
                    self.frame_count["right"] += 1
                    last_process_time = current_time
                    last_frame = frame
                    last_left_result = left_result
                    last_right_result = right_result
                    
                    # Print status
                    print(f"Processed dual frame. "
                          f"Left: {len(left_result.connection_graph.components)} components, "
                          f"Right: {len(right_result.connection_graph.components)} components")
                
                # Display results
                if self.display_results:
                    if last_frame is not None and last_left_result is not None and last_right_result is not None:
                        display_frame = self.annotate_dual_frame(frame, last_left_result, last_right_result)
                        cv2.imshow("Dual Circuit Board Vision", display_frame)
                    else:
                        # Draw split line on raw frame
                        height, width = frame.shape[:2]
                        split_x = int(width * self.split_ratio)
                        cv2.line(frame, (split_x, 0), (split_x, height), (255, 255, 255), 2)
                        cv2.imshow("Dual Circuit Board Vision", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and last_frame is not None:
                    # Save current frame
                    save_path = self.output_dir / f"dual_manual_save_{int(time.time())}.jpg"
                    annotated = self.annotate_dual_frame(last_frame, last_left_result, last_right_result)
                    cv2.imwrite(str(save_path), annotated)
                    print(f"Dual frame saved to {save_path}")
                elif key == ord('p'):
                    paused = not paused
                    print(f"Detection {'paused' if paused else 'resumed'}")
        
        except KeyboardInterrupt:
            print("\nStopping dual detection...")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Dual system cleanup complete.")


def main():
    """Main entry point for dual circuit board system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Circuit Board Vision System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--model", type=str, help="Path to trained YOLOv8 model")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    parser.add_argument("--no-save", action="store_true", help="Disable saving outputs")
    parser.add_argument("--validate", action="store_true", help="Enable circuit validation")
    parser.add_argument("--split-ratio", type=float, default=0.5, help="Split ratio (0.5 = equal halves)")
    
    args = parser.parse_args()
    
    # Initialize dual system
    system = DualCircuitBoardSystem(
        model_path=args.model,
        save_outputs=not args.no_save,
        display_results=not args.no_display,
        enable_validation=args.validate,
        split_ratio=args.split_ratio
    )
    
    try:
        system.run_dual_real_time(args.camera)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()
