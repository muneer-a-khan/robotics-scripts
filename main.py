"""
Main application for the Snap Circuit computer vision system.
Orchestrates the complete pipeline from video input to circuit analysis.
"""

import cv2
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from config import VIDEO_CONFIG, OUTPUT_CONFIG, YOLO_CONFIG
from models.component_detector import ComponentDetector
from vision.connection_detector import ConnectionDetector
from circuit.graph_builder import CircuitGraphBuilder
from data_structures import DetectionResult


class SnapCircuitVisionSystem:
    """
    Main class orchestrating the complete Snap Circuit vision pipeline.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 save_outputs: bool = True,
                 display_results: bool = True):
        """
        Initialize the vision system.
        
        Args:
            model_path: Path to trained YOLOv8 model
            save_outputs: Whether to save detection outputs
            display_results: Whether to display real-time results
        """
        self.save_outputs = save_outputs
        self.display_results = display_results
        
        # Initialize pipeline components
        print("Initializing Snap Circuit Vision System...")
        
        self.component_detector = ComponentDetector(model_path)
        self.connection_detector = ConnectionDetector()
        self.graph_builder = CircuitGraphBuilder()
        
        # Video capture setup
        self.cap = None
        self.frame_count = 0
        
        # Output directories
        if self.save_outputs:
            self.output_dir = Path("output")
            self.frames_dir = self.output_dir / "frames"
            self.data_dir = self.output_dir / "data"
            
            for dir_path in [self.output_dir, self.frames_dir, self.data_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        print("Vision system initialized successfully!")
    
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
        
        # Set camera properties
        resolution = VIDEO_CONFIG["resolution"]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, VIDEO_CONFIG["fps"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, VIDEO_CONFIG["buffer_size"])
        
        print("Camera started successfully!")
        return True
    
    def process_frame(self, image: np.ndarray) -> DetectionResult:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            image: Input image (BGR format)
            
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
                components, connections, start_time, self.frame_count
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = DetectionResult(
                connection_graph=connection_graph,
                raw_detections=[comp.to_dict() for comp in components],
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error processing frame: {e}")
            
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
    
    def annotate_frame(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Annotate frame with detection results.
        
        Args:
            image: Input image
            result: Detection result
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw components
        annotated = self.component_detector.annotate_image(
            annotated, result.connection_graph.components
        )
        
        # Draw connections
        annotated = self.connection_detector.visualize_connections(
            annotated, 
            result.connection_graph.components,
            result.connection_graph.edges
        )
        
        # Add text overlay with circuit state
        state = result.connection_graph.state
        text_lines = [
            f"Components: {len(result.connection_graph.components)}",
            f"Connections: {len(result.connection_graph.edges)}",
            f"Circuit Closed: {state.is_circuit_closed}",
            f"Power On: {state.power_on}",
            f"Active: {len(state.active_components)}",
            f"Processing: {result.processing_time:.3f}s"
        ]
        
        # Draw text background
        text_height = 25
        background_height = len(text_lines) * text_height + 20
        cv2.rectangle(annotated, (10, 10), (300, background_height), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (300, background_height), (255, 255, 255), 2)
        
        # Draw text
        for i, line in enumerate(text_lines):
            y_pos = 30 + i * text_height
            cv2.putText(annotated, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(annotated, f"Frame: {self.frame_count}", 
                   (annotated.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def save_results(self, image: np.ndarray, result: DetectionResult) -> None:
        """
        Save detection results to disk.
        
        Args:
            image: Original image
            result: Detection result
        """
        if not self.save_outputs:
            return
        
        timestamp = int(time.time() * 1000)
        
        # Save annotated frame
        if OUTPUT_CONFIG["save_annotated_frames"]:
            annotated = self.annotate_frame(image, result)
            frame_path = self.frames_dir / f"frame_{timestamp}_{self.frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated)
        
        # Save detection data
        if OUTPUT_CONFIG["save_detection_data"]:
            data_path = self.data_dir / f"detection_{timestamp}_{self.frame_count:06d}.json"
            with open(data_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
    
    def run_real_time(self, camera_id: Optional[int] = None) -> None:
        """
        Run real-time detection on camera feed.
        
        Args:
            camera_id: Camera device ID
        """
        if not self.start_camera(camera_id):
            return
        
        print("Starting real-time detection...")
        print("Press 'q' to quit, 's' to save current frame, 'p' to pause")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read frame from camera")
                        break
                    
                    # Process frame
                    result = self.process_frame(frame)
                    
                    # Save results
                    self.save_results(frame, result)
                    
                    # Update frame counter
                    self.frame_count += 1
                    
                    # Print status
                    if self.frame_count % 30 == 0:  # Every 30 frames
                        print(f"Processed {self.frame_count} frames. "
                              f"Components: {len(result.connection_graph.components)}, "
                              f"Circuit closed: {result.connection_graph.state.is_circuit_closed}")
                
                # Display results
                if self.display_results and 'frame' in locals():
                    annotated = self.annotate_frame(frame, result)
                    cv2.imshow("Snap Circuit Vision", annotated)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and 'frame' in locals():
                    # Save current frame manually
                    save_path = self.output_dir / f"manual_save_{int(time.time())}.jpg"
                    annotated = self.annotate_frame(frame, result)
                    cv2.imwrite(str(save_path), annotated)
                    print(f"Frame saved to {save_path}")
                elif key == ord('p'):
                    paused = not paused
                    print(f"Detection {'paused' if paused else 'resumed'}")
        
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            self.cleanup()
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Process a video file and save results.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.process_frame(frame)
                
                # Annotate frame
                annotated = self.annotate_frame(frame, result)
                
                # Write to output video
                if writer:
                    writer.write(annotated)
                
                # Save results
                self.save_results(frame, result)
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                
                # Display if enabled
                if self.display_results:
                    cv2.imshow("Video Processing", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
        print(f"Video processing complete. Processed {frame_count} frames.")
    
    def process_image(self, image_path: str) -> DetectionResult:
        """
        Process a single image file.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Detection result
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        result = self.process_frame(image)
        
        # Save results
        self.save_results(image, result)
        
        # Display if enabled
        if self.display_results:
            annotated = self.annotate_frame(image, result)
            cv2.imshow("Image Processing", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Snap Circuit Vision System")
    parser.add_argument("--mode", choices=["camera", "video", "image"], 
                       default="camera", help="Processing mode")
    parser.add_argument("--input", type=str, help="Input file path (for video/image modes)")
    parser.add_argument("--output", type=str, help="Output file path (for video mode)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--model", type=str, help="Path to trained YOLOv8 model")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    parser.add_argument("--no-save", action="store_true", help="Disable saving outputs")
    
    args = parser.parse_args()
    
    # Initialize system
    system = SnapCircuitVisionSystem(
        model_path=args.model,
        save_outputs=not args.no_save,
        display_results=not args.no_display
    )
    
    try:
        if args.mode == "camera":
            system.run_real_time(args.camera)
        elif args.mode == "video":
            if not args.input:
                print("Error: Video mode requires --input argument")
                return
            system.process_video_file(args.input, args.output)
        elif args.mode == "image":
            if not args.input:
                print("Error: Image mode requires --input argument")
                return
            result = system.process_image(args.input)
            print(f"Image processed. Found {len(result.connection_graph.components)} components")
            print(f"Circuit closed: {result.connection_graph.state.is_circuit_closed}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        system.cleanup()


if __name__ == "__main__":
    main() 