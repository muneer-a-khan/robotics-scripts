#!/usr/bin/env python3
"""
Dual Board Live Detection System

This system provides real-time detection and analysis for two circuit boards
simultaneously from a single camera feed. It includes:

- Green tape coverage detection (blocks processing when hands are detected)
- Dual board component detection with separate processing for each side
- Graph generation for each board with connectivity analysis
- Component orientation validation
- Circuit scoring based on correctness of connections and orientations
- Live visualization with separate windows for each board
- JSON output for each board with timestamp synchronization

The system splits the camera frame in half and processes each side independently,
generating separate graph outputs and scoring for each board.
"""

import cv2
import numpy as np
import time
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from collections import deque
import queue

# Import existing system components
from config import VIDEO_CONFIG, OUTPUT_CONFIG, YOLO_CONFIG
from models.component_detector import ComponentDetector
from vision.connection_detector import ConnectionDetector
from circuit.graph_builder import CircuitGraphBuilder
from data_structures import DetectionResult, ConnectionGraph, CircuitState, ComponentDetection, BoundingBox
from graph_output_converter import DetectionToGraphConverter
from live_circuit_visualizer import create_live_visualization
from circuit_validator import CircuitValidator
from enhanced_orientation_detector import EnhancedOrientationDetector


@dataclass
class GreenTapeStatus:
    """Green tape detection results"""
    left_side_covered: bool
    right_side_covered: bool
    left_coverage_percentage: float
    right_coverage_percentage: float
    overall_coverage_percentage: float
    should_skip: bool
    detection_confidence: float


@dataclass
class BoardAnalysisResult:
    """Analysis result for a single board"""
    board_side: str  # 'left' or 'right'
    detection_result: DetectionResult
    orientation_score: float  # 0-100, percentage of correctly oriented components
    connectivity_score: float  # 0-100, percentage of expected connections made
    overall_score: float  # 0-100, combined score
    component_count: int
    connection_count: int
    validation_issues: List[str]
    processing_time: float
    timestamp: float


@dataclass
class DualBoardAnalysisResult:
    """Combined analysis result for both boards"""
    left_board: BoardAnalysisResult
    right_board: BoardAnalysisResult
    tape_status: GreenTapeStatus
    frame_number: int
    timestamp: float
    should_process: bool


class GreenTapeDetector:
    """Specialized green tape detection for hand blocking detection"""
    
    def __init__(self):
        """Initialize green tape detector"""
        # HSV color ranges for green tape (adjustable)
        self.green_lower = np.array([35, 40, 40])    # Lower HSV bound
        self.green_upper = np.array([85, 255, 255])  # Upper HSV bound
        
        # Detection parameters
        self.min_tape_area_ratio = 0.005  # Minimum tape area to consider valid
        self.coverage_threshold = 0.7     # If tape coverage drops below this, assume covered
        
        # Smoothing for temporal stability
        self.history_size = 5
        self.left_coverage_history = deque(maxlen=self.history_size)
        self.right_coverage_history = deque(maxlen=self.history_size)
    
    def detect_tape_coverage(self, image: np.ndarray, split_ratio: float = 0.5) -> GreenTapeStatus:
        """
        Detect green tape coverage to determine if hands are blocking components
        
        Args:
            image: Input image
            split_ratio: Where to split the image (0.5 = middle)
            
        Returns:
            GreenTapeStatus with detailed coverage information
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for green areas
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Split image to analyze each side
        height, width = image.shape[:2]
        split_x = int(width * split_ratio)
        
        left_mask = green_mask[:, :split_x]
        right_mask = green_mask[:, split_x:]
        
        # Calculate coverage for each side
        left_total_pixels = left_mask.shape[0] * left_mask.shape[1]
        right_total_pixels = right_mask.shape[0] * right_mask.shape[1]
        
        left_green_pixels = np.sum(left_mask > 0)
        right_green_pixels = np.sum(right_mask > 0)
        
        left_coverage = left_green_pixels / left_total_pixels
        right_coverage = right_green_pixels / right_total_pixels
        
        # Add to history for smoothing
        self.left_coverage_history.append(left_coverage)
        self.right_coverage_history.append(right_coverage)
        
        # Calculate smoothed coverage
        left_avg_coverage = np.mean(self.left_coverage_history)
        right_avg_coverage = np.mean(self.right_coverage_history)
        
        # Determine if each side is covered (tape not visible = hand blocking)
        left_covered = left_avg_coverage < self.min_tape_area_ratio
        right_covered = right_avg_coverage < self.min_tape_area_ratio
        
        # Overall coverage
        total_green_pixels = left_green_pixels + right_green_pixels
        total_pixels = left_total_pixels + right_total_pixels
        overall_coverage = total_green_pixels / total_pixels
        
        # Calculate detection confidence based on coverage stability
        left_stability = 1.0 - np.std(self.left_coverage_history) if len(self.left_coverage_history) > 1 else 0.5
        right_stability = 1.0 - np.std(self.right_coverage_history) if len(self.right_coverage_history) > 1 else 0.5
        detection_confidence = (left_stability + right_stability) / 2
        
        # Determine if we should skip processing
        should_skip = left_covered or right_covered
        
        return GreenTapeStatus(
            left_side_covered=left_covered,
            right_side_covered=right_covered,
            left_coverage_percentage=left_avg_coverage * 100,
            right_coverage_percentage=right_avg_coverage * 100,
            overall_coverage_percentage=overall_coverage * 100,
            should_skip=should_skip,
            detection_confidence=detection_confidence
        )
    
    def visualize_tape_detection(self, image: np.ndarray, 
                               tape_status: GreenTapeStatus, 
                               split_ratio: float = 0.5) -> np.ndarray:
        """Create visualization overlay for tape detection"""
        overlay = image.copy()
        height, width = image.shape[:2]
        split_x = int(width * split_ratio)
        
        # Create green mask for visualization
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Highlight green areas
        green_overlay = np.zeros_like(image)
        green_overlay[green_mask > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.8, green_overlay, 0.2, 0)
        
        # Add status text for each side
        left_color = (0, 0, 255) if tape_status.left_side_covered else (0, 255, 0)
        right_color = (0, 0, 255) if tape_status.right_side_covered else (0, 255, 0)
        
        # Left side status
        left_text = f"LEFT: {'BLOCKED' if tape_status.left_side_covered else 'CLEAR'}"
        cv2.putText(overlay, left_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
        cv2.putText(overlay, f"Coverage: {tape_status.left_coverage_percentage:.1f}%", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
        
        # Right side status
        right_text = f"RIGHT: {'BLOCKED' if tape_status.right_side_covered else 'CLEAR'}"
        cv2.putText(overlay, right_text, (split_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
        cv2.putText(overlay, f"Coverage: {tape_status.right_coverage_percentage:.1f}%", 
                   (split_x + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
        
        # Overall status
        overall_color = (0, 0, 255) if tape_status.should_skip else (0, 255, 0)
        overall_text = f"STATUS: {'SKIP PROCESSING' if tape_status.should_skip else 'PROCESSING'}"
        cv2.putText(overlay, overall_text, (width//2 - 100, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, overall_color, 2)
        
        # Split line
        cv2.line(overlay, (split_x, 0), (split_x, height), (255, 255, 255), 2)
        
        return overlay


class BoardAnalyzer:
    """Analyzer for individual board components and connectivity"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize board analyzer"""
        # Detection components
        self.component_detector = ComponentDetector(model_path)
        self.connection_detector = ConnectionDetector()
        self.graph_builder = CircuitGraphBuilder()
        self.graph_converter = DetectionToGraphConverter()
        
        # Validation components (optional)
        self.circuit_validator = CircuitValidator()
        self.orientation_detector = EnhancedOrientationDetector()
        
        # Expected component counts for scoring (adjustable)
        self.expected_components = {
            'battery_holder': 1,
            'switch': 1,
            'led': 1,
            'wire': 2,
            'connection_node': 2
        }
        
        # Expected minimum connections
        self.min_expected_connections = 3
    
    def analyze_board(self, image: np.ndarray, side: str, frame_number: int) -> BoardAnalysisResult:
        """
        Perform complete analysis of a single board
        
        Args:
            image: Board image (already cropped to one side)
            side: 'left' or 'right'
            frame_number: Current frame number
            
        Returns:
            BoardAnalysisResult with complete analysis
        """
        start_time = time.time()
        timestamp = start_time
        
        try:
            # Step 1: Detect components
            components = self.component_detector.detect(image)
            
            # Step 2: Detect connections
            connections = self.connection_detector.detect_connections(image, components)
            
            # Step 3: Build circuit graph
            connection_graph = self.graph_builder.build_graph(
                components, connections, timestamp, frame_number
            )
            
            # Step 4: Enhanced orientation detection
            orientation_results = self.orientation_detector.validate_all_orientations(
                components, image
            )
            
            # Step 5: Circuit validation
            validation_result = self.circuit_validator.validate_circuit(connection_graph)
            
            # Step 6: Calculate scores
            orientation_score = self._calculate_orientation_score(orientation_results)
            connectivity_score = self._calculate_connectivity_score(connection_graph)
            overall_score = (orientation_score + connectivity_score) / 2
            
            # Step 7: Extract validation issues
            validation_issues = self._extract_validation_issues(validation_result)
            
            # Create detection result
            processing_time = time.time() - start_time
            detection_result = DetectionResult(
                connection_graph=connection_graph,
                raw_detections=[comp.to_dict() for comp in components],
                processing_time=processing_time,
                validation_result=validation_result
            )
            
            return BoardAnalysisResult(
                board_side=side,
                detection_result=detection_result,
                orientation_score=orientation_score,
                connectivity_score=connectivity_score,
                overall_score=overall_score,
                component_count=len(components),
                connection_count=len(connections),
                validation_issues=validation_issues,
                processing_time=processing_time,
                timestamp=timestamp
            )
            
        except Exception as e:
            # Return error result
            processing_time = time.time() - start_time
            
            # Create empty graph for error case
            from data_structures import ConnectionGraph, CircuitState
            empty_graph = ConnectionGraph(
                components=[],
                edges=[],
                state=CircuitState(is_circuit_closed=False, power_on=False),
                timestamp=timestamp
            )
            
            empty_result = DetectionResult(
                connection_graph=empty_graph,
                raw_detections=[],
                processing_time=processing_time,
                error_message=str(e)
            )
            
            return BoardAnalysisResult(
                board_side=side,
                detection_result=empty_result,
                orientation_score=0.0,
                connectivity_score=0.0,
                overall_score=0.0,
                component_count=0,
                connection_count=0,
                validation_issues=[f"Analysis failed: {str(e)}"],
                processing_time=processing_time,
                timestamp=timestamp
            )
    
    def _calculate_orientation_score(self, orientation_results: List[Dict]) -> float:
        """Calculate orientation correctness score (0-100)"""
        if not orientation_results:
            return 0.0
        
        correct_orientations = sum(1 for result in orientation_results 
                                 if result.get('orientation_correct', False))
        
        return (correct_orientations / len(orientation_results)) * 100
    
    def _calculate_connectivity_score(self, graph: ConnectionGraph) -> float:
        """Calculate connectivity score based on expected connections (0-100)"""
        actual_connections = len(graph.edges)
        
        if actual_connections >= self.min_expected_connections:
            # Bonus for having more connections than minimum
            base_score = 80
            bonus = min(20, (actual_connections - self.min_expected_connections) * 5)
            return min(100, base_score + bonus)
        else:
            # Partial score based on how close we are to minimum
            return (actual_connections / self.min_expected_connections) * 80
    
    def _extract_validation_issues(self, validation_result: Optional[Dict]) -> List[str]:
        """Extract human-readable validation issues"""
        if not validation_result:
            return []
        
        issues = []
        
        # Extract issues from validation result structure
        if 'issues' in validation_result:
            issues.extend(validation_result['issues'])
        
        if 'warnings' in validation_result:
            issues.extend([f"Warning: {w}" for w in validation_result['warnings']])
        
        return issues


class DualBoardLiveSystem:
    """Main live detection system for dual board setup"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 split_ratio: float = 0.5,
                 processing_interval: float = 2.0,
                 save_outputs: bool = True,
                 display_results: bool = True):
        """
        Initialize the dual board live system
        
        Args:
            model_path: Path to trained model
            split_ratio: Where to split the frame (0.5 = middle)
            processing_interval: Seconds between processing frames
            save_outputs: Whether to save detection outputs
            display_results: Whether to display live results
        """
        self.model_path = model_path
        self.split_ratio = split_ratio
        self.processing_interval = processing_interval
        self.save_outputs = save_outputs
        self.display_results = display_results
        
        # Initialize components
        print("🎯 Initializing Dual Board Live System...")
        self.tape_detector = GreenTapeDetector()
        self.board_analyzer = BoardAnalyzer(model_path)
        
        # Video capture
        self.cap = None
        self.frame_count = 0
        self.last_process_time = 0
        
        # Output setup
        if self.save_outputs:
            self.output_dir = Path("dual_board_output")
            self.left_output_dir = self.output_dir / "left"
            self.right_output_dir = self.output_dir / "right"
            self.combined_output_dir = self.output_dir / "combined"
            
            for dir_path in [self.output_dir, self.left_output_dir, 
                           self.right_output_dir, self.combined_output_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Threading for parallel processing
        self.processing_executor = ThreadPoolExecutor(max_workers=2)
        
        print("✅ Dual Board Live System initialized!")
        print(f"   • Split ratio: {split_ratio}")
        print(f"   • Processing interval: {processing_interval}s")
        print(f"   • Save outputs: {save_outputs}")
        print(f"   • Display results: {display_results}")
    
    def start_camera(self, camera_id: int = 0) -> bool:
        """Start camera capture"""
        print(f"📹 Starting camera {camera_id}...")
        
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {camera_id}")
            return False
        
        # Set camera properties
        resolution = VIDEO_CONFIG.get("resolution", (1920, 1080))
        fps = VIDEO_CONFIG.get("fps", 30)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        print("✅ Camera started successfully!")
        return True
    
    def split_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split frame into left and right boards"""
        height, width = frame.shape[:2]
        split_x = int(width * self.split_ratio)
        
        left_frame = frame[:, :split_x]
        right_frame = frame[:, split_x:]
        
        return left_frame, right_frame
    
    def process_dual_frame(self, frame: np.ndarray) -> DualBoardAnalysisResult:
        """
        Process complete dual frame analysis
        
        Args:
            frame: Full camera frame
            
        Returns:
            DualBoardAnalysisResult with analysis for both sides
        """
        start_time = time.time()
        
        # Check green tape status first
        tape_status = self.tape_detector.detect_tape_coverage(frame, self.split_ratio)
        
        # Determine if we should process based on tape status
        should_process = not tape_status.should_skip
        
        if should_process:
            # Split frame
            left_frame, right_frame = self.split_frame(frame)
            
            # Process both sides in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                left_future = executor.submit(
                    self.board_analyzer.analyze_board, left_frame, "left", self.frame_count
                )
                right_future = executor.submit(
                    self.board_analyzer.analyze_board, right_frame, "right", self.frame_count
                )
                
                left_result = left_future.result()
                right_result = right_future.result()
        else:
            # Create empty results when skipping
            empty_result = self._create_empty_board_result("left", start_time)
            left_result = empty_result
            right_result = self._create_empty_board_result("right", start_time)
        
        return DualBoardAnalysisResult(
            left_board=left_result,
            right_board=right_result,
            tape_status=tape_status,
            frame_number=self.frame_count,
            timestamp=start_time,
            should_process=should_process
        )
    
    def _create_empty_board_result(self, side: str, timestamp: float) -> BoardAnalysisResult:
        """Create empty board result for when processing is skipped"""
        from data_structures import ConnectionGraph, CircuitState
        
        empty_graph = ConnectionGraph(
            components=[],
            edges=[],
            state=CircuitState(is_circuit_closed=False, power_on=False),
            timestamp=timestamp
        )
        
        empty_detection = DetectionResult(
            connection_graph=empty_graph,
            raw_detections=[],
            processing_time=0.0
        )
        
        return BoardAnalysisResult(
            board_side=side,
            detection_result=empty_detection,
            orientation_score=0.0,
            connectivity_score=0.0,
            overall_score=0.0,
            component_count=0,
            connection_count=0,
            validation_issues=["Processing skipped - tape covered"],
            processing_time=0.0,
            timestamp=timestamp
        )
    
    def save_analysis_results(self, frame: np.ndarray, 
                            analysis: DualBoardAnalysisResult) -> None:
        """Save complete analysis results"""
        if not self.save_outputs:
            return
        
        timestamp_str = str(int(analysis.timestamp * 1000))
        
        # Save individual board results
        self._save_board_result(analysis.left_board, frame, timestamp_str)
        self._save_board_result(analysis.right_board, frame, timestamp_str)
        
        # Save combined analysis
        combined_path = self.combined_output_dir / f"analysis_{timestamp_str}.json"
        with open(combined_path, 'w') as f:
            json.dump(asdict(analysis), f, indent=2, default=str)
        
        # Save annotated full frame
        annotated_frame = self._create_annotated_frame(frame, analysis)
        frame_path = self.combined_output_dir / f"frame_{timestamp_str}.jpg"
        cv2.imwrite(str(frame_path), annotated_frame)
    
    def _save_board_result(self, board_result: BoardAnalysisResult, 
                          full_frame: np.ndarray, timestamp_str: str) -> None:
        """Save results for individual board"""
        side = board_result.board_side
        output_dir = self.left_output_dir if side == "left" else self.right_output_dir
        
        # Save detection result JSON
        detection_path = output_dir / f"detection_{timestamp_str}.json"
        with open(detection_path, 'w') as f:
            json.dump(board_result.detection_result.to_dict(), f, indent=2)
        
        # Save graph format
        graph_path = output_dir / f"graph_{timestamp_str}.json"
        circuit_graph = self.board_analyzer.graph_converter.convert_detection_result(
            board_result.detection_result
        )
        with open(graph_path, 'w') as f:
            f.write(circuit_graph.to_json())
        
        # Save board-specific annotated image
        left_frame, right_frame = self.split_frame(full_frame)
        board_image = left_frame if side == "left" else right_frame
        
        # Annotate board image
        annotated_board = self._annotate_board_image(board_image, board_result)
        image_path = output_dir / f"board_{timestamp_str}.jpg"
        cv2.imwrite(str(image_path), annotated_board)
        
        # Generate live circuit visualization
        try:
            graph_data = json.loads(circuit_graph.to_json())
            visualization_path = create_live_visualization(
                graph_data,
                timestamp=int(float(timestamp_str)),
                output_dir=str(output_dir),
                validation_data=board_result.detection_result.validation_result
            )
            
            if visualization_path:
                # Rename to include side information
                viz_path = output_dir / f"circuit_visual_{timestamp_str}.png"
                import shutil
                shutil.move(visualization_path, viz_path)
                
                # Update latest visualization
                latest_viz_path = output_dir / "latest_circuit_visual.png"
                shutil.copy2(viz_path, latest_viz_path)
                
        except Exception as e:
            print(f"⚠️  Could not generate visualization for {side} side: {e}")
    
    def _create_annotated_frame(self, frame: np.ndarray, 
                              analysis: DualBoardAnalysisResult) -> np.ndarray:
        """Create annotated full frame with dual board analysis"""
        annotated = frame.copy()
        height, width = frame.shape[:2]
        split_x = int(width * self.split_ratio)
        
        # Draw split line
        cv2.line(annotated, (split_x, 0), (split_x, height), (255, 255, 255), 2)
        
        # Add component detection bounding boxes for both sides
        self._draw_component_detections(annotated, analysis, split_x)
        
        # Add tape detection overlay if needed
        if analysis.tape_status.should_skip:
            annotated = self.tape_detector.visualize_tape_detection(
                annotated, analysis.tape_status, self.split_ratio
            )
        
        # Add analysis overlays for each side
        self._add_board_overlay(annotated, analysis.left_board, "left")
        self._add_board_overlay(annotated, analysis.right_board, "right")
        
        # Add global info
        global_info = [
            f"Frame: {analysis.frame_number}",
            f"Processing: {'ACTIVE' if analysis.should_process else 'SKIPPED'}",
            f"Tape Status: {'OK' if not analysis.tape_status.should_skip else 'BLOCKED'}"
        ]
        
        for i, info in enumerate(global_info):
            cv2.putText(annotated, info, (width - 300, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def _draw_component_detections(self, image: np.ndarray, 
                                 analysis: DualBoardAnalysisResult, 
                                 split_x: int) -> None:
        """Draw component detection bounding boxes on the full frame"""
        # Get the original frame dimensions
        height, width = image.shape[:2]
        
        # Extract components from detection results
        left_components = analysis.left_board.detection_result.connection_graph.components
        right_components = analysis.right_board.detection_result.connection_graph.components
        
        # Draw left side components
        left_frame = image[:, :split_x]
        if len(left_components) > 0:
            left_annotated = self.board_analyzer.component_detector.annotate_image(
                left_frame, left_components
            )
            image[:, :split_x] = left_annotated
        
        # Draw right side components (need to adjust coordinates)
        right_frame = image[:, split_x:]
        if len(right_components) > 0:
            # Adjust component coordinates for right side offset
            adjusted_components = []
            for comp in right_components:
                # Create a copy with adjusted coordinates
                adjusted_comp = ComponentDetection(
                    id=comp.id,
                    label=comp.label,
                    bbox=BoundingBox(
                        comp.bbox.x1 - split_x,  # Adjust for right side offset
                        comp.bbox.y1,
                        comp.bbox.x2 - split_x,
                        comp.bbox.y2
                    ),
                    orientation=comp.orientation,
                    confidence=comp.confidence,
                    component_type=comp.component_type,
                    switch_state=comp.switch_state,
                    connection_points=comp.connection_points,
                    metadata=comp.metadata
                )
                adjusted_components.append(adjusted_comp)
            
            right_annotated = self.board_analyzer.component_detector.annotate_image(
                right_frame, adjusted_components
            )
            image[:, split_x:] = right_annotated
    
    def _add_board_overlay(self, image: np.ndarray, 
                          board_result: BoardAnalysisResult, side: str) -> None:
        """Add analysis overlay for one board"""
        height, width = image.shape[:2]
        split_x = int(width * self.split_ratio)
        
        # Determine position based on side
        if side == "left":
            x_offset = 10
        else:
            x_offset = split_x + 10
        
        # Text information
        info_lines = [
            f"{side.upper()} BOARD",
            f"Score: {board_result.overall_score:.1f}%",
            f"Components: {board_result.component_count}",
            f"Connections: {board_result.connection_count}",
            f"Orientation: {board_result.orientation_score:.1f}%",
            f"Connectivity: {board_result.connectivity_score:.1f}%"
        ]
        
        # Background rectangle
        rect_height = len(info_lines) * 25 + 20
        cv2.rectangle(image, (x_offset - 5, height - rect_height - 10), 
                     (x_offset + 250, height - 10), (0, 0, 0), -1)
        
        # Text
        color = (0, 255, 0) if board_result.overall_score > 70 else (0, 255, 255) if board_result.overall_score > 40 else (0, 0, 255)
        
        for i, line in enumerate(info_lines):
            y_pos = height - rect_height + 20 + i * 25
            text_color = color if i == 1 else (255, 255, 255)  # Highlight score line
            cv2.putText(image, line, (x_offset, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    def _annotate_board_image(self, board_image: np.ndarray, 
                            board_result: BoardAnalysisResult) -> np.ndarray:
        """Create annotated image for individual board"""
        # Use component detector to annotate detections
        annotated = self.board_analyzer.component_detector.annotate_image(
            board_image, board_result.detection_result.connection_graph.components
        )
        
        # Add connection visualization
        annotated = self.board_analyzer.connection_detector.visualize_connections(
            annotated,
            board_result.detection_result.connection_graph.components,
            board_result.detection_result.connection_graph.edges
        )
        
        # Add score overlay
        score_text = f"Score: {board_result.overall_score:.1f}%"
        score_color = (0, 255, 0) if board_result.overall_score > 70 else (0, 255, 255) if board_result.overall_score > 40 else (0, 0, 255)
        
        cv2.putText(annotated, score_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, score_color, 2)
        
        return annotated
    
    def run_live_detection(self, camera_id: int = 0) -> None:
        """
        Run the live dual board detection system
        
        Args:
            camera_id: Camera device ID
        """
        if not self.start_camera(camera_id):
            return
        
        print("🚀 Starting Dual Board Live Detection System")
        print(f"   📹 Camera: {camera_id}")
        print(f"   ⏱️  Processing interval: {self.processing_interval}s")
        print(f"   🎯 Split ratio: {self.split_ratio}")
        print(f"   💾 Save outputs: {self.save_outputs}")
        print(f"   🖥️  Display results: {self.display_results}")
        print("\n🎮 Controls:")
        print("   • 'q': Quit")
        print("   • 's': Save current frame")  
        print("   • 'p': Pause/Resume processing")
        print("   • 't': Toggle tape detection overlay")
        print("   • SPACE: Force process current frame")
        
        paused = False
        show_tape_overlay = False
        last_analysis = None
        
        try:
            while True:
                current_time = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                # Check if it's time to process
                should_process_time = (current_time - self.last_process_time) >= self.processing_interval
                
                if not paused and should_process_time:
                    print(f"🔄 Processing frame {self.frame_count}...")
                    
                    # Process frame
                    analysis = self.process_dual_frame(frame)
                    
                    # Save results
                    if self.save_outputs:
                        self.save_analysis_results(frame, analysis)
                    
                    # Update state
                    self.frame_count += 1
                    self.last_process_time = current_time
                    last_analysis = analysis
                    
                    # Print status
                    left_score = analysis.left_board.overall_score
                    right_score = analysis.right_board.overall_score
                    tape_status = "OK" if not analysis.tape_status.should_skip else "BLOCKED"
                    
                    print(f"   📊 Results: Left={left_score:.1f}%, Right={right_score:.1f}%, Tape={tape_status}")
                
                # Display results
                if self.display_results:
                    if last_analysis and not show_tape_overlay:
                        display_frame = self._create_annotated_frame(frame, last_analysis)
                    elif show_tape_overlay:
                        tape_status = self.tape_detector.detect_tape_coverage(frame, self.split_ratio)
                        display_frame = self.tape_detector.visualize_tape_detection(
                            frame, tape_status, self.split_ratio
                        )
                    else:
                        # Just show split line
                        display_frame = frame.copy()
                        height, width = frame.shape[:2]
                        split_x = int(width * self.split_ratio)
                        cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 2)
                    
                    cv2.imshow('Dual Board Live Detection', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and last_analysis:
                    # Save current frame
                    manual_save_path = self.output_dir / f"manual_save_{int(time.time())}.jpg"
                    annotated = self._create_annotated_frame(frame, last_analysis)
                    cv2.imwrite(str(manual_save_path), annotated)
                    print(f"💾 Manual save: {manual_save_path}")
                elif key == ord('p'):
                    paused = not paused
                    print(f"⏸️  Processing {'paused' if paused else 'resumed'}")
                elif key == ord('t'):
                    show_tape_overlay = not show_tape_overlay
                    print(f"🎭 Tape overlay {'enabled' if show_tape_overlay else 'disabled'}")
                elif key == ord(' '):
                    # Force process current frame
                    print("🔄 Force processing current frame...")
                    analysis = self.process_dual_frame(frame)
                    if self.save_outputs:
                        self.save_analysis_results(frame, analysis)
                    last_analysis = analysis
                    self.frame_count += 1
                    print(f"   ✅ Forced processing complete")
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.processing_executor.shutdown(wait=True)
        print("✅ Cleanup complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Board Live Detection System")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device ID")
    parser.add_argument("--model", "-m", help="Path to trained model")
    parser.add_argument("--split", "-s", type=float, default=0.5, help="Frame split ratio")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="Processing interval (seconds)")
    parser.add_argument("--no-display", action="store_true", help="Disable live display")
    parser.add_argument("--no-save", action="store_true", help="Disable saving outputs")
    
    args = parser.parse_args()
    
    # Initialize system
    system = DualBoardLiveSystem(
        model_path=args.model,
        split_ratio=args.split,
        processing_interval=args.interval,
        save_outputs=not args.no_save,
        display_results=not args.no_display
    )
    
    # Run live detection
    system.run_live_detection(args.camera)


if __name__ == "__main__":
    main()
