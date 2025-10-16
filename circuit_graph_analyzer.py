#!/usr/bin/env python3
"""
Circuit Graph Analyzer
Analyzes component bounding boxes to create a connectivity graph
"""

import json
import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import time
import cv2
import numpy as np

# Import LED orientation detector
try:
    from led_orientation_detector import LEDOrientationDetector
    LED_DETECTOR_AVAILABLE = True
except ImportError:
    LED_DETECTOR_AVAILABLE = False
    print("⚠️ LED orientation detector not available")

# Import Horn orientation detector (using simple grayscale version like LED)
try:
    from horn_orientation_detector_simple import SimpleHornOrientationDetector as HornOrientationDetector
    HORN_DETECTOR_AVAILABLE = True
except ImportError:
    HORN_DETECTOR_AVAILABLE = False
    print("⚠️ Horn orientation detector not available")

# Import terminal-based analyzer
try:
    from terminal_based_circuit_analyzer import TerminalBasedCircuitAnalyzer
    TERMINAL_ANALYZER_AVAILABLE = True
except ImportError:
    TERMINAL_ANALYZER_AVAILABLE = False
    print("⚠️ Terminal-based circuit analyzer not available")

# Import component name mapper
try:
    from component_name_mapper import map_component_name
    NAME_MAPPER_AVAILABLE = True
except ImportError:
    NAME_MAPPER_AVAILABLE = False
    def map_component_name(name):
        return name

@dataclass
class ComponentNode:
    """Represents a component in the circuit graph"""
    id: str
    type: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    center: Tuple[float, float]
    grid_row: int
    grid_col: int
    board_side: str  # 'left' or 'right'

@dataclass
class Connection:
    """Represents a connection between two components"""
    component1_id: str
    component2_id: str
    distance: float
    connection_type: str  # 'direct', 'close', 'wire'

class CircuitGraphAnalyzer:
    def __init__(self, connection_threshold: float = 50.0):
        """
        Initialize circuit graph analyzer
        
        Args:
            connection_threshold: Max pixel distance to consider components connected
        """
        self.connection_threshold = connection_threshold
        self.nodes: Dict[str, ComponentNode] = {}
        self.connections: List[Connection] = []
        self.graph_data = {}
        self.led_orientations = {}  # Store LED orientation results
        self.horn_orientations = {}  # Store Horn orientation results
        self.horn_reclassifications = {}  # Track which box indices should be reclassified as Horn
        self.frame = None  # Store frame for orientation detection
        
        # Initialize LED orientation detector if available
        if LED_DETECTOR_AVAILABLE:
            try:
                self.led_detector = LEDOrientationDetector()
            except Exception as e:
                print(f"⚠️ Could not initialize LED detector: {e}")
                self.led_detector = None
        else:
            self.led_detector = None
        
        # Initialize Horn orientation detector if available
        if HORN_DETECTOR_AVAILABLE:
            try:
                self.horn_detector = HornOrientationDetector()
            except Exception as e:
                print(f"⚠️ Could not initialize Horn detector: {e}")
                self.horn_detector = None
        else:
            self.horn_detector = None
        
        # Initialize terminal-based analyzer if available
        if TERMINAL_ANALYZER_AVAILABLE:
            self.terminal_analyzer = TerminalBasedCircuitAnalyzer(min_overlap_percentage=15.0)
        else:
            self.terminal_analyzer = None
        
    def add_detections(self, left_boxes, right_boxes, model_names, frame_width, frame_height):
        """Add detected components to the graph"""
        self.nodes.clear()
        self.connections.clear()
        
        # Clear terminal analyzer if available
        if self.terminal_analyzer:
            self.terminal_analyzer.components.clear()
            self.terminal_analyzer.terminals.clear()
            self.terminal_analyzer.connections.clear()
        
        # Process left side detections
        component_counter = 0
        for i, box in left_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            raw_class_name = model_names[class_id]
            confidence = float(box.conf[0])
            
            # Skip green tape for graph analysis
            if raw_class_name == "Green tape":
                continue
            
            # Apply name mapping (e.g., Photoresistor -> Horn)
            class_name = map_component_name(raw_class_name)
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            center = (float(center_x), float(center_y))
            
            # Check if this component was reclassified as Horn based on red plus detection
            reclass_key = f"L_{component_counter}"
            if reclass_key in self.horn_reclassifications:
                class_name = 'Horn'
            
            # Create node ID with the (possibly reclassified) class name
            node_id = f"L_{component_counter}_{class_name}"
            
            # Store orientation with the final node_id if we had a reclassification
            if reclass_key in self.horn_reclassifications and 'orientation' in self.horn_reclassifications[reclass_key]:
                self.horn_orientations[node_id] = self.horn_reclassifications[reclass_key]['orientation']
            
            component_counter += 1
            
            self.nodes[node_id] = ComponentNode(
                id=node_id,
                type=class_name,
                confidence=confidence,
                bbox=bbox,
                center=center,
                grid_row=0,  # Will be calculated later if needed
                grid_col=0,
                board_side='left'
            )
            
            # Add to terminal analyzer
            if self.terminal_analyzer:
                led_orientation = self.led_orientations.get(node_id)
                horn_orientation = self.horn_orientations.get(node_id)
                # Pass orientation data (either LED or Horn)
                orientation_data = led_orientation if led_orientation else horn_orientation
                self.terminal_analyzer.add_component(
                    component_id=node_id,
                    component_type=class_name,
                    bbox=bbox,
                    confidence=confidence,
                    board_side='left',
                    led_orientation=orientation_data
                )
        
        # Process right side detections
        for i, box in right_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            raw_class_name = model_names[class_id]
            confidence = float(box.conf[0])
            
            # Skip green tape for graph analysis
            if raw_class_name == "Green tape":
                continue
            
            # Apply name mapping (e.g., Photoresistor -> Horn)
            class_name = map_component_name(raw_class_name)
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            center = (float(center_x), float(center_y))
            
            # Check if this component was reclassified as Horn based on red plus detection
            reclass_key = f"R_{component_counter}"
            if reclass_key in self.horn_reclassifications:
                class_name = 'Horn'
            
            # Create node ID with the (possibly reclassified) class name
            node_id = f"R_{component_counter}_{class_name}"
            
            # Store orientation with the final node_id if we had a reclassification
            if reclass_key in self.horn_reclassifications and 'orientation' in self.horn_reclassifications[reclass_key]:
                self.horn_orientations[node_id] = self.horn_reclassifications[reclass_key]['orientation']
            
            component_counter += 1
            
            self.nodes[node_id] = ComponentNode(
                id=node_id,
                type=class_name,
                confidence=confidence,
                bbox=bbox,
                center=center,
                grid_row=0,
                grid_col=0,
                board_side='right'
            )
            
            # Add to terminal analyzer
            if self.terminal_analyzer:
                led_orientation = self.led_orientations.get(node_id)
                horn_orientation = self.horn_orientations.get(node_id)
                # Pass orientation data (either LED or Horn)
                orientation_data = led_orientation if led_orientation else horn_orientation
                self.terminal_analyzer.add_component(
                    component_id=node_id,
                    component_type=class_name,
                    bbox=bbox,
                    confidence=confidence,
                    board_side='right',
                    led_orientation=orientation_data
                )
    
    def calculate_distance(self, node1: ComponentNode, node2: ComponentNode) -> float:
        """Calculate Euclidean distance between two component centers"""
        x1, y1 = node1.center
        x2, y2 = node2.center
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def bbox_overlap_or_close(self, node1: ComponentNode, node2: ComponentNode) -> bool:
        """Check if two components' bounding boxes overlap or are very close"""
        x1_min, y1_min, x1_max, y1_max = node1.bbox
        x2_min, y2_min, x2_max, y2_max = node2.bbox
        
        # Add padding for "close" detection
        padding = 20.0
        
        # Check if bounding boxes overlap or are within padding distance
        overlap_x = (x1_min - padding <= x2_max) and (x2_min - padding <= x1_max)
        overlap_y = (y1_min - padding <= y1_max) and (y2_min - padding <= y1_max)
        
        return overlap_x and overlap_y
    
    def determine_connection_type(self, node1: ComponentNode, node2: ComponentNode, distance: float) -> str:
        """Determine the type of connection between two components"""
        # Direct connection (overlapping or very close)
        if self.bbox_overlap_or_close(node1, node2):
            return 'direct'
        
        # Wire-based connection (wires connect to other components)
        if node1.type == 'Wire' or node2.type == 'Wire':
            if distance <= self.connection_threshold * 1.5:  # Wires can bridge longer distances
                return 'wire'
        
        # Close proximity connection
        if distance <= self.connection_threshold:
            return 'close'
        
        return 'none'
    
    def analyze_connections(self):
        """Analyze all components and determine connections (only within same side)"""
        self.connections.clear()
        
        # Compare every pair of components
        node_list = list(self.nodes.values())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node1 = node_list[i]
                node2 = node_list[j]
                
                # Only check connections if both components are on the same side
                if node1.board_side != node2.board_side:
                    continue
                
                distance = self.calculate_distance(node1, node2)
                connection_type = self.determine_connection_type(node1, node2, distance)
                
                if connection_type != 'none':
                    self.connections.append(Connection(
                        component1_id=node1.id,
                        component2_id=node2.id,
                        distance=distance,
                        connection_type=connection_type
                    ))
    
    def build_graph_data(self):
        """Build the graph data structure"""
        # Create adjacency list representation
        adjacency_list = {}
        
        # Initialize adjacency list
        for node_id in self.nodes.keys():
            adjacency_list[node_id] = []
        
        # Add connections
        for connection in self.connections:
            adjacency_list[connection.component1_id].append({
                'connected_to': connection.component2_id,
                'distance': connection.distance,
                'connection_type': connection.connection_type
            })
            adjacency_list[connection.component2_id].append({
                'connected_to': connection.component1_id,
                'distance': connection.distance,
                'connection_type': connection.connection_type
            })
        
        # Build nodes with LED orientation info
        nodes_dict = {}
        for node_id, node in self.nodes.items():
            node_data = {
                'type': node.type,
                'confidence': node.confidence,
                'bbox': node.bbox,
                'center': node.center,
                'board_side': node.board_side
            }
            
            # Add LED orientation if available
            if node_id in self.led_orientations:
                node_data['led_orientation'] = self.led_orientations[node_id]
            
            # Add Horn orientation if available
            if node_id in self.horn_orientations:
                node_data['horn_orientation'] = self.horn_orientations[node_id]
            
            nodes_dict[node_id] = node_data
        
        # Add terminal-based analysis data if available
        terminal_data = None
        if self.terminal_analyzer and TERMINAL_ANALYZER_AVAILABLE:
            # Build terminal connections data
            terminal_connections = []
            for conn in self.terminal_analyzer.connections:
                terminal_connections.append({
                    'from_component': conn.from_component_id,
                    'from_terminal': conn.from_terminal,
                    'to_component': conn.to_component_id,
                    'to_terminal': conn.to_terminal,
                    'overlap_percentage': conn.overlap_percentage,
                    'distance': conn.distance
                })
            
            terminal_data = {
                'connections': terminal_connections,
                'summary': self.terminal_analyzer.get_summary()
            }
        
        self.graph_data = {
            'timestamp': time.time(),
            'nodes': nodes_dict,
            'adjacency_list': adjacency_list,
            'connections': [
                {
                    'component1': conn.component1_id,
                    'component2': conn.component2_id,
                    'distance': conn.distance,
                    'type': conn.connection_type
                }
                for conn in self.connections
            ],
            'led_orientations': self.led_orientations,  # Add summary of all LED orientations
            'horn_orientations': self.horn_orientations,  # Add summary of all Horn orientations
            'terminal_analysis': terminal_data  # Add terminal-based analysis results
        }
    
    def get_connection_summary(self) -> str:
        """Generate a human-readable connection summary"""
        summary_lines = []
        
        # === PRIMARY: TERMINAL-BASED CIRCUIT ANALYSIS ===
        if self.terminal_analyzer and TERMINAL_ANALYZER_AVAILABLE:
            summary_lines.append("="*60)
            summary_lines.append("TERMINAL-BASED CIRCUIT ANALYSIS")
            summary_lines.append("="*60)
            summary_lines.append(f"Total components: {len(self.nodes)}")
            summary_lines.append("")
            
            # Group components by board side
            left_components = [node for node in self.nodes.values() if node.board_side == 'left']
            right_components = [node for node in self.nodes.values() if node.board_side == 'right']
            
            summary_lines.append(f"LEFT BOARD: {len(left_components)} components")
            summary_lines.append(f"RIGHT BOARD: {len(right_components)} components")
            summary_lines.append("")
            
            # Group components by type
            components_by_type = {}
            for node in self.nodes.values():
                if node.type not in components_by_type:
                    components_by_type[node.type] = []
                components_by_type[node.type].append(node)
            
            summary_lines.append("Components by Type:")
            for comp_type, nodes in sorted(components_by_type.items()):
                summary_lines.append(f"  {comp_type}: {len(nodes)}")
            summary_lines.append("")
            
            # All components list with board side
            summary_lines.append("All Components:")
            
            # Show left board components first
            summary_lines.append("  LEFT BOARD:")
            for node_id, node in sorted(self.nodes.items()):
                if node.board_side == 'left':
                    comp_line = f"    {node_id}: {node.type} [conf: {node.confidence:.2f}]"
                    
                    # Add LED orientation if available
                    if node_id in self.led_orientations:
                        orientation_info = self.led_orientations[node_id]
                        comp_line += f" - LED: {orientation_info['orientation']}"
                        if orientation_info['orientation'] != 'UNKNOWN':
                            comp_line += f" ({orientation_info['led_orientation']}, +{orientation_info['plus_position']})"
                    
                    # Add Horn orientation if available
                    if node_id in self.horn_orientations:
                        orientation_info = self.horn_orientations[node_id]
                        comp_line += f" - Horn: {orientation_info['orientation']}"
                        if orientation_info['orientation'] != 'UNKNOWN':
                            comp_line += f" ({orientation_info['horn_orientation']}, +{orientation_info['plus_position']})"
                    
                    summary_lines.append(comp_line)
            
            # Show right board components
            summary_lines.append("  RIGHT BOARD:")
            for node_id, node in sorted(self.nodes.items()):
                if node.board_side == 'right':
                    comp_line = f"    {node_id}: {node.type} [conf: {node.confidence:.2f}]"
                    
                    # Add LED orientation if available
                    if node_id in self.led_orientations:
                        orientation_info = self.led_orientations[node_id]
                        comp_line += f" - LED: {orientation_info['orientation']}"
                        if orientation_info['orientation'] != 'UNKNOWN':
                            comp_line += f" ({orientation_info['led_orientation']}, +{orientation_info['plus_position']})"
                    
                    # Add Horn orientation if available
                    if node_id in self.horn_orientations:
                        orientation_info = self.horn_orientations[node_id]
                        comp_line += f" - Horn: {orientation_info['orientation']}"
                        if orientation_info['orientation'] != 'UNKNOWN':
                            comp_line += f" ({orientation_info['horn_orientation']}, +{orientation_info['plus_position']})"
                    
                    summary_lines.append(comp_line)
            summary_lines.append("")
            
            # LED Orientation Summary (if any LEDs detected)
            if self.led_orientations:
                summary_lines.append("LED Orientation Details:")
                for led_id, orientation_info in self.led_orientations.items():
                    summary_lines.append(f"  {led_id}:")
                    summary_lines.append(f"    Status: {orientation_info['orientation']}")
                    summary_lines.append(f"    Confidence: {orientation_info['confidence']:.2f}")
                    if 'led_orientation' in orientation_info:
                        summary_lines.append(f"    LED Position: {orientation_info['led_orientation']}")
                    if 'plus_position' in orientation_info and orientation_info['plus_position']:
                        summary_lines.append(f"    '+' Position: {orientation_info['plus_position']}")
                    if 'bbox_width' in orientation_info and 'bbox_height' in orientation_info:
                        summary_lines.append(f"    Bounding Box: {orientation_info['bbox_width']:.0f}x{orientation_info['bbox_height']:.0f} pixels")
                    if 'reason' in orientation_info:
                        summary_lines.append(f"    Details: {orientation_info['reason']}")
                summary_lines.append("")
            
            # Horn Orientation Summary (if any Horns detected)
            # Note: This includes Photoresistors since they're actually Horns
            if self.horn_orientations:
                summary_lines.append("Horn Orientation Details (includes Photoresistor/Lamp if detected):")
                for horn_id, orientation_info in self.horn_orientations.items():
                    summary_lines.append(f"  {horn_id}:")
                    summary_lines.append(f"    Status: {orientation_info['orientation']}")
                    summary_lines.append(f"    Confidence: {orientation_info['confidence']:.2f}")
                    if 'horn_orientation' in orientation_info:
                        summary_lines.append(f"    Horn Position: {orientation_info['horn_orientation']}")
                    if 'plus_position' in orientation_info and orientation_info['plus_position']:
                        summary_lines.append(f"    '+' Position: {orientation_info['plus_position']}")
                    if 'bbox_width' in orientation_info and 'bbox_height' in orientation_info:
                        summary_lines.append(f"    Bounding Box: {orientation_info['bbox_width']:.0f}x{orientation_info['bbox_height']:.0f} pixels")
                    if 'reason' in orientation_info:
                        summary_lines.append(f"    Details: {orientation_info['reason']}")
                summary_lines.append("")
            
            # Terminal connections summary
            summary_lines.append(self.terminal_analyzer.get_summary())
        
        # === COMMENTED OUT: Traditional connectivity analysis ===
        # summary_lines.append("")
        # summary_lines.append("="*60)
        # summary_lines.append("TRADITIONAL CONNECTIVITY ANALYSIS (for reference)")
        # summary_lines.append("="*60)
        # summary_lines.append(f"Total connections: {len(self.connections)}")
        # summary_lines.append("")
        # 
        # # Connected pairs
        # summary_lines.append("Connected Components:")
        # if self.connections:
        #     for connection in self.connections:
        #         node1 = self.nodes[connection.component1_id]
        #         node2 = self.nodes[connection.component2_id]
        #         summary_lines.append(
        #             f"  {node1.type} ({node1.board_side}) --[{connection.connection_type}]--> "
        #             f"{node2.type} ({node2.board_side}) [distance: {connection.distance:.1f}px]"
        #         )
        # else:
        #     summary_lines.append("  No connections detected")
        
        return "\n".join(summary_lines)
    
    def save_to_json(self, filepath: str):
        """Save graph data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.graph_data, f, indent=2, default=str)
        print(f"✅ Circuit graph saved to: {filepath}")
    
    def save_summary_to_text(self, filepath: str):
        """Save human-readable summary to text file"""
        with open(filepath, 'w') as f:
            f.write(self.get_connection_summary())
        print(f"✅ Circuit summary saved to: {filepath}")
    
    def analyze_circuit(self, left_boxes, right_boxes, model_names, frame_width, frame_height, frame=None):
        """
        Complete circuit analysis workflow
        
        Args:
            frame: Optional - original frame for LED and Horn orientation detection
        """
        print("🔍 Analyzing circuit connectivity...")
        
        # Store frame for orientation detection
        self.frame = frame
        
        # STEP 1: Detect and reclassify horns based on red plus sign (post-processing)
        # This must happen FIRST so reclassifications are available during node creation
        if frame is not None and self.horn_detector is not None:
            self.detect_and_reclassify_horns(left_boxes, right_boxes, model_names, frame)
        
        # STEP 2: Detect LED and Horn orientations (before adding detections)
        # This way orientations are available when adding components to terminal analyzer
        if frame is not None and (self.led_detector is not None or self.horn_detector is not None):
            print("   Detecting component orientations (LED and Horn)...")
            self.detect_led_orientations_from_boxes(left_boxes, right_boxes, model_names, frame)
        
        # Add detections (will also add to terminal analyzer with LED/Horn orientations)
        self.add_detections(left_boxes, right_boxes, model_names, frame_width, frame_height)
        print(f"   Added {len(self.nodes)} components")
        
        # Run terminal-based analysis if available
        if self.terminal_analyzer and TERMINAL_ANALYZER_AVAILABLE:
            print("   Running terminal-based circuit analysis...")
            self.terminal_analyzer.analyze_terminal_connections()
            print(f"   Found {len(self.terminal_analyzer.connections)} terminal connections")
        
        # Also run traditional connection analysis
        self.analyze_connections()
        print(f"   Found {len(self.connections)} traditional connections")
        
        # Build graph data
        self.build_graph_data()
        print("   Built graph data structure")
        
        return self.graph_data
    
    def detect_and_reclassify_horns(self, left_boxes, right_boxes, model_names, frame):
        """
        Post-process detection: Look for red plus signs to identify and reclassify horns.
        If Lamp, Photoresistor, or Horn is detected, check for red plus. If found, it's a Horn.
        Stores reclassifications for use during node creation.
        """
        if frame is None or self.horn_detector is None:
            return
        
        print("   Post-processing: Checking for red plus signs to identify Horns...")
        
        # Clear previous reclassifications
        self.horn_reclassifications.clear()
        
        # Components that might be horns (if they have a red plus)
        horn_candidate_types = ['Lamp', 'Photoresistor', 'Horn']
        
        reclassified_count = 0
        component_counter = 0
        
        # Process left boxes
        for i, box in left_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            raw_class_name = model_names[class_id]
            
            if raw_class_name == "Green tape":
                component_counter += 1
                continue
            
            # Check if this is a potential horn candidate (based on RAW model output)
            if raw_class_name in horn_candidate_types:
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                # Try to detect red plus sign
                result = self.horn_detector.detect_orientation(frame, bbox, debug=True)  # Enable debug
                
                # If we found a red plus with good confidence, it's a Horn!
                if result['confidence'] >= 0.5:
                    # Store reclassification using component_counter as key
                    # This key will be used in add_detections to match the same component
                    reclass_key = f"L_{component_counter}"
                    self.horn_reclassifications[reclass_key] = {
                        'original_type': raw_class_name,
                        'new_type': 'Horn',
                        'confidence': result['confidence'],
                        'orientation': result
                    }
                    if raw_class_name != 'Horn':
                        print(f"   ✓ Reclassified {raw_class_name} → Horn (red plus detected, conf: {result['confidence']:.2f})")
                        reclassified_count += 1
            
            component_counter += 1
        
        # Process right boxes  
        for i, box in right_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            raw_class_name = model_names[class_id]
            
            if raw_class_name == "Green tape":
                component_counter += 1
                continue
            
            # Check if this is a potential horn candidate (based on RAW model output)
            if raw_class_name in horn_candidate_types:
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                # Try to detect red plus sign
                result = self.horn_detector.detect_orientation(frame, bbox, debug=True)  # Enable debug
                
                # If we found a red plus with good confidence, it's a Horn!
                if result['confidence'] >= 0.5:
                    # Store reclassification using component_counter as key
                    # This key will be used in add_detections to match the same component
                    reclass_key = f"R_{component_counter}"
                    self.horn_reclassifications[reclass_key] = {
                        'original_type': raw_class_name,
                        'new_type': 'Horn',
                        'confidence': result['confidence'],
                        'orientation': result
                    }
                    if raw_class_name != 'Horn':
                        print(f"   ✓ Reclassified {raw_class_name} → Horn (red plus detected, conf: {result['confidence']:.2f})")
                        reclassified_count += 1
            
            component_counter += 1
        
        if reclassified_count > 0:
            print(f"   Reclassified {reclassified_count} component(s) as Horn based on red plus detection")
    
    def detect_led_orientations_from_boxes(self, left_boxes, right_boxes, model_names, frame):
        """Detect orientation of all LED_2 (Red) and Horn components from raw boxes (before creating nodes)"""
        if frame is None:
            return
        
        self.led_orientations.clear()
        self.horn_orientations.clear()
        led_count = 0
        horn_count = 0
        horn_reclassified_count = 0
        component_counter = 0
        
        # Process left boxes
        for i, box in left_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            raw_class_name = model_names[class_id]
            
            if raw_class_name == "Green tape":
                continue
            
            class_name = map_component_name(raw_class_name)
            node_id = f"L_{component_counter}_{class_name}"
            component_counter += 1
            
            # Detect LED orientation
            if 'LED_2 (Red)' in class_name and self.led_detector is not None:
                led_count += 1
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                result = self.led_detector.detect_orientation(frame, bbox, debug=False)
                self.led_orientations[node_id] = result
                
                if result['orientation'] != 'UNKNOWN':
                    print(f"   LED {node_id}: {result['orientation']} (confidence: {result['confidence']:.2f})")
            
            # Detect Horn orientation (skip if already processed during reclassification)
            # Note: Reclassified horns already have their orientation in self.horn_orientations
            # Also check for Photoresistor and Lamp since they could be Horns
            if ('Horn' in class_name or 'Photoresistor' in class_name or 'Lamp' in class_name) and self.horn_detector is not None and node_id not in self.horn_orientations:
                horn_count += 1
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                result = self.horn_detector.detect_orientation(frame, bbox, debug=True)  # Enable debug
                self.horn_orientations[node_id] = result
                
                if result['orientation'] != 'UNKNOWN':
                    # Show as "Horn" if it's Photoresistor or Lamp (with high confidence red plus)
                    if 'Photoresistor' in class_name and result['confidence'] >= 0.5:
                        display_name = 'Horn (detected as Photoresistor)'
                    elif 'Lamp' in class_name and result['confidence'] >= 0.5:
                        display_name = 'Horn (detected as Lamp)'
                    else:
                        display_name = class_name
                    print(f"   {display_name} {node_id}: {result['orientation']} (confidence: {result['confidence']:.2f})")
        
        # Process right boxes
        for i, box in right_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            raw_class_name = model_names[class_id]
            
            if raw_class_name == "Green tape":
                continue
            
            class_name = map_component_name(raw_class_name)
            node_id = f"R_{component_counter}_{class_name}"
            component_counter += 1
            
            # Detect LED orientation
            if 'LED_2 (Red)' in class_name and self.led_detector is not None:
                led_count += 1
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                result = self.led_detector.detect_orientation(frame, bbox, debug=False)
                self.led_orientations[node_id] = result
                
                if result['orientation'] != 'UNKNOWN':
                    print(f"   LED {node_id}: {result['orientation']} (confidence: {result['confidence']:.2f})")
            
            # Detect Horn orientation (skip if already processed during reclassification)
            # Note: Reclassified horns already have their orientation in self.horn_orientations
            # Also check for Photoresistor and Lamp since they could be Horns
            if ('Horn' in class_name or 'Photoresistor' in class_name or 'Lamp' in class_name) and self.horn_detector is not None and node_id not in self.horn_orientations:
                horn_count += 1
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                result = self.horn_detector.detect_orientation(frame, bbox, debug=True)  # Enable debug
                self.horn_orientations[node_id] = result
                
                if result['orientation'] != 'UNKNOWN':
                    # Show as "Horn" if it's Photoresistor or Lamp (with high confidence red plus)
                    if 'Photoresistor' in class_name and result['confidence'] >= 0.5:
                        display_name = 'Horn (detected as Photoresistor)'
                    elif 'Lamp' in class_name and result['confidence'] >= 0.5:
                        display_name = 'Horn (detected as Lamp)'
                    else:
                        display_name = class_name
                    print(f"   {display_name} {node_id}: {result['orientation']} (confidence: {result['confidence']:.2f})")
        
        if led_count > 0:
            print(f"   Detected orientation for {led_count} LED component(s)")
        if horn_count > 0:
            print(f"   Detected orientation for {horn_count} Horn component(s)")
    
    def detect_led_orientations(self):
        """Detect orientation of all LED_2 (Red) and Horn components (from existing nodes)"""
        if self.frame is None:
            return
        
        led_count = 0
        horn_count = 0
        
        for node_id, node in self.nodes.items():
            # Detect LED orientation
            if 'LED_2 (Red)' in node.type and self.led_detector is not None:
                led_count += 1
                result = self.led_detector.detect_orientation(self.frame, node.bbox, debug=False)
                self.led_orientations[node_id] = result
                
                if result['orientation'] != 'UNKNOWN':
                    print(f"   LED {node_id}: {result['orientation']} (confidence: {result['confidence']:.2f})")
            
            # Detect Horn orientation
            if 'Horn' in node.type and self.horn_detector is not None:
                horn_count += 1
                result = self.horn_detector.detect_orientation(self.frame, node.bbox, debug=False)
                self.horn_orientations[node_id] = result
                
                if result['orientation'] != 'UNKNOWN':
                    print(f"   Horn {node_id}: {result['orientation']} (confidence: {result['confidence']:.2f})")
        
        if led_count > 0:
            print(f"   Detected orientation for {led_count} LED component(s)")
        if horn_count > 0:
            print(f"   Detected orientation for {horn_count} Horn component(s)")

if __name__ == "__main__":
    # Demo usage
    analyzer = CircuitGraphAnalyzer()
    print("Circuit Graph Analyzer initialized")
    print("Use analyze_circuit() method with detection results")
