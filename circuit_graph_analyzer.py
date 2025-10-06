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
        
    def add_detections(self, left_boxes, right_boxes, model_names, frame_width, frame_height):
        """Add detected components to the graph"""
        self.nodes.clear()
        self.connections.clear()
        
        # Process left side detections
        component_counter = 0
        for i, box in left_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = model_names[class_id]
            confidence = float(box.conf[0])
            
            # Skip green tape for graph analysis
            if class_name == "Green tape":
                continue
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            node_id = f"L_{component_counter}_{class_name}"
            component_counter += 1
            
            self.nodes[node_id] = ComponentNode(
                id=node_id,
                type=class_name,
                confidence=confidence,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                center=(float(center_x), float(center_y)),
                grid_row=0,  # Will be calculated later if needed
                grid_col=0,
                board_side='left'
            )
        
        # Process right side detections
        for i, box in right_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = model_names[class_id]
            confidence = float(box.conf[0])
            
            # Skip green tape for graph analysis
            if class_name == "Green tape":
                continue
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            node_id = f"R_{component_counter}_{class_name}"
            component_counter += 1
            
            self.nodes[node_id] = ComponentNode(
                id=node_id,
                type=class_name,
                confidence=confidence,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                center=(float(center_x), float(center_y)),
                grid_row=0,
                grid_col=0,
                board_side='right'
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
        """Analyze all components and determine connections"""
        self.connections.clear()
        
        # Compare every pair of components
        node_list = list(self.nodes.values())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node1 = node_list[i]
                node2 = node_list[j]
                
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
        
        self.graph_data = {
            'timestamp': time.time(),
            'nodes': {
                node_id: {
                    'type': node.type,
                    'confidence': node.confidence,
                    'bbox': node.bbox,
                    'center': node.center,
                    'board_side': node.board_side
                }
                for node_id, node in self.nodes.items()
            },
            'adjacency_list': adjacency_list,
            'connections': [
                {
                    'component1': conn.component1_id,
                    'component2': conn.component2_id,
                    'distance': conn.distance,
                    'type': conn.connection_type
                }
                for conn in self.connections
            ]
        }
    
    def get_connection_summary(self) -> str:
        """Generate a human-readable connection summary"""
        summary_lines = []
        summary_lines.append("=== CIRCUIT CONNECTIVITY ANALYSIS ===")
        summary_lines.append(f"Total components: {len(self.nodes)}")
        summary_lines.append(f"Total connections: {len(self.connections)}")
        summary_lines.append("")
        
        # Group components by type
        components_by_type = {}
        for node in self.nodes.values():
            if node.type not in components_by_type:
                components_by_type[node.type] = []
            components_by_type[node.type].append(node)
        
        summary_lines.append("Components by type:")
        for comp_type, nodes in components_by_type.items():
            summary_lines.append(f"  {comp_type}: {len(nodes)}")
        summary_lines.append("")
        
        # All components list
        summary_lines.append("All Components:")
        for node_id, node in self.nodes.items():
            summary_lines.append(f"  {node_id}: {node.type} ({node.board_side}) [conf: {node.confidence:.2f}]")
        summary_lines.append("")
        
        # Connected pairs
        summary_lines.append("Connected Components:")
        if self.connections:
            for connection in self.connections:
                node1 = self.nodes[connection.component1_id]
                node2 = self.nodes[connection.component2_id]
                summary_lines.append(
                    f"  {node1.type} ({node1.board_side}) --[{connection.connection_type}]--> "
                    f"{node2.type} ({node2.board_side}) [distance: {connection.distance:.1f}px]"
                )
        else:
            summary_lines.append("  No connections detected")
        summary_lines.append("")
        
        # Disconnected pairs (components that are NOT connected)
        summary_lines.append("Disconnected Components:")
        connected_pairs = set()
        for connection in self.connections:
            # Add both directions since connections are bidirectional
            connected_pairs.add((connection.component1_id, connection.component2_id))
            connected_pairs.add((connection.component2_id, connection.component1_id))
        
        disconnected_count = 0
        node_list = list(self.nodes.values())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node1 = node_list[i]
                node2 = node_list[j]
                
                # Check if this pair is NOT connected
                if (node1.id, node2.id) not in connected_pairs:
                    distance = self.calculate_distance(node1, node2)
                    summary_lines.append(
                        f"  {node1.type} ({node1.board_side}) --[NO CONNECTION]--> "
                        f"{node2.type} ({node2.board_side}) [distance: {distance:.1f}px]"
                    )
                    disconnected_count += 1
        
        if disconnected_count == 0:
            summary_lines.append("  All components are connected to each other!")
        summary_lines.append("")
        
        # Connection statistics
        total_possible_connections = len(self.nodes) * (len(self.nodes) - 1) // 2
        connection_percentage = (len(self.connections) / total_possible_connections * 100) if total_possible_connections > 0 else 0
        
        summary_lines.append("Connection Statistics:")
        summary_lines.append(f"  Total possible connections: {total_possible_connections}")
        summary_lines.append(f"  Actual connections: {len(self.connections)}")
        summary_lines.append(f"  Disconnected pairs: {disconnected_count}")
        summary_lines.append(f"  Connection density: {connection_percentage:.1f}%")
        
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
    
    def analyze_circuit(self, left_boxes, right_boxes, model_names, frame_width, frame_height):
        """Complete circuit analysis workflow"""
        print("🔍 Analyzing circuit connectivity...")
        
        # Add detections
        self.add_detections(left_boxes, right_boxes, model_names, frame_width, frame_height)
        print(f"   Added {len(self.nodes)} components")
        
        # Analyze connections
        self.analyze_connections()
        print(f"   Found {len(self.connections)} connections")
        
        # Build graph data
        self.build_graph_data()
        print("   Built graph data structure")
        
        return self.graph_data

if __name__ == "__main__":
    # Demo usage
    analyzer = CircuitGraphAnalyzer()
    print("Circuit Graph Analyzer initialized")
    print("Use analyze_circuit() method with detection results")
