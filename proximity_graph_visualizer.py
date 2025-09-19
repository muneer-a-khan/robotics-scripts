#!/usr/bin/env python3
"""
Proximity-Based Graph Visualizer

Creates visual connectivity graphs for dual board circuit detection.
Components are nodes, connections are edges based on proximity distance.
Generates separate graphs for left and right sides.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import math
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ComponentNode:
    """Represents a component node in the graph"""
    id: str
    name: str
    component_type: str
    center: Tuple[float, float]
    confidence: float
    side: str  # 'left' or 'right'


class ProximityGraphVisualizer:
    """
    Creates proximity-based connectivity graphs for circuit boards.
    
    Components within a certain distance threshold are considered connected.
    """
    
    def __init__(self, 
                 connection_threshold: float = 150.0,
                 output_dir: str = "graph_output",
                 image_size: Tuple[int, int] = (12, 8)):
        """
        Initialize the graph visualizer
        
        Args:
            connection_threshold: Maximum distance for components to be connected (pixels)
            output_dir: Directory to save graph images
            image_size: Size of the output graph images (width, height in inches)
        """
        self.connection_threshold = connection_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.image_size = image_size
        
        # Component type colors for visualization
        self.component_colors = {
            'battery_holder': '#FF4444',    # Red
            'wire': '#4444FF',              # Blue  
            'switch': '#44FF44',            # Green
            'button': '#FF8844',            # Orange
            'led': '#FFFF44',               # Yellow
            'speaker': '#FF44FF',           # Magenta
            'music_circuit': '#44FFFF',     # Cyan
            'motor': '#8844FF',             # Purple
            'resistor': '#CCCCCC',          # Gray
            'connection_node': '#888888',   # Dark gray
            'lamp': '#FFCC44',              # Light orange
            'fan': '#88CCFF',               # Light blue
            'buzzer': '#CC88FF',            # Light purple
            'photoresistor': '#FFCC88',     # Light orange-gray
            'microphone': '#88FFCC',        # Light teal
            'alarm': '#CC4444',             # Dark red
        }
        
        print(f"🔗 Proximity Graph Visualizer initialized")
        print(f"   Connection threshold: {connection_threshold} pixels")
        print(f"   Output directory: {output_dir}")
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def create_component_nodes(self, detections: List[Dict]) -> Tuple[List[ComponentNode], List[ComponentNode]]:
        """
        Convert detections to component nodes, separated by side
        
        Args:
            detections: List of detection dictionaries from YOLO
            
        Returns:
            Tuple of (left_nodes, right_nodes)
        """
        left_nodes = []
        right_nodes = []
        
        for i, detection in enumerate(detections):
            node = ComponentNode(
                id=f"{detection['class_name']}_{i}",
                name=f"{detection['class_name'].title()} {i+1}",
                component_type=detection['class_name'],
                center=detection['center'],
                confidence=detection['confidence'],
                side=detection['side']
            )
            
            if detection['side'] == 'left':
                left_nodes.append(node)
            else:
                right_nodes.append(node)
        
        return left_nodes, right_nodes
    
    def create_connectivity_graph(self, nodes: List[ComponentNode]) -> nx.Graph:
        """
        Create a NetworkX graph with proximity-based connections
        
        Args:
            nodes: List of component nodes
            
        Returns:
            NetworkX graph with nodes and edges
        """
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node.id, 
                      name=node.name,
                      component_type=node.component_type,
                      center=node.center,
                      confidence=node.confidence,
                      color=self.component_colors.get(node.component_type, '#CCCCCC'))
        
        # Add edges based on proximity
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:  # Avoid duplicate pairs
                distance = self.calculate_distance(node1.center, node2.center)
                
                if distance <= self.connection_threshold:
                    G.add_edge(node1.id, node2.id, 
                              distance=distance,
                              weight=1.0 - (distance / self.connection_threshold))  # Closer = stronger
        
        return G
    
    def create_spatial_positions(self, graph: nx.Graph, side: str, image_width: int = 1920) -> Dict:
        """
        Create node positions based on actual component locations
        
        Args:
            graph: NetworkX graph
            side: 'left' or 'right'
            image_width: Width of the source image for scaling
            
        Returns:
            Dictionary of node positions
        """
        pos = {}
        
        if len(graph.nodes()) == 0:
            return pos
        
        # Get all component centers
        centers = []
        for node in graph.nodes():
            center = graph.nodes[node]['center']
            centers.append(center)
        
        # Convert to numpy array for easier processing
        centers = np.array(centers)
        
        # Normalize positions to fit nicely in the plot
        if side == 'left':
            # For left side, use x coordinates as-is but normalize
            x_coords = centers[:, 0]
            x_min, x_max = x_coords.min(), min(x_coords.max(), image_width // 2)
        else:
            # For right side, shift x coordinates and normalize
            x_coords = centers[:, 0] - (image_width // 2)
            x_min, x_max = x_coords.min(), x_coords.max()
        
        y_coords = centers[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Normalize coordinates to fit in a reasonable range (0-10 for better visualization)
        if x_max - x_min > 0:
            x_normalized = 10 * (x_coords - x_min) / (x_max - x_min)
        else:
            x_normalized = np.full_like(x_coords, 5)  # Center if all same x
            
        if y_max - y_min > 0:
            # Flip y coordinates (in images, y=0 is top, in plots y=0 is bottom)
            y_normalized = 10 * (1 - (y_coords - y_min) / (y_max - y_min))
        else:
            y_normalized = np.full_like(y_coords, 5)  # Center if all same y
        
        # Create position dictionary
        for i, node in enumerate(graph.nodes()):
            pos[node] = (x_normalized[i], y_normalized[i])
        
        return pos
    
    def generate_dual_graphs(self, detections: List[Dict], 
                           save_images: bool = True,
                           show_images: bool = False) -> plt.Figure:
        """
        Generate side-by-side connectivity graphs for both left and right sides
        
        Args:
            detections: List of detection dictionaries
            save_images: Whether to save the graph images
            show_images: Whether to display the images
            
        Returns:
            Combined matplotlib figure with both graphs
        """
        print(f"\n🔗 Generating dual connectivity graphs...")
        print(f"   Total detections: {len(detections)}")
        
        # Create component nodes
        left_nodes, right_nodes = self.create_component_nodes(detections)
        
        print(f"   Left side components: {len(left_nodes)}")
        print(f"   Right side components: {len(right_nodes)}")
        
        # Create graphs
        left_graph = self.create_connectivity_graph(left_nodes)
        right_graph = self.create_connectivity_graph(right_nodes)
        
        print(f"   Left connections: {len(left_graph.edges())}")
        print(f"   Right connections: {len(right_graph.edges())}")
        
        # Create side-by-side figure
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Generate spatial positions (preserving actual component locations)
        left_pos = self.create_spatial_positions(left_graph, 'left')
        right_pos = self.create_spatial_positions(right_graph, 'right')
        
        # Draw left side graph
        self.draw_single_graph(left_graph, left_pos, ax_left, 
                              f"Left Board ({len(left_nodes)} components)", "left")
        
        # Draw right side graph  
        self.draw_single_graph(right_graph, right_pos, ax_right,
                              f"Right Board ({len(right_nodes)} components)", "right")
        
        # Overall title
        total_connections = len(left_graph.edges()) + len(right_graph.edges())
        fig.suptitle(f'Dual Board Circuit Connectivity - {len(detections)} Components, {total_connections} Connections', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Add threshold info
        fig.text(0.5, 0.02, f'Connection Threshold: {self.connection_threshold}px | Positions based on actual component locations', 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save image if requested
        if save_images:
            timestamp = int(time.time())
            combined_path = self.output_dir / f"dual_board_graph_{timestamp}.png"
            fig.savefig(combined_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved combined graph: {combined_path}")
        
        # Show image if requested
        if show_images:
            plt.show()
        
        return fig
    
    def draw_single_graph(self, graph: nx.Graph, pos: Dict, ax: plt.Axes, title: str, side: str):
        """
        Draw a single graph on the given axes
        
        Args:
            graph: NetworkX graph to draw
            pos: Node positions dictionary
            ax: Matplotlib axes
            title: Graph title
            side: 'left' or 'right'
        """
        if len(graph.nodes()) == 0:
            ax.text(0.5, 0.5, f'No {side} components detected', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=14, color='gray')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
            return
        
        # Get node attributes
        node_colors = [graph.nodes[node]['color'] for node in graph.nodes()]
        node_labels = {node: graph.nodes[node]['name'] for node in graph.nodes()}
        
        # Draw edges first (behind nodes)
        nx.draw_networkx_edges(graph, pos, ax=ax,
                              edge_color='gray',
                              width=2,
                              alpha=0.6)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, ax=ax,
                              node_color=node_colors,
                              node_size=800,
                              alpha=0.9,
                              edgecolors='black',
                              linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax,
                               font_size=7,
                               font_weight='bold',
                               font_color='white')
        
        # Add connection info
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())
        
        # Title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Info box
        info_text = f"Nodes: {num_nodes}\nConnections: {num_edges}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Status
        if len(list(nx.isolates(graph))) == 0 and num_nodes > 0:
            status_text = "✅ All connected"
            status_color = 'lightgreen'
        elif num_edges > 0:
            isolated = len(list(nx.isolates(graph)))
            status_text = f"⚠️ {isolated} isolated"
            status_color = 'lightyellow'
        else:
            status_text = "❌ No connections"
            status_color = 'lightcoral'
        
        ax.text(0.02, 0.02, status_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))
        
        # Component legend (smaller for side-by-side layout)
        if num_nodes > 0:
            component_types = set(graph.nodes[node]['component_type'] for node in graph.nodes())
            legend_elements = []
            
            for comp_type in sorted(component_types):
                color = self.component_colors.get(comp_type, '#CCCCCC')
                legend_elements.append(plt.scatter([], [], c=color, s=60, label=comp_type.title()))
            
            if legend_elements and len(legend_elements) <= 6:  # Only show legend if not too many types
                ax.legend(handles=legend_elements, loc='upper right', 
                         bbox_to_anchor=(0.98, 0.85), frameon=True, fancybox=True, fontsize=8)
        
        # Set equal aspect ratio and clean axes
        ax.set_aspect('equal')
        ax.axis('off')
    
    def analyze_connectivity(self, detections: List[Dict]) -> Dict:
        """
        Analyze connectivity patterns and return statistics
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary with connectivity analysis
        """
        left_nodes, right_nodes = self.create_component_nodes(detections)
        left_graph = self.create_connectivity_graph(left_nodes)
        right_graph = self.create_connectivity_graph(right_nodes)
        
        analysis = {
            'left_side': {
                'components': len(left_nodes),
                'connections': len(left_graph.edges()),
                'component_types': {},
                'isolated_nodes': list(nx.isolates(left_graph))
            },
            'right_side': {
                'components': len(right_nodes),
                'connections': len(right_graph.edges()),
                'component_types': {},
                'isolated_nodes': list(nx.isolates(right_graph))
            },
            'connectivity_threshold': self.connection_threshold
        }
        
        # Count component types per side
        for node in left_nodes:
            comp_type = node.component_type
            analysis['left_side']['component_types'][comp_type] = \
                analysis['left_side']['component_types'].get(comp_type, 0) + 1
        
        for node in right_nodes:
            comp_type = node.component_type
            analysis['right_side']['component_types'][comp_type] = \
                analysis['right_side']['component_types'].get(comp_type, 0) + 1
        
        return analysis
    
    def print_connectivity_report(self, detections: List[Dict]):
        """Print a detailed connectivity analysis report"""
        analysis = self.analyze_connectivity(detections)
        
        print(f"\n📊 CONNECTIVITY ANALYSIS REPORT")
        print("=" * 50)
        
        for side in ['left_side', 'right_side']:
            side_name = side.replace('_', ' ').title()
            data = analysis[side]
            
            print(f"\n🔌 {side_name}:")
            print(f"   Components: {data['components']}")
            print(f"   Connections: {data['connections']}")
            
            if data['component_types']:
                print(f"   Component breakdown:")
                for comp_type, count in sorted(data['component_types'].items()):
                    print(f"      • {comp_type}: {count}")
            
            if data['isolated_nodes']:
                print(f"   ⚠️  Isolated components: {len(data['isolated_nodes'])}")
            else:
                print(f"   ✅ All components connected")
        
        print(f"\n🔧 Settings:")
        print(f"   Connection threshold: {analysis['connectivity_threshold']} pixels")


def create_sample_detections():
    """Create sample detection data for testing"""
    return [
        {'class_name': 'battery_holder', 'confidence': 0.95, 'center': (200, 300), 'side': 'left'},
        {'class_name': 'wire', 'confidence': 0.85, 'center': (250, 280), 'side': 'left'},
        {'class_name': 'wire', 'confidence': 0.90, 'center': (180, 350), 'side': 'left'},
        {'class_name': 'wire', 'confidence': 0.88, 'center': (220, 380), 'side': 'left'},
        
        {'class_name': 'battery_holder', 'confidence': 0.92, 'center': (800, 300), 'side': 'right'},
        {'class_name': 'wire', 'confidence': 0.87, 'center': (850, 280), 'side': 'right'},
        {'class_name': 'wire', 'confidence': 0.91, 'center': (780, 350), 'side': 'right'},
        {'class_name': 'wire', 'confidence': 0.89, 'center': (820, 380), 'side': 'right'},
    ]


if __name__ == "__main__":
    # Test the visualizer with sample data
    visualizer = ProximityGraphVisualizer(connection_threshold=100.0)
    
    print("🧪 Testing with sample detection data...")
    sample_detections = create_sample_detections()
    
    # Generate graphs
    combined_fig = visualizer.generate_dual_graphs(
        sample_detections, 
        save_images=True, 
        show_images=True
    )
    
    # Print analysis
    visualizer.print_connectivity_report(sample_detections)
    
    print("\n✅ Test complete! Check graph_output directory for saved images.")
