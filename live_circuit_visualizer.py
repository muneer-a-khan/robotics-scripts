import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict
import os
import time
from pathlib import Path
from scipy.spatial.distance import cdist

class LiveCircuitVisualizer:
    """
    Live circuit visualizer that creates snap circuit board visualizations from detection data.
    """
    
    def __init__(self, confidence_threshold=0.75, output_dir="output"):
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define colors for different component types
        self.component_colors = {
            'wire': '#FFD700',           # Gold
            'resistor': '#FF6B6B',       # Red
            'switch': '#4ECDC4',         # Teal
            'battery_holder': '#45B7D1', # Blue
            'button': '#96CEB4',         # Green
            'music_circuit': '#FFEAA7',  # Light Yellow
            'led': '#FD79A8',           # Pink
            'motor': '#A29BFE',         # Purple
            'lamp': '#FDCB6E',          # Orange
            'buzzer': '#6C5CE7',        # Violet
            'alarm': '#E17055',         # Dark Orange
            'photoresistor': '#74B9FF', # Light Blue
            'speaker': '#E84393'        # Magenta
        }
        
        # Default color for unknown components
        self.default_color = '#95A5A6'  # Gray
    
    def find_connections_with_tolerance(self, components, tolerance=35):
        """
        Find connections between components using tolerance-based clustering.
        
        Args:
            components: List of components with connection points
            tolerance: Distance tolerance for considering points connected (pixels)
        
        Returns:
            List of component pairs that are connected
        """
        connections = []
        component_points = {}
        
        # Collect all connection points for each component
        for component in components:
            comp_id = component.get("id", "unknown")
            points = component.get("connection_points", [])
            component_points[comp_id] = np.array(points) if points else np.array([]).reshape(0, 2)
        
        # Compare each pair of components
        component_ids = list(component_points.keys())
        for i in range(len(component_ids)):
            for j in range(i + 1, len(component_ids)):
                comp1_id = component_ids[i]
                comp2_id = component_ids[j]
                
                points1 = component_points[comp1_id]
                points2 = component_points[comp2_id]
                
                if len(points1) == 0 or len(points2) == 0:
                    continue
                
                # Calculate pairwise distances between all connection points
                distances = cdist(points1, points2)
                min_distance = np.min(distances)
                
                # If any points are within tolerance, consider components connected
                if min_distance <= tolerance:
                    connections.append((comp1_id, comp2_id, min_distance))
        
        return connections
    
    def create_snap_circuit_board_background(self, ax, width, height):
        """Create a snap circuit board background with hexagonal grid pattern"""
        # Set background color to light gray/white
        ax.set_facecolor('#F8F8F8')
        
        # Board border
        board = patches.Rectangle((0, 0), width, height, 
                                linewidth=3, edgecolor='#CCCCCC', 
                                facecolor='#F0F0F0', alpha=0.8)
        ax.add_patch(board)
        
        # Create hexagonal grid pattern
        hex_size = min(width, height) * 0.03  # Adjust size based on image
        hex_spacing_x = hex_size * 1.5
        hex_spacing_y = hex_size * np.sqrt(3)
        
        # Calculate grid dimensions
        rows = int(height / hex_spacing_y) + 2
        cols = int(width / hex_spacing_x) + 2
        
        for row in range(rows):
            for col in range(cols):
                # Offset every other row for hexagonal pattern
                x_offset = (hex_spacing_x / 2) if row % 2 == 1 else 0
                x = col * hex_spacing_x + x_offset
                y = row * hex_spacing_y
                
                if 0 <= x <= width and 0 <= y <= height:
                    # Create hexagon
                    angles = np.linspace(0, 2*np.pi, 7)
                    hex_x = x + hex_size * 0.4 * np.cos(angles)
                    hex_y = y + hex_size * 0.4 * np.sin(angles)
                    
                    hexagon = patches.Polygon(list(zip(hex_x, hex_y)), 
                                            closed=True, linewidth=0.5, 
                                            edgecolor='#DDDDDD', 
                                            facecolor='none', alpha=0.6)
                    ax.add_patch(hexagon)
                    
                    # Add small connection points
                    circle = patches.Circle((x, y), hex_size * 0.1, 
                                          facecolor='#BBBBBB', 
                                          edgecolor='#999999', 
                                          linewidth=0.5, alpha=0.7)
                    ax.add_patch(circle)
    
    def visualize_from_graph_data(self, graph_data, timestamp=None, save_path=None, validation_data=None):
        """
        Create visualization from graph data dictionary.
        
        Args:
            graph_data: Dictionary containing connection graph data
            timestamp: Optional timestamp for filename
            save_path: Optional specific save path
            validation_data: Optional validation results to display
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        
        if save_path is None:
            save_path = self.output_dir / f"live_circuit_visual_{timestamp}.png"
        
        # Filter components with confidence > threshold
        high_confidence_components = []
        components = graph_data.get("connection_graph", {}).get("components", [])
        
        for component in components:
            if component.get("confidence", 0) > self.confidence_threshold:
                high_confidence_components.append(component)
        
        print(f"Showing {len(high_confidence_components)} components with >{self.confidence_threshold*100}% confidence")
        for comp in high_confidence_components:
            print(f"  {comp['component_type']}: {comp['confidence']:.3f}")
        
        if not high_confidence_components:
            print("No high confidence components found - skipping visualization")
            return None
        
        # Create graph
        G = nx.Graph()
        
        # Use improved tolerance-based connection detection
        connections = self.find_connections_with_tolerance(high_confidence_components, tolerance=35)
        print(f"Found {len(connections)} connections with tolerance matching")
        
        # Get image dimensions from bbox coordinates
        all_coords = []
        for component in high_confidence_components:
            bbox = component.get("bbox", [0, 0, 100, 100])
            all_coords.extend([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        if all_coords:
            min_x, max_x = min(all_coords[::2]), max(all_coords[::2])
            min_y, max_y = min(all_coords[1::2]), max(all_coords[1::2])
            img_width = max_x - min_x
            img_height = max_y - min_y
            print(f"Image dimensions: {img_width:.0f} x {img_height:.0f}")
        else:
            img_width, img_height = 1000, 1000
            min_x, min_y = 0, 0
        
        # Store component info and positions
        component_info = {}
        positions = {}
        
        for component in high_confidence_components:
            comp_id = component.get("id", "unknown")
            label = component.get("component_type", "unknown")
            confidence = component.get("confidence", 0.0)
            bbox = component.get("bbox", [0, 0, 100, 100])
            
            # Calculate center position from bbox
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Convert to relative coordinates (0-1 range)
            if img_width > 0 and img_height > 0:
                rel_x = (center_x - min_x) / img_width
                rel_y = 1 - ((center_y - min_y) / img_height)  # Flip Y to match matplotlib
            else:
                rel_x, rel_y = 0.5, 0.5
            
            positions[comp_id] = (rel_x, rel_y)
            
            num_connections = len(component.get("connection_points", []))
            
            # Store component info
            component_info[comp_id] = {
                'type': label,
                'confidence': confidence,
                'connections': num_connections,
                'bbox': bbox
            }
            
            G.add_node(comp_id, 
                      label=label,
                      confidence=confidence,
                      connections=num_connections,
                      color=self.component_colors.get(label, self.default_color))
        
        # Print detailed component information
        print(f"\n=== High Confidence Component Details ===")
        for comp_id, info in component_info.items():
            print(f"{comp_id}: {info['type']} (confidence: {info['confidence']:.3f}, connections: {info['connections']})")
            print(f"  Position: {positions[comp_id][0]:.3f}, {positions[comp_id][1]:.3f}")
            print(f"  Bbox: {info['bbox']}")
            print()
        
        # Add edges based on tolerance-based connection detection
        for comp1, comp2, distance in connections:
            if comp1 in G.nodes() and comp2 in G.nodes():
                G.add_edge(comp1, comp2, distance=distance)
                print(f"Added edge: {comp1} <-> {comp2} (distance: {distance:.1f}px)")
        
        # Create the visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
        # Create snap circuit board background
        self.create_snap_circuit_board_background(ax, 1.0, 1.0)
        
        # Draw edges first (so they appear behind nodes)
        for edge in G.edges():
            if edge[0] in positions and edge[1] in positions:
                x_coords = [positions[edge[0]][0], positions[edge[1]][0]]
                y_coords = [positions[edge[0]][1], positions[edge[1]][1]]
                
                # Get edge distance for color coding
                edge_data = G.edges[edge]
                distance = edge_data.get('distance', 0)
                
                # Color code by connection quality
                if distance <= 5:
                    edge_color = '#00AA00'  # Green for close connections
                    line_width = 4
                elif distance <= 20:
                    edge_color = '#FFA500'  # Orange for medium connections  
                    line_width = 3
                else:
                    edge_color = '#FF6666'  # Red for distant connections
                    line_width = 2
                
                ax.plot(x_coords, y_coords, 
                       color=edge_color, linewidth=line_width, alpha=0.8, zorder=2)
        
        # Draw nodes at their actual positions
        for node in G.nodes():
            if node in positions:
                x, y = positions[node]
                color = G.nodes[node]['color']
                
                # Draw component circle
                circle = patches.Circle((x, y), 0.04, 
                                      facecolor=color, 
                                      edgecolor='black',
                                      linewidth=2, alpha=0.9, zorder=3)
                ax.add_patch(circle)
                
                # Add label
                info = component_info[node]
                label_text = f"{info['type']}\nconf: {info['confidence']:.2f}"
                ax.text(x, y-0.08, label_text, 
                       ha='center', va='top', fontsize=9, 
                       fontweight='bold', 
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='white', alpha=0.8),
                       zorder=4)
        
        # Create a legend for component types
        legend_elements = []
        unique_types = set(component_info[comp]['type'] for comp in component_info)
        for comp_type in sorted(unique_types):
            color = self.component_colors.get(comp_type, self.default_color)
            legend_elements.append(plt.scatter([], [], c=color, s=200, label=comp_type))
        
        if legend_elements:
            ax.legend(handles=legend_elements, 
                     title="Component Types (Live)",
                     bbox_to_anchor=(1.02, 1), 
                     loc='upper left',
                     fontsize=10)
        
        # Add title with timestamp
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp/1000))
        ax.set_title(f"Live Snap Circuit Board - {time_str}\n(High Confidence Components: {len(G.nodes())}, Tolerance-Based Connections)", 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add validation scores in top right if available
        if validation_data:
            self._add_validation_display(ax, validation_data)
        
        # Add statistics
        total_components = len(G.nodes())
        total_connections = len(G.edges())
        ax.text(0.5, -0.05, f"Components: {total_components} | Connections: {total_connections}", 
               ha='center', va='top', transform=ax.transAxes,
               fontsize=12, style='italic')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save the graph
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        print(f"Live circuit visualization saved to: {save_path}")
        return save_path
    
    def _add_validation_display(self, ax, validation_data):
        """Add validation scores and status to the top right of the visualization."""
        # Extract validation information
        overall_result = validation_data.get("overall_result", "unknown")
        summary = validation_data.get("summary", {})
        score = summary.get("score", 0)
        errors = summary.get("errors", 0)
        warnings = summary.get("warnings", 0)
        total_issues = summary.get("total_issues", 0)
        
        # Choose colors and symbols based on result
        if overall_result == "correct":
            status_color = '#4CAF50'  # Green
            status_symbol = '✅'
            border_color = '#2E7D32'
        elif overall_result == "partial":
            status_color = '#FF9800'  # Orange
            status_symbol = '⚠️'
            border_color = '#F57C00'
        elif overall_result == "incorrect":
            status_color = '#F44336'  # Red
            status_symbol = '❌'
            border_color = '#C62828'
        else:
            status_color = '#9E9E9E'  # Gray
            status_symbol = '❓'
            border_color = '#616161'
        
        # Create validation info box - simplified
        validation_text = [
            f"{status_symbol} {overall_result.upper()}",
            f"Score: {score}%"
        ]
        
        # Position in top right (using axes coordinates)
        box_x = 0.98
        box_y = 0.98
        
        # Create background box
        from matplotlib.patches import FancyBboxPatch
        
        # Calculate box size based on text - smaller for simplified display
        box_width = 0.15
        box_height = 0.08
        
        # Create rounded rectangle background
        validation_box = FancyBboxPatch(
            (box_x - box_width, box_y - box_height), 
            box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor='white',
            edgecolor=border_color,
            linewidth=2,
            alpha=0.95,
            transform=ax.transAxes,
            zorder=10
        )
        ax.add_patch(validation_box)
        
        # Add validation text
        text_y_start = box_y - 0.02
        text_spacing = 0.025
        
        for i, text_line in enumerate(validation_text):
            text_y = text_y_start - (i * text_spacing)
            
            # First line (status) gets special formatting
            if i == 0:
                ax.text(box_x - 0.01, text_y, text_line,
                       ha='right', va='top', transform=ax.transAxes,
                       fontsize=11, fontweight='bold', color=status_color,
                       zorder=11)
            else:
                # Regular formatting for other lines
                ax.text(box_x - 0.01, text_y, text_line,
                       ha='right', va='top', transform=ax.transAxes,
                       fontsize=9, color='black',
                       zorder=11)
        
        # Add a subtle drop shadow effect
        shadow_box = FancyBboxPatch(
            (box_x - box_width + 0.002, box_y - box_height - 0.002), 
            box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor='gray',
            alpha=0.3,
            transform=ax.transAxes,
            zorder=9
        )
        ax.add_patch(shadow_box)
    
    def visualize_from_file(self, json_file_path):
        """
        Create visualization from a JSON file.
        
        Args:
            json_file_path: Path to the graph JSON file
        """
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # Extract timestamp from filename if possible
            filename = Path(json_file_path).stem
            timestamp = None
            if 'graph_' in filename:
                try:
                    timestamp_str = filename.split('_')[1]
                    timestamp = int(timestamp_str)
                except:
                    timestamp = int(time.time() * 1000)
            
            return self.visualize_from_graph_data(data, timestamp, save_path=None, validation_data=None)
            
        except Exception as e:
            print(f"Error creating visualization from {json_file_path}: {e}")
            return None
    
    def get_latest_graph_file(self, data_dir="output/data"):
        """
        Get the most recent graph JSON file.
        
        Args:
            data_dir: Directory containing graph files
            
        Returns:
            Path to latest graph file or None
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            return None
        
        graph_files = list(data_path.glob("graph_*.json"))
        if not graph_files:
            return None
        
        # Sort by modification time and return newest
        latest_file = max(graph_files, key=os.path.getmtime)
        return latest_file
    
    def monitor_and_visualize(self, data_dir="output/data", check_interval=1):
        """
        Monitor for new graph files and create visualizations automatically.
        
        Args:
            data_dir: Directory to monitor
            check_interval: How often to check for new files (seconds)
        """
        print(f"Monitoring {data_dir} for new graph files...")
        print("Press Ctrl+C to stop monitoring")
        
        processed_files = set()
        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True)
        
        try:
            while True:
                # Find all graph files
                graph_files = list(data_path.glob("graph_*.json"))
                
                # Process new files
                for graph_file in graph_files:
                    if graph_file not in processed_files:
                        print(f"New graph file detected: {graph_file.name}")
                        self.visualize_from_file(graph_file)
                        processed_files.add(graph_file)
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nStopped monitoring.")


# Function to integrate with main system
def create_live_visualization(graph_data, timestamp=None, output_dir="output", validation_data=None):
    """
    Convenience function to create a live visualization from graph data.
    This can be called directly from the main detection system.
    
    Args:
        graph_data: Dictionary containing connection graph data
        timestamp: Optional timestamp
        output_dir: Output directory for visualization
        validation_data: Optional validation results to display
    
    Returns:
        Path to saved visualization or None if failed
    """
    visualizer = LiveCircuitVisualizer(output_dir=output_dir)
    return visualizer.visualize_from_graph_data(graph_data, timestamp, save_path=None, validation_data=validation_data)


if __name__ == "__main__":
    # Can be run standalone to monitor for new files
    visualizer = LiveCircuitVisualizer()
    
    # Try to visualize the latest graph file
    latest_file = visualizer.get_latest_graph_file()
    if latest_file:
        print(f"Visualizing latest graph file: {latest_file}")
        visualizer.visualize_from_file(latest_file)
    else:
        print("No graph files found. Starting monitor mode...")
        visualizer.monitor_and_visualize() 