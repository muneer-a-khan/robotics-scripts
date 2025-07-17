#!/usr/bin/env python3
"""
Visual Graph Output Converter
Converts JSON graph output into visual circuit diagrams that are easy to understand.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
from datetime import datetime


class CircuitVisualizer:
    """Convert JSON graph output to visual circuit diagrams."""
    
    def __init__(self):
        # Component styling
        self.component_colors = {
            'battery_holder': '#FF6B6B',  # Red
            'led': '#4ECDC4',             # Teal  
            'switch': '#45B7D1',          # Blue
            'button': '#96CEB4',          # Green
            'wire': '#FECA57',            # Yellow
            'motor': '#FF9FF3',           # Pink
            'resistor': '#54A0FF',        # Light Blue
            'speaker': '#5F27CD',         # Purple
            'buzzer': '#FF9F43',          # Orange
            'lamp': '#FECA57',            # Bright Yellow
            'photoresistor': '#A55EEA',   # Violet
            'music_circuit': '#FD79A8',   # Rose
            'alarm': '#E84393',           # Magenta
            'fan': '#00B894',             # Sea Green
            'microphone': '#6C5CE7',      # Periwinkle
            'connection_node': '#DDD'     # Gray
        }
        
        self.component_shapes = {
            'battery_holder': 'rectangle',
            'led': 'circle',
            'switch': 'rectangle',
            'button': 'circle',
            'wire': 'line',
            'motor': 'rectangle',
            'resistor': 'rectangle',
            'speaker': 'circle',
            'buzzer': 'circle',
            'lamp': 'circle',
            'photoresistor': 'rectangle',
            'music_circuit': 'rectangle',
            'alarm': 'circle',
            'fan': 'circle',
            'microphone': 'circle',
            'connection_node': 'circle'
        }
    
    def load_graph_data(self, json_file: str) -> Dict[str, Any]:
        """Load graph data from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    
    def create_network_graph(self, graph_data: Dict[str, Any]) -> Tuple[nx.Graph, Dict]:
        """Create NetworkX graph from JSON data."""
        G = nx.Graph()
        pos = {}
        
        # Add nodes (components)
        graph = graph_data.get('graph', {})
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        
        for node in nodes:
            component_id = node['id']
            component_type = node['component_type']
            state = node.get('state', 'unknown')
            confidence = node.get('detection_confidence', 0.0)
            position = node.get('position', {})
            
            # Add node with attributes (avoid duplicate keys)
            node_attrs = {k: v for k, v in node.items() if k not in ['component_type', 'state', 'detection_confidence']}
            G.add_node(component_id, 
                      component_type=component_type,
                      state=state,
                      confidence=confidence,
                      **node_attrs)
            
            # Set position for layout (normalize coordinates)
            x = position.get('x', 0) / 1000  # Scale down for display
            y = position.get('y', 0) / 1000
            pos[component_id] = (x, -y)  # Flip Y for proper display
        
        # Add edges (connections)
        for edge in edges:
            comp1 = edge.get('component1_id')
            comp2 = edge.get('component2_id') 
            strength = edge.get('connection_strength', 1.0)
            
            if comp1 and comp2:
                G.add_edge(comp1, comp2, weight=strength, **edge)
        
        return G, pos
    
    def create_visual_diagram(self, json_file: str, output_file: str = None) -> None:
        """Create comprehensive visual diagram from JSON graph data."""
        # Load data
        data = self.load_graph_data(json_file)
        G, pos = self.create_network_graph(data)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Main circuit diagram
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        # Component details
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        
        # Circuit analysis  
        ax3 = plt.subplot2grid((3, 3), (1, 2))
        
        # Timeline/metadata
        ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        
        # === MAIN CIRCUIT DIAGRAM ===
        self.draw_circuit_diagram(G, pos, ax1, data)
        
        # === COMPONENT DETAILS ===
        self.draw_component_details(data, ax2)
        
        # === CIRCUIT ANALYSIS ===
        self.draw_circuit_analysis(data, ax3)
        
        # === METADATA & TIMELINE ===
        self.draw_metadata(data, ax4)
        
        # Title and layout
        timestamp = data.get('graph', {}).get('timestamp', 0)
        dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
        
        fig.suptitle(f'Snap Circuit Analysis - {dt.strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📁 Visual diagram saved: {output_file}")
        else:
            plt.show()
    
    def draw_circuit_diagram(self, G: nx.Graph, pos: Dict, ax: plt.Axes, data: Dict) -> None:
        """Draw the main circuit diagram."""
        ax.set_title('🔌 Circuit Topology', fontsize=14, fontweight='bold')
        
        if not G.nodes():
            ax.text(0.5, 0.5, 'No components detected', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        # Draw edges (connections)
        if G.edges():
            edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, ax=ax, 
                                 width=[w*3 for w in edge_weights],
                                 edge_color='green', alpha=0.7)
        
        # Draw nodes (components)
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            component_type = node_data.get('component_type', 'unknown')
            state = node_data.get('state', 'unknown')
            confidence = node_data.get('confidence', 0.0)
            
            x, y = pos[node_id]
            color = self.component_colors.get(component_type, '#CCCCCC')
            
            # Node color based on state
            if state == 'powered_on':
                color = '#00FF00'  # Bright green for powered
            elif state == 'inactive':
                color = '#FF6B6B'  # Red for inactive
            elif confidence < 0.5:
                color = '#FFCCCC'  # Light red for low confidence
            
            # Draw component
            circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # Label
            ax.text(x, y-0.5, f'{component_type}\n{confidence:.2f}', 
                   ha='center', va='top', fontsize=8, fontweight='bold')
        
        # Circuit state indicators
        circuit_analysis = data.get('graph', {}).get('circuit_analysis', {})
        circuit_complete = circuit_analysis.get('circuit_complete', False)
        power_sources = circuit_analysis.get('power_sources', [])
        
        status_text = f"Circuit: {'✅ Complete' if circuit_complete else '❌ Incomplete'}\n"
        status_text += f"Power: {'🔋 ' + ', '.join(power_sources) if power_sources else '❌ No power'}"
        
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
               va='top', ha='left', fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def draw_component_details(self, data: Dict, ax: plt.Axes) -> None:
        """Draw component details table."""
        ax.set_title('📦 Components', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        nodes = data.get('graph', {}).get('nodes', [])
        
        if not nodes:
            ax.text(0.5, 0.5, 'No components\ndetected', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Create table data
        table_data = []
        for node in nodes:
            comp_type = node.get('component_type', 'unknown')
            state = node.get('state', 'unknown')
            confidence = node.get('detection_confidence', 0.0)
            
            # Status emoji
            if state == 'powered_on':
                status = '🟢'
            elif state == 'inactive':
                status = '🔴'  
            else:
                status = '⚪'
            
            table_data.append([
                status,
                comp_type,
                f'{confidence:.2f}'
            ])
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Status', 'Type', 'Conf'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color cells based on confidence
        for i, row in enumerate(table_data):
            conf = float(row[2])
            if conf >= 0.8:
                color = '#C8E6C9'  # Light green
            elif conf >= 0.5:
                color = '#FFF9C4'  # Light yellow
            else:
                color = '#FFCDD2'  # Light red
                
            table[(i+1, 2)].set_facecolor(color)
    
    def draw_circuit_analysis(self, data: Dict, ax: plt.Axes) -> None:
        """Draw circuit analysis results."""
        ax.set_title('⚡ Analysis', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        circuit_analysis = data.get('graph', {}).get('circuit_analysis', {})
        connectivity = circuit_analysis.get('connectivity', {})
        
        analysis_text = []
        
        # Circuit completeness
        complete = circuit_analysis.get('circuit_complete', False)
        analysis_text.append(f"Complete: {'✅' if complete else '❌'}")
        
        # Power sources
        power_sources = circuit_analysis.get('power_sources', [])
        analysis_text.append(f"Power: {len(power_sources)} sources")
        
        # Connections
        connected_components = connectivity.get('connected_components', [])
        analysis_text.append(f"Groups: {len(connected_components)}")
        
        # Isolated components
        isolated = connectivity.get('isolated_components', [])
        analysis_text.append(f"Isolated: {len(isolated)}")
        
        # Active paths
        active_paths = circuit_analysis.get('active_paths', [])
        analysis_text.append(f"Active paths: {len(active_paths)}")
        
        # Display analysis
        ax.text(0.1, 0.9, '\n'.join(analysis_text), 
               transform=ax.transAxes, va='top', ha='left', 
               fontsize=10, fontfamily='monospace')
    
    def draw_metadata(self, data: Dict, ax: plt.Axes) -> None:
        """Draw metadata and timing information."""
        ax.set_title('📊 Session Information', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Extract metadata
        timestamp = data.get('graph', {}).get('timestamp', 0)
        dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
        
        nodes = data.get('graph', {}).get('nodes', [])
        total_components = len(nodes)
        
        # Sample learning context from first node
        learning_context = {}
        if nodes:
            learning_context = nodes[0].get('learning_context', {})
        
        # Create metadata text
        metadata_text = f"""
🕒 Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}
📦 Total Components: {total_components}
🎓 Skill Level: {learning_context.get('skill_level', 'Unknown')}
🔄 Session ID: {learning_context.get('session_id', 'Unknown')}
📝 Attempts: {learning_context.get('attempts', 'Unknown')}
        """.strip()
        
        ax.text(0.1, 0.5, metadata_text, 
               transform=ax.transAxes, va='center', ha='left', 
               fontsize=10, fontfamily='monospace')


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Visualize Snap Circuit graph output')
    parser.add_argument('json_file', help='Path to JSON graph output file')
    parser.add_argument('--output', '-o', help='Output image file (PNG/PDF)')
    parser.add_argument('--show', '-s', action='store_true', 
                       help='Show interactive plot')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"❌ File not found: {args.json_file}")
        return
    
    # Create visualizer
    visualizer = CircuitVisualizer()
    
    # Generate output filename if not provided
    output_file = args.output
    if not output_file and not args.show:
        input_path = Path(args.json_file)
        output_file = input_path.parent / f"{input_path.stem}_visual.png"
    
    # Create visualization
    print(f"🎨 Creating visual diagram from: {args.json_file}")
    
    try:
        visualizer.create_visual_diagram(args.json_file, output_file)
        
        if args.show:
            plt.show()
            
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 