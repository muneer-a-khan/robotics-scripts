#!/usr/bin/env python3
"""
Network Graph Visualizer for Snap Circuit Analysis
Creates professional network diagrams similar to Cytoscape/Gephi style.
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
import matplotlib.patches as mpatches
from collections import defaultdict


class NetworkGraphVisualizer:
    """Create professional network graph visualizations."""
    
    def __init__(self):
        # Node colors by component type
        self.node_colors = {
            'battery_holder': '#FF6B6B',  # Red - Power source
            'led': '#4ECDC4',             # Teal - Output
            'switch': '#45B7D1',          # Blue - Control
            'button': '#96CEB4',          # Green - Input
            'wire': '#FFF176',            # Yellow - Connection
            'motor': '#FF9FF3',           # Pink - Output
            'resistor': '#81C784',        # Light Green - Passive
            'speaker': '#9C27B0',         # Purple - Output
            'buzzer': '#FF9800',          # Orange - Output
            'lamp': '#FFEB3B',            # Bright Yellow - Output
            'photoresistor': '#BA68C8',   # Violet - Sensor
            'music_circuit': '#E91E63',   # Pink - IC
            'alarm': '#F44336',           # Red - Output
            'fan': '#00BCD4',             # Cyan - Output
            'microphone': '#673AB7',      # Deep Purple - Sensor
            'connection_node': '#9E9E9E'  # Gray - Connection
        }
        
        # Edge colors by connection strength
        self.edge_colors = {
            'strong': '#2E7D32',      # Dark Green
            'medium': '#FFA000',      # Amber
            'weak': '#D32F2F',        # Red
            'potential': '#757575'    # Gray
        }
    
    def load_graph_data(self, json_file: str) -> Dict[str, Any]:
        """Load graph data from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    
    def create_network_graph(self, graph_data: Dict[str, Any]) -> Tuple[nx.DiGraph, Dict, Dict]:
        """Create NetworkX directed graph from JSON data."""
        G = nx.DiGraph()
        node_attributes = {}
        edge_attributes = {}
        
        # Extract graph data
        graph = graph_data.get('graph', {})
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        
        # Add nodes with attributes
        for node in nodes:
            node_id = node['id']
            component_type = node['component_type']
            state = node.get('state', 'unknown')
            confidence = node.get('detection_confidence', 0.0)
            
            # Create clean node label
            clean_id = node_id.replace('_', '').upper()
            
            G.add_node(node_id)
            node_attributes[node_id] = {
                'label': clean_id,
                'component_type': component_type,
                'state': state,
                'confidence': confidence,
                'color': self.node_colors.get(component_type, '#CCCCCC')
            }
        
        # Add edges with attributes
        for edge in edges:
            source = edge.get('component1_id')
            target = edge.get('component2_id')
            strength = edge.get('connection_strength', 0.5)
            
            if source and target and source in G.nodes() and target in G.nodes():
                G.add_edge(source, target)
                
                # Determine edge color based on strength
                if strength >= 0.8:
                    edge_color = self.edge_colors['strong']
                elif strength >= 0.5:
                    edge_color = self.edge_colors['medium']
                else:
                    edge_color = self.edge_colors['weak']
                
                edge_attributes[(source, target)] = {
                    'strength': strength,
                    'color': edge_color,
                    'width': max(1, strength * 3)
                }
        
        return G, node_attributes, edge_attributes
    
    def create_network_visualization(self, json_file: str, output_file: str = None) -> None:
        """Create professional network graph visualization."""
        # Load data
        data = self.load_graph_data(json_file)
        G, node_attrs, edge_attrs = self.create_network_graph(data)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No components detected', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=20, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            return
        
        # Choose layout algorithm based on graph size
        if len(G.nodes()) <= 10:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # Draw edges first (so they appear behind nodes)
        if G.edges():
            for edge in G.edges():
                if edge in edge_attrs:
                    edge_data = edge_attrs[edge]
                    x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                    y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                    
                    # Draw edge line
                    ax.plot(x_coords, y_coords, 
                           color=edge_data['color'], 
                           linewidth=edge_data['width'],
                           alpha=0.7,
                           zorder=1)
                    
                    # Add arrowhead
                    dx = x_coords[1] - x_coords[0]
                    dy = y_coords[1] - y_coords[0]
                    length = np.sqrt(dx**2 + dy**2)
                    
                    if length > 0:
                        # Normalize direction
                        dx_norm = dx / length
                        dy_norm = dy / length
                        
                        # Arrow position (closer to target node)
                        arrow_pos_x = x_coords[1] - dx_norm * 0.15
                        arrow_pos_y = y_coords[1] - dy_norm * 0.15
                        
                        # Draw arrowhead
                        ax.annotate('', xy=(x_coords[1], y_coords[1]), 
                                   xytext=(arrow_pos_x, arrow_pos_y),
                                   arrowprops=dict(arrowstyle='->', 
                                                 color=edge_data['color'],
                                                 lw=edge_data['width']*0.8),
                                   zorder=2)
        
        # Draw nodes
        for node in G.nodes():
            x, y = pos[node]
            attrs = node_attrs[node]
            
            # Node color based on state
            base_color = attrs['color']
            if attrs['state'] == 'powered_on':
                node_color = '#00FF00'  # Bright green for powered
                edge_color = '#006600'  # Dark green border
            elif attrs['state'] == 'inactive':
                node_color = base_color
                edge_color = '#333333'  # Dark border
            else:
                node_color = base_color
                edge_color = '#666666'  # Medium border
            
            # Create oval node
            oval = patches.Ellipse((x, y), 0.25, 0.12, 
                                 facecolor=node_color, 
                                 edgecolor=edge_color,
                                 linewidth=2,
                                 alpha=0.8,
                                 zorder=3)
            ax.add_patch(oval)
            
            # Add node label
            ax.text(x, y, attrs['label'], 
                   ha='center', va='center', 
                   fontsize=9, fontweight='bold',
                   color='black',
                   zorder=4)
            
            # Add confidence indicator (small circle)
            confidence = attrs['confidence']
            if confidence >= 0.8:
                conf_color = '#4CAF50'  # Green
            elif confidence >= 0.5:
                conf_color = '#FF9800'  # Orange
            else:
                conf_color = '#F44336'  # Red
            
            conf_circle = patches.Circle((x + 0.15, y + 0.08), 0.02,
                                       facecolor=conf_color,
                                       edgecolor='white',
                                       linewidth=1,
                                       zorder=5)
            ax.add_patch(conf_circle)
        
        # Create legend
        self.create_legend(ax, node_attrs, edge_attrs)
        
        # Add title with metadata
        timestamp = data.get('graph', {}).get('timestamp', 0)
        dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
        circuit_analysis = data.get('graph', {}).get('circuit_analysis', {})
        
        title = f"Snap Circuit Network Analysis - {dt.strftime('%H:%M:%S')}"
        subtitle = f"Components: {len(G.nodes())} | Connections: {len(G.edges())} | "
        subtitle += f"Circuit: {'Complete' if circuit_analysis.get('circuit_complete', False) else 'Incomplete'}"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
               ha='center', va='top', fontsize=12, style='italic')
        
        # Clean up axes
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"📁 Network graph saved: {output_file}")
        else:
            plt.show()
    
    def create_legend(self, ax: plt.Axes, node_attrs: Dict, edge_attrs: Dict) -> None:
        """Create legend for the network graph."""
        # Component type legend
        component_types = set(attrs['component_type'] for attrs in node_attrs.values())
        legend_elements = []
        
        for comp_type in sorted(component_types):
            color = self.node_colors.get(comp_type, '#CCCCCC')
            legend_elements.append(mpatches.Patch(color=color, label=comp_type.replace('_', ' ').title()))
        
        # State indicators
        legend_elements.extend([
            mpatches.Patch(color='#00FF00', label='Powered On'),
            mpatches.Patch(color='#4CAF50', label='High Confidence'),
            mpatches.Patch(color='#FF9800', label='Medium Confidence'),
            mpatches.Patch(color='#F44336', label='Low Confidence')
        ])
        
        # Create legend
        ax.legend(handles=legend_elements, 
                 loc='upper left', 
                 bbox_to_anchor=(0.02, 0.98),
                 fontsize=8,
                 frameon=True,
                 fancybox=True,
                 shadow=True)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Create network graph visualization')
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
    visualizer = NetworkGraphVisualizer()
    
    # Generate output filename if not provided
    output_file = args.output
    if not output_file and not args.show:
        input_path = Path(args.json_file)
        output_file = input_path.parent / f"{input_path.stem}_network.png"
    
    # Create visualization
    print(f"🔗 Creating network graph from: {args.json_file}")
    
    try:
        visualizer.create_network_visualization(args.json_file, output_file)
        
        if args.show:
            plt.show()
            
    except Exception as e:
        print(f"❌ Error creating network graph: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 