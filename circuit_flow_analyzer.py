#!/usr/bin/env python3
"""
Circuit Flow Analyzer - Analyzes circuit connections and flow
Works with existing component detection to show circuit paths and next components
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
import networkx as nx
from collections import defaultdict, deque
import cv2

class CircuitFlowAnalyzer:
    def __init__(self):
        """Initialize circuit flow analyzer"""
        self.components = {}
        self.connections = defaultdict(list)
        self.circuit_paths = []
        self.component_positions = {}
        
        # Define component connection rules
        self.connection_rules = {
            'Battery Holder': {
                'outputs': ['Wire', 'Slide switch', 'Press switch'],
                'role': 'power_source',
                'max_connections': 2  # + and - terminals
            },
            'Wire': {
                'outputs': ['any'],  # Can connect to anything
                'role': 'conductor',
                'max_connections': 'unlimited'
            },
            'Slide switch': {
                'outputs': ['Wire'],
                'role': 'control',
                'max_connections': 2
            },
            'Press switch': {
                'outputs': ['Wire'],
                'role': 'control', 
                'max_connections': 2
            },
            'LED_1 (Yellow)': {
                'outputs': ['Wire'],
                'role': 'load',
                'max_connections': 2,
                'requires': ['Resistor']  # LEDs typically need resistors
            },
            'LED_2 (Red)': {
                'outputs': ['Wire'],
                'role': 'load',
                'max_connections': 2,
                'requires': ['Resistor']
            },
            'Resistor': {
                'outputs': ['Wire', 'LED_1 (Yellow)', 'LED_2 (Red)'],
                'role': 'passive',
                'max_connections': 2
            },
            'Lamp': {
                'outputs': ['Wire'],
                'role': 'load',
                'max_connections': 2
            },
            'Speaker': {
                'outputs': ['Wire', 'U_1 blue music circuit', 'U_2 red alarm circuit'],
                'role': 'output',
                'max_connections': 2
            },
            'U_1 blue music circuit': {
                'outputs': ['Speaker', 'Wire'],
                'inputs': ['Wire', 'Battery Holder'],
                'role': 'processor',
                'max_connections': 5  # Multiple ports
            },
            'U_2 red alarm circuit': {
                'outputs': ['Speaker', 'Wire'],
                'inputs': ['Wire', 'Battery Holder'],
                'role': 'processor',
                'max_connections': 5
            }
        }
    
    def add_component(self, comp_id, comp_type, position, confidence=1.0):
        """Add a component to the circuit"""
        self.components[comp_id] = {
            'type': comp_type,
            'position': position,
            'confidence': confidence,
            'connections': []
        }
        self.component_positions[comp_id] = position
    
    def analyze_proximity_connections(self, proximity_threshold=2.0):
        """
        Analyze which components are close enough to potentially connect
        Based on grid positions
        """
        potential_connections = []
        
        for comp1_id, comp1 in self.components.items():
            for comp2_id, comp2 in self.components.items():
                if comp1_id != comp2_id:
                    pos1 = comp1['position']
                    pos2 = comp2['position']
                    
                    # Calculate distance (simplified grid distance)
                    if isinstance(pos1, dict) and isinstance(pos2, dict):
                        # Handle component bounds
                        center1 = ((pos1['x1'] + pos1['x2'])/2, (pos1['y1'] + pos1['y2'])/2)
                        center2 = ((pos2['x1'] + pos2['x2'])/2, (pos2['y1'] + pos2['y2'])/2)
                        
                        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                        
                        if distance <= proximity_threshold:
                            # Check if connection is electrically valid
                            if self.is_valid_connection(comp1['type'], comp2['type']):
                                potential_connections.append({
                                    'comp1': comp1_id,
                                    'comp2': comp2_id,
                                    'distance': distance,
                                    'type1': comp1['type'],
                                    'type2': comp2['type']
                                })
        
        return potential_connections
    
    def is_valid_connection(self, comp_type1, comp_type2):
        """Check if two component types can be electrically connected"""
        rules1 = self.connection_rules.get(comp_type1, {})
        rules2 = self.connection_rules.get(comp_type2, {})
        
        outputs1 = rules1.get('outputs', [])
        outputs2 = rules2.get('outputs', [])
        
        # Check if comp1 can output to comp2 or vice versa
        can_connect = (comp_type2 in outputs1 or 
                      comp_type1 in outputs2 or 
                      'any' in outputs1 or 
                      'any' in outputs2)
        
        return can_connect
    
    def find_circuit_paths(self):
        """Find complete circuit paths (from power source back to power source)"""
        paths = []
        
        # Find power sources (batteries)
        power_sources = [comp_id for comp_id, comp in self.components.items() 
                        if comp['type'] == 'Battery Holder']
        
        for power_source in power_sources:
            # Use graph traversal to find circular paths
            paths.extend(self._find_paths_from_source(power_source))
        
        self.circuit_paths = paths
        return paths
    
    def _find_paths_from_source(self, source_id, visited=None, path=None):
        """Find all paths from a power source back to itself (complete circuits)"""
        if visited is None:
            visited = set()
        if path is None:
            path = []
        
        if source_id in visited and len(path) > 2:
            # Found a complete circuit
            return [path + [source_id]]
        
        if source_id in visited:
            return []
        
        visited.add(source_id)
        path.append(source_id)
        
        paths = []
        # Get connected components
        connections = self.connections[source_id]
        
        for connected_id in connections:
            sub_paths = self._find_paths_from_source(connected_id, visited.copy(), path.copy())
            paths.extend(sub_paths)
        
        return paths
    
    def predict_next_components(self, current_components):
        """
        Predict what components might be added next to complete circuits
        """
        suggestions = []
        
        # Analyze current circuit state
        power_sources = [comp for comp in current_components if comp.get('type') == 'Battery Holder']
        loads = [comp for comp in current_components if self.connection_rules.get(comp.get('type', ''), {}).get('role') == 'load']
        controls = [comp for comp in current_components if self.connection_rules.get(comp.get('type', ''), {}).get('role') == 'control']
        
        # Basic circuit analysis
        if power_sources and not loads:
            suggestions.append({
                'component': 'LED or Lamp',
                'reason': 'Need a load component to complete the circuit',
                'priority': 'high',
                'position_suggestion': 'Connect after current components'
            })
        
        if power_sources and loads and not controls:
            suggestions.append({
                'component': 'Switch',
                'reason': 'Add a switch to control the circuit',
                'priority': 'medium',
                'position_suggestion': 'Between power source and load'
            })
        
        # Check for LEDs without resistors
        leds = [comp for comp in current_components if 'LED' in comp.get('type', '')]
        resistors = [comp for comp in current_components if comp.get('type') == 'Resistor']
        
        if leds and len(resistors) < len(leds):
            suggestions.append({
                'component': 'Resistor',
                'reason': 'LEDs need current-limiting resistors',
                'priority': 'high',
                'position_suggestion': 'In series with LED'
            })
        
        # Check for incomplete circuits (need wires)
        isolated_components = self._find_isolated_components(current_components)
        if isolated_components:
            suggestions.append({
                'component': 'Wire',
                'reason': f'{len(isolated_components)} components appear isolated',
                'priority': 'high',
                'position_suggestion': 'Connect isolated components'
            })
        
        return suggestions
    
    def _find_isolated_components(self, components):
        """Find components that appear to be isolated (not connected)"""
        # This is a simplified version - would need actual position analysis
        # For now, assume components are isolated if they're far from others
        isolated = []
        
        for i, comp1 in enumerate(components):
            is_isolated = True
            pos1 = comp1.get('position', {})
            
            for j, comp2 in enumerate(components):
                if i != j:
                    pos2 = comp2.get('position', {})
                    
                    # Check if components are close (simplified)
                    if self._are_components_close(pos1, pos2):
                        is_isolated = False
                        break
            
            if is_isolated:
                isolated.append(comp1)
        
        return isolated
    
    def _are_components_close(self, pos1, pos2, threshold=3):
        """Check if two components are close enough to be connected"""
        if not isinstance(pos1, dict) or not isinstance(pos2, dict):
            return False
        
        center1 = ((pos1.get('x1', 0) + pos1.get('x2', 0))/2, 
                  (pos1.get('y1', 0) + pos1.get('y2', 0))/2)
        center2 = ((pos2.get('x1', 0) + pos2.get('x2', 0))/2, 
                  (pos2.get('y1', 0) + pos2.get('y2', 0))/2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance <= threshold
    
    def create_circuit_flow_diagram(self, components_data):
        """Create a visual diagram showing circuit flow"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Component connections
        self._draw_connection_diagram(ax1, components_data)
        
        # Right: Flow analysis and suggestions
        self._draw_flow_analysis(ax2, components_data)
        
        plt.tight_layout()
        return fig
    
    def _draw_connection_diagram(self, ax, components_data):
        """Draw component connection diagram"""
        ax.set_title("Circuit Component Connections", fontsize=14, fontweight='bold')
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes for each component
        component_list = []
        for comp_type, detections in components_data.items():
            for i, detection in enumerate(detections):
                comp_id = f"{comp_type}_{i}"
                component_list.append({'id': comp_id, 'type': comp_type, 'data': detection})
                G.add_node(comp_id, type=comp_type)
        
        # Add edges based on proximity and connection rules
        for i, comp1 in enumerate(component_list):
            for j, comp2 in enumerate(component_list[i+1:], i+1):
                if self._are_components_close(comp1['data'], comp2['data']) and \
                   self.is_valid_connection(comp1['type'], comp2['type']):
                    G.add_edge(comp1['id'], comp2['id'])
        
        if len(G.nodes()) > 0:
            # Layout the graph
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes with component-specific colors
            for comp_type in set(comp['type'] for comp in component_list):
                nodes = [node for node in G.nodes() if G.nodes[node]['type'] == comp_type]
                color = self._get_component_color(comp_type)
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, 
                                     node_size=800, alpha=0.8, ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, alpha=0.6, ax=ax)
            
            # Draw labels
            labels = {node: node.split('_')[0] for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No components detected', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_flow_analysis(self, ax, components_data):
        """Draw flow analysis and suggestions"""
        ax.set_title("Circuit Analysis & Suggestions", fontsize=14, fontweight='bold')
        
        # Convert components_data to list format for analysis
        current_components = []
        for comp_type, detections in components_data.items():
            for detection in detections:
                current_components.append({
                    'type': comp_type,
                    'position': detection
                })
        
        # Get suggestions
        suggestions = self.predict_next_components(current_components)
        
        # Display current circuit status
        status_text = f"Current Circuit Status:\n"
        status_text += f"• Total Components: {len(current_components)}\n"
        
        # Count component types
        comp_counts = defaultdict(int)
        for comp in current_components:
            comp_counts[comp['type']] += 1
        
        for comp_type, count in comp_counts.items():
            status_text += f"• {comp_type}: {count}\n"
        
        ax.text(0.05, 0.95, status_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        # Display suggestions
        if suggestions:
            suggestions_text = "Suggestions for Circuit Completion:\n\n"
            for i, suggestion in enumerate(suggestions[:5]):  # Show top 5
                priority_icon = "🔴" if suggestion['priority'] == 'high' else "🟡"
                suggestions_text += f"{priority_icon} {suggestion['component']}\n"
                suggestions_text += f"   Reason: {suggestion['reason']}\n"
                suggestions_text += f"   Position: {suggestion['position_suggestion']}\n\n"
            
            ax.text(0.05, 0.6, suggestions_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        else:
            ax.text(0.05, 0.6, "No specific suggestions at this time.\nCircuit appears complete!", 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _get_component_color(self, comp_type):
        """Get color for component type (matching enhanced visualizer)"""
        color_map = {
            'Wire': 'green',
            'Battery Holder': 'red', 
            'LED_1 (Yellow)': 'gold',
            'LED_2 (Red)': 'orangered',
            'Resistor': 'brown',
            'Lamp': 'orange',
            'Speaker': 'blue',
            'Slide switch': 'forestgreen',
            'Press switch': 'darkcyan',
            'U_1 blue music circuit': 'dodgerblue',
            'U_2 red alarm circuit': 'firebrick'
        }
        return color_map.get(comp_type, 'gray')

# Integration function for existing detection system
def analyze_circuit_from_detection(left_detections, right_detections):
    """
    Analyze circuit from detection results
    
    Args:
        left_detections: Left side detection results
        right_detections: Right side detection results
    
    Returns:
        CircuitFlowAnalyzer with analysis results
    """
    analyzer = CircuitFlowAnalyzer()
    
    # Add components from left side
    comp_id_counter = 0
    for comp_type, detections in left_detections.items():
        for detection in detections:
            analyzer.add_component(
                comp_id=f"left_{comp_id_counter}",
                comp_type=comp_type,
                position=detection,
                confidence=detection.get('confidence', 1.0)
            )
            comp_id_counter += 1
    
    # Add components from right side
    for comp_type, detections in right_detections.items():
        for detection in detections:
            analyzer.add_component(
                comp_id=f"right_{comp_id_counter}",
                comp_type=comp_type,
                position=detection,
                confidence=detection.get('confidence', 1.0)
            )
            comp_id_counter += 1
    
    # Analyze connections
    potential_connections = analyzer.analyze_proximity_connections()
    
    # Store connections
    for connection in potential_connections:
        analyzer.connections[connection['comp1']].append(connection['comp2'])
        analyzer.connections[connection['comp2']].append(connection['comp1'])
    
    return analyzer

if __name__ == "__main__":
    # Example usage
    analyzer = CircuitFlowAnalyzer()
    
    # Example detection data
    sample_detections = {
        'Battery Holder': [{'x1': 9, 'y1': 13, 'x2': 10, 'y2': 14, 'confidence': 0.9}],
        'LED_2 (Red)': [{'x1': 3, 'y1': 8, 'x2': 4, 'y2': 9, 'confidence': 0.85}],
        'Resistor': [{'x1': 5, 'y1': 8, 'x2': 6, 'y2': 9, 'confidence': 0.92}],
        'Wire': [
            {'x1': 5, 'y1': 4, 'x2': 8, 'y2': 4, 'confidence': 0.88},
            {'x1': 8, 'y1': 4, 'x2': 8, 'y2': 12, 'confidence': 0.91}
        ]
    }
    
    fig = analyzer.create_circuit_flow_diagram(sample_detections)
    plt.show()
