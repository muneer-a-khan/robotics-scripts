#!/usr/bin/env python3
"""
Enhanced Board Visualizer - Advanced circuit board visualization
Shows component placement, connections, and circuit flow
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
from collections import defaultdict, deque

class EnhancedBoardVisualizer:
    def __init__(self, rows=13, cols=15, cell_size=60):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.board_width = cols * cell_size
        self.board_height = rows * cell_size
        
        # Enhanced color mapping with better visibility
        self.component_colors = {
            'Wire': '#2E8B57',           # Sea Green
            'Battery Holder': '#DC143C',  # Crimson
            'LED_1 (Yellow)': '#FFD700', # Gold
            'LED_2 (Red)': '#FF4500',    # Orange Red
            'Resistor': '#8B4513',       # Saddle Brown
            'Lamp': '#FFA500',           # Orange
            'Photoresistor': '#9370DB',  # Medium Purple
            'U_1 blue music circuit': '#1E90FF',     # Dodger Blue
            'U_2 red alarm circuit': '#B22222',      # Fire Brick
            'U_3 green space war circuit': '#32CD32', # Lime Green
            'Speaker': '#4169E1',        # Royal Blue
            'Slide switch': '#228B22',   # Forest Green
            'Press switch': '#008B8B',   # Dark Cyan
            'Whistle chip': '#DDA0DD',   # Plum
            'Green tape': '#90EE90'      # Light Green
        }
        
        # Component symbols for better identification
        self.component_symbols = {
            'Wire': '═',
            'Battery Holder': '⚡',
            'LED_1 (Yellow)': '💡',
            'LED_2 (Red)': '🔴',
            'Resistor': '▬',
            'Lamp': '💡',
            'Photoresistor': '👁',
            'U_1 blue music circuit': '♪',
            'U_2 red alarm circuit': '⏰',
            'U_3 green space war circuit': '🚀',
            'Speaker': '🔊',
            'Slide switch': '⚪',
            'Press switch': '⭕',
            'Whistle chip': '🎵',
            'Green tape': '▦'
        }

    def create_enhanced_board(self, detection_results, component_list=None):
        """
        Create an enhanced visual board representation
        
        Args:
            detection_results: Dict with component positions and types
            component_list: List of component objects for connection info
        """
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left panel: Component placement
        self._draw_component_board(ax1, detection_results)
        
        # Right panel: Circuit flow (if component_list provided)
        if component_list:
            self._draw_circuit_flow(ax2, detection_results, component_list)
        else:
            self._draw_connection_analysis(ax2, detection_results)
        
        plt.tight_layout()
        return fig

    def _draw_component_board(self, ax, detection_results):
        """Draw the main component placement board"""
        ax.set_title("Circuit Board - Component Placement", fontsize=16, fontweight='bold')
        
        # Draw grid
        for i in range(self.rows + 1):
            ax.axhline(y=i, color='lightgray', linewidth=0.5)
        for j in range(self.cols + 1):
            ax.axvline(x=j, color='lightgray', linewidth=0.5)
        
        # Draw components
        component_counts = defaultdict(int)
        
        for component_type, positions in detection_results.items():
            if component_type in self.component_colors:
                color = self.component_colors[component_type]
                symbol = self.component_symbols.get(component_type, '●')
                
                for pos in positions:
                    if isinstance(pos, dict):
                        # Handle component with bounds
                        x1, y1, x2, y2 = pos['x1'], pos['y1'], pos['x2'], pos['y2']
                        
                        # Draw filled rectangle for component area
                        rect = patches.Rectangle((y1, self.rows-x2-1), y2-y1+1, x2-x1+1, 
                                               linewidth=2, edgecolor='black', 
                                               facecolor=color, alpha=0.7)
                        ax.add_patch(rect)
                        
                        # Add component label
                        center_x = y1 + (y2-y1+1)/2
                        center_y = self.rows-x2-1 + (x2-x1+1)/2
                        
                        # Component symbol
                        ax.text(center_x, center_y+0.15, symbol, 
                               ha='center', va='center', fontsize=20, fontweight='bold')
                        
                        # Component type (abbreviated)
                        comp_name = component_type.replace('_', '\n').replace(' circuit', '\ncircuit')
                        ax.text(center_x, center_y-0.2, comp_name, 
                               ha='center', va='center', fontsize=8, fontweight='bold')
                        
                        component_counts[component_type] += 1
                    else:
                        # Handle simple position
                        x, y = pos
                        ax.scatter(y, self.rows-x-1, c=color, s=200, alpha=0.8, edgecolors='black')
                        ax.text(y, self.rows-x-1, symbol, ha='center', va='center', 
                               fontsize=12, fontweight='bold')
        
        # Add component legend
        self._add_component_legend(ax, component_counts)
        
        ax.set_xlim(-0.5, self.cols-0.5)
        ax.set_ylim(-0.5, self.rows-0.5)
        ax.set_xlabel('Column', fontweight='bold')
        ax.set_ylabel('Row', fontweight='bold')
        ax.set_aspect('equal')

    def _draw_circuit_flow(self, ax, detection_results, component_list):
        """Draw circuit flow and connections"""
        ax.set_title("Circuit Flow Analysis", fontsize=16, fontweight='bold')
        
        # Create a simplified view focusing on connections
        # This would integrate with your existing circuit analysis
        
        # For now, create a connection matrix visualization
        self._draw_connection_matrix(ax, detection_results)

    def _draw_connection_analysis(self, ax, detection_results):
        """Draw connection analysis when component objects aren't available"""
        ax.set_title("Component Analysis", fontsize=16, fontweight='bold')
        
        # Component count analysis
        component_counts = {}
        total_components = 0
        
        for comp_type, positions in detection_results.items():
            count = len(positions) if isinstance(positions, list) else 1
            component_counts[comp_type] = count
            total_components += count
        
        # Create bar chart of components
        if component_counts:
            comp_names = list(component_counts.keys())
            comp_counts_list = list(component_counts.values())
            colors = [self.component_colors.get(name, '#gray') for name in comp_names]
            
            bars = ax.bar(range(len(comp_names)), comp_counts_list, color=colors, alpha=0.7)
            ax.set_xlabel('Component Type', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_xticks(range(len(comp_names)))
            ax.set_xticklabels([name.replace('_', '\n').replace(' circuit', '\ncircuit') for name in comp_names], 
                              rotation=45, ha='right', fontsize=10)
            
            # Add count labels on bars
            for bar, count in zip(bars, comp_counts_list):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       str(count), ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f"Components Detected (Total: {total_components})", 
                        fontsize=14, fontweight='bold')

    def _draw_connection_matrix(self, ax, detection_results):
        """Draw a connection possibility matrix"""
        component_types = list(detection_results.keys())
        n_components = len(component_types)
        
        if n_components == 0:
            ax.text(0.5, 0.5, 'No components detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            return
        
        # Create connection matrix (simplified - based on typical connections)
        connection_matrix = np.zeros((n_components, n_components))
        
        # Define typical component connections (simplified)
        connection_rules = {
            'Battery Holder': ['Wire', 'Slide switch', 'Press switch'],
            'Wire': ['Battery Holder', 'LED_1 (Yellow)', 'LED_2 (Red)', 'Lamp', 'Resistor', 
                    'Speaker', 'U_1 blue music circuit', 'U_2 red alarm circuit'],
            'Slide switch': ['Wire', 'Battery Holder'],
            'Press switch': ['Wire', 'Battery Holder'],
            'LED_1 (Yellow)': ['Wire', 'Resistor'],
            'LED_2 (Red)': ['Wire', 'Resistor'],
            'Resistor': ['Wire', 'LED_1 (Yellow)', 'LED_2 (Red)'],
            'Lamp': ['Wire'],
            'Speaker': ['Wire', 'U_1 blue music circuit', 'U_2 red alarm circuit'],
        }
        
        # Fill connection matrix
        for i, comp1 in enumerate(component_types):
            for j, comp2 in enumerate(component_types):
                if comp1 in connection_rules and comp2 in connection_rules[comp1]:
                    connection_matrix[i][j] = 1
        
        # Draw heatmap
        im = ax.imshow(connection_matrix, cmap='RdYlGn', aspect='equal', alpha=0.8)
        
        # Add labels
        ax.set_xticks(range(n_components))
        ax.set_yticks(range(n_components))
        ax.set_xticklabels([comp.replace('_', '\n') for comp in component_types], 
                          rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels([comp.replace('_', '\n') for comp in component_types], fontsize=10)
        
        # Add text annotations
        for i in range(n_components):
            for j in range(n_components):
                if connection_matrix[i][j] == 1:
                    ax.text(j, i, '✓', ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=16)
        
        ax.set_title('Component Connection Possibilities', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _add_component_legend(self, ax, component_counts):
        """Add a legend showing component types and counts"""
        legend_elements = []
        for comp_type, count in component_counts.items():
            color = self.component_colors.get(comp_type, 'gray')
            symbol = self.component_symbols.get(comp_type, '●')
            legend_elements.append(patches.Patch(color=color, 
                                                label=f'{symbol} {comp_type} ({count})'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize=10, framealpha=0.9)

    def create_live_board_overlay(self, frame, detection_results, board_bounds):
        """
        Create an overlay on the live camera feed showing component identification
        
        Args:
            frame: Camera frame
            detection_results: Detection results with positions
            board_bounds: Board boundary coordinates
        """
        overlay = frame.copy()
        
        # Draw board grid overlay
        if board_bounds:
            x1, y1, x2, y2 = board_bounds
            
            # Draw grid lines
            for i in range(self.rows + 1):
                y = int(y1 + (i / self.rows) * (y2 - y1))
                cv2.line(overlay, (x1, y), (x2, y), (255, 255, 255), 1)
            
            for j in range(self.cols + 1):
                x = int(x1 + (j / self.cols) * (x2 - x1))
                cv2.line(overlay, (x, y1), (x, y2), (255, 255, 255), 1)
        
        # Draw component overlays
        for comp_type, positions in detection_results.items():
            color_hex = self.component_colors.get(comp_type, '#FFFFFF')
            # Convert hex to BGR for OpenCV
            color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
            
            for pos in positions:
                if isinstance(pos, dict) and 'bbox' in pos:
                    # Draw bounding box
                    x1, y1, x2, y2 = pos['bbox']
                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                                color_bgr, 2)
                    
                    # Add label
                    label = f"{comp_type}: {pos.get('confidence', 0):.2f}"
                    cv2.putText(overlay, label, (int(x1), int(y1-10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
        
        return overlay

def process_detection_for_visualization(left_components, right_components, left_classes, right_classes):
    """
    Convert detection results to format suitable for visualization
    
    Args:
        left_components: Left side component count
        right_components: Right side component count  
        left_classes: Left side component classes dict
        right_classes: Right side component classes dict
    """
    # Combine left and right results
    all_results = {}
    
    # Process left side
    for comp_type, count in left_classes.items():
        if comp_type not in all_results:
            all_results[comp_type] = []
        # Add placeholder positions (would be replaced with actual positions)
        for i in range(count):
            all_results[comp_type].append({
                'side': 'left',
                'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1,  # Placeholder
                'confidence': 0.8  # Placeholder
            })
    
    # Process right side
    for comp_type, count in right_classes.items():
        if comp_type not in all_results:
            all_results[comp_type] = []
        for i in range(count):
            all_results[comp_type].append({
                'side': 'right', 
                'x1': 7, 'y1': 0, 'x2': 8, 'y2': 1,  # Placeholder
                'confidence': 0.8  # Placeholder
            })
    
    return all_results

if __name__ == "__main__":
    # Example usage
    visualizer = EnhancedBoardVisualizer()
    
    # Example detection results (replace with your actual results)
    sample_results = {
        'U_2 red alarm circuit': [{'x1': 2, 'y1': 4, 'x2': 3, 'y2': 5}],
        'Slide switch': [{'x1': 4, 'y1': 6, 'x2': 4, 'y2': 7}],
        'Wire': [
            {'x1': 5, 'y1': 4, 'x2': 8, 'y2': 4},
            {'x1': 8, 'y1': 4, 'x2': 8, 'y2': 12}
        ],
        'Battery Holder': [{'x1': 9, 'y1': 13, 'x2': 10, 'y2': 14}],
        'Lamp': [{'x1': 8, 'y1': 6, 'x2': 9, 'y2': 7}],
        'Resistor': [{'x1': 3, 'y1': 8, 'x2': 4, 'y2': 9}]
    }
    
    fig = visualizer.create_enhanced_board(sample_results)
    plt.show()
