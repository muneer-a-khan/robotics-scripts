#!/usr/bin/env python3
"""
Dual Board Visualizer - Simple side-by-side board representation
Shows left and right boards with components filling in grid squares
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import time

class DualBoardVisualizer:
    def __init__(self, cell_size=40):
        # Fixed board dimensions: 7 columns x 5 rows for each board (matching calibration)
        self.cols = 7
        self.rows = 5
        self.cell_size = cell_size
        
        # Simple, clear colors for components (keeping existing colors)
        self.component_colors = {
            'Wire': '#228B22',           # Forest Green
            'Battery Holder': '#DC143C',  # Crimson Red
            'LED_1 (Yellow)': '#FFD700', # Gold
            'LED_2 (Red)': '#FF4500',    # Orange Red
            'Resistor': '#8B4513',       # Brown
            'Lamp': '#FFA500',           # Orange
            'Photoresistor': '#9370DB',  # Purple
            'U_1 blue music circuit': '#1E90FF',     # Blue
            'U_2 red alarm circuit': '#B22222',      # Dark Red
            'U_3 green space war circuit': '#32CD32', # Lime Green
            'Speaker': '#4169E1',        # Royal Blue
            'Slide switch': '#008B8B',   # Dark Cyan
            'Press switch': '#006400',   # Dark Green
            'Whistle chip': '#DDA0DD'    # Plum
            # Removed Green tape from visualization
        }
        
        # Component abbreviations for labels
        self.component_labels = {
            'Wire': 'W',
            'Battery Holder': 'BAT',
            'LED_1 (Yellow)': 'LED1',
            'LED_2 (Red)': 'LED2',
            'Resistor': 'RES',
            'Lamp': 'LAMP',
            'Photoresistor': 'PHR',
            'U_1 blue music circuit': 'MC1',
            'U_2 red alarm circuit': 'MC2',
            'U_3 green space war circuit': 'MC3',
            'Speaker': 'SPK',
            'Slide switch': 'SW1',
            'Press switch': 'SW2',
            'Whistle chip': 'WHI'
            # Removed Green tape from labels
        }

    def create_dual_board_visualization(self, left_detections, right_detections):
        """
        Create side-by-side board visualization
        
        Args:
            left_detections: Dict of {component_type: [detection_info, ...]}
            right_detections: Dict of {component_type: [detection_info, ...]}
        """
        # Create figure with two side-by-side subplots
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Dual Circuit Board Detection', fontsize=16, fontweight='bold')
        
        # Draw left board
        self._draw_single_board(ax_left, left_detections, "LEFT BOARD")
        
        # Draw right board  
        self._draw_single_board(ax_right, right_detections, "RIGHT BOARD")
        
        # Add component legend
        self._add_legend(fig, left_detections, right_detections)
        
        plt.tight_layout()
        return fig

    def _draw_single_board(self, ax, detections, title):
        """Draw a single 7x5 board with components marking occupied grid spots"""
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Create empty board grid (7 cols x 5 rows)
        board_grid = np.ones((self.rows, self.cols, 3)) * 0.95  # Light gray background
        
        # Track which grid spots are occupied and their components
        grid_components = {}  # (row, col) -> [(component_type, confidence), ...]
        
        # Collect all components for each grid spot
        for comp_type, detection_list in detections.items():
            # Skip green tape since we don't visualize it anymore
            if comp_type == "Green tape" or comp_type not in self.component_colors:
                continue
            
            for detection in detection_list:
                # Get grid position (expecting simple row, col coordinates)
                if isinstance(detection, dict):
                    row = detection.get('row', 0)
                    col = detection.get('col', 0)
                    confidence = detection.get('confidence', 1.0)
                else:
                    # Fallback for simple format
                    row = 0
                    col = 0
                    confidence = 1.0
                
                # Ensure bounds
                row = max(0, min(row, self.rows - 1))
                col = max(0, min(col, self.cols - 1))
                
                # Add to grid components list
                grid_pos = (row, col)
                if grid_pos not in grid_components:
                    grid_components[grid_pos] = []
                grid_components[grid_pos].append((comp_type, confidence))
        
        # Fill grid spots with the topmost (highest confidence) component
        occupied_spots = {}  # (row, col) -> component_type
        for (row, col), components in grid_components.items():
            # Sort by confidence (highest first) to get "topmost" component
            components.sort(key=lambda x: x[1], reverse=True)
            topmost_component = components[0][0]  # Get component type with highest confidence
            
            color_hex = self.component_colors[topmost_component]
            # Convert hex to RGB
            color_rgb = [int(color_hex[i:i+2], 16)/255 for i in (1, 3, 5)]
            
            # Mark this grid spot as occupied with the topmost component
            board_grid[row, col] = color_rgb
            occupied_spots[(row, col)] = topmost_component
        
        # Display the board
        ax.imshow(board_grid, aspect='equal', origin='upper')
        
        # Add grid lines
        for i in range(self.rows + 1):
            ax.axhline(y=i-0.5, color='black', linewidth=1)
        for j in range(self.cols + 1):
            ax.axvline(x=j-0.5, color='black', linewidth=1)
        
        # Add component labels on occupied spots
        for (row, col), comp_type in occupied_spots.items():
            if comp_type in self.component_labels:
                label = self.component_labels[comp_type]
                ax.text(col, row, label, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Set axis properties
        ax.set_xlim(-0.5, self.cols-0.5)
        ax.set_ylim(self.rows-0.5, -0.5)
        ax.set_xlabel('Column', fontweight='bold')
        ax.set_ylabel('Row', fontweight='bold')
        
        # Add row and column numbers
        ax.set_xticks(range(self.cols))
        ax.set_xticklabels(range(1, self.cols+1))
        ax.set_yticks(range(self.rows))
        ax.set_yticklabels(range(1, self.rows+1))

    def _add_legend(self, fig, left_detections, right_detections):
        """Add component legend (excluding green tape)"""
        # Combine all detected components (excluding green tape)
        all_components = set()
        for detections in [left_detections, right_detections]:
            for comp_type in detections.keys():
                if comp_type != "Green tape":
                    all_components.add(comp_type)
        
        # Create legend
        legend_elements = []
        for comp_type in sorted(all_components):
            if comp_type in self.component_colors:
                color = self.component_colors[comp_type]
                label = self.component_labels.get(comp_type, comp_type[:6])
                
                # Count total occurrences
                left_count = len(left_detections.get(comp_type, []))
                right_count = len(right_detections.get(comp_type, []))
                total_count = left_count + right_count
                
                legend_elements.append(
                    patches.Patch(color=color, label=f'{label}: {total_count} ({left_count}L, {right_count}R)')
                )
        
        if legend_elements:
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, 0.02), 
                      loc='lower center', ncol=3, fontsize=10)

    def create_opencv_visualization(self, left_detections, right_detections):
        """
        Create OpenCV version for real-time display (simple 7x5 grids)
        
        Returns:
            numpy array image that can be displayed with cv2.imshow()
        """
        # Calculate image dimensions
        board_width = self.cols * self.cell_size   
        board_height = self.rows * self.cell_size  
        gap = 20  # Gap between boards
        total_width = board_width * 2 + gap
        total_height = board_height + 60  # Extra space for titles
        
        # Create blank image
        img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 240  # Light gray
        
        # Draw left board
        left_board = self._draw_opencv_board(left_detections, "LEFT")
        img[40:40+board_height, 0:board_width] = left_board
        
        # Draw right board
        right_board = self._draw_opencv_board(right_detections, "RIGHT")
        img[40:40+board_height, board_width+gap:board_width+gap+board_width] = right_board
        
        # Add titles
        cv2.putText(img, "LEFT BOARD", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "RIGHT BOARD", (board_width + gap + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return img

    def _draw_opencv_board(self, detections, side):
        """Draw single 7x5 board using OpenCV"""
        board_width = self.cols * self.cell_size   
        board_height = self.rows * self.cell_size  
        
        # Create board image
        board = np.ones((board_height, board_width, 3), dtype=np.uint8) * 245  # Light background
        
        # Draw grid lines
        for i in range(self.rows + 1):
            y = i * self.cell_size
            cv2.line(board, (0, y), (board_width, y), (128, 128, 128), 1)
        for j in range(self.cols + 1):
            x = j * self.cell_size
            cv2.line(board, (x, 0), (x, board_height), (128, 128, 128), 1)
        
        # Track which grid spots are occupied and their components
        grid_components = {}  # (row, col) -> [(component_type, confidence), ...]
        
        # Collect all components for each grid spot
        for comp_type, detection_list in detections.items():
            # Skip green tape since we don't visualize it anymore
            if comp_type == "Green tape" or comp_type not in self.component_colors:
                continue
            
            for detection in detection_list:
                # Get grid position
                if isinstance(detection, dict):
                    row = detection.get('row', 0)
                    col = detection.get('col', 0)
                    confidence = detection.get('confidence', 1.0)
                else:
                    # Fallback for simple format
                    row = 0
                    col = 0
                    confidence = 1.0
                
                # Ensure bounds
                row = max(0, min(row, self.rows - 1))
                col = max(0, min(col, self.cols - 1))
                
                # Add to grid components list
                grid_pos = (row, col)
                if grid_pos not in grid_components:
                    grid_components[grid_pos] = []
                grid_components[grid_pos].append((comp_type, confidence))
        
        # Fill grid spots with the topmost (highest confidence) component
        for (row, col), components in grid_components.items():
            # Sort by confidence (highest first) to get "topmost" component
            components.sort(key=lambda x: x[1], reverse=True)
            topmost_component = components[0][0]  # Get component type with highest confidence
            
            color_hex = self.component_colors[topmost_component]
            # Convert hex to BGR for OpenCV
            color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
            
            # Convert grid to pixel coordinates
            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            
            # Fill rectangle
            cv2.rectangle(board, (x1, y1), (x2, y2), color_bgr, -1)
            
            # Add label
            label = self.component_labels.get(topmost_component, topmost_component[:3])
            
            # Center text in cell
            center_x = x1 + self.cell_size // 2
            center_y = y1 + self.cell_size // 2
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            
            cv2.putText(board, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return board

def convert_simple_detections_to_grid(left_classes, right_classes):
    """
    Convert your simple detection format to grid-based format for visualization
    This is a fallback when real coordinates aren't available
    
    Args:
        left_classes: Dict like {'Component': count} from your detection
        right_classes: Dict like {'Component': count} from your detection
    """
    left_detections = {}
    right_detections = {}
    
    # Convert left side
    pos_counter = 0
    for comp_type, count in left_classes.items():
        left_detections[comp_type] = []
        for i in range(count):
            # Simple grid positioning for left side (columns 0-6)
            grid_x = pos_counter % 7
            grid_y = (pos_counter // 7) % 15
            
            left_detections[comp_type].append({
                'x1': grid_x, 'y1': grid_y,
                'x2': grid_x + 1, 'y2': grid_y + 1
            })
            pos_counter += 1
    
    # Convert right side  
    pos_counter = 0
    for comp_type, count in right_classes.items():
        right_detections[comp_type] = []
        for i in range(count):
            # Simple grid positioning for right side (columns 7-12)
            grid_x = 7 + (pos_counter % 6)
            grid_y = (pos_counter // 6) % 15
            
            right_detections[comp_type].append({
                'x1': grid_x, 'y1': grid_y,
                'x2': grid_x + 1, 'y2': grid_y + 1
            })
            pos_counter += 1
    
    return left_detections, right_detections

def convert_detections_to_7x5_grid(left_boxes, right_boxes, model_names, frame_width, frame_height, split_ratio=0.5):
    """
    Convert YOLO detection results to exact 7x5 grid positions for each board
    Uses calibrated pixel boundaries for precise mapping
    
    Args:
        left_boxes: List of (index, box) for left side detections
        right_boxes: List of (index, box) for right side detections  
        model_names: Model class names dict
        frame_width: Camera frame width
        frame_height: Camera frame height
        split_ratio: Split ratio for left/right division
    
    Returns:
        left_detections: Dict of {component_type: [{'row': int, 'col': int}, ...]}
        right_detections: Dict of {component_type: [{'row': int, 'col': int}, ...]}
    """
    left_detections = {}
    right_detections = {}
    
    # Calibrated board boundaries (from your calibration session)
    left_board_bounds = {
        'x1': 290, 'y1': 345, 'x2': 850, 'y2': 765
    }
    right_board_bounds = {
        'x1': 1095, 'y1': 325, 'x2': 1705, 'y2': 785
    }
    
    def pixel_to_grid_7x5(x, y, board_bounds):
        """Convert pixel coordinates to 7x5 grid coordinates using calibrated bounds"""
        # Check if point is within board bounds
        if not (board_bounds['x1'] <= x <= board_bounds['x2'] and 
                board_bounds['y1'] <= y <= board_bounds['y2']):
            return None, None
        
        # Convert to relative coordinates within board
        rel_x = (x - board_bounds['x1']) / (board_bounds['x2'] - board_bounds['x1'])
        rel_y = (y - board_bounds['y1']) / (board_bounds['y2'] - board_bounds['y1'])
        
        # Map to 7x5 grid: 7 cols x 5 rows
        grid_col = int(rel_x * 7)  # 0-6
        grid_row = int(rel_y * 5)  # 0-4
        
        # Clamp to bounds
        grid_col = max(0, min(grid_col, 6))
        grid_row = max(0, min(grid_row, 4))
        
        return grid_row, grid_col
    
    def get_component_grid_size(class_name):
        """Get the standard grid size for each component type"""
        # Format: (width_across, height_up_down)
        component_sizes = {
            'Wire': (2, 1),  # Can be 2x1 or 3x1, we'll use 2x1 as default
            'Battery Holder': (3, 3),
            'U_1 blue music circuit': (3, 2),
            'U_2 red alarm circuit': (3, 2),
            'U_3 green space war circuit': (3, 2),
            # Everything else: 3x1
            'LED_1 (Yellow)': (3, 1),
            'LED_2 (Red)': (3, 1),
            'Resistor': (3, 1),
            'Lamp': (3, 1),
            'Photoresistor': (3, 1),
            'Slide switch': (3, 1),
            'Press switch': (3, 1),
            'Speaker': (3, 1),
            'Whistle chip': (3, 1),
        }
        
        # Handle different wire sizes - check if it's a longer wire (basic heuristic)
        if class_name == 'Wire':
            return (2, 1)  # Default wire size, could be expanded to (3, 1) based on detection
        
        return component_sizes.get(class_name, (3, 1))  # Default to 3x1 for unknown components
    
    def get_covered_grid_spots_by_component(center_x, center_y, class_name, board_bounds, bbox=None):
        """Get grid spots covered by a component based on its standard size"""
        # Get center grid position
        result = pixel_to_grid_7x5(center_x, center_y, board_bounds)
        if result[0] is None:
            return []
        
        center_row, center_col = result
        width_across, height_up_down = get_component_grid_size(class_name)
        
        # Special handling for wires - determine orientation and size based on bounding box
        if class_name == 'Wire' and bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Determine if wire is horizontal or vertical
            is_horizontal = bbox_width > bbox_height
            aspect_ratio = max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1)
            
            if is_horizontal:
                # Wire is oriented horizontally - spans across columns
                if aspect_ratio > 2.5:  # Long horizontal wire
                    width_across = 3
                    height_up_down = 1
                else:  # Short horizontal wire
                    width_across = 2
                    height_up_down = 1
            else:
                # Wire is oriented vertically - spans across rows
                if aspect_ratio > 2.5:  # Long vertical wire
                    width_across = 1
                    height_up_down = 3
                else:  # Short vertical wire
                    width_across = 1
                    height_up_down = 2
        
        covered_spots = []
        
        # Calculate the span from center
        half_width = width_across // 2
        half_height = height_up_down // 2
        
        # For even widths/heights, we need to decide which side gets the extra space
        extra_width_right = width_across % 2 == 0
        extra_height_down = height_up_down % 2 == 0
        
        # Calculate the grid bounds for this component
        if extra_width_right:
            col_start = center_col - half_width + 1
            col_end = center_col + half_width
        else:
            col_start = center_col - half_width
            col_end = center_col + half_width
            
        if extra_height_down:
            row_start = center_row - half_height + 1
            row_end = center_row + half_height
        else:
            row_start = center_row - half_height
            row_end = center_row + half_height
        
        # Clamp to board boundaries (7 cols x 5 rows)
        col_start = max(0, col_start)
        col_end = min(6, col_end)
        row_start = max(0, row_start)
        row_end = min(4, row_end)
        
        # Generate all covered grid spots
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                covered_spots.append((row, col))
        
        return covered_spots
    
    # Process left side detections
    for i, box in left_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0])
        class_name = model_names[class_id]
        confidence = float(box.conf[0])
        
        # Skip green tape for visualization
        if class_name == "Green tape":
            continue
        
        # Get center point of bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Get grid spots covered by this component based on its standard size
        covered_spots = get_covered_grid_spots_by_component(center_x, center_y, class_name, left_board_bounds, [x1, y1, x2, y2])
        
        if covered_spots:
            if class_name not in left_detections:
                left_detections[class_name] = []
            
            # Add each covered grid spot as a separate detection entry
            for grid_row, grid_col in covered_spots:
                left_detections[class_name].append({
                    'row': grid_row,
                    'col': grid_col,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
    
    # Process right side detections
    for i, box in right_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0])
        class_name = model_names[class_id]
        confidence = float(box.conf[0])
        
        # Skip green tape for visualization
        if class_name == "Green tape":
            continue
        
        # Get center point of bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Get grid spots covered by this component based on its standard size
        covered_spots = get_covered_grid_spots_by_component(center_x, center_y, class_name, right_board_bounds, [x1, y1, x2, y2])
        
        if covered_spots:
            if class_name not in right_detections:
                right_detections[class_name] = []
            
            # Add each covered grid spot as a separate detection entry
            for grid_row, grid_col in covered_spots:
                right_detections[class_name].append({
                    'row': grid_row,
                    'col': grid_col,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
    
    return left_detections, right_detections

if __name__ == "__main__":
    # Demo usage
    visualizer = DualBoardVisualizer()
    
    # Sample data
    sample_left = {
        'U_2 red alarm circuit': [{}],
        'Slide switch': [{}],
        'Wire': [{}, {}]
    }
    
    sample_right = {
        'Battery Holder': [{}],
        'Lamp': [{}],
        'Wire': [{}]
    }
    
    # Create matplotlib visualization
    fig = visualizer.create_dual_board_visualization(sample_left, sample_right)
    plt.show()
    
    # Create OpenCV visualization
    opencv_img = visualizer.create_opencv_visualization(sample_left, sample_right)
    cv2.imshow('Dual Board', opencv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
