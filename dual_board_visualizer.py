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
    def __init__(self, rows=13, cols=15, cell_size=30):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        
        # Simple, clear colors for components
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
            'Whistle chip': '#DDA0DD',   # Plum
            'Green tape': '#90EE90'      # Light Green
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
            'Whistle chip': 'WHI',
            'Green tape': 'TAPE'
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
        """Draw a single board with components (rotated 90 degrees clockwise)"""
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Create empty board grid (rotated: cols become rows, rows become cols)
        board_grid = np.ones((self.cols, self.rows, 3)) * 0.95  # Light gray background
        
        # Track which cells are occupied
        occupied_cells = set()
        
        # Fill in detected components
        component_count = 0
        for comp_type, detection_list in detections.items():
            if comp_type not in self.component_colors:
                continue
                
            color_hex = self.component_colors[comp_type]
            # Convert hex to RGB
            color_rgb = [int(color_hex[i:i+2], 16)/255 for i in (1, 3, 5)]
            
            for detection in detection_list:
                # Get grid position
                if isinstance(detection, dict):
                    # Use actual grid coordinates if available
                    orig_x1 = detection.get('x1', component_count % self.rows)
                    orig_y1 = detection.get('y1', component_count % self.cols)
                    orig_x2 = detection.get('x2', orig_x1 + 1)
                    orig_y2 = detection.get('y2', orig_y1 + 1)
                    
                    # Make wires extend 2 boxes horizontally (parallel to tape)
                    if comp_type == "Wire":
                        orig_y2 = min(orig_y1 + 2, self.cols)  # Extend by 2 in y direction (horizontal)
                    # Make tape extend full width of board
                    elif comp_type == "Green tape":
                        orig_y1 = 0  # Start from left edge
                        orig_y2 = self.cols  # Extend to right edge
                else:
                    # Simple positioning for demo
                    orig_x1 = component_count % self.rows
                    orig_y1 = (component_count // self.rows) % self.cols
                    orig_x2 = orig_x1 + 1
                    orig_y2 = orig_y1 + 1
                    
                    # Make wires extend 2 boxes horizontally (parallel to tape)
                    if comp_type == "Wire":
                        orig_y2 = min(orig_y1 + 2, self.cols)  # Extend by 2 in y direction (horizontal)
                    # Make tape extend full width of board
                    elif comp_type == "Green tape":
                        orig_y1 = 0  # Start from left edge
                        orig_y2 = self.cols  # Extend to right edge
                
                # Rotate coordinates 90 degrees clockwise: (x,y) -> (y, rows-1-x)
                # But we need to be careful about the mapping
                x1 = orig_y1  # New x = old y
                y1 = self.rows - 1 - orig_x2 + 1  # New y = rows - 1 - old x (flipped)
                x2 = orig_y2
                y2 = self.rows - 1 - orig_x1 + 1
                
                # Ensure bounds for rotated grid (now cols x rows)
                x1 = max(0, min(x1, self.cols-1))
                y1 = max(0, min(y1, self.rows-1))
                x2 = max(x1+1, min(x2, self.cols))
                y2 = max(y1+1, min(y2, self.rows))
                
                # Fill the grid cells
                for x in range(x1, x2):
                    for y in range(y1, y2):
                        if 0 <= x < self.cols and 0 <= y < self.rows:
                            board_grid[x, y] = color_rgb
                            occupied_cells.add((x, y))
                
                component_count += 1
        
        # Display the rotated board
        ax.imshow(board_grid, aspect='equal', origin='upper')
        
        # Add grid lines (adjusted for rotated dimensions)
        for i in range(self.cols + 1):
            ax.axhline(y=i-0.5, color='black', linewidth=0.5)
        for j in range(self.rows + 1):
            ax.axvline(x=j-0.5, color='black', linewidth=0.5)
        
        # Add component labels on occupied cells
        label_count = defaultdict(int)
        for comp_type, detection_list in detections.items():
            if comp_type not in self.component_labels:
                continue
                
            label = self.component_labels[comp_type]
            
            for i, detection in enumerate(detection_list):
                # Get original position
                if isinstance(detection, dict):
                    orig_x1 = detection.get('x1', i % self.rows)
                    orig_y1 = detection.get('y1', i % self.cols)
                else:
                    orig_x1 = i % self.rows
                    orig_y1 = (i // self.rows) % self.cols
                
                # Rotate coordinates for label placement
                label_x = orig_y1
                label_y = self.rows - 1 - orig_x1
                
                # Ensure bounds
                label_x = max(0, min(label_x, self.cols-1))
                label_y = max(0, min(label_y, self.rows-1))
                
                # Add label
                display_label = f"{label}"
                if len(detection_list) > 1:
                    display_label = f"{label}{i+1}"
                
                ax.text(label_y, label_x, display_label, ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Set axis properties (swapped for rotation)
        ax.set_xlim(-0.5, self.rows-0.5)
        ax.set_ylim(self.cols-0.5, -0.5)
        ax.set_xlabel('Column', fontweight='bold')
        ax.set_ylabel('Row', fontweight='bold')
        
        # Add row and column numbers (adjusted for rotation)
        ax.set_xticks(range(self.rows))
        ax.set_xticklabels(range(1, self.rows+1))
        ax.set_yticks(range(self.cols))
        ax.set_yticklabels(range(1, self.cols+1))

    def _add_legend(self, fig, left_detections, right_detections):
        """Add component legend"""
        # Combine all detected components
        all_components = set()
        for detections in [left_detections, right_detections]:
            all_components.update(detections.keys())
        
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
        Create OpenCV version for real-time display (rotated layout)
        
        Returns:
            numpy array image that can be displayed with cv2.imshow()
        """
        # Calculate image dimensions (swapped for rotation)
        board_width = self.rows * self.cell_size   # Now using rows for width
        board_height = self.cols * self.cell_size  # Now using cols for height  
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
        """Draw single board using OpenCV (rotated layout)"""
        board_width = self.rows * self.cell_size   # Swapped for rotation
        board_height = self.cols * self.cell_size  # Swapped for rotation
        
        # Create board image
        board = np.ones((board_height, board_width, 3), dtype=np.uint8) * 245  # Light background
        
        # Draw grid lines (adjusted for rotation)
        for i in range(self.cols + 1):  # Now using cols for vertical lines
            y = i * self.cell_size
            cv2.line(board, (0, y), (board_width, y), (128, 128, 128), 1)
        for j in range(self.rows + 1):  # Now using rows for horizontal lines
            x = j * self.cell_size
            cv2.line(board, (x, 0), (x, board_height), (128, 128, 128), 1)
        
        # Fill in components
        component_count = 0
        for comp_type, detection_list in detections.items():
            if comp_type not in self.component_colors:
                continue
                
            color_hex = self.component_colors[comp_type]
            # Convert hex to BGR for OpenCV
            color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
            
            for i, detection in enumerate(detection_list):
                # Get original grid position
                if isinstance(detection, dict):
                    orig_x1 = detection.get('x1', component_count % self.rows)
                    orig_y1 = detection.get('y1', component_count % self.cols)
                    orig_x2 = detection.get('x2', orig_x1 + 1)
                    orig_y2 = detection.get('y2', orig_y1 + 1)
                    
                    # Make wires extend 2 boxes horizontally (parallel to tape)
                    if comp_type == "Wire":
                        orig_y2 = min(orig_y1 + 2, self.cols)  # Extend by 2 in y direction (horizontal)
                    # Make tape extend full width of board
                    elif comp_type == "Green tape":
                        orig_y1 = 0  # Start from left edge
                        orig_y2 = self.cols  # Extend to right edge
                else:
                    orig_x1 = component_count % self.rows
                    orig_y1 = (component_count // self.rows) % self.cols
                    orig_x2 = orig_x1 + 1
                    orig_y2 = orig_y1 + 1
                    
                    # Make wires extend 2 boxes horizontally (parallel to tape)
                    if comp_type == "Wire":
                        orig_y2 = min(orig_y1 + 2, self.cols)  # Extend by 2 in y direction (horizontal)
                    # Make tape extend full width of board
                    elif comp_type == "Green tape":
                        orig_y1 = 0  # Start from left edge
                        orig_y2 = self.cols  # Extend to right edge
                
                # Rotate coordinates 90 degrees clockwise for display
                x1 = orig_y1  # New x = old y
                y1 = self.rows - 1 - orig_x2 + 1  # New y = rows - 1 - old x (flipped)
                x2 = orig_y2
                y2 = self.rows - 1 - orig_x1 + 1
                
                # Ensure bounds for rotated grid (now cols x rows)
                x1 = max(0, min(x1, self.cols-1))
                y1 = max(0, min(y1, self.rows-1))
                x2 = max(x1+1, min(x2, self.cols))
                y2 = max(y1+1, min(y2, self.rows))
                
                # Convert grid to pixel coordinates (adjusted for rotation)
                px1 = y1 * self.cell_size  # Using y for x pixel coordinate 
                py1 = x1 * self.cell_size  # Using x for y pixel coordinate
                px2 = y2 * self.cell_size
                py2 = x2 * self.cell_size
                
                # Fill rectangle
                cv2.rectangle(board, (px1, py1), (px2, py2), color_bgr, -1)
                
                # Add label
                label = self.component_labels.get(comp_type, comp_type[:3])
                if len(detection_list) > 1:
                    label = f"{label}{i+1}"
                
                # Center text in cell
                center_x = px1 + (px2 - px1) // 2
                center_y = py1 + (py2 - py1) // 2
                
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                
                cv2.putText(board, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                component_count += 1
        
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

def convert_detections_with_positions(left_boxes, right_boxes, model_names, frame_width, frame_height, split_ratio=0.5):
    """
    Convert YOLO detection results to grid positions for accurate board visualization
    (Results will be used with rotated board layout)
    
    Args:
        left_boxes: List of (index, box) for left side detections
        right_boxes: List of (index, box) for right side detections  
        model_names: Model class names dict
        frame_width: Camera frame width
        frame_height: Camera frame height
        split_ratio: Split ratio for left/right division
    """
    left_detections = {}
    right_detections = {}
    
    def pixel_to_grid(x, y, side):
        """Convert pixel coordinates to grid coordinates (before rotation)"""
        if side == 'left':
            # Left side: map to original grid coordinates (13 rows x 15 cols)
            grid_x = int((y / frame_height) * 13)  # y maps to rows
            grid_y = int((x / (frame_width * split_ratio)) * 7)  # x maps to partial cols
            grid_x = max(0, min(grid_x, 12))
            grid_y = max(0, min(grid_y, 6))
        else:
            # Right side: map to original grid coordinates
            relative_x = x - (frame_width * split_ratio)
            grid_x = int((y / frame_height) * 13)  # y maps to rows
            grid_y = 7 + int((relative_x / (frame_width * (1 - split_ratio))) * 8)  # x maps to remaining cols
            grid_x = max(0, min(grid_x, 12))
            grid_y = max(7, min(grid_y, 14))
        
        return grid_x, grid_y
    
    # Process left side detections
    for i, box in left_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0])
        class_name = model_names[class_id]
        confidence = float(box.conf[0])
        
        # Convert center point to grid coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        grid_x, grid_y = pixel_to_grid(center_x, center_y, 'left')
        
        if class_name not in left_detections:
            left_detections[class_name] = []
        
        # Make wires extend 2 boxes horizontally (parallel to tape)
        if class_name == "Wire":
            y2_extent = min(grid_y + 2, 14)  # Extend by 2 horizontally for wires
            x2_extent = grid_x + 1
        # Make tape extend full width
        elif class_name == "Green tape":
            y2_extent = 14  # Full width
            grid_y = 0  # Start from edge
            x2_extent = grid_x + 1
        else:
            y2_extent = grid_y + 1
            x2_extent = grid_x + 1
            
        left_detections[class_name].append({
            'x1': grid_x, 'y1': grid_y,
            'x2': x2_extent, 'y2': y2_extent,
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2]
        })
    
    # Process right side detections
    for i, box in right_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0])
        class_name = model_names[class_id]
        confidence = float(box.conf[0])
        
        # Convert center point to grid coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        grid_x, grid_y = pixel_to_grid(center_x, center_y, 'right')
        
        if class_name not in right_detections:
            right_detections[class_name] = []
        
        # Make wires extend 2 boxes horizontally (parallel to tape)
        if class_name == "Wire":
            y2_extent = min(grid_y + 2, 14)  # Extend by 2 horizontally for wires
            x2_extent = grid_x + 1
        # Make tape extend full width
        elif class_name == "Green tape":
            y2_extent = 14  # Full width  
            grid_y = 0  # Start from edge
            x2_extent = grid_x + 1
        else:
            y2_extent = grid_y + 1
            x2_extent = grid_x + 1
            
        right_detections[class_name].append({
            'x1': grid_x, 'y1': grid_y,
            'x2': x2_extent, 'y2': y2_extent,
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
