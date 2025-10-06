#!/usr/bin/env python3
"""
7x5 Grid Calibration Tool - Interactive Version

This tool lets you drag the board overlays to match your physical boards exactly.
- Drag the board outlines to position them over your real boards
- The grids are rotated to match your setup
- Get exact pixel coordinates for perfect 1:1 mapping
"""

import cv2
import numpy as np
from pathlib import Path

class GridCalibrator:
    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480
        self.split_ratio = 0.5
        
        # Grid parameters - wider to fully cover 7 columns
        self.left_board_bounds = {
            'x1': 10,   # Left edge of left board (pixels) - even wider
            'y1': 80,   # Top edge of left board (pixels)
            'x2': 350,  # Right edge of left board (pixels) - much wider
            'y2': 280   # Bottom edge of left board (pixels)
        }
        
        self.right_board_bounds = {
            'x1': 325,  # Left edge of right board (pixels) - moved closer
            'y1': 80,   # Top edge of right board (pixels)
            'x2': 675,  # Right edge of right board (pixels) - much wider
            'y2': 280   # Bottom edge of right board (pixels)
        }
        
        # Control state
        self.selected_board = 'left'  # 'left' or 'right'
        self.move_step = 15  # Pixels to move per keypress
        self.resize_step = 10  # Pixels to resize per keypress
        
        # Rotation state (90 degrees clockwise from original)
        self.rotated = True
    
    def get_board_center(self, bounds):
        """Get center point of a board"""
        center_x = (bounds['x1'] + bounds['x2']) // 2
        center_y = (bounds['y1'] + bounds['y2']) // 2
        return center_x, center_y
    
    def is_point_in_board(self, x, y, bounds):
        """Check if point is inside board bounds"""
        return (bounds['x1'] <= x <= bounds['x2'] and 
                bounds['y1'] <= y <= bounds['y2'])
    
    def draw_rotated_grid_overlay(self, frame):
        """Draw 7x5 grid overlay on both boards (rotated for your setup)"""
        height, width = frame.shape[:2]
        self.frame_width = width
        self.frame_height = height
        
        # Draw split line
        split_x = int(width * self.split_ratio)
        cv2.line(frame, (split_x, 0), (split_x, height), (255, 255, 255), 2)
        
        # Draw left board grid (7 cols x 5 rows for display)
        self.draw_single_rotated_board(frame, self.left_board_bounds, "LEFT", (0, 255, 0))
        
        # Draw right board grid (7 cols x 5 rows for display)  
        self.draw_single_rotated_board(frame, self.right_board_bounds, "RIGHT", (0, 255, 255))
        
        return frame
    
    def draw_single_rotated_board(self, frame, bounds, side, color):
        """Draw a single board (7 cols x 5 rows for visual display)"""
        board_width = bounds['x2'] - bounds['x1']
        board_height = bounds['y2'] - bounds['y1']
        
        # Highlight selected board with thicker outline and brighter color
        is_selected = side.lower() == self.selected_board
        thickness = 4 if is_selected else 2
        display_color = color if not is_selected else tuple(min(255, c + 50) for c in color)
        
        # Draw board outline
        cv2.rectangle(frame, (bounds['x1'], bounds['y1']), 
                     (bounds['x2'], bounds['y2']), display_color, thickness)
        
        # Now showing: 7 columns (horizontal) x 5 rows (vertical)
        display_cols = 7
        display_rows = 5
        
        # Draw grid lines - COLUMNS (vertical lines)
        for col in range(display_cols + 1):  # 8 lines for 7 columns
            x = bounds['x1'] + int((col / display_cols) * board_width)
            cv2.line(frame, (x, bounds['y1']), (x, bounds['y2']), display_color, 1)
        
        # Draw grid lines - ROWS (horizontal lines)
        for row in range(display_rows + 1):  # 6 lines for 5 rows
            y = bounds['y1'] + int((row / display_rows) * board_height)
            cv2.line(frame, (bounds['x1'], y), (bounds['x2'], y), display_color, 1)
        
        # Add grid labels for the rotated view
        self.add_rotated_grid_labels(frame, bounds, side, display_color, display_cols, display_rows)
        
        # Add selection indicator
        if is_selected:
            cv2.putText(frame, "SELECTED", (bounds['x1'], bounds['y1'] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def add_rotated_grid_labels(self, frame, bounds, side, color, cols, rows):
        """Add row/column labels for grid"""
        board_width = bounds['x2'] - bounds['x1']
        board_height = bounds['y2'] - bounds['y1']
        
        # Column labels (1-7) - horizontal
        for col in range(cols):
            x = bounds['x1'] + int((col + 0.5) / cols * board_width)
            y = bounds['y1'] - 15
            cv2.putText(frame, str(col + 1), (x - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Row labels (1-5) - vertical
        label_x = bounds['x1'] - 25 if side == "LEFT" else bounds['x2'] + 10
        for row in range(rows):
            y = bounds['y1'] + int((row + 0.5) / rows * board_height)
            cv2.putText(frame, str(row + 1), (label_x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add title
        title_y = bounds['y1'] - 35
        title = f"{side} BOARD (7x5)"
        cv2.putText(frame, title, (bounds['x1'], title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add center marker
        center_x, center_y = self.get_board_center(bounds)
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
    
    def pixel_to_rotated_grid(self, x, y):
        """Convert pixel coordinates to 7x5 grid coordinates"""
        # Check which board the point is in
        if self.is_point_in_board(x, y, self.left_board_bounds):
            bounds = self.left_board_bounds
            side = 'LEFT'
        elif self.is_point_in_board(x, y, self.right_board_bounds):
            bounds = self.right_board_bounds
            side = 'RIGHT'
        else:
            return None, None, None
        
        # Convert to relative coordinates within board
        rel_x = (x - bounds['x1']) / (bounds['x2'] - bounds['x1'])
        rel_y = (y - bounds['y1']) / (bounds['y2'] - bounds['y1'])
        
        # For 7x5 grid: 7 cols x 5 rows
        grid_col = int(rel_x * 7)  # 0-6
        grid_row = int(rel_y * 5)  # 0-4
        
        # Clamp to bounds
        grid_col = max(0, min(grid_col, 6))
        grid_row = max(0, min(grid_row, 4))
        
        return side, grid_row, grid_col, grid_row, grid_col
    
    def move_board(self, direction):
        """Move the selected board in the given direction"""
        bounds = self.left_board_bounds if self.selected_board == 'left' else self.right_board_bounds
        
        if direction == 'up':
            bounds['y1'] -= self.move_step
            bounds['y2'] -= self.move_step
        elif direction == 'down':
            bounds['y1'] += self.move_step
            bounds['y2'] += self.move_step
        elif direction == 'left':
            bounds['x1'] -= self.move_step
            bounds['x2'] -= self.move_step
        elif direction == 'right':
            bounds['x1'] += self.move_step
            bounds['x2'] += self.move_step
        
        # Ensure bounds stay within frame
        bounds['x1'] = max(0, bounds['x1'])
        bounds['y1'] = max(0, bounds['y1'])
        bounds['x2'] = min(self.frame_width, bounds['x2'])
        bounds['y2'] = min(self.frame_height, bounds['y2'])
        
        print(f"Moved {self.selected_board.upper()} board {direction}")
    
    def resize_board(self, direction):
        """Resize the selected board"""
        bounds = self.left_board_bounds if self.selected_board == 'left' else self.right_board_bounds
        
        if direction == 'bigger':
            bounds['x1'] -= self.resize_step
            bounds['y1'] -= self.resize_step
            bounds['x2'] += self.resize_step
            bounds['y2'] += self.resize_step
        elif direction == 'smaller':
            bounds['x1'] += self.resize_step
            bounds['y1'] += self.resize_step
            bounds['x2'] -= self.resize_step
            bounds['y2'] -= self.resize_step
        elif direction == 'wider':
            bounds['x1'] -= self.resize_step
            bounds['x2'] += self.resize_step
        elif direction == 'narrower':
            bounds['x1'] += self.resize_step
            bounds['x2'] -= self.resize_step
        elif direction == 'taller':
            bounds['y1'] -= self.resize_step
            bounds['y2'] += self.resize_step
        elif direction == 'shorter':
            bounds['y1'] += self.resize_step
            bounds['y2'] -= self.resize_step
        
        # Ensure minimum size
        if bounds['x2'] - bounds['x1'] < 50:
            bounds['x2'] = bounds['x1'] + 50
        if bounds['y2'] - bounds['y1'] < 50:
            bounds['y2'] = bounds['y1'] + 50
            
        # Ensure bounds stay within frame
        bounds['x1'] = max(0, bounds['x1'])
        bounds['y1'] = max(0, bounds['y1'])
        bounds['x2'] = min(self.frame_width, bounds['x2'])
        bounds['y2'] = min(self.frame_height, bounds['y2'])
        
        print(f"Resized {self.selected_board.upper()} board {direction}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to show grid coordinates"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Click to show grid coordinates
            result = self.pixel_to_rotated_grid(x, y)
            if result[0]:
                side, grid_row, grid_col, _, _ = result
                print(f"{side} Board:")
                print(f"  Grid: Row {grid_row + 1}, Col {grid_col + 1} [Array: {grid_row}, {grid_col}]")
                print(f"  Pixel: ({x}, {y})")
            else:
                print(f"Click outside board area: ({x}, {y})")
    
    def save_calibration(self):
        """Save the calibrated coordinates"""
        calib_data = {
            'left_board': self.left_board_bounds.copy(),
            'right_board': self.right_board_bounds.copy(),
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'split_ratio': self.split_ratio
        }
        
        print("\n📋 CALIBRATION RESULTS:")
        print("=" * 40)
        print("Left Board Bounds:")
        print(f"  x1={calib_data['left_board']['x1']}, y1={calib_data['left_board']['y1']}")
        print(f"  x2={calib_data['left_board']['x2']}, y2={calib_data['left_board']['y2']}")
        print()
        print("Right Board Bounds:")
        print(f"  x1={calib_data['right_board']['x1']}, y1={calib_data['right_board']['y1']}")
        print(f"  x2={calib_data['right_board']['x2']}, y2={calib_data['right_board']['y2']}")
        print()
        print("Copy these values into your convert_detections_to_7x5_grid function!")
        
        return calib_data
    
    def run_calibration(self):
        """Run the interactive calibration tool"""
        print("🎯 INTERACTIVE 7x5 GRID CALIBRATION TOOL")
        print("=" * 45)
        print()
        print("Instructions:")  
        print("• Use keyboard controls to position the GREEN and CYAN board outlines")
        print("• The grids show your board orientation: 7 columns x 5 rows")
        print("• Click inside boards to see grid coordinates")
        print("• The SELECTED board is highlighted and can be moved/resized")
        print()
        print("Keyboard Controls:")
        print("SELECTION:")
        print("• TAB: Switch between LEFT and RIGHT board")
        print()
        print("MOVEMENT (WASD):")
        print("• W: Move selected board UP")
        print("• A: Move selected board LEFT")
        print("• S: Move selected board DOWN") 
        print("• D: Move selected board RIGHT")
        print()
        print("RESIZING:")
        print("• +/=: Make board BIGGER")
        print("• -: Make board SMALLER")
        print("• ↑: Make board TALLER")
        print("• ↓: Make board SHORTER")
        print("• →: Make board WIDER")
        print("• ←: Make board NARROWER")
        print()
        print("OTHER:")
        print("• CLICK in board: Show grid coordinates")
        print("• 'c': Save calibration and show results")
        print("• 'r': Reset boards to default positions")
        print("• 's': Save current frame")
        print("• 'q': Quit")
        print()
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        cv2.namedWindow('Keyboard-Controlled Grid Calibration')
        cv2.setMouseCallback('Keyboard-Controlled Grid Calibration', self.mouse_callback)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update frame dimensions
                height, width = frame.shape[:2]
                self.frame_width = width
                self.frame_height = height
                
                # Draw rotated grid overlay
                frame_with_grid = self.draw_rotated_grid_overlay(frame.copy())
                
                # Add control instructions
                cv2.putText(frame_with_grid, f"Selected: {self.selected_board.upper()} | TAB to switch | WASD to move | +/- to resize", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame_with_grid, "Arrow keys: resize specific dimensions | 'c' to save calibration", 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imshow('Keyboard-Controlled Grid Calibration', frame_with_grid)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Handle keyboard controls
                if key == ord('q'):
                    break
                elif key == 9:  # TAB key
                    self.selected_board = 'right' if self.selected_board == 'left' else 'left'
                    print(f"Selected {self.selected_board.upper()} board")
                
                # Movement controls (WASD)
                elif key == ord('w'):
                    self.move_board('up')
                elif key == ord('a'):
                    self.move_board('left')
                elif key == ord('s'):
                    self.move_board('down')
                elif key == ord('d'):
                    self.move_board('right')
                
                # Resize controls
                elif key == ord('=') or key == ord('+'):
                    self.resize_board('bigger')
                elif key == ord('-'):
                    self.resize_board('smaller')
                
                # Arrow keys for specific dimension resizing
                elif key == 82:  # Up arrow
                    self.resize_board('taller')
                elif key == 84:  # Down arrow
                    self.resize_board('shorter')
                elif key == 83:  # Right arrow
                    self.resize_board('wider')
                elif key == 81:  # Left arrow
                    self.resize_board('narrower')
                
                # Other controls
                elif key == ord('c'):
                    # Save calibration
                    self.save_calibration()
                elif key == ord('r'):
                    # Reset to defaults
                    self.__init__()
                    print("🔄 Reset boards to default positions")
                elif key == ord('s'):
                    cv2.imwrite('keyboard_calibration_frame.jpg', frame_with_grid)
                    print("📸 Calibration frame saved: keyboard_calibration_frame.jpg")
        
        except KeyboardInterrupt:
            print("\n🛑 Calibration interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Calibration complete")

def main():
    calibrator = GridCalibrator()
    calibrator.run_calibration()

if __name__ == "__main__":
    main()
