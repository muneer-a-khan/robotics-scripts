#!/usr/bin/env python3
"""
LED Orientation Detector
Detects the orientation of LED_2 (Red) components by finding the white '+' symbol
"""

import cv2
import numpy as np
from pathlib import Path

class LEDOrientationDetector:
    def __init__(self, template_folder="white_plus_folder"):
        """
        Initialize LED orientation detector with template images
        
        Args:
            template_folder: Path to folder containing white '+' template images
        """
        self.template_folder = Path(template_folder)
        self.templates = []
        self.load_templates()
    
    def load_templates(self):
        """Load all template images from the template folder"""
        template_files = list(self.template_folder.glob("*.png")) + list(self.template_folder.glob("*.jpg"))
        
        for template_file in template_files:
            template = cv2.imread(str(template_file))
            if template is not None:
                # Convert to grayscale for matching
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                self.templates.append({
                    'original': template,
                    'gray': template_gray,
                    'height': template.shape[0],
                    'width': template.shape[1],
                    'name': template_file.name
                })
                print(f"   Loaded template: {template_file.name} ({template.shape[1]}x{template.shape[0]})")
        
        if not self.templates:
            print(f"⚠️ Warning: No templates found in {self.template_folder}")
    
    def detect_orientation(self, frame, bbox, debug=False):
        """
        Detect LED orientation based on '+' symbol position
        
        Args:
            frame: Full image frame
            bbox: Bounding box [x1, y1, x2, y2] of the LED component
            debug: If True, save debug images
            
        Returns:
            dict with 'orientation' ('CORRECT', 'REVERSED', 'UNKNOWN'), 'confidence', and 'plus_position'
        """
        if not self.templates:
            return {
                'orientation': 'UNKNOWN',
                'confidence': 0.0,
                'reason': 'No templates loaded',
                'plus_position': None
            }
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Add some padding to ensure we capture the full LED
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        # Crop LED region
        led_region = frame[y1:y2, x1:x2]
        
        if led_region.size == 0:
            return {
                'orientation': 'UNKNOWN',
                'confidence': 0.0,
                'reason': 'Invalid bounding box',
                'plus_position': None
            }
        
        # Convert to grayscale
        led_gray = cv2.cvtColor(led_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate center of LED bounding box
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_center_x = bbox_width / 2
        bbox_center_y = bbox_height / 2
        
        # Determine if LED is horizontal or vertical based on aspect ratio
        is_horizontal = bbox_width > bbox_height
        
        best_match = None
        best_confidence = 0.0
        best_location = None
        best_template = None
        
        # Try all templates
        for template_data in self.templates:
            template_gray = template_data['gray']
            
            # Skip if template is larger than LED region
            if (template_gray.shape[0] > led_gray.shape[0] or 
                template_gray.shape[1] > led_gray.shape[1]):
                continue
            
            # Try multiple scales (including very small scales for far away LEDs)
            for scale in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                scaled_width = int(template_gray.shape[1] * scale)
                scaled_height = int(template_gray.shape[0] * scale)
                
                # Skip if too small
                if scaled_width < 5 or scaled_height < 5:
                    continue
                
                # Skip if scaled template is too large
                if (scaled_height > led_gray.shape[0] or 
                    scaled_width > led_gray.shape[1]):
                    continue
                
                scaled_template = cv2.resize(template_gray, (scaled_width, scaled_height))
                
                # Template matching
                result = cv2.matchTemplate(led_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_location = max_loc
                    best_match = scaled_template
                    best_template = template_data['name']
        
        # Confidence threshold for valid detection
        confidence_threshold = 0.5
        
        if best_confidence < confidence_threshold:
            return {
                'orientation': 'UNKNOWN',
                'confidence': float(best_confidence),
                'reason': f'Low confidence match ({best_confidence:.2f} < {confidence_threshold})',
                'plus_position': None,
                'led_orientation': 'unknown'
            }
        
        # Calculate '+' position relative to LED center
        plus_x = best_location[0] + (best_match.shape[1] / 2)
        plus_y = best_location[1] + (best_match.shape[0] / 2)
        
        # Determine orientation based on whether LED is horizontal or vertical
        if is_horizontal:
            # LED is sideways (horizontal)
            # CORRECT = '+' on right side, REVERSED = '+' on left side
            plus_position = 'RIGHT' if plus_x > bbox_center_x else 'LEFT'
            orientation = 'CORRECT' if plus_position == 'RIGHT' else 'REVERSED'
            led_orientation_type = 'horizontal'
        else:
            # LED is upright (vertical)
            # CORRECT = '+' on bottom half, REVERSED = '+' on top half
            plus_position = 'BOTTOM' if plus_y > bbox_center_y else 'TOP'
            orientation = 'CORRECT' if plus_position == 'BOTTOM' else 'REVERSED'
            led_orientation_type = 'vertical'
        
        result = {
            'orientation': orientation,
            'confidence': float(best_confidence),
            'plus_position': plus_position,
            'plus_x_relative': float(plus_x),
            'plus_y_relative': float(plus_y),
            'bbox_center_x': float(bbox_center_x),
            'bbox_center_y': float(bbox_center_y),
            'bbox_width': float(bbox_width),
            'bbox_height': float(bbox_height),
            'led_orientation': led_orientation_type,
            'template_used': best_template
        }
        
        # Debug visualization
        if debug:
            debug_img = led_region.copy()
            # Draw bounding box around matched area
            top_left = best_location
            bottom_right = (top_left[0] + best_match.shape[1], top_left[1] + best_match.shape[0])
            cv2.rectangle(debug_img, top_left, bottom_right, (0, 255, 0), 2)
            
            # Draw center lines based on LED orientation
            if is_horizontal:
                # Draw vertical center line for horizontal LED
                cv2.line(debug_img, (int(bbox_center_x), 0), (int(bbox_center_x), debug_img.shape[0]), 
                        (255, 0, 0), 1)
                # Add text indicating which side
                cv2.putText(debug_img, "LEFT", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(debug_img, "RIGHT", (debug_img.shape[1] - 50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Draw horizontal center line for vertical LED
                cv2.line(debug_img, (0, int(bbox_center_y)), (debug_img.shape[1], int(bbox_center_y)), 
                        (255, 0, 0), 1)
                # Add text indicating which side
                cv2.putText(debug_img, "TOP", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(debug_img, "BOTTOM", (5, debug_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save debug image
            debug_path = f"led_orientation_debug_{led_orientation_type}_{orientation}_{best_confidence:.2f}.jpg"
            cv2.imwrite(debug_path, debug_img)
            print(f"   Debug image saved: {debug_path}")
        
        return result

if __name__ == "__main__":
    # Test the detector
    detector = LEDOrientationDetector()
    print(f"Loaded {len(detector.templates)} templates")

