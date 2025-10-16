#!/usr/bin/env python3
"""
Simple Horn Orientation Detector - Grayscale Version
Uses simple grayscale matching like the LED detector (which works great)
"""

import cv2
import numpy as np
from pathlib import Path

class SimpleHornOrientationDetector:
    def __init__(self, template_folder="red_plus_folder"):
        """
        Initialize Horn orientation detector with template images
        Uses simple grayscale matching (proven to work for LED)
        """
        self.template_folder = Path(template_folder)
        self.templates = []
        self.load_templates()
    
    def load_templates(self):
        """Load all template images - simple grayscale conversion"""
        template_files = list(self.template_folder.glob("*.png")) + list(self.template_folder.glob("*.jpg"))
        
        for template_file in template_files:
            template = cv2.imread(str(template_file))
            if template is not None:
                # Simple grayscale conversion (like LED detector)
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                self.templates.append({
                    'original': template,
                    'gray': template_gray,
                    'height': template.shape[0],
                    'width': template.shape[1],
                    'name': template_file.name
                })
                print(f"   Loaded horn template (grayscale): {template_file.name} ({template.shape[1]}x{template.shape[0]})")
        
        if not self.templates:
            print(f"⚠️ Warning: No templates found in {self.template_folder}")
    
    def detect_orientation(self, frame, bbox, debug=True):
        """
        Detect Horn orientation - same logic as LED detector
        """
        if not self.templates:
            return {
                'orientation': 'UNKNOWN',
                'confidence': 0.0,
                'reason': 'No templates loaded',
                'plus_position': None
            }
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Add padding
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        # Crop region
        horn_region = frame[y1:y2, x1:x2]
        
        if horn_region.size == 0:
            return {
                'orientation': 'UNKNOWN',
                'confidence': 0.0,
                'reason': 'Invalid bounding box',
                'plus_position': None
            }
        
        # Simple grayscale conversion (like LED)
        horn_gray = cv2.cvtColor(horn_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate center
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_center_x = bbox_width / 2
        bbox_center_y = bbox_height / 2
        
        # Determine orientation
        is_horizontal = bbox_width > bbox_height
        
        best_match = None
        best_confidence = 0.0
        best_location = None
        best_template = None
        
        # Try all templates at multiple scales
        for template_data in self.templates:
            template_gray = template_data['gray']
            
            if (template_gray.shape[0] > horn_gray.shape[0] or 
                template_gray.shape[1] > horn_gray.shape[1]):
                continue
            
            # Try multiple scales - but avoid too small (prevents matching noise)
            for scale in [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                scaled_width = int(template_gray.shape[1] * scale)
                scaled_height = int(template_gray.shape[0] * scale)
                
                # Minimum size to avoid matching random noise
                if scaled_width < 15 or scaled_height < 15:
                    continue
                
                if (scaled_height > horn_gray.shape[0] or 
                    scaled_width > horn_gray.shape[1]):
                    continue
                
                scaled_template = cv2.resize(template_gray, (scaled_width, scaled_height))
                
                # Template matching
                result = cv2.matchTemplate(horn_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_location = max_loc
                    best_match = scaled_template
                    best_template = template_data['name']
        
        # Lower threshold for initial testing
        confidence_threshold = 0.3
        
        if best_confidence < confidence_threshold:
            return {
                'orientation': 'UNKNOWN',
                'confidence': float(best_confidence),
                'reason': f'Low confidence match ({best_confidence:.2f} < {confidence_threshold})',
                'plus_position': None,
                'horn_orientation': 'unknown'
            }
        
        # Calculate position
        plus_x = best_location[0] + (best_match.shape[1] / 2)
        plus_y = best_location[1] + (best_match.shape[0] / 2)
        
        # Debug: print position info
        if debug:
            print(f"   Horn detection debug:")
            print(f"     - Bbox dimensions: {bbox_width:.0f}x{bbox_height:.0f}")
            print(f"     - Bbox center: ({bbox_center_x:.0f}, {bbox_center_y:.0f})")
            print(f"     - Match location (top-left): ({best_location[0]}, {best_location[1]})")
            print(f"     - Template: {best_template}, size: {best_match.shape[1]}x{best_match.shape[0]}")
            print(f"     - Plus center calculated at: ({plus_x:.0f}, {plus_y:.0f})")
            print(f"     - Confidence: {best_confidence:.2f}")
            print(f"     - Orientation: {'HORIZONTAL' if is_horizontal else 'VERTICAL'}")
        
        # Same logic as LED
        if is_horizontal:
            plus_position = 'RIGHT' if plus_x > bbox_center_x else 'LEFT'
            orientation = 'CORRECT' if plus_position == 'RIGHT' else 'REVERSED'
            horn_orientation_type = 'horizontal'
            if debug:
                print(f"     - Plus is on {'RIGHT' if plus_x > bbox_center_x else 'LEFT'} side (plus_x={plus_x:.0f} vs center={bbox_center_x:.0f})")
        else:
            plus_position = 'BOTTOM' if plus_y > bbox_center_y else 'TOP'
            orientation = 'CORRECT' if plus_position == 'BOTTOM' else 'REVERSED'
            horn_orientation_type = 'vertical'
            if debug:
                print(f"     - Plus is on {'BOTTOM' if plus_y > bbox_center_y else 'TOP'} side (plus_y={plus_y:.0f} vs center={bbox_center_y:.0f})")
        
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
            'horn_orientation': horn_orientation_type,
            'template_used': best_template
        }
        
        # Debug
        if debug:
            debug_img = horn_region.copy()
            
            # Draw the matched template location
            top_left = best_location
            bottom_right = (top_left[0] + best_match.shape[1], top_left[1] + best_match.shape[0])
            cv2.rectangle(debug_img, top_left, bottom_right, (0, 255, 0), 2)
            
            # Draw center lines and labels
            if is_horizontal:
                # Vertical line at center for horizontal horn
                cv2.line(debug_img, (int(bbox_center_x), 0), (int(bbox_center_x), debug_img.shape[0]), (0, 0, 255), 2)
                cv2.putText(debug_img, "CENTER", (int(bbox_center_x)-30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Label left and right sides
                cv2.putText(debug_img, "LEFT", (10, debug_img.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(debug_img, "RIGHT", (debug_img.shape[1]-60, debug_img.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Horizontal line at center for vertical horn
                cv2.line(debug_img, (0, int(bbox_center_y)), (debug_img.shape[1], int(bbox_center_y)), (0, 0, 255), 2)
                cv2.putText(debug_img, "CENTER", (10, int(bbox_center_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Label top and bottom
                cv2.putText(debug_img, "TOP", (debug_img.shape[1]//2-20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(debug_img, "BOTTOM", (debug_img.shape[1]//2-40, debug_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mark the detected plus center with a big circle
            cv2.circle(debug_img, (int(plus_x), int(plus_y)), 8, (255, 0, 0), -1)
            cv2.putText(debug_img, "PLUS", (int(plus_x)+15, int(plus_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            debug_path = f"horn_simple_debug_{horn_orientation_type}_{orientation}_{best_confidence:.2f}.jpg"
            cv2.imwrite(debug_path, debug_img)
            print(f"   Horn debug saved: {debug_path}")
        
        return result

