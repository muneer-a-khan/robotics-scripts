#!/usr/bin/env python3
"""
Simple manual annotation helper for YOLO format.
Since LabelImg has compatibility issues, this helps create annotations manually.
"""

import cv2
import numpy as np
from pathlib import Path
import json

class SimpleAnnotator:
    def __init__(self, image_path, classes_file):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.original_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        
        # Load classes
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.annotations = []
        self.current_class = 0
        self.drawing = False
        self.start_point = None
        
        print(f"Loaded image: {image_path}")
        print(f"Image size: {self.width}x{self.height}")
        print(f"Classes: {self.classes}")
        print(f"Current class: {self.classes[self.current_class]}")
        print("\nControls:")
        print("- Click and drag to draw bounding box")
        print("- Press number keys (0-9) to change class")
        print("- Press 's' to save annotations")
        print("- Press 'r' to reset current image")
        print("- Press 'q' to quit")
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_image = self.image.copy()
                cv2.rectangle(temp_image, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.putText(temp_image, self.classes[self.current_class], 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('Annotator', temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                self.add_annotation(self.start_point, end_point)
                self.draw_annotations()
                
    def add_annotation(self, start, end):
        # Convert to YOLO format (normalized center coordinates)
        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
        x2, y2 = max(start[0], end[0]), max(start[1], end[1])
        
        center_x = (x1 + x2) / 2 / self.width
        center_y = (y1 + y2) / 2 / self.height
        bbox_width = (x2 - x1) / self.width
        bbox_height = (y2 - y1) / self.height
        
        annotation = {
            'class_id': self.current_class,
            'class_name': self.classes[self.current_class],
            'center_x': center_x,
            'center_y': center_y,
            'width': bbox_width,
            'height': bbox_height
        }
        
        self.annotations.append(annotation)
        print(f"Added annotation: {self.classes[self.current_class]} at ({center_x:.3f}, {center_y:.3f})")
        
    def draw_annotations(self):
        self.image = self.original_image.copy()
        for i, ann in enumerate(self.annotations):
            # Convert back to pixel coordinates for display
            center_x = ann['center_x'] * self.width
            center_y = ann['center_y'] * self.height
            bbox_width = ann['width'] * self.width
            bbox_height = ann['height'] * self.height
            
            x1 = int(center_x - bbox_width / 2)
            y1 = int(center_y - bbox_height / 2)
            x2 = int(center_x + bbox_width / 2)
            y2 = int(center_y + bbox_height / 2)
            
            # Different colors for different classes
            color = ((i * 80) % 255, (i * 120) % 255, (i * 160) % 255)
            cv2.rectangle(self.image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.image, f"{ann['class_name']} ({i})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow('Annotator', self.image)
        
    def save_annotations(self):
        output_path = Path(self.image_path).with_suffix('.txt')
        output_path = Path('data/training/labels/train') / output_path.name
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for ann in self.annotations:
                line = f"{ann['class_id']} {ann['center_x']:.6f} {ann['center_y']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
                f.write(line + '\n')
        
        print(f"Saved {len(self.annotations)} annotations to {output_path}")
        
    def run(self):
        cv2.namedWindow('Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Annotator', self.mouse_callback)
        cv2.imshow('Annotator', self.image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Number keys to change class
            if key >= ord('0') and key <= ord('9'):
                class_idx = key - ord('0')
                if class_idx < len(self.classes):
                    self.current_class = class_idx
                    print(f"Current class: {self.classes[self.current_class]}")
                    
            elif key == ord('s'):
                self.save_annotations()
                
            elif key == ord('r'):
                self.annotations = []
                self.image = self.original_image.copy()
                cv2.imshow('Annotator', self.image)
                print("Reset annotations")
                
            elif key == ord('q'):
                break
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python manual_annotate.py <image_path> <classes_file>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    classes_file = sys.argv[2]
    
    annotator = SimpleAnnotator(image_path, classes_file)
    annotator.run() 