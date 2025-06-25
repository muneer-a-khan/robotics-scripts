#!/usr/bin/env python3
"""
Batch annotation tool for multiple circuit images.
Helps you efficiently annotate all circuit images with proper component-level annotations.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from manual_annotate import SimpleAnnotator

class BatchAnnotator:
    def __init__(self, image_directory, classes_file):
        self.image_directory = Path(image_directory)
        self.classes_file = classes_file
        
        # Find all circuit images
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(self.image_directory.glob(ext))
        
        print(f"📸 Found {len(self.image_files)} images to annotate:")
        for i, img_path in enumerate(self.image_files):
            print(f"   {i+1}. {img_path.name}")
        
        self.current_index = 0
        
    def annotate_all(self):
        """Annotate all images in sequence."""
        print(f"\n🎯 BATCH ANNOTATION WORKFLOW")
        print("="*50)
        print("For each image, follow the ANNOTATION GUIDELINES:")
        print("• Label COMPLETE components, not individual parts")
        print("• battery_holder: ONE box around entire battery compartment")
        print("• switch: ONE box around complete switch assembly")  
        print("• connection_node: ONE large box covering wire segments")
        print("• Press 's' to save and move to next image")
        print("• Press 'q' to quit batch process")
        print("="*50)
        
        while self.current_index < len(self.image_files):
            current_image = self.image_files[self.current_index]
            
            print(f"\n📸 Annotating image {self.current_index + 1}/{len(self.image_files)}: {current_image.name}")
            
            # Check if already annotated
            annotation_path = Path('data/training/labels/train') / (current_image.stem + '.txt')
            if annotation_path.exists():
                print(f"   ⚠️  Annotation already exists: {annotation_path}")
                response = input("   Continue anyway? (y/n/skip): ").lower()
                if response == 'n':
                    return
                elif response == 'skip':
                    self.current_index += 1
                    continue
            
            # Annotate current image
            annotator = SimpleAnnotator(str(current_image), self.classes_file)
            annotator.run()
            
            # Check if annotations were saved
            if annotation_path.exists():
                print(f"   ✅ Saved annotations for {current_image.name}")
                self.current_index += 1
            else:
                print(f"   ❌ No annotations saved for {current_image.name}")
                response = input("   Continue to next image anyway? (y/n): ").lower()
                if response == 'y':
                    self.current_index += 1
                elif response == 'n':
                    break
        
        print(f"\n🎉 Batch annotation complete!")
        print(f"   Processed: {self.current_index}/{len(self.image_files)} images")
        
        # Show summary
        self.show_annotation_summary()
        
    def show_annotation_summary(self):
        """Show summary of annotated images."""
        labels_dir = Path('data/training/labels/train')
        annotated_files = list(labels_dir.glob('*.txt'))
        
        print(f"\n📊 ANNOTATION SUMMARY")
        print("="*30)
        print(f"Total annotation files: {len(annotated_files)}")
        
        total_annotations = 0
        component_counts = {}
        
        for label_file in annotated_files:
            if label_file.stat().st_size > 0:  # Non-empty file
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    image_annotations = len([line for line in lines if line.strip()])
                    total_annotations += image_annotations
                    
                    print(f"   {label_file.stem}: {image_annotations} components")
                    
                    # Count component types
                    for line in lines:
                        if line.strip():
                            class_id = int(line.split()[0])
                            # Map class ID to name
                            with open(self.classes_file, 'r') as cf:
                                class_names = [line.strip() for line in cf.readlines()]
                                if class_id < len(class_names):
                                    component_name = class_names[class_id]
                                    component_counts[component_name] = component_counts.get(component_name, 0) + 1
        
        print(f"\nTotal components annotated: {total_annotations}")
        print("Component distribution:")
        for component, count in sorted(component_counts.items()):
            print(f"   • {component}: {count}")
            
    def annotate_specific_image(self, image_name):
        """Annotate a specific image by name."""
        image_path = None
        for img in self.image_files:
            if img.name.lower() == image_name.lower():
                image_path = img
                break
        
        if not image_path:
            print(f"❌ Image not found: {image_name}")
            print("Available images:")
            for img in self.image_files:
                print(f"   • {img.name}")
            return
        
        print(f"📸 Annotating: {image_path.name}")
        annotator = SimpleAnnotator(str(image_path), self.classes_file)
        annotator.run()

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python batch_annotate.py all                    # Annotate all images")
        print("  python batch_annotate.py <image_name>          # Annotate specific image")
        print("  python batch_annotate.py circuit_02.jpg        # Example")
        sys.exit(1)
    
    classes_file = "classes.txt"
    image_directory = "data/expanded_training/images/train"
    
    batch_annotator = BatchAnnotator(image_directory, classes_file)
    
    if sys.argv[1].lower() == "all":
        batch_annotator.annotate_all()
    else:
        image_name = sys.argv[1]
        batch_annotator.annotate_specific_image(image_name)

if __name__ == "__main__":
    main() 