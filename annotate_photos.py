#!/usr/bin/env python3
"""
Photos Annotation Script for Dual Board System

This script sets up and runs annotation for the photos folder using the
dual board annotation system with green tape detection support.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dual_board_annotator import DualBoardAnnotator


def main():
    """Main annotation function for photos folder"""
    print("🎯 PHOTOS ANNOTATION FOR DUAL BOARD SYSTEM")
    print("=" * 60)
    print()
    
    # Set up paths
    photos_dir = project_root / "photos"
    classes_file = project_root / "classes.txt"
    output_dir = project_root / "photos_annotations"
    
    # Validate paths
    if not photos_dir.exists():
        print(f"❌ Photos directory not found: {photos_dir}")
        return
    
    if not classes_file.exists():
        print(f"❌ Classes file not found: {classes_file}")
        return
    
    # Count photos
    photo_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    photo_files = []
    for ext in photo_extensions:
        photo_files.extend(photos_dir.glob(ext))
    
    if not photo_files:
        print(f"❌ No photos found in {photos_dir}")
        return
    
    print(f"📁 Photos directory: {photos_dir}")
    print(f"📸 Found {len(photo_files)} photos to annotate")
    print(f"📋 Classes file: {classes_file}")
    print(f"💾 Annotations will be saved to: {output_dir}")
    print()
    
    # Load classes and show them
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print(f"🏷️  CLASSES ({len(classes)}):")
    for i, cls in enumerate(classes):
        print(f"   {i:2d}: {cls}")
    print()
    
    print("🎮 ANNOTATION FEATURES:")
    print("   • Dual board split-screen annotation")  
    print("   • Navigate all 15 classes easily")
    print("   • YOLO format output")
    print("   • Keyboard shortcuts for efficiency")
    print()
    
    print("🎮 CONTROLS:")
    print("   • Mouse: Click + drag to draw bounding box")
    print("   • Numbers 0-9: Direct class selection (0-9)")
    print("   • UP/DOWN arrows: Cycle through ALL classes")
    print("   • +/- keys: Next/Previous class (alternative)")
    print("   • TAB: Switch between left/right side")
    print("   • 'c': Show all classes with current selection")
    print("   • 's': Save annotations and continue")
    print("   • 'r': Reset current image")
    print("   • 'q': Quit")
    print("   • 'h': Show help")
    print()
    
    # Ask for confirmation
    response = input("🚀 Ready to start annotation? (y/n): ").lower().strip()
    if response != 'y':
        print("❌ Annotation cancelled.")
        return
    
    print()
    print("🎯 STARTING DUAL BOARD ANNOTATION...")
    print("=" * 60)
    
    try:
        # Initialize the dual board annotator
        annotator = DualBoardAnnotator(str(classes_file))
        
        # Start batch annotation
        annotator.batch_annotate(str(photos_dir), str(output_dir))
        
        print()
        print("🎉 ANNOTATION SESSION COMPLETE!")
        print(f"💾 Annotations saved in: {output_dir}")
        
        # Count generated annotation files
        annotation_files = list(output_dir.glob("*.txt"))
        print(f"📝 Generated {len(annotation_files)} annotation files")
        
    except KeyboardInterrupt:
        print("\n⏹️  Annotation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during annotation: {e}")
        return
    
    print()
    print("📋 NEXT STEPS:")
    print("   1. Review annotations in photos_annotations/")
    print("   2. Create training dataset with annotated photos")
    print("   3. Train new dual board model")
    print("   4. Test model performance")


if __name__ == "__main__":
    main()
