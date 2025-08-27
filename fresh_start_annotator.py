"""
Fresh Start Annotator
Helper tool for re-annotating images if needed
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml
import shutil

from fresh_start_config import FreshStartConfig

class FreshStartAnnotator:
    def __init__(self):
        self.config = FreshStartConfig()
        
    def check_labelimg_installation(self):
        """Check if labelImg is installed and accessible"""
        try:
            result = subprocess.run(['labelImg', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install_labelimg(self):
        """Install labelImg if not already installed"""
        print("Installing labelImg...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'labelImg'], 
                         check=True)
            print("labelImg installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install labelImg: {e}")
            return False
    
    def setup_annotation_environment(self):
        """Setup the annotation environment"""
        print("Setting up annotation environment...")
        
        # Check if labelImg is installed
        if not self.check_labelimg_installation():
            print("labelImg not found. Installing...")
            if not self.install_labelimg():
                print("Failed to install labelImg. Please install manually:")
                print("pip install labelImg")
                return False
        
        # Create classes.txt in the right location
        classes_file = self.config.TRAINING_DATA / "classes.txt"
        classes_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(classes_file, 'w') as f:
            for class_name in self.config.COMPONENT_CLASSES:
                f.write(f"{class_name}\n")
        
        print(f"Classes file created at: {classes_file}")
        print("Annotation environment setup complete!")
        return True
    
    def start_annotation(self, image_dir=None):
        """Start labelImg for annotation"""
        if image_dir is None:
            image_dir = self.config.TRAINING_DATA / "images" / "train"
        
        if not Path(image_dir).exists():
            print(f"Image directory not found: {image_dir}")
            return False
        
        print(f"Starting labelImg for directory: {image_dir}")
        print("Instructions:")
        print("1. Press 'W' to create a bounding box")
        print("2. Press 'D' to go to next image")
        print("3. Press 'A' to go to previous image")
        print("4. Press 'Ctrl+S' to save")
        print("5. Make sure to select the correct class for each component")
        
        try:
            subprocess.run([
                'labelImg', 
                str(image_dir),
                str(self.config.TRAINING_DATA / "labels" / "train"),
                str(self.config.TRAINING_DATA / "classes.txt")
            ])
            return True
        except Exception as e:
            print(f"Failed to start labelImg: {e}")
            return False
    
    def validate_annotations(self):
        """Validate existing annotations"""
        print("Validating annotations...")
        
        labels_dir = self.config.TRAINING_DATA / "labels"
        images_dir = self.config.TRAINING_DATA / "images"
        
        issues = []
        
        # Check each split
        for split in ['train', 'val', 'test']:
            split_labels = labels_dir / split
            split_images = images_dir / split
            
            if not split_labels.exists():
                issues.append(f"Labels directory missing: {split_labels}")
                continue
                
            if not split_images.exists():
                issues.append(f"Images directory missing: {split_images}")
                continue
            
            # Count files
            label_files = list(split_labels.glob("*.txt"))
            image_files = list(split_images.glob("*.jpg")) + list(split_images.glob("*.png"))
            
            print(f"{split}: {len(label_files)} labels, {len(image_files)} images")
            
            # Check for missing labels
            for img_file in image_files:
                label_file = split_labels / f"{img_file.stem}.txt"
                if not label_file.exists():
                    issues.append(f"Missing label for {img_file}")
        
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("All annotations validated successfully!")
        
        return len(issues) == 0
    
    def create_annotation_script(self):
        """Create a batch script for easy annotation"""
        script_content = f"""@echo off
echo Starting Snap Circuit Annotation Tool
echo.
echo This will open labelImg for annotating your snap circuit images
echo.
echo Make sure you have the following classes available:
echo {', '.join(self.config.COMPONENT_CLASSES)}
echo.
pause

cd /d "{self.config.PROJECT_ROOT}"
labelImg "{self.config.TRAINING_DATA}\\images\\train" "{self.config.TRAINING_DATA}\\labels\\train" "{self.config.TRAINING_DATA}\\classes.txt"

echo.
echo Annotation session complete!
pause
"""
        
        script_path = self.config.PROJECT_ROOT / "start_annotation.bat"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Annotation script created: {script_path}")
        return script_path
    
    def backup_annotations(self):
        """Create a backup of current annotations"""
        backup_dir = self.config.OUTPUT_DIR / "annotation_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        source_labels = self.config.TRAINING_DATA / "labels"
        if source_labels.exists():
            shutil.copytree(source_labels, backup_dir / "labels", dirs_exist_ok=True)
            print(f"Annotations backed up to: {backup_dir}")
        else:
            print("No existing annotations to backup")

def main():
    """Main annotation setup function"""
    annotator = FreshStartAnnotator()
    
    print("=== Snap Circuit Annotation Setup ===")
    print()
    
    # Setup environment
    if not annotator.setup_annotation_environment():
        print("Failed to setup annotation environment")
        return
    
    # Validate existing annotations
    print()
    print("=== Validating Existing Annotations ===")
    annotator.validate_annotations()
    
    # Create backup
    print()
    print("=== Creating Backup ===")
    annotator.backup_annotations()
    
    # Create annotation script
    print()
    print("=== Creating Annotation Script ===")
    script_path = annotator.create_annotation_script()
    
    print()
    print("=== Setup Complete ===")
    print(f"To start annotating, run: {script_path}")
    print("Or use the command:")
    print(f"labelImg {annotator.config.TRAINING_DATA}/images/train {annotator.config.TRAINING_DATA}/labels/train {annotator.config.TRAINING_DATA}/classes.txt")

if __name__ == "__main__":
    main() 