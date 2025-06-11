# Copy training files
import shutil
from pathlib import Path

# Assuming you save your image as 'snap_circuit_image.jpg'
if Path('snap_circuit_image.jpg').exists():
    # Copy to train directory
    shutil.copy('snap_circuit_image.jpg', 'data/training/images/train/')
    shutil.copy('snap_circuit_training.txt', 'data/training/labels/train/snap_circuit_image.txt')
    
    # Copy to val directory for validation
    shutil.copy('snap_circuit_image.jpg', 'data/training/images/val/')
    shutil.copy('snap_circuit_training.txt', 'data/training/labels/val/snap_circuit_image.txt')
    
    print("✅ Training files copied successfully!")
else:
    print("❌ Please save your image as 'snap_circuit_image.jpg' first")
