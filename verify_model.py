#!/usr/bin/env python3
"""
Quick Model Verification Script

Verifies that your newly trained dual board model can be loaded and is ready for testing.
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def verify_model(model_path: str) -> bool:
    """
    Verify that a YOLO model can be loaded and is functional.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if model is valid, False otherwise
    """
    try:
        # Check if file exists
        model_file = Path(model_path)
        if not model_file.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        # Check file size
        file_size = model_file.stat().st_size
        print(f"📄 Model file size: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1024 * 1024:  # Less than 1MB is suspicious
            print("⚠️  Warning: Model file seems very small (< 1MB)")
        
        # Try to load the model
        print("🔄 Loading model...")
        model = YOLO(model_path)
        
        # Check model info
        print("✅ Model loaded successfully!")
        
        # Get model details
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            class_names = model.model.names
            print(f"📋 Number of classes: {len(class_names)}")
            print(f"🏷️  Class names: {list(class_names.values())}")
        
        # Check device capability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Target device: {device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("💻 Using CPU (consider GPU for better performance)")
        
        print("\n✅ Model verification PASSED!")
        print("🎯 Your model is ready for dual board testing!")
        return True
        
    except Exception as e:
        print(f"❌ Model verification FAILED: {e}")
        print("\n🔍 Possible issues:")
        print("   • Model file is corrupted")
        print("   • Wrong model format")
        print("   • Missing dependencies")
        print("   • Insufficient permissions")
        return False


def main():
    """Main verification function"""
    model_path = "models/weights/dual_board_dual_board_1758049257.pt"
    
    print("🔍 Dual Board Model Verification")
    print("=" * 40)
    print(f"Model: {model_path}")
    print()
    
    success = verify_model(model_path)
    
    if success:
        print("\n🚀 Next steps:")
        print("   1. Run: python test_new_dual_board_model.py")
        print("   2. Or use: ./test_dual_model.sh (Unix) or test_dual_model.bat (Windows)")
        print("   3. Position two circuit boards in camera view")
        print("   4. Watch live detection and graph generation!")
        return 0
    else:
        print("\n🛠️  Troubleshooting:")
        print("   1. Check if model training completed successfully")
        print("   2. Verify file permissions and integrity")
        print("   3. Re-run training if model is corrupted")
        print("   4. Check ultralytics installation: pip install ultralytics")
        return 1


if __name__ == "__main__":
    sys.exit(main())
