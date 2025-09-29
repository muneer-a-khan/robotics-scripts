#!/usr/bin/env python3
"""
Start Live Detection with Photos Model

This script starts the dual board live detection system using
the newly trained photos model with updated 15 classes.
"""

import sys
from pathlib import Path
from dual_board_live_system import DualBoardLiveSystem


def find_photos_model():
    """Find the trained photos model"""
    model_paths = [
        Path("dual_board_training/photos_model_fixed/weights/best.pt"),
        Path("dual_board_training/photos_model_fixed/weights/last.pt"),
        Path("dual_board_training/photos_model/weights/best.pt"),
        Path("dual_board_training/photos_model/weights/last.pt")
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            return str(model_path)
    
    return None


def main():
    """Main function"""
    print("🎯 STARTING LIVE DETECTION WITH PHOTOS MODEL")
    print("=" * 60)
    
    # Find the model
    model_path = find_photos_model()
    
    if not model_path:
        print("❌ No trained photos model found!")
        print("   Please ensure training is complete.")
        print("   Expected locations:")
        print("   • dual_board_training/photos_model_fixed/weights/best.pt")
        print("   • dual_board_training/photos_model_fixed/weights/last.pt")
        return
    
    print(f"🤖 Using model: {model_path}")
    print(f"🎥 Starting live detection system...")
    print()
    print("🎮 CONTROLS:")
    print("   • ESC: Quit")
    print("   • SPACE: Toggle detection")
    print("   • 'c': Capture screenshot")
    print("   • 's': Save current detections")
    print()
    print("🟢 Green tape detection: Enabled")
    print("   System will pause detection when green tape is obscured")
    print()
    
    try:
        # Initialize dual board live system
        live_system = DualBoardLiveSystem(
            model_path=model_path,
            camera_id=0,
            split_ratio=0.5,
            processing_interval=0.5  # Faster processing
        )
        
        # Start live detection
        live_system.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Detection stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure your camera is connected and accessible")


if __name__ == "__main__":
    main()
