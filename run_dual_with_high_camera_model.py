#!/usr/bin/env python3
"""
Run Dual Camera System with High Camera Model

This script runs the dual camera system using the newly trained high camera model.
It automatically uses your trained model and enables validation for real-time graph generation.

Usage:
    # Run with validation and display
    python run_dual_with_high_camera_model.py
    
    # Run without display (headless mode)
    python run_dual_with_high_camera_model.py --no-display
    
    # Use different camera
    python run_dual_with_high_camera_model.py --camera 1
"""

import sys
import argparse
from pathlib import Path
import subprocess
import time

# Path to your newly trained high camera model
HIGH_CAMERA_MODEL_PATH = "high_camera_training/high_camera_model2/weights/best.pt"

def check_model_exists():
    """Check if the high camera model exists."""
    model_path = Path(HIGH_CAMERA_MODEL_PATH)
    if not model_path.exists():
        print(f"❌ High camera model not found at: {model_path}")
        print("Available models:")
        
        # Check for alternative model paths
        alternative_paths = [
            "high_camera_training/high_camera_model/weights/best.pt",
            "models/weights/latest_trained_model.pt",
            "latest_trained_model.pt"
        ]
        
        for alt_path in alternative_paths:
            if Path(alt_path).exists():
                print(f"✅ Found alternative: {alt_path}")
                return alt_path
        
        print("Please run the high camera training first!")
        return None
    
    print(f"✅ High camera model found: {model_path}")
    print(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    return str(model_path)

def run_dual_system(camera_id=0, validate=True, no_display=False, split_ratio=0.5):
    """Run the dual camera system with high camera model."""
    
    model_path = check_model_exists()
    if not model_path:
        return False
    
    print("🚀 Starting Dual Circuit Board System with High Camera Model")
    print("=" * 60)
    print(f"🎯 Model: {model_path}")
    print(f"📹 Camera: {camera_id}")
    print(f"✅ Validation: {'ENABLED' if validate else 'DISABLED'}")
    print(f"📺 Display: {'ENABLED' if not no_display else 'DISABLED'}")
    print(f"📊 Split Ratio: {split_ratio}")
    print("=" * 60)
    print()
    print("📝 This system will:")
    print("   • Split your camera feed into LEFT and RIGHT circuit boards")
    print("   • Detect components on both boards simultaneously")
    print("   • Generate circuit graphs for each board")
    print("   • Create live visualizations with validation data")
    print("   • Save outputs to output/data/left/ and output/data/right/")
    print()
    
    # Build command for dual camera system
    cmd = [
        sys.executable, 
        "dual_camera_system.py",
        "--camera", str(camera_id),
        "--model", model_path,
        "--split-ratio", str(split_ratio)
    ]
    
    if validate:
        cmd.append("--validate")
    
    if no_display:
        cmd.append("--no-display")
    
    print("🔄 Starting dual camera processing...")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        # Run the dual camera system
        subprocess.run(cmd)
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Dual camera system stopped by user")
        return True
        
    except Exception as e:
        print(f"\n❌ Error running dual camera system: {e}")
        return False

def start_visualization_monitor():
    """Start the dual visualization monitor in background."""
    print("🎨 Starting dual visualization monitor...")
    
    try:
        # Start dual visualization monitor
        cmd = [sys.executable, "dual_visualization_monitor.py"]
        subprocess.Popen(cmd)
        print("✅ Dual visualization monitor started in background")
        time.sleep(2)  # Give it time to start
        return True
        
    except Exception as e:
        print(f"❌ Error starting visualization monitor: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Dual Camera System with High Camera Model")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--no-display", action="store_true", help="Disable real-time display")
    parser.add_argument("--no-validation", action="store_true", help="Disable circuit validation")
    parser.add_argument("--split-ratio", type=float, default=0.5, help="Camera split ratio (default: 0.5)")
    parser.add_argument("--no-monitor", action="store_true", help="Don't start visualization monitor")
    
    args = parser.parse_args()
    
    print("🎯 High Camera Model Dual System Launcher")
    print("=" * 50)
    
    # Start visualization monitor first (unless disabled)
    if not args.no_monitor:
        if not start_visualization_monitor():
            print("⚠️  Continuing without visualization monitor...")
    
    # Run the dual camera system
    success = run_dual_system(
        camera_id=args.camera,
        validate=not args.no_validation,
        no_display=args.no_display,
        split_ratio=args.split_ratio
    )
    
    if success:
        print("✅ Dual camera system completed successfully!")
    else:
        print("❌ Dual camera system failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()