#!/usr/bin/env python3
"""
Run Dual Camera System with High Camera Model

This script runs the dual camera system using the newly trained high camera model.
The model was specifically trained for high-up camera perspectives with excellent results:
- mAP50: 99.11%
- mAP50-95: 94.12%
"""

import sys
import argparse
from pathlib import Path
import subprocess

# Path to the trained high camera model
HIGH_CAMERA_MODEL_PATH = "output/results/high_camera_gpu_20250827_090037/weights/best.pt"

def check_model_exists():
    """Check if the high camera model exists."""
    model_path = Path(HIGH_CAMERA_MODEL_PATH)
    if not model_path.exists():
        print(f"❌ High camera model not found at: {model_path}")
        print("Please run the high camera training first!")
        return False
    
    print(f"✅ High camera model found: {model_path}")
    print(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    return True

def run_dual_system(camera_id=0, validate=True, no_display=False, split_ratio=0.5):
    """Run the dual camera system with the high camera model."""
    
    if not check_model_exists():
        return False
    
    print("🚀 Starting Dual Camera System with High Camera Model")
    print("=" * 60)
    print(f"🎯 Model: {HIGH_CAMERA_MODEL_PATH}")
    print(f"📹 Camera ID: {camera_id}")
    print(f"🔄 Split Ratio: {split_ratio} (Left: {split_ratio*100:.0f}%, Right: {(1-split_ratio)*100:.0f}%)")
    print(f"✅ Validation: {'ENABLED' if validate else 'DISABLED'}")
    print(f"📺 Display: {'DISABLED' if no_display else 'ENABLED'}")
    print("=" * 60)
    
    # Build command
    cmd = [
        sys.executable, 
        "dual_camera_system.py",
        "--model", HIGH_CAMERA_MODEL_PATH,
        "--camera", str(camera_id),
        "--split-ratio", str(split_ratio)
    ]
    
    if validate:
        cmd.append("--validate")
    
    if no_display:
        cmd.append("--no-display")
    
    print("🎬 Starting dual camera system...")
    print("📝 Commands:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save current frame")
    print("   - Press 'p' to pause/resume")
    print("=" * 60)
    
    try:
        subprocess.run(cmd)
        print("\n✅ Dual camera system completed successfully!")
        return True
    except KeyboardInterrupt:
        print("\n🛑 Dual camera system stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error running dual camera system: {e}")
        return False

def run_with_monitor(camera_id=0, validate=True):
    """Run dual system with both camera and monitor."""
    print("🚀 Starting Complete Dual System (Camera + Monitor)")
    print("=" * 60)
    
    # Use start_dual_system.py for complete setup
    cmd = [
        sys.executable,
        "start_dual_system.py",
        "--mode", "all",
        "--camera-id", str(camera_id),
        "--model", HIGH_CAMERA_MODEL_PATH
    ]
    
    if validate:
        cmd.append("--validate")
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\n🛑 Complete dual system stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error running complete dual system: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Dual Camera System with High Camera Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic dual camera with validation
    python run_dual_with_high_camera_model.py
    
    # Custom camera and split ratio
    python run_dual_with_high_camera_model.py --camera 1 --split-ratio 0.6
    
    # Headless mode (no display)
    python run_dual_with_high_camera_model.py --no-display
    
    # Complete system with monitor
    python run_dual_with_high_camera_model.py --with-monitor
    
    # No validation (faster processing)
    python run_dual_with_high_camera_model.py --no-validate
        """
    )
    
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device ID (default: 0)")
    parser.add_argument("--split-ratio", type=float, default=0.5,
                       help="Camera split ratio (default: 0.5 = equal halves)")
    parser.add_argument("--no-validate", action="store_true",
                       help="Disable circuit validation (faster processing)")
    parser.add_argument("--no-display", action="store_true",
                       help="Run in headless mode (no display)")
    parser.add_argument("--with-monitor", action="store_true",
                       help="Run complete system with visualization monitor")
    
    args = parser.parse_args()
    
    print("🎯 High Camera Model Dual System Launcher")
    print("=" * 50)
    
    # Validate split ratio
    if not 0.1 <= args.split_ratio <= 0.9:
        print("❌ Split ratio must be between 0.1 and 0.9")
        return False
    
    success = False
    
    if args.with_monitor:
        success = run_with_monitor(
            camera_id=args.camera,
            validate=not args.no_validate
        )
    else:
        success = run_dual_system(
            camera_id=args.camera,
            validate=not args.no_validate,
            no_display=args.no_display,
            split_ratio=args.split_ratio
        )
    
    if success:
        print("\n🎉 Session completed!")
        print("📁 Check output/ directory for results:")
        print("   - output/data/left/     - Left board graphs")
        print("   - output/data/right/    - Right board graphs")
        print("   - output/frames/        - Annotated frames")
        print("   - output/live_circuit_visual_*.png - Visualizations")
    
    return success

if __name__ == "__main__":
    main()