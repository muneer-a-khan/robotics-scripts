#!/usr/bin/env python3
"""
Start Dual Circuit Board System

This script provides an easy way to start the complete dual circuit board system:
1. Dual camera processing (splits single camera into left/right boards)
2. Dual visualization monitoring (creates separate visualizations for each side)
3. Model retraining preparation (for high-up camera optimization)

Usage Examples:
    # Start dual camera system with validation
    python start_dual_system.py --mode dual-camera --validate
    
    # Start dual visualization monitor only
    python start_dual_system.py --mode dual-monitor
    
    # Prepare training data for high camera
    python start_dual_system.py --mode train-prep --collect-data
    
    # Start everything (camera + monitor)
    python start_dual_system.py --mode all
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def start_dual_camera(validate=False, camera_id=0, no_display=False, model_path=None):
    """Start the dual camera system."""
    print("🎥 Starting Dual Camera System...")
    print("=" * 50)
    
    cmd = [sys.executable, "dual_camera_system.py", "--camera", str(camera_id)]
    
    if model_path:
        cmd.extend(["--model", model_path])
        print(f"🎯 Using model: {model_path}")
    
    if validate:
        cmd.append("--validate")
        print("✅ Circuit validation ENABLED")
    
    if no_display:
        cmd.append("--no-display")
        print("📺 Display DISABLED")
    
    print(f"📹 Using camera ID: {camera_id}")
    print("🔄 Camera feed will be split into LEFT and RIGHT boards")
    print("📁 Outputs saved to: output/data/left/ and output/data/right/")
    print("=" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Dual camera system stopped")


def start_dual_monitor(process_existing=False):
    """Start the dual visualization monitor."""
    print("👀 Starting Dual Visualization Monitor...")
    print("=" * 50)
    
    cmd = [sys.executable, "dual_visualization_monitor.py"]
    
    if process_existing:
        cmd.append("--process-existing")
        print("📄 Will process existing graph files first")
    
    print("📁 Monitoring: output/data/left/ and output/data/right/")
    print("🎨 Creating: live_circuit_visual_left_*.png and live_circuit_visual_right_*.png")
    print("📊 Validation data included in visualizations")
    print("=" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Dual monitor stopped")


def start_training_prep(collect_data=False, duration=10):
    """Start training data preparation."""
    print("🎓 Starting Training Data Preparation...")
    print("=" * 50)
    
    if collect_data:
        cmd = [sys.executable, "high_camera_training_prep.py", 
               "--mode", "collect", "--duration", str(duration)]
        print(f"📸 Collecting training data for {duration} minutes")
        print("📋 Position circuit components on both sides of camera view")
        print("⌨️ Press SPACE to capture images during collection")
    else:
        print("📚 Training preparation tools available:")
        print("  • Collect live data: --collect-data")
        print("  • Create resolution variants")
        print("  • Create zoomed-out variants")
        print("  • Create dual board layouts")
        print("  • Train optimized model")
        return
    
    print("=" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Training preparation stopped")


def start_all_systems(validate=False, camera_id=0):
    """Start both camera and monitor systems in parallel."""
    print("🚀 Starting Complete Dual Circuit Board System...")
    print("=" * 60)
    print("This will start:")
    print("  1. 🎥 Dual Camera System (splits camera feed)")
    print("  2. 👀 Dual Visualization Monitor (creates live graphs)")
    print("=" * 60)
    
    def run_camera():
        cmd = [sys.executable, "dual_camera_system.py", "--camera", str(camera_id)]
        if validate:
            cmd.append("--validate")
        subprocess.run(cmd)
    
    def run_monitor():
        # Wait a bit for camera to start generating files
        time.sleep(5)
        cmd = [sys.executable, "dual_visualization_monitor.py", "--process-existing"]
        subprocess.run(cmd)
    
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            camera_future = executor.submit(run_camera)
            monitor_future = executor.submit(run_monitor)
            
            # Wait for both to complete
            camera_future.result()
            monitor_future.result()
    
    except KeyboardInterrupt:
        print("\n🛑 All systems stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dual Circuit Board System Launcher")
    parser.add_argument("--mode", choices=["dual-camera", "dual-monitor", "train-prep", "all"],
                       default="dual-camera", help="System mode to start")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--validate", action="store_true", help="Enable circuit validation")
    parser.add_argument("--no-display", action="store_true", help="Disable camera display")
    parser.add_argument("--process-existing", action="store_true", help="Process existing files first")
    parser.add_argument("--collect-data", action="store_true", help="Collect training data")
    parser.add_argument("--duration", type=int, default=10, help="Data collection duration (minutes)")
    
    args = parser.parse_args()
    
    print("🤖 Dual Circuit Board Vision System")
    print("=" * 60)
    print("New Features:")
    print("✅ Single camera → Dual circuit board detection")
    print("✅ Separate processing for LEFT and RIGHT boards")
    print("✅ Independent visualizations with validation")
    print("✅ High-camera training data preparation")
    print("=" * 60)
    
    if args.mode == "dual-camera":
        start_dual_camera(args.validate, args.camera, args.no_display)
    
    elif args.mode == "dual-monitor":
        start_dual_monitor(args.process_existing)
    
    elif args.mode == "train-prep":
        start_training_prep(args.collect_data, args.duration)
    
    elif args.mode == "all":
        start_all_systems(args.validate, args.camera)
    
    else:
        print("❌ Invalid mode selected")
        parser.print_help()


if __name__ == "__main__":
    main()
