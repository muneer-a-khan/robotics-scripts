#!/usr/bin/env python3
"""
Test Script for New Dual Board Model

This script tests your newly trained dual board model with:
- Live camera setup with real-time image splitting
- Graph generation for connectivity analysis 
- Real-time visualization with circuit diagrams
- Component detection and scoring for both boards

Usage:
    python test_new_dual_board_model.py [options]
"""

import argparse
import sys
from pathlib import Path

# Import the dual board live system
from dual_board_live_system import DualBoardLiveSystem


def main():
    """Main entry point for testing the new dual board model"""
    
    parser = argparse.ArgumentParser(
        description="Test your newly trained dual board model with live camera and graph generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and camera settings
    parser.add_argument(
        "--model", "-m", 
        default="models/weights/dual_board_dual_board_1758049257.pt",
        help="Path to your trained dual board model"
    )
    parser.add_argument(
        "--camera", "-c", 
        type=int, 
        default=0,
        help="Camera device ID (0 for default camera)"
    )
    
    # System configuration
    parser.add_argument(
        "--split-ratio", "-s", 
        type=float, 
        default=0.5,
        help="Image split ratio (0.5 = equal halves)"
    )
    parser.add_argument(
        "--interval", "-i", 
        type=float, 
        default=2.0,
        help="Processing interval in seconds"
    )
    
    # Display and output options
    parser.add_argument(
        "--no-display", 
        action="store_true",
        help="Disable live display window"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Disable saving outputs to disk"
    )
    
    # Performance options
    parser.add_argument(
        "--fast-mode", 
        action="store_true",
        help="Use faster processing (1 second intervals)"
    )
    
    args = parser.parse_args()
    
    # Verify model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        print("\n💡 Available model files:")
        models_dir = Path("models/weights")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pt"):
                print(f"   • {model_file}")
        else:
            print("   • No models directory found")
        return 1
    
    # Adjust processing interval for fast mode
    if args.fast_mode:
        processing_interval = 1.0
        print("🚀 Fast mode enabled - processing every 1 second")
    else:
        processing_interval = args.interval
    
    # Display configuration
    print("🎯 Dual Board Model Test Configuration")
    print("=" * 50)
    print(f"📱 Model: {model_path.name}")
    print(f"📹 Camera: {args.camera}")
    print(f"✂️  Split ratio: {args.split_ratio} ({args.split_ratio*100:.0f}% left, {(1-args.split_ratio)*100:.0f}% right)")
    print(f"⏱️  Processing interval: {processing_interval}s")
    print(f"🖥️  Display: {'Disabled' if args.no_display else 'Enabled'}")
    print(f"💾 Save outputs: {'Disabled' if args.no_save else 'Enabled'}")
    print()
    
    print("🎮 Live Detection Controls:")
    print("   • 'q': Quit system")
    print("   • 's': Save current frame manually")
    print("   • 'p': Pause/Resume processing")
    print("   • 't': Toggle green tape detection overlay")
    print("   • SPACE: Force process current frame immediately")
    print()
    
    print("📊 What you'll see:")
    print("   • Live camera feed split down the middle")
    print("   • Real-time component detection on both sides")
    print("   • Connectivity graphs generated for each board")
    print("   • Scoring for component orientation and connections")
    print("   • Circuit visualizations saved as PNG files")
    print()
    
    if not args.no_save:
        print("📁 Output locations:")
        print("   • Left board: dual_board_output/left/")
        print("   • Right board: dual_board_output/right/")
        print("   • Combined analysis: dual_board_output/combined/")
        print("   • Latest visualizations: **/latest_circuit_visual.png")
        print()
    
    try:
        # Initialize the dual board live system
        print("🔧 Initializing dual board system...")
        system = DualBoardLiveSystem(
            model_path=str(model_path),
            split_ratio=args.split_ratio,
            processing_interval=processing_interval,
            save_outputs=not args.no_save,
            display_results=not args.no_display
        )
        
        # Start live detection
        print("🚀 Starting live dual board detection...")
        print("   Make sure both circuit boards are positioned in camera view!")
        print("   Use green tape on boards for hand detection functionality.")
        print()
        
        system.run_live_detection(args.camera)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Detection stopped by user")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("   1. Check that your camera is connected and working")
        print("   2. Verify the model file is valid and not corrupted")
        print("   3. Ensure you have sufficient lighting for both boards")
        print("   4. Try a different camera ID if default doesn't work")
        print("   5. Check that no other applications are using the camera")
        return 1


if __name__ == "__main__":
    sys.exit(main())
