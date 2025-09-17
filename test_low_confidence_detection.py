#!/usr/bin/env python3
"""
Low Confidence Detection Test

Tests your dual board model with much lower confidence thresholds to see all detections.
This helps debug detection issues by showing everything the model sees.
"""

import argparse
import sys
from pathlib import Path

# Import the dual board live system
from dual_board_live_system import DualBoardLiveSystem


def main():
    """Main entry point for low confidence testing"""
    
    parser = argparse.ArgumentParser(
        description="Test dual board model with low confidence thresholds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model", "-m", 
        default="models/weights/dual_board_dual_board_1758049257.pt",
        help="Path to your trained dual board model"
    )
    parser.add_argument(
        "--camera", "-c", 
        type=int, 
        default=0,
        help="Camera device ID"
    )
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.25,
        help="Lower confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--interval", "-i", 
        type=float, 
        default=2.0,
        help="Processing interval in seconds"
    )
    
    args = parser.parse_args()
    
    # Verify model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        return 1
    
    print("🔍 LOW CONFIDENCE DETECTION TEST")
    print("=" * 50)
    print(f"📱 Model: {model_path.name}")
    print(f"📹 Camera: {args.camera}")
    print(f"🎯 Confidence threshold: {args.confidence*100:.1f}% (LOWERED for debugging)")
    print(f"⏱️  Processing interval: {args.interval}s")
    print()
    
    print("🎮 Controls:")
    print("   • 'q': Quit system")
    print("   • 's': Save current frame")
    print("   • 'p': Pause/Resume")
    print("   • SPACE: Force process frame")
    print()
    
    print("🔍 This will show ALL detections above {:.1f}% confidence!".format(args.confidence*100))
    print("   You should see many more detections now.")
    print()
    
    try:
        # Initialize with lower confidence threshold
        print("🔧 Initializing dual board system with LOW confidence threshold...")
        system = DualBoardLiveSystem(
            model_path=str(model_path),
            split_ratio=0.5,
            processing_interval=args.interval,
            save_outputs=True,
            display_results=True
        )
        
        # HACK: Modify the visualizer confidence threshold directly
        # This bypasses the normal high confidence filtering
        if hasattr(system.board_analyzer.graph_converter, 'visualizer'):
            system.board_analyzer.graph_converter.visualizer.confidence_threshold = args.confidence
        
        # Also modify the component detector confidence if possible
        if hasattr(system.board_analyzer.component_detector, 'model'):
            # Try to set YOLO confidence threshold
            try:
                system.board_analyzer.component_detector.model.conf = args.confidence
                print(f"✅ Set YOLO confidence to {args.confidence}")
            except:
                print(f"⚠️  Could not set YOLO confidence, using default")
        
        print(f"🚀 Starting LOW confidence detection (showing detections >{args.confidence*100:.1f}%)...")
        system.run_live_detection(args.camera)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Detection stopped by user")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
