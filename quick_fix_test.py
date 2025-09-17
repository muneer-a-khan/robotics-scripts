#!/usr/bin/env python3
"""
Quick fix test with very low thresholds
"""

import sys
from dual_board_live_system import DualBoardLiveSystem

def main():
    print("🔧 QUICK FIX TEST - Ultra-low confidence thresholds")
    print("=" * 60)
    
    # Initialize system
    system = DualBoardLiveSystem(
        model_path="models/weights/dual_board_dual_board_1758049257.pt",
        split_ratio=0.5,
        processing_interval=1.0,
        save_outputs=True,
        display_results=True
    )
    
    # Set ultra-low confidence in the component detector
    system.board_analyzer.component_detector.confidence_threshold = 0.1
    system.board_analyzer.component_detector.model.conf = 0.1
    
    print("✅ Set all confidence thresholds to 10%")
    print("🚀 Starting detection with ultra-low thresholds...")
    
    try:
        system.run_live_detection(0)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
