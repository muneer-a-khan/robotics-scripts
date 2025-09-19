#!/usr/bin/env python3
"""
Launch Dual Board System

Easy launcher for the complete dual board system with clean model and graph generation.
"""

import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from test_clean_system_with_graphs import test_clean_system_with_graphs

def main():
    print("🚀 LAUNCHING DUAL BOARD SYSTEM")
    print("=" * 50)
    print("✅ Clean model: 100% symmetric detection")
    print("🔗 Graph generation: Side-by-side with spatial positioning")
    print("📊 Live connectivity analysis")
    print("🎨 High-quality PNG output")
    print()
    print("🎮 Controls while running:")
    print("   • 'g': Generate graphs immediately")
    print("   • 'd': Detailed detection analysis")
    print("   • 'r': Print connectivity report")
    print("   • 't': Change connection threshold")
    print("   • 'q': Quit")
    print()
    print("📁 Graphs saved to: graph_output/")
    print()
    
    try:
        test_clean_system_with_graphs()
    except KeyboardInterrupt:
        print("\n👋 System shutdown requested")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    
    print("✅ Dual board system stopped")

if __name__ == "__main__":
    main()
