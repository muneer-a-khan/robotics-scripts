#!/usr/bin/env python3
"""
Quick setup script for improved Snap Circuit model training.
This script helps you prepare for better training with individual component images.
"""

import os
import sys
from pathlib import Path
import subprocess


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = ['cv2', 'numpy', 'ultralytics', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install opencv-python numpy ultralytics torch")
        return False
    
    return True


def check_current_training_data():
    """Check what training data currently exists."""
    print("\n📊 Checking current training data...")
    
    # Check for original training data
    original_data = Path("data/training")
    if original_data.exists():
        train_images = list((original_data / "images" / "train").glob("*"))
        val_images = list((original_data / "images" / "val").glob("*"))
        print(f"✅ Original dataset: {len(train_images)} train, {len(val_images)} val images")
    else:
        print("⚠️  No original training data found")
    
    # Check for individual component images
    individual_data = Path("data/individual_components")
    if individual_data.exists():
        component_counts = {}
        for component_dir in individual_data.iterdir():
            if component_dir.is_dir():
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    images.extend(component_dir.glob(ext))
                if images:
                    component_counts[component_dir.name] = len(images)
        
        if component_counts:
            print("✅ Individual component images found:")
            for component, count in component_counts.items():
                print(f"   {component}: {count} images")
        else:
            print("⚠️  Individual component directories exist but no images found")
    else:
        print("⚠️  No individual component images found")
    
    return original_data.exists(), bool(component_counts if 'component_counts' in locals() else False)


def create_training_instructions():
    """Create detailed instructions for taking component photos."""
    instructions = """
# 📸 COMPONENT PHOTOGRAPHY GUIDE

## Setup
1. **Lighting**: Use bright, diffused lighting (window light or LED panels)
2. **Background**: Plain white paper or poster board
3. **Camera**: Phone camera works fine, ensure good focus
4. **Distance**: Component should fill 60-80% of the frame

## For Each Component Type:

### Battery Holder
- Take 5-7 photos from different angles
- Include close-ups of connection points
- Show both empty and with battery if available

### LEDs (Light Emitting Diodes)
- Take photos of different colored LEDs if available
- Show both on and off states
- Include close-ups of the clear plastic housing

### Switches
- Capture in both on/off positions
- Multiple angles showing the lever mechanism
- Close-ups of connection points

### Speakers/Buzzers
- Show the speaker cone clearly
- Multiple angles including side view
- Connection points visible

### Motors
- Show the motor housing
- Include the output shaft
- Connection wire clearly visible

### Wires/Connectors
- Different wire colors and lengths
- Show connection ends clearly
- Both straight and bent positions

## Photo Tips:
- Take 5-10 photos per component type
- Include variations (colors, sizes, angles)
- Ensure sharp focus on the component
- Avoid reflections on metallic parts
- Keep backgrounds simple and clean

## File Naming:
- battery_holder_01.jpg, battery_holder_02.jpg, etc.
- led_red_01.jpg, led_blue_01.jpg, etc.
- switch_on_01.jpg, switch_off_01.jpg, etc.
"""
    
    with open("COMPONENT_PHOTO_GUIDE.md", "w", encoding='utf-8') as f:
        f.write(instructions)
    
    print("✅ Created photography guide: COMPONENT_PHOTO_GUIDE.md")


def setup_training_environment():
    """Set up the improved training environment."""
    print("\n🔧 Setting up improved training environment...")
    
    # Run the individual component dataset setup
    if Path("create_individual_component_dataset.py").exists():
        print("📁 Creating component directories...")
        os.system("python create_individual_component_dataset.py")
    else:
        print("❌ create_individual_component_dataset.py not found")
        return False
    
    return True


def provide_next_steps(has_original_data, has_individual_data):
    """Provide clear next steps based on current data status."""
    print("\n" + "="*60)
    print("📋 RECOMMENDED NEXT STEPS")
    print("="*60)
    
    if not has_original_data and not has_individual_data:
        print("🚀 STARTING FROM SCRATCH:")
        print("1. First, take photos of your complete circuit board setup")
        print("2. Save as 'snap_circuit_image.jpg' in project root")
        print("3. Run: python create_training_data.py")
        print("4. Then take individual component photos (see guide)")
        print("5. Run: python create_individual_component_dataset.py --process")
        print("6. Train: python train_improved_model.py")
    
    elif has_original_data and not has_individual_data:
        print("📸 ADDING INDIVIDUAL COMPONENTS:")
        print("1. Take photos of individual components (see COMPONENT_PHOTO_GUIDE.md)")
        print("2. Save in data/individual_components/[component_name]/")
        print("3. Run: python create_individual_component_dataset.py --process")
        print("4. Train improved model: python train_improved_model.py")
    
    elif has_individual_data and not has_original_data:
        print("🎯 CREATING COMPLETE DATASET:")
        print("1. Take photos of complete circuit board setups")
        print("2. Run: python create_training_data.py")
        print("3. Run: python create_individual_component_dataset.py --process")
        print("4. Train: python train_improved_model.py")
    
    else:
        print("✅ READY FOR IMPROVED TRAINING:")
        print("1. Run: python create_individual_component_dataset.py --process")
        print("2. Train improved model: python train_improved_model.py")
        print("3. Test with optimized live camera: python main.py --mode camera")
    
    print("\n🎥 OPTIMIZED LIVE CAMERA:")
    print("- Now processes every 1-2 seconds instead of continuous")
    print("- Use +/- keys during live mode to adjust processing interval")
    print("- Much more efficient for real-time monitoring")


def main():
    """Main setup function."""
    print("🎯 SNAP CIRCUIT TRAINING IMPROVEMENT SETUP")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Check current data
    has_original_data, has_individual_data = check_current_training_data()
    
    # Create photography guide
    create_training_instructions()
    
    # Setup environment
    if not setup_training_environment():
        print("❌ Failed to set up training environment")
        return
    
    # Provide next steps
    provide_next_steps(has_original_data, has_individual_data)
    
    print("\n" + "="*50)
    print("🎉 Setup complete! Follow the next steps above.")
    print("💡 Pro tip: Start with 3-5 component types, then expand")
    print("📖 Read 'COMPONENT_PHOTO_GUIDE.md' for detailed photo instructions")


if __name__ == "__main__":
    main() 