# Snap Circuit Computer Vision System 

A comprehensive computer vision pipeline for analyzing Snap Circuit boards using YOLOv8 object detection, OpenCV connection detection, and NetworkX graph analysis.

## System Overview

This system provides real-time analysis of Snap Circuit boards through:

- **Component Detection**: YOLOv8-based identification of circuit pieces (LEDs, batteries, wires, etc.)
- **Connection Analysis**: OpenCV-based wire tracing and component connection detection  
- **Circuit State Inference**: NetworkX graph analysis to determine if circuits are closed, powered, and functional
- **Live Circuit Visualization**: Automatic generation of beautiful circuit board visualizations in real-time
- **Graph Format Output**: TD-BKT compatible graph format for downstream algorithms
- **Real-time Processing**: Live camera feed analysis with annotated output

## Quick Start

### Prerequisites

- Python 3.8+
- OpenCV-compatible camera (for real-time mode)
- CUDA-compatible GPU (recommended for faster inference)

### Installation

**Clone and setup environment:**
```bash
git clone <repository-url>
cd robotics-research
pip install -r requirements.txt
```

## Usage

### Real-time Camera Analysis

```bash
# Basic real-time detection (outputs JSON + live visualizations)
python main.py --mode camera

# Use specific camera and custom model
python main.py --mode camera --camera 1 --model models/weights/snap_circuit_yolov8.pt

# Disable display for headless operation
python main.py --mode camera --no-display

# The system will automatically:
# Process frames every 3 seconds
# Generate live circuit visualizations (PNG files)
# Save detection data (JSON files)
# Display real-time annotated camera feed

# Enable circuit validation for correctness checking
python main.py --mode camera --validate

# The system will additionally:
# Validate circuit correctness in real-time
# Check component orientations and polarity
# Display validation scores on visualizations
# Show specific issues and suggestions
```

**Real-time Controls:**
- `q`: Quit
- `s`: Save current frame
- `p`: Pause/resume detection
- `+/-`: Adjust processing interval

### Live Circuit Visualization

The system automatically generates circuit board visualizations in real-time:

**Automatic Generation**: 
- Processes camera feed every 3 seconds
- Creates PNG visualizations automatically
- No manual intervention required

**Visual Features**:
- **Hexagonal grid background** mimicking actual snap circuit boards
- **Color-coded components** with distinct colors for each type:
  - Battery holders (blue), LEDs (pink), switches (teal)
  - Buttons (green), wires (gold), motors (purple)
  - Resistors (red), speakers (magenta), and 10+ other types
- **Connection lines** showing component relationships
- **Component labels** with confidence scores
- **Circuit status** indicating if circuit is complete and powered
- **Validation scores** (when validation is enabled) showing:
  - Overall result (✅ Correct, ⚠️ Partial, ❌ Incorrect)
  - Quality score (0-100%)
  - Issue counts (errors and warnings)
- **Timestamp information** for each visualization

**Features**:
- Real-time digital twin of your snap circuit board
- Intelligent connection detection between nearby components
- High-confidence component filtering (>75% confidence)
- Actual layout based on camera-detected positions

### Single Image Analysis

```bash
# Analyze single image
python main.py --mode image --input circuit_image.jpg
```


## Configuration

Key settings in `config.py`:

```python
# Performance optimized for real-time
VIDEO_CONFIG = {
    "processing_interval": 3.0,  # Process every 3 seconds (optimized for visualization)
    "resolution": [1920, 1080],
    "fps": 30
}

# YOLO Detection
YOLO_CONFIG = {
    "confidence_threshold": 0.5,
    "device": "cuda"  # Uses GPU if available
}

# Live Visualization
VISUALIZATION_CONFIG = {
    "confidence_threshold": 0.75,  # Only show high-confidence components
    "connection_tolerance": 35,     # Pixel distance for connection detection
    "save_latest": True            # Always save latest_circuit_visual.png
}
```
