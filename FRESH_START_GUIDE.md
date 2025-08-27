# Fresh Start Guide: Snap Circuit Component Detection

This guide will help you completely restart your snap circuit component detection and connection detection process with a clean, modern approach.

## Overview

The fresh start pipeline consists of 4 main steps:

1. **Environment Setup** - Prepare directories, validate data, setup annotation tools
2. **Data Annotation** - Annotate images with component bounding boxes (if needed)
3. **Model Training** - Train YOLOv8 model on annotated data
4. **Detection Testing** - Test the trained model on sample images

## Quick Start

### Option 1: Complete Pipeline (Recommended)
```bash
python fresh_start_main.py
```

### Option 2: Step-by-Step
```bash
# Step 1: Setup environment only
python fresh_start_main.py --setup-only

# Step 2: Annotate data (if needed)
python fresh_start_main.py --step 2

# Step 3: Train model
python fresh_start_main.py --step 3

# Step 4: Test detection
python fresh_start_main.py --step 4
```

### Option 3: Force Re-annotation
```bash
# If you want to re-annotate all data
python fresh_start_main.py --force-reannotate
```

## Detailed Steps

### Step 1: Environment Setup

The setup process:
- Creates necessary output directories
- Validates existing dataset structure
- Sets up annotation tools (labelImg)
- Creates backup of existing annotations

**Files created:**
- `output/models/` - For trained models
- `output/results/` - For training results
- `output/visualizations/` - For detection visualizations
- `output/logs/` - For training logs

### Step 2: Data Annotation

If you need to re-annotate your images:

1. **Automatic Setup**: The script will install labelImg if needed
2. **Manual Annotation**: Use labelImg to draw bounding boxes around components
3. **Classes**: 16 component types are supported:
   - wire, switch, button, battery_holder, led
   - speaker, music_circuit, motor, resistor
   - connection_node, lamp, fan, buzzer
   - photoresistor, microphone, alarm

**Annotation Tips:**
- Press 'W' to create bounding box
- Press 'D' for next image
- Press 'A' for previous image
- Press 'Ctrl+S' to save
- Make sure to select correct class for each component

### Step 3: Model Training

The training process uses YOLOv8 with optimized settings:

**Configuration:**
- Model: YOLOv8n (nano) - fast and efficient
- Epochs: 100 with early stopping (patience=20)
- Batch size: 16
- Image size: 640x640
- Optimizer: Auto (Adam/SGD)
- Learning rate: 0.01 with cosine annealing

**Training Features:**
- Automatic validation
- Early stopping to prevent overfitting
- Model checkpointing every 10 epochs
- Comprehensive logging
- Performance metrics tracking

### Step 4: Detection Testing

The detection pipeline includes:

**Component Detection:**
- YOLO-based object detection
- Confidence threshold: 0.25
- NMS threshold: 0.45
- Maximum detections: 100

**Connection Detection:**
- Distance-based connection detection
- Connection types: wire, direct, aligned, proximity
- Distance threshold: 50 pixels
- Angle threshold: 30 degrees

**Visualization:**
- Component bounding boxes with labels
- Connection lines with different styles
- Confidence scores
- Color-coded by component type

## Configuration

All settings are in `fresh_start_config.py`:

```python
# Training settings
MODEL_SIZE = "n"  # n, s, m, l, x
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
PATIENCE = 20

# Detection settings
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
MAX_DETECTIONS = 100

# Connection detection
CONNECTION_DISTANCE_THRESHOLD = 50
CONNECTION_ANGLE_THRESHOLD = 30
```

## Dataset Structure

Your dataset should be organized as:

```
data/augmented_training/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Output Files

After running the pipeline, you'll find:

**Models:**
- `output/models/snap_circuit_detector.pt` - Trained model

**Results:**
- `output/results/fresh_start_YYYYMMDD_HHMMSS/` - Training results
- Confusion matrix, PR curves, training plots

**Visualizations:**
- `output/visualizations/detection_YYYYMMDD_HHMMSS.png` - Detection results

**Analysis:**
- `output/circuit_analysis.json` - Detailed analysis results

## Troubleshooting

### Common Issues

1. **labelImg not found**
   ```bash
   pip install labelImg
   ```

2. **CUDA out of memory**
   - Reduce batch size in config
   - Use smaller model size (n instead of s/m/l/x)

3. **No detections**
   - Lower confidence threshold
   - Check if model was trained properly
   - Verify annotations are correct

4. **Poor detection accuracy**
   - Add more training data
   - Improve annotation quality
   - Increase training epochs
   - Try larger model size

### Performance Optimization

- **Fast training**: Use YOLOv8n (nano)
- **Better accuracy**: Use YOLOv8s (small) or larger
- **GPU training**: Ensure CUDA is available
- **Data augmentation**: Already included in YOLOv8

## Advanced Usage

### Custom Model Configuration

Edit `fresh_start_config.py` to customize:

```python
# For better accuracy (slower training)
MODEL_SIZE = "s"  # or "m", "l", "x"
EPOCHS = 200
BATCH_SIZE = 8

# For faster training (lower accuracy)
MODEL_SIZE = "n"
EPOCHS = 50
BATCH_SIZE = 32
```

### Custom Connection Detection

Modify connection detection logic in `fresh_start_detector.py`:

```python
def _determine_connection_type(self, comp1, comp2, distance, angle):
    # Add your custom logic here
    pass
```

### Batch Processing

To process multiple images:

```python
from fresh_start_detector import FreshStartDetector

detector = FreshStartDetector()
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

for image_path in image_paths:
    analysis = detector.analyze_circuit(image_path)
    print(f"Analysis for {image_path}: {analysis['total_components']} components")
```

## Next Steps

After successful training:

1. **Deploy the model** for real-time detection
2. **Fine-tune** on specific circuit types
3. **Add more components** to the dataset
4. **Improve connection detection** with more sophisticated algorithms
5. **Create a web interface** for easy usage

## Support

If you encounter issues:

1. Check the logs in `output/logs/`
2. Verify your dataset structure
3. Ensure all dependencies are installed
4. Check GPU memory if using CUDA

The fresh start pipeline provides a solid foundation for snap circuit component detection that you can build upon for your specific needs. 