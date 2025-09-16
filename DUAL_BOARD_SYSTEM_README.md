# Dual Board Snap Circuit Detection System

A comprehensive computer vision system for simultaneous detection and analysis of two snap circuit boards from a single camera feed. The system provides real-time component detection, orientation validation, connectivity analysis, and circuit scoring with green tape coverage detection to handle hands blocking components during assembly.

## 🎯 System Overview

This system extends your existing snap circuit detection capabilities to handle dual board setups with the following key features:

### Core Capabilities
- **Dual Board Detection**: Split camera feed processing for two circuit boards simultaneously
- **Green Tape Detection**: Automatically skip processing when hands are detected (tape covered)
- **Component Detection**: Accurate detection of all snap circuit components on each board
- **Orientation Analysis**: Validate component orientations and provide correction scores
- **Connectivity Analysis**: Detect connections between components and score circuit completeness
- **Graph Generation**: Create separate network graphs for each board with JSON output
- **Live Visualization**: Real-time display with separate scoring for each board
- **Data Pipeline**: Complete annotation → augmentation → training → deployment workflow

### Technical Features
- **Advanced Data Augmentation**: Rotations (90°, 180°, 270°), color variations, brightness/contrast
- **GPU Acceleration**: Optimized training with automatic batch size calculation
- **Temporal Stability**: Smoothed detection with history-based filtering
- **Parallel Processing**: Simultaneous analysis of both boards for optimal performance
- **Comprehensive Logging**: Detailed training and validation metrics

## 🚀 Quick Start

### Complete Workflow (Recommended)
```bash
# Run entire pipeline from annotation to live detection
python setup_dual_board_system.py --complete-workflow --images new_images/
```

### Step-by-Step Workflow
```bash
# 1. Manual annotation with dual board support
python setup_dual_board_system.py --step annotate --images new_images/

# 2. Generate augmented training data  
python setup_dual_board_system.py --step augment --images new_images/ --annotations dual_board_annotations/

# 3. Train custom dual board model
python setup_dual_board_system.py --step train --data dual_board_augmented_dataset/data.yaml

# 4. Run live detection system
python setup_dual_board_system.py --step live --model models/weights/dual_board_*.pt
```

### Check System Status
```bash
python setup_dual_board_system.py --status
```

## 📋 System Requirements

### Hardware
- **GPU Recommended**: NVIDIA GPU with 6GB+ VRAM for training (RTX 3060+)
- **CPU Minimum**: Intel i5/AMD Ryzen 5 for CPU-only training
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 5GB+ free space for dataset and models
- **Camera**: USB camera or built-in webcam

### Software
- **Python**: 3.8+
- **OpenCV**: 4.5+
- **PyTorch**: 1.9+ with CUDA support (for GPU)
- **Ultralytics**: Latest YOLO implementation
- **NumPy, Pandas, Matplotlib**: Standard data science libraries

### Installation
```bash
# Install required packages
pip install ultralytics opencv-python numpy pandas matplotlib pathlib dataclasses

# Ensure project structure
python setup_dual_board_system.py --status
```

## 📝 Manual Annotation Guide

The dual board annotator provides an advanced interface for annotating two circuit boards simultaneously.

### Annotation Interface
- **Split Screen**: Image automatically split at configurable ratio (default 50/50)
- **Green Tape Detection**: Visual feedback on tape coverage and hand blocking
- **Side Selection**: Automatic side detection based on mouse position
- **Class Selection**: Number keys (0-9) to select component classes
- **Real-time Preview**: Live annotation feedback with bounding boxes

### Controls
| Key/Mouse | Action |
|-----------|---------|
| **Mouse Drag** | Draw bounding box |
| **Right Click** | Delete last annotation |
| **0-9** | Select component class |
| **TAB** | Switch between left/right side |
| **SPACE** | Toggle green tape detection overlay |
| **s** | Save annotations and continue to next image |
| **r** | Reset current image annotations |
| **q** | Quit annotation session |
| **h** | Show help |

### Annotation Guidelines
- **Complete Components**: Label entire functional units, not individual parts
- **Consistent Orientation**: Use standard orientation for similar components
- **Side-Specific**: Each side gets separate annotation files (`*_left.txt`, `*_right.txt`)
- **Green Tape Awareness**: Skip images where tape is covered (hands detected)

### Output Format
```
annotations/
├── image001_left.txt     # Left board annotations (YOLO format)
├── image001_right.txt    # Right board annotations (YOLO format)
├── image002_left.txt
├── image002_right.txt
└── ...
```

## 🎨 Data Augmentation

The augmentation pipeline creates diverse training data to improve model robustness.

### Augmentation Types
1. **Geometric Transformations**
   - Rotations: 90°, 180°, 270° (with proper annotation adjustment)
   - Maintains annotation accuracy through coordinate transformation

2. **Color Variations**
   - HSV adjustments: Hue shift (-20° to +20°)
   - Saturation multiplier: 0.8x to 1.2x
   - Brightness multiplier: 0.8x to 1.2x

3. **Quality Variations**
   - Brightness offset: -30 to +30
   - Contrast multiplier: 0.8x to 1.2x
   - Gaussian noise addition (configurable intensity)

4. **Combined Augmentations**
   - Rotation + Color variations
   - Multiple random combinations per image

### Configuration
```python
config = AugmentationConfig(
    enable_rotations=True,
    rotation_angles=[90, 180, 270],
    enable_color_variations=True,
    max_augmentations_per_image=12,
    preserve_original=True
)
```

### Output Dataset Structure
```
dual_board_augmented_dataset/
├── data.yaml                    # Dataset configuration
├── images/
│   ├── train/                   # Training images
│   ├── val/                     # Validation images  
│   └── test/                    # Test images
└── labels/
    ├── train/                   # Training annotations
    ├── val/                     # Validation annotations
    └── test/                    # Test annotations
```

## 🏋️‍♂️ Model Training

Optimized training pipeline specifically designed for circuit component detection.

### Training Features
- **Smart Batch Sizing**: Automatic calculation based on GPU memory
- **Circuit-Optimized Augmentation**: Parameters tuned for circuit stability
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: Cosine annealing for optimal convergence
- **Comprehensive Logging**: Training metrics, validation results, and summaries

### Training Parameters
```python
training_params = {
    'epochs': 200,
    'optimizer': 'AdamW',
    'lr0': 0.001,              # Initial learning rate
    'cos_lr': True,            # Cosine learning rate schedule
    'patience': 30,            # Early stopping patience
    
    # Loss weights (optimized for circuits)
    'box': 7.5,                # Box loss weight
    'cls': 0.5,                # Classification loss weight
    
    # Augmentation (conservative for circuits)
    'degrees': 5.0,            # Small rotation range
    'flipud': 0.0,             # No vertical flip (orientation matters)
    'mosaic': 0.3,             # Light mosaic augmentation
}
```

### GPU Memory Optimization
| GPU Memory | Batch Size | Expected Performance |
|-----------|------------|---------------------|
| 24GB+ (RTX 4090) | 32 | Excellent |
| 16GB (RTX 4080) | 24 | Very Good |
| 12GB (RTX 4070 Ti) | 16 | Good |
| 8GB (RTX 3070) | 12 | Moderate |
| 6GB (RTX 3060) | 8 | Basic |

### Training Output
```
models/weights/
├── dual_board_[experiment].pt   # Best trained model
└── training_summaries/
    └── dual_board_[experiment]_summary.json
```

## 📹 Live Detection System

Real-time dual board detection with comprehensive analysis and scoring.

### System Components

#### 1. Green Tape Detector
- **Purpose**: Detect when green tape is covered (hands blocking components)
- **Algorithm**: HSV color space analysis with temporal smoothing
- **Output**: Coverage percentage for each side, processing recommendations

#### 2. Board Analyzer
- **Component Detection**: YOLO-based detection of all circuit components
- **Connection Detection**: Analysis of electrical connections between components
- **Orientation Validation**: Verification of correct component orientations
- **Connectivity Scoring**: Assessment of circuit completeness and correctness

#### 3. Dual Processing Pipeline
- **Frame Splitting**: Automatic division of camera feed at configurable ratio
- **Parallel Processing**: Simultaneous analysis of both boards using ThreadPoolExecutor
- **Result Synchronization**: Coordinated output with timestamp alignment

### Live System Features

#### Real-time Analysis
- **Processing Interval**: Configurable (default 2 seconds)
- **Frame Rate Independent**: Maintains consistent analysis regardless of camera FPS
- **Adaptive Processing**: Skips frames when hands are detected (tape covered)

#### Scoring System
- **Orientation Score**: Percentage of correctly oriented components (0-100%)
- **Connectivity Score**: Assessment of proper connections (0-100%)
- **Overall Score**: Combined metric for circuit correctness (0-100%)

#### Output Generation
- **JSON Files**: Detailed detection data for each board
- **Graph Files**: Network representation of circuit connectivity
- **Visualizations**: Circuit diagrams with component and connection overlays
- **Annotated Images**: Visual feedback with bounding boxes and scores

### Live System Controls
| Key | Action |
|-----|--------|
| **q** | Quit system |
| **s** | Save current frame manually |
| **p** | Pause/Resume processing |
| **t** | Toggle green tape detection overlay |
| **SPACE** | Force process current frame |

### Output Directory Structure
```
dual_board_output/
├── left/                        # Left board outputs
│   ├── detection_[timestamp].json
│   ├── graph_[timestamp].json
│   ├── board_[timestamp].jpg
│   ├── circuit_visual_[timestamp].png
│   └── latest_circuit_visual.png
├── right/                       # Right board outputs
│   └── [same structure as left]
└── combined/                    # Combined analysis
    ├── analysis_[timestamp].json
    └── frame_[timestamp].jpg
```

## 🎯 Advanced Configuration

### Green Tape Detection Tuning
```python
# Adjust HSV color ranges for your green tape
green_lower = np.array([35, 40, 40])    # Lower HSV bound
green_upper = np.array([85, 255, 255])  # Upper HSV bound

# Coverage thresholds
min_tape_area_ratio = 0.005      # Minimum visible tape
coverage_threshold = 0.7         # Coverage drop indicating hands
```

### Camera Configuration
```python
camera_config = {
    "resolution": (1920, 1080),  # Full HD recommended
    "fps": 30,                   # Standard frame rate
    "buffer_size": 1,            # Minimize latency
    "split_ratio": 0.5,          # Adjust board position
}
```

### Model Performance Tuning
```python
detection_config = {
    "confidence_threshold": 0.25,  # Lower for better recall
    "iou_threshold": 0.4,          # Balanced NMS
    "max_detections": 50,          # Sufficient for circuit boards
}
```

## 📊 Performance Metrics

### Training Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision**: True Positive / (True Positive + False Positive)
- **Recall**: True Positive / (True Positive + False Negative)

### Live Detection Metrics
- **Processing Time**: Time per frame analysis
- **Detection Count**: Components detected per board
- **Orientation Score**: Percentage correctly oriented
- **Connectivity Score**: Percentage properly connected
- **Overall Score**: Combined circuit correctness

## 🛠️ Troubleshooting

### Common Issues

#### 1. Camera Not Found
```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"

# Try different camera IDs
python setup_dual_board_system.py --step live --camera 1
```

#### 2. GPU Training Issues
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Force CPU training if GPU issues
python train_dual_board_model.py --cpu --data dataset/data.yaml
```

#### 3. Green Tape Detection Problems
- Adjust HSV color ranges in `DualBoardLiveSystem`
- Check lighting conditions and tape color consistency
- Use tape detection overlay (`t` key) to debug

#### 4. Poor Detection Accuracy
- Increase training epochs (200+ recommended)
- Add more diverse training images
- Check annotation quality and consistency
- Verify proper data augmentation

### Performance Optimization

#### Memory Issues
```bash
# Reduce batch size for low-memory GPUs
python train_dual_board_model.py --batch 4 --data dataset/data.yaml

# Use smaller base model
python setup_dual_board_system.py --step train --base-model yolov8n.pt
```

#### Speed Optimization
```python
# Adjust processing interval for real-time needs
system = DualBoardLiveSystem(processing_interval=1.0)  # Process every second

# Reduce image resolution for faster processing
camera_config = {"resolution": (1280, 720)}
```

## 📈 System Extensions

### Adding New Component Classes
1. Update `classes.txt` with new component names
2. Re-annotate images with new components
3. Retrain model with updated dataset
4. Update validation logic if needed

### Custom Scoring Algorithms
```python
class CustomBoardAnalyzer(BoardAnalyzer):
    def _calculate_connectivity_score(self, graph):
        # Implement custom connectivity scoring
        return custom_score
```

### Integration with External Systems
- **Robot Integration**: Use JSON outputs for robotic assembly guidance
- **Database Logging**: Store results in database for historical analysis  
- **Web Interface**: Create web dashboard for remote monitoring
- **Alert Systems**: Implement notifications for circuit completion/errors

## 📚 API Reference

### Core Classes

#### DualBoardAnnotator
```python
annotator = DualBoardAnnotator("classes.txt")
annotator.batch_annotate("images/", "annotations/")
```

#### DualBoardAugmentationPipeline
```python
pipeline = DualBoardAugmentationPipeline(config)
data_yaml = pipeline.create_augmented_dataset(images_dir, annotations_dir, output_dir)
```

#### DualBoardModelTrainer
```python
trainer = DualBoardModelTrainer(base_model="yolov8x.pt")
model_path = trainer.train_model(data_yaml_path, epochs=200)
```

#### DualBoardLiveSystem
```python
system = DualBoardLiveSystem(model_path=model_path, split_ratio=0.5)
system.run_live_detection(camera_id=0)
```

## 🤝 Contributing

This system is designed to be extensible. Key areas for contribution:

1. **New Component Types**: Add support for additional snap circuit components
2. **Detection Algorithms**: Improve component and connection detection accuracy
3. **Scoring Methods**: Develop more sophisticated circuit analysis
4. **User Interface**: Create GUI applications for easier use
5. **Integration**: Connect with external systems and databases

## 📄 License and Credits

This dual board system extends the original snap circuit detection system with advanced dual board capabilities, automated annotation tools, comprehensive data augmentation, and real-time analysis features.

---

## 🚀 Getting Started Now

1. **Check your setup**:
   ```bash
   python setup_dual_board_system.py --status
   ```

2. **Run complete workflow**:
   ```bash
   python setup_dual_board_system.py --complete-workflow --images new_images/
   ```

3. **Or start with annotation**:
   ```bash
   python setup_dual_board_system.py --step annotate --images new_images/
   ```

The system will guide you through each step with detailed feedback and instructions. For best results, ensure good lighting, clear green tape visibility, and consistent camera positioning for dual board setups.
