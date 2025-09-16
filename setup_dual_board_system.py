#!/usr/bin/env python3
"""
Dual Board System Setup Script

This script provides a complete setup and workflow for the dual board snap circuit
detection system. It guides users through the entire process from annotation to
training to live detection.

Workflow:
1. Annotate images manually with dual board setup
2. Generate augmented training data
3. Train the model
4. Run live detection system

Usage Examples:
  python setup_dual_board_system.py --step annotate --images new_images/
  python setup_dual_board_system.py --step augment --annotations annotations/
  python setup_dual_board_system.py --step train --data augmented_dataset/data.yaml
  python setup_dual_board_system.py --step live --model trained_model.pt
  python setup_dual_board_system.py --complete-workflow --images new_images/
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
import json

# Import our dual board components
from dual_board_annotator import DualBoardAnnotator
from dual_board_data_augmentation import DualBoardAugmentationPipeline, AugmentationConfig
from train_dual_board_model import DualBoardModelTrainer
from dual_board_live_system import DualBoardLiveSystem


class DualBoardSystemSetup:
    """Complete setup manager for dual board system"""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the setup manager"""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.classes_file = self.project_root / "classes.txt"
        
        print("🎯 Dual Board System Setup Manager")
        print(f"📁 Project root: {self.project_root}")
        
        # Ensure classes file exists
        if not self.classes_file.exists():
            print(f"⚠️  Classes file not found: {self.classes_file}")
            print("   Creating default classes file...")
            self.create_default_classes_file()
    
    def create_default_classes_file(self):
        """Create default classes.txt file"""
        default_classes = [
            "wire",
            "switch", 
            "button",
            "battery_holder",
            "led",
            "speaker",
            "music_circuit", 
            "motor",
            "resistor",
            "connection_node",
            "lamp",
            "fan",
            "buzzer",
            "photoresistor",
            "microphone",
            "alarm"
        ]
        
        with open(self.classes_file, 'w') as f:
            for class_name in default_classes:
                f.write(f"{class_name}\n")
        
        print(f"✅ Created default classes file: {self.classes_file}")
        print(f"   Classes: {', '.join(default_classes)}")
    
    def step_annotate(self, images_dir: str, output_dir: str = "annotations") -> bool:
        """
        Step 1: Manual annotation of dual board images
        
        Args:
            images_dir: Directory containing images to annotate
            output_dir: Output directory for annotations
            
        Returns:
            True if annotations were created
        """
        print("📝 STEP 1: MANUAL ANNOTATION")
        print("="*40)
        
        images_path = Path(images_dir)
        if not images_path.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return False
        
        # Count images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(images_path.glob(ext))
        
        if not image_files:
            print(f"❌ No images found in {images_dir}")
            return False
        
        print(f"📸 Found {len(image_files)} images to annotate")
        print(f"💾 Annotations will be saved to: {output_dir}")
        print(f"📋 Using classes from: {self.classes_file}")
        
        print("\n🎯 Starting dual board annotation...")
        print("Features:")
        print("  • Split-screen view for dual boards")
        print("  • Green tape detection for hand blocking")
        print("  • Separate annotations for left/right sides")
        print("  • YOLO format output")
        
        # Initialize annotator
        try:
            annotator = DualBoardAnnotator(str(self.classes_file))
            annotator.batch_annotate(images_dir, output_dir)
            
            # Verify annotations were created
            output_path = Path(output_dir)
            if output_path.exists():
                annotation_files = list(output_path.glob('*.txt'))
                if annotation_files:
                    print(f"✅ Created {len(annotation_files)} annotation files")
                    return True
                else:
                    print("⚠️  No annotation files were created")
                    return False
            else:
                print("⚠️  Output directory was not created")
                return False
                
        except Exception as e:
            print(f"❌ Annotation failed: {e}")
            return False
    
    def step_augment(self, images_dir: str, annotations_dir: str, 
                    output_dataset_dir: str = "dual_board_dataset", 
                    max_augmentations: int = 12) -> Optional[str]:
        """
        Step 2: Data augmentation pipeline
        
        Args:
            images_dir: Directory containing original images
            annotations_dir: Directory containing annotations from step 1
            output_dataset_dir: Output directory for augmented dataset
            max_augmentations: Maximum augmentations per image
            
        Returns:
            Path to data.yaml file if successful
        """
        print("🎨 STEP 2: DATA AUGMENTATION")
        print("="*35)
        
        images_path = Path(images_dir)
        annotations_path = Path(annotations_dir)
        
        if not images_path.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return None
        
        if not annotations_path.exists():
            print(f"❌ Annotations directory not found: {annotations_dir}")
            return None
        
        print(f"📸 Images: {images_dir}")
        print(f"📋 Annotations: {annotations_dir}")
        print(f"💾 Output dataset: {output_dataset_dir}")
        print(f"🎨 Max augmentations per image: {max_augmentations}")
        
        print("\n🎨 Augmentations to apply:")
        print("  • Rotations: 90°, 180°, 270°")
        print("  • Color variations: HSV adjustments")
        print("  • Brightness/contrast changes")
        print("  • Noise addition")
        print("  • Combined augmentations")
        
        try:
            # Create augmentation configuration
            config = AugmentationConfig(
                enable_rotations=True,
                rotation_angles=[90, 180, 270],
                enable_color_variations=True,
                enable_noise=True,
                max_augmentations_per_image=max_augmentations
            )
            
            # Initialize pipeline
            pipeline = DualBoardAugmentationPipeline(config)
            pipeline.load_classes(str(self.classes_file))
            
            # Create augmented dataset
            data_yaml_path = pipeline.create_augmented_dataset(
                images_dir, annotations_dir, output_dataset_dir
            )
            
            if data_yaml_path and Path(data_yaml_path).exists():
                print(f"✅ Augmented dataset created successfully!")
                print(f"📄 Dataset config: {data_yaml_path}")
                return data_yaml_path
            else:
                print("❌ Failed to create augmented dataset")
                return None
                
        except Exception as e:
            print(f"❌ Augmentation failed: {e}")
            return None
    
    def step_train(self, data_yaml_path: str, 
                  base_model: str = "yolov8x.pt",
                  epochs: int = 200,
                  experiment_name: Optional[str] = None) -> Optional[str]:
        """
        Step 3: Model training
        
        Args:
            data_yaml_path: Path to augmented dataset configuration
            base_model: Base YOLO model to start from
            epochs: Number of training epochs
            experiment_name: Name for training experiment
            
        Returns:
            Path to trained model if successful
        """
        print("🏋️‍♂️ STEP 3: MODEL TRAINING")
        print("="*30)
        
        data_path = Path(data_yaml_path)
        if not data_path.exists():
            print(f"❌ Dataset config not found: {data_yaml_path}")
            return None
        
        if not experiment_name:
            experiment_name = f"dual_board_{int(time.time())}"
        
        print(f"📄 Dataset: {data_yaml_path}")
        print(f"🎯 Base model: {base_model}")
        print(f"🔄 Epochs: {epochs}")
        print(f"🏷️  Experiment: {experiment_name}")
        
        print("\n🏋️‍♂️ Training optimizations:")
        print("  • GPU acceleration (if available)")
        print("  • Optimal batch size calculation")
        print("  • Circuit-specific augmentation parameters")
        print("  • Early stopping with patience")
        print("  • Learning rate scheduling")
        
        try:
            # Initialize trainer
            trainer = DualBoardModelTrainer(
                base_model=base_model,
                experiment_name=experiment_name,
                use_gpu=True
            )
            
            print("\n🚀 Starting training...")
            print("   This may take 30 minutes to several hours depending on:")
            print("   • Dataset size")
            print("   • Number of epochs") 
            print("   • Hardware capabilities")
            
            # Train model
            model_path = trainer.train_model(
                data_yaml_path=data_yaml_path,
                epochs=epochs
            )
            
            if model_path and Path(model_path).exists():
                print(f"✅ Training completed successfully!")
                print(f"🎉 Trained model: {model_path}")
                
                # Run validation
                print("\n🧪 Running model validation...")
                metrics = trainer.validate_model(model_path, data_yaml_path)
                
                if metrics:
                    print("📊 Validation Results:")
                    for metric, value in metrics.items():
                        if value is not None:
                            print(f"   • {metric}: {value:.3f}")
                
                return model_path
            else:
                print("❌ Training failed!")
                return None
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            return None
    
    def step_live(self, model_path: Optional[str] = None,
                 camera_id: int = 0,
                 split_ratio: float = 0.5,
                 processing_interval: float = 2.0) -> bool:
        """
        Step 4: Live detection system
        
        Args:
            model_path: Path to trained model (optional)
            camera_id: Camera device ID
            split_ratio: Where to split the frame
            processing_interval: Seconds between processing
            
        Returns:
            True if system started successfully
        """
        print("📹 STEP 4: LIVE DETECTION SYSTEM")
        print("="*40)
        
        if model_path:
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                print(f"❌ Model not found: {model_path}")
                return False
            print(f"🎯 Using trained model: {model_path}")
        else:
            print("⚠️  No model specified - using base YOLO model")
        
        print(f"📹 Camera: {camera_id}")
        print(f"🔄 Split ratio: {split_ratio}")
        print(f"⏱️  Processing interval: {processing_interval}s")
        
        print("\n🎯 Live system features:")
        print("  • Dual board detection with frame splitting")
        print("  • Green tape coverage detection")
        print("  • Real-time component and connection detection")
        print("  • Graph generation for each board")
        print("  • Component orientation validation")
        print("  • Circuit connectivity scoring")
        print("  • Live visualization windows")
        print("  • JSON output with timestamps")
        
        try:
            # Initialize live system
            system = DualBoardLiveSystem(
                model_path=model_path,
                split_ratio=split_ratio,
                processing_interval=processing_interval,
                save_outputs=True,
                display_results=True
            )
            
            print(f"\n🚀 Starting live detection system...")
            print("   Controls:")
            print("     • 'q': Quit")
            print("     • 's': Save current frame")
            print("     • 'p': Pause/Resume")
            print("     • 't': Toggle tape detection overlay")
            print("     • SPACE: Force process current frame")
            
            # Run live detection
            system.run_live_detection(camera_id)
            
            print("✅ Live system completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Live detection error: {e}")
            return False
    
    def complete_workflow(self, images_dir: str,
                         base_model: str = "yolov8x.pt",
                         epochs: int = 200,
                         camera_id: int = 0) -> bool:
        """
        Run complete workflow from annotation to live detection
        
        Args:
            images_dir: Directory containing images to process
            base_model: Base YOLO model for training
            epochs: Training epochs
            camera_id: Camera for live detection
            
        Returns:
            True if complete workflow succeeded
        """
        print("🚀 COMPLETE DUAL BOARD WORKFLOW")
        print("="*50)
        
        print("This workflow will:")
        print("1. 📝 Guide you through manual annotation")
        print("2. 🎨 Generate augmented training data")
        print("3. 🏋️‍♂️ Train a custom dual board model")
        print("4. 📹 Launch live detection system")
        print()
        
        # Step 1: Annotation
        print("🔄 Starting Step 1: Annotation...")
        if not self.step_annotate(images_dir, "dual_board_annotations"):
            print("❌ Workflow stopped at annotation step")
            return False
        
        # Step 2: Augmentation
        print("\n🔄 Starting Step 2: Data Augmentation...")
        data_yaml_path = self.step_augment(
            images_dir, 
            "dual_board_annotations", 
            "dual_board_augmented_dataset"
        )
        
        if not data_yaml_path:
            print("❌ Workflow stopped at augmentation step")
            return False
        
        # Step 3: Training
        print("\n🔄 Starting Step 3: Model Training...")
        model_path = self.step_train(
            data_yaml_path,
            base_model=base_model,
            epochs=epochs,
            experiment_name="dual_board_complete_workflow"
        )
        
        if not model_path:
            print("❌ Workflow stopped at training step")
            return False
        
        # Step 4: Live Detection
        print("\n🔄 Starting Step 4: Live Detection...")
        success = self.step_live(
            model_path=model_path,
            camera_id=camera_id
        )
        
        if success:
            print("\n🎉 COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
            print(f"   • Annotations: dual_board_annotations/")
            print(f"   • Dataset: dual_board_augmented_dataset/")
            print(f"   • Model: {model_path}")
            print(f"   • Live outputs: dual_board_output/")
            return True
        else:
            print("❌ Workflow stopped at live detection step")
            return False
    
    def print_system_status(self):
        """Print current system status and available options"""
        print("📊 DUAL BOARD SYSTEM STATUS")
        print("="*40)
        
        # Check for existing components
        status = {
            "classes_file": self.classes_file.exists(),
            "annotations": Path("dual_board_annotations").exists(),
            "augmented_dataset": Path("dual_board_augmented_dataset").exists(),
            "trained_models": len(list(Path("models/weights").glob("dual_board_*.pt"))) if Path("models/weights").exists() else 0
        }
        
        print("✅" if status["classes_file"] else "❌", f"Classes file: {self.classes_file}")
        print("✅" if status["annotations"] else "❌", "Annotations directory: dual_board_annotations/")
        print("✅" if status["augmented_dataset"] else "❌", "Augmented dataset: dual_board_augmented_dataset/")
        print(f"{'✅' if status['trained_models'] > 0 else '❌'} Trained models: {status['trained_models']} found")
        
        print("\n💡 Recommended next steps:")
        if not status["annotations"]:
            print("   1. Run annotation: --step annotate --images new_images/")
        elif not status["augmented_dataset"]:
            print("   2. Run augmentation: --step augment --annotations dual_board_annotations/")
        elif status["trained_models"] == 0:
            print("   3. Run training: --step train --data dual_board_augmented_dataset/data.yaml")
        else:
            print("   4. Run live detection: --step live --model models/weights/dual_board_*.pt")
            print("   OR run complete workflow: --complete-workflow --images new_images/")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Dual Board System Setup")
    
    # Main workflow options
    parser.add_argument("--step", choices=["annotate", "augment", "train", "live"], 
                       help="Run specific step")
    parser.add_argument("--complete-workflow", action="store_true", 
                       help="Run complete workflow from annotation to live detection")
    parser.add_argument("--status", action="store_true", 
                       help="Show system status and recommendations")
    
    # Input/output paths
    parser.add_argument("--images", help="Images directory for annotation")
    parser.add_argument("--annotations", help="Annotations directory for augmentation")
    parser.add_argument("--data", help="Data.yaml path for training")
    parser.add_argument("--model", help="Model path for live detection")
    parser.add_argument("--output", help="Output directory (default varies by step)")
    
    # Training options
    parser.add_argument("--base-model", default="yolov8x.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--experiment", help="Training experiment name")
    
    # Live detection options  
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--split", type=float, default=0.5, help="Frame split ratio")
    parser.add_argument("--interval", type=float, default=2.0, help="Processing interval")
    
    # Augmentation options
    parser.add_argument("--max-aug", type=int, default=12, help="Max augmentations per image")
    
    args = parser.parse_args()
    
    # Initialize setup manager
    setup = DualBoardSystemSetup()
    
    # Handle status check
    if args.status:
        setup.print_system_status()
        return
    
    # Handle complete workflow
    if args.complete_workflow:
        if not args.images:
            print("❌ --images required for complete workflow")
            return
        
        success = setup.complete_workflow(
            images_dir=args.images,
            base_model=args.base_model,
            epochs=args.epochs,
            camera_id=args.camera
        )
        
        if not success:
            sys.exit(1)
        return
    
    # Handle individual steps
    if args.step == "annotate":
        if not args.images:
            print("❌ --images required for annotation step")
            return
        
        output_dir = args.output or "dual_board_annotations"
        success = setup.step_annotate(args.images, output_dir)
        if not success:
            sys.exit(1)
    
    elif args.step == "augment":
        if not args.images or not args.annotations:
            print("❌ --images and --annotations required for augmentation step")
            return
        
        output_dir = args.output or "dual_board_augmented_dataset"
        data_yaml = setup.step_augment(
            args.images, args.annotations, output_dir, args.max_aug
        )
        if not data_yaml:
            sys.exit(1)
    
    elif args.step == "train":
        if not args.data:
            print("❌ --data required for training step")
            return
        
        model_path = setup.step_train(
            args.data, args.base_model, args.epochs, args.experiment
        )
        if not model_path:
            sys.exit(1)
    
    elif args.step == "live":
        success = setup.step_live(
            args.model, args.camera, args.split, args.interval
        )
        if not success:
            sys.exit(1)
    
    else:
        # No specific action - show help
        setup.print_system_status()
        print("\n💡 Use --help to see all available options")
        print("   Example workflows:")
        print("   • Complete: python setup_dual_board_system.py --complete-workflow --images new_images/")
        print("   • Step by step:")
        print("     1. python setup_dual_board_system.py --step annotate --images new_images/")
        print("     2. python setup_dual_board_system.py --step augment --images new_images/ --annotations dual_board_annotations/")
        print("     3. python setup_dual_board_system.py --step train --data dual_board_augmented_dataset/data.yaml")
        print("     4. python setup_dual_board_system.py --step live --model models/weights/dual_board_*.pt")


if __name__ == "__main__":
    main()
