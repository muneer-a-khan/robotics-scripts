"""
Fresh Start Main Script
Complete restart of snap circuit component detection pipeline
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from fresh_start_config import FreshStartConfig
from fresh_start_annotator import FreshStartAnnotator
from fresh_start_trainer import FreshStartTrainer
from fresh_start_detector import FreshStartDetector

class FreshStartPipeline:
    def __init__(self):
        self.config = FreshStartConfig()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def step_1_setup_environment(self):
        """Step 1: Setup the complete environment"""
        self.logger.info("=== STEP 1: Setting up environment ===")
        
        # Create directories
        self.config.create_directories()
        
        # Setup annotation environment
        annotator = FreshStartAnnotator()
        if not annotator.setup_annotation_environment():
            self.logger.error("Failed to setup annotation environment")
            return False
        
        # Validate existing data
        if not annotator.validate_annotations():
            self.logger.warning("Some annotation issues found")
        
        self.logger.info("Environment setup complete!")
        return True
    
    def step_2_annotate_data(self, force_reannotate=False):
        """Step 2: Annotate data if needed"""
        self.logger.info("=== STEP 2: Data annotation ===")
        
        annotator = FreshStartAnnotator()
        
        # Check if annotations exist
        labels_dir = self.config.TRAINING_DATA / "labels"
        if labels_dir.exists() and not force_reannotate:
            self.logger.info("Annotations already exist. Skipping annotation step.")
            self.logger.info("Use --force-reannotate to re-annotate all data.")
            return True
        
        # Create backup of existing annotations
        if labels_dir.exists():
            self.logger.info("Creating backup of existing annotations...")
            annotator.backup_annotations()
        
        # Start annotation
        self.logger.info("Starting annotation process...")
        self.logger.info("This will open labelImg. Please annotate your images.")
        self.logger.info("Press Enter when annotation is complete...")
        
        input("Press Enter to continue...")
        
        # Validate annotations after completion
        if annotator.validate_annotations():
            self.logger.info("Annotation validation successful!")
            return True
        else:
            self.logger.error("Annotation validation failed!")
            return False
    
    def step_3_train_model(self, experiment_name=None):
        """Step 3: Train the model"""
        self.logger.info("=== STEP 3: Model training ===")
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"fresh_start_{timestamp}"
        
        trainer = FreshStartTrainer()
        
        try:
            self.logger.info(f"Starting training experiment: {experiment_name}")
            results, model = trainer.train_model(experiment_name)
            
            # Validate the trained model
            model_path = self.config.get_model_save_path(experiment_name)
            if model_path.exists():
                self.logger.info("Validating trained model...")
                trainer.validate_model(model_path)
            
            self.logger.info("Training completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return False
    
    def step_4_test_detection(self, test_image=None):
        """Step 4: Test the detection pipeline"""
        self.logger.info("=== STEP 4: Testing detection ===")
        
        # Find a test image
        if test_image is None:
            test_images = list(self.config.TRAINING_DATA.glob("images/val/*.jpg"))
            if not test_images:
                test_images = list(self.config.TRAINING_DATA.glob("images/train/*.jpg"))
            
            if test_images:
                test_image = str(test_images[0])
            else:
                self.logger.error("No test images found!")
                return False
        
        try:
            detector = FreshStartDetector()
            
            # Analyze circuit
            self.logger.info(f"Testing detection on: {test_image}")
            analysis = detector.analyze_circuit(test_image)
            
            # Create visualizations
            detections = [detector.ComponentDetection(**comp) for comp in analysis["components"]]
            connections = detector.detect_connections(detections)
            
            visualization_path = detector.visualize_detections(
                test_image, detections, connections
            )
            
            # Print results
            self.logger.info(f"Detection test complete!")
            self.logger.info(f"Components detected: {analysis['total_components']}")
            self.logger.info(f"Connections detected: {analysis['total_connections']}")
            self.logger.info(f"Visualization saved to: {visualization_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Detection test failed: {str(e)}")
            return False
    
    def run_complete_pipeline(self, force_reannotate=False, test_image=None):
        """Run the complete fresh start pipeline"""
        self.logger.info("=== FRESH START PIPELINE ===")
        self.logger.info("Starting complete restart of snap circuit detection...")
        
        # Step 1: Setup environment
        if not self.step_1_setup_environment():
            self.logger.error("Environment setup failed!")
            return False
        
        # Step 2: Annotate data
        if not self.step_2_annotate_data(force_reannotate):
            self.logger.error("Data annotation failed!")
            return False
        
        # Step 3: Train model
        if not self.step_3_train_model():
            self.logger.error("Model training failed!")
            return False
        
        # Step 4: Test detection
        if not self.step_4_test_detection(test_image):
            self.logger.error("Detection test failed!")
            return False
        
        self.logger.info("=== PIPELINE COMPLETE ===")
        self.logger.info("Fresh start completed successfully!")
        return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Fresh Start Snap Circuit Detection Pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], 
                       help="Run specific step only")
    parser.add_argument("--force-reannotate", action="store_true",
                       help="Force re-annotation of all data")
    parser.add_argument("--test-image", type=str,
                       help="Path to test image for detection")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only setup environment and exit")
    
    args = parser.parse_args()
    
    pipeline = FreshStartPipeline()
    
    if args.setup_only:
        pipeline.step_1_setup_environment()
        return
    
    if args.step:
        # Run specific step
        if args.step == 1:
            pipeline.step_1_setup_environment()
        elif args.step == 2:
            pipeline.step_2_annotate_data(args.force_reannotate)
        elif args.step == 3:
            pipeline.step_3_train_model()
        elif args.step == 4:
            pipeline.step_4_test_detection(args.test_image)
    else:
        # Run complete pipeline
        pipeline.run_complete_pipeline(args.force_reannotate, args.test_image)

if __name__ == "__main__":
    main() 