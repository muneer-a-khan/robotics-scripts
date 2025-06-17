"""
Tools for preparing YOLOv8 training data for Snap Circuit components.
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict

from config import COMPONENT_CLASSES, TRAINING_DATA_DIR, TRAINING_CONFIG


class DatasetPreparer:
    """
    Prepares datasets for YOLOv8 training with Snap Circuit components.
    """
    
    def __init__(self, dataset_root: Optional[str] = None):
        """
        Initialize the dataset preparer.
        
        Args:
            dataset_root: Root directory for dataset (uses config default if None)
        """
        self.dataset_root = Path(dataset_root) if dataset_root else TRAINING_DATA_DIR
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        # YOLO dataset structure
        self.images_dir = self.dataset_root / "images"
        self.labels_dir = self.dataset_root / "labels"
        
        # Split directories
        self.splits = ["train", "val", "test"]
        for split in self.splits:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    def create_data_yaml(self, 
                        train_split: float = 0.7,
                        val_split: float = 0.2,
                        test_split: float = 0.1) -> str:
        """
        Create the data.yaml file required for YOLOv8 training.
        
        Args:
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            
        Returns:
            Path to created data.yaml file
        """
        # Verify splits sum to 1
        if abs(train_split + val_split + test_split - 1.0) > 0.01:
            raise ValueError("Data splits must sum to 1.0")
        
        # Create data configuration
        data_config = {
            "path": str(self.dataset_root.absolute()),
            "train": "images/train",
            "val": "images/val", 
            "test": "images/test",
            "nc": len(COMPONENT_CLASSES),  # Number of classes
            "names": COMPONENT_CLASSES
        }
        
        # Save to YAML file
        yaml_path = self.dataset_root / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"Created data.yaml at {yaml_path}")
        print(f"Dataset configuration:")
        print(f"  Classes: {len(COMPONENT_CLASSES)}")
        print(f"  Train split: {train_split:.1%}")
        print(f"  Val split: {val_split:.1%}")
        print(f"  Test split: {test_split:.1%}")
        
        return str(yaml_path)
    
    def convert_annotations(self, 
                          annotations_dir: str,
                          format_type: str = "coco") -> None:
        """
        Convert annotations from various formats to YOLO format.
        
        Args:
            annotations_dir: Directory containing annotation files
            format_type: Format of input annotations ("coco", "pascal_voc", "yolo")
        """
        annotations_path = Path(annotations_dir)
        
        if not annotations_path.exists():
            raise ValueError(f"Annotations directory not found: {annotations_dir}")
        
        if format_type.lower() == "coco":
            self._convert_from_coco(annotations_path)
        elif format_type.lower() == "pascal_voc":
            self._convert_from_pascal_voc(annotations_path)
        elif format_type.lower() == "yolo":
            self._copy_yolo_annotations(annotations_path)
        else:
            raise ValueError(f"Unsupported annotation format: {format_type}")
    
    def _convert_from_coco(self, annotations_path: Path) -> None:
        """Convert COCO format annotations to YOLO format."""
        try:
            import json
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError("pycocotools required for COCO conversion. Install with: pip install pycocotools")
        
        # Find COCO annotation files
        coco_files = list(annotations_path.glob("*.json"))
        
        for coco_file in coco_files:
            print(f"Converting COCO annotations from {coco_file}")
            
            coco = COCO(str(coco_file))
            
            # Get image and annotation info
            img_ids = coco.getImgIds()
            
            for img_id in img_ids:
                img_info = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                
                # Convert to YOLO format
                yolo_annotations = []
                for ann in anns:
                    category_id = ann['category_id']
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # Convert to YOLO format (normalized center coordinates)
                    img_width = img_info['width']
                    img_height = img_info['height']
                    
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    yolo_annotations.append(f"{category_id} {x_center} {y_center} {width} {height}")
                
                # Save YOLO annotation file
                img_name = img_info['file_name']
                base_name = Path(img_name).stem
                label_file = self.labels_dir / "train" / f"{base_name}.txt"
                
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
    
    def _convert_from_pascal_voc(self, annotations_path: Path) -> None:
        """Convert Pascal VOC format annotations to YOLO format."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("xml.etree.ElementTree required for VOC conversion")
        
        # Find XML annotation files
        xml_files = list(annotations_path.glob("*.xml"))
        
        for xml_file in xml_files:
            print(f"Converting VOC annotations from {xml_file}")
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image dimensions
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            # Convert objects to YOLO format
            yolo_annotations = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                
                # Map class name to index
                if class_name in COMPONENT_CLASSES:
                    class_id = COMPONENT_CLASSES.index(class_name)
                else:
                    print(f"Warning: Unknown class '{class_name}' in {xml_file}")
                    continue
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Convert to YOLO format
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
            
            # Save YOLO annotation file
            base_name = xml_file.stem
            label_file = self.labels_dir / "train" / f"{base_name}.txt"
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    def _copy_yolo_annotations(self, annotations_path: Path) -> None:
        """Copy existing YOLO format annotations."""
        txt_files = list(annotations_path.glob("*.txt"))
        
        for txt_file in txt_files:
            dest_file = self.labels_dir / "train" / txt_file.name
            shutil.copy2(txt_file, dest_file)
            print(f"Copied {txt_file} to {dest_file}")
    
    def organize_images(self, 
                       images_dir: str,
                       train_split: float = 0.7,
                       val_split: float = 0.2,
                       test_split: float = 0.1,
                       seed: int = 42) -> None:
        """
        Organize images into train/val/test splits.
        
        Args:
            images_dir: Directory containing all images
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for testing
            seed: Random seed for reproducible splits
        """
        images_path = Path(images_dir)
        
        if not images_path.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")
        
        print(f"Found {len(image_files)} images")
        
        # Shuffle images with fixed seed
        random.seed(seed)
        random.shuffle(image_files)
        
        # Calculate split indices
        total_images = len(image_files)
        train_end = int(total_images * train_split)
        val_end = train_end + int(total_images * val_split)
        
        # Split images
        splits = {
            "train": image_files[:train_end],
            "val": image_files[train_end:val_end],
            "test": image_files[val_end:]
        }
        
        # Copy images to appropriate directories
        for split_name, split_files in splits.items():
            print(f"Copying {len(split_files)} images to {split_name} split")
            
            for img_file in split_files:
                dest_file = self.images_dir / split_name / img_file.name
                shutil.copy2(img_file, dest_file)
                
                # Also move corresponding label file if it exists
                label_file = self.labels_dir / "train" / f"{img_file.stem}.txt"
                if label_file.exists():
                    dest_label = self.labels_dir / split_name / f"{img_file.stem}.txt"
                    if split_name != "train":  # Don't move if already in train
                        shutil.move(label_file, dest_label)
        
        print("Image organization complete!")
        print(f"Train: {len(splits['train'])} images")
        print(f"Val: {len(splits['val'])} images")
        print(f"Test: {len(splits['test'])} images")
    
    def validate_dataset(self) -> Dict[str, any]:
        """
        Validate the prepared dataset.
        
        Returns:
            Validation report
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check if data.yaml exists
        yaml_path = self.dataset_root / "data.yaml"
        if not yaml_path.exists():
            report["errors"].append("data.yaml file not found")
            report["valid"] = False
        
        # Check splits
        split_stats = {}
        for split in self.splits:
            img_dir = self.images_dir / split
            label_dir = self.labels_dir / split
            
            img_files = list(img_dir.glob("*.*"))
            label_files = list(label_dir.glob("*.txt"))
            
            split_stats[split] = {
                "images": len(img_files),
                "labels": len(label_files)
            }
            
            # Check for missing labels
            missing_labels = []
            for img_file in img_files:
                label_file = label_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    missing_labels.append(img_file.name)
            
            if missing_labels:
                report["warnings"].append(
                    f"{split} split: {len(missing_labels)} images missing labels"
                )
            
            # Check for empty label files
            empty_labels = []
            for label_file in label_files:
                if label_file.stat().st_size == 0:
                    empty_labels.append(label_file.name)
            
            if empty_labels:
                report["warnings"].append(
                    f"{split} split: {len(empty_labels)} empty label files"
                )
        
        report["statistics"] = split_stats
        
        # Class distribution analysis
        class_counts = defaultdict(int)
        for split in self.splits:
            label_dir = self.labels_dir / split
            for label_file in label_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            if 0 <= class_id < len(COMPONENT_CLASSES):
                                class_counts[COMPONENT_CLASSES[class_id]] += 1
        
        report["statistics"]["class_distribution"] = dict(class_counts)
        
        return report
    
    def create_sample_annotations(self, num_samples: int = 10) -> None:
        """
        Create sample annotation files for demonstration.
        
        Args:
            num_samples: Number of sample files to create
        """
        print(f"Creating {num_samples} sample annotation files...")
        
        sample_dir = self.dataset_root / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        for i in range(num_samples):
            # Create sample annotation with random bounding boxes
            annotations = []
            num_objects = random.randint(1, 3)
            
            for _ in range(num_objects):
                class_id = random.randint(0, len(COMPONENT_CLASSES) - 1)
                x_center = random.uniform(0.2, 0.8)
                y_center = random.uniform(0.2, 0.8)
                width = random.uniform(0.05, 0.3)
                height = random.uniform(0.05, 0.3)
                
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save sample file
            sample_file = sample_dir / f"sample_{i:03d}.txt"
            with open(sample_file, 'w') as f:
                f.write('\n'.join(annotations))
        
        print(f"Sample annotations created in {sample_dir}")


def main():
    """Main function for data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare YOLOv8 dataset for Snap Circuits")
    parser.add_argument("--images", type=str, help="Directory containing images")
    parser.add_argument("--annotations", type=str, help="Directory containing annotations")
    parser.add_argument("--format", choices=["coco", "pascal_voc", "yolo"], 
                       default="yolo", help="Annotation format")
    parser.add_argument("--dataset-root", type=str, help="Root directory for dataset")
    parser.add_argument("--train-split", type=float, default=0.7, help="Training split fraction")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction") 
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split fraction")
    parser.add_argument("--create-samples", action="store_true", help="Create sample annotations")
    parser.add_argument("--validate", action="store_true", help="Validate prepared dataset")
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = DatasetPreparer(args.dataset_root)
    
    # Create sample annotations if requested
    if args.create_samples:
        preparer.create_sample_annotations()
        return
    
    # Convert annotations if provided
    if args.annotations:
        preparer.convert_annotations(args.annotations, args.format)
    
    # Organize images if provided
    if args.images:
        preparer.organize_images(
            args.images, 
            args.train_split, 
            args.val_split, 
            args.test_split
        )
    
    # Create data.yaml
    yaml_path = preparer.create_data_yaml(args.train_split, args.val_split, args.test_split)
    
    # Validate dataset if requested
    if args.validate:
        report = preparer.validate_dataset()
        print("\nDataset Validation Report:")
        print(f"Valid: {report['valid']}")
        
        if report['errors']:
            print("Errors:")
            for error in report['errors']:
                print(f"  - {error}")
        
        if report['warnings']:
            print("Warnings:")
            for warning in report['warnings']:
                print(f"  - {warning}")
        
        print("Statistics:")
        for split, stats in report['statistics'].items():
            if isinstance(stats, dict) and 'images' in stats:
                print(f"  {split}: {stats['images']} images, {stats['labels']} labels")
    
    print(f"\nDataset preparation complete!")
    print(f"Data configuration saved to: {yaml_path}")
    print(f"Ready for YOLOv8 training with: yolo train data={yaml_path}")


if __name__ == "__main__":
    main() 