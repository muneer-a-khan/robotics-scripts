#!/usr/bin/env python3
"""
Model Class Name Renamer
Utility to fix mislabeled classes in trained models
"""

def rename_model_classes(model):
    """
    Rename mislabeled classes in a YOLO model
    
    Args:
        model: YOLO model instance
        
    Returns:
        dict: Mapping of renamed classes {old_name: new_name}
    """
    renamed_classes = {}
    
    # Photoresistor doesn't exist - it's actually Horn (mislabeled during training)
    # Try multiple ways to access and modify the names dictionary
    names_dicts_to_update = []
    
    # Method 1: model.model.names (the actual underlying dict)
    if hasattr(model, 'model') and hasattr(model.model, 'names'):
        names_dicts_to_update.append(model.model.names)
    
    # Method 2: model.predictor if it exists
    if hasattr(model, 'predictor') and model.predictor and hasattr(model.predictor, 'model'):
        if hasattr(model.predictor.model, 'names'):
            names_dicts_to_update.append(model.predictor.model.names)
    
    # Update all found dictionaries
    for names_dict in names_dicts_to_update:
        if 'Photoresistor' in names_dict.values():
            for class_id, class_name in list(names_dict.items()):
                if class_name == 'Photoresistor':
                    names_dict[class_id] = 'Horn'
                    renamed_classes['Photoresistor'] = 'Horn'
                    print(f"   ✓ Renamed class {class_id}: 'Photoresistor' → 'Horn'")
    
    return renamed_classes

def get_corrected_class_name(class_name):
    """
    Get the corrected class name for a component
    
    Args:
        class_name: Original class name from model
        
    Returns:
        str: Corrected class name
    """
    corrections = {
        'Photoresistor': 'Horn',  # Photoresistor doesn't exist, it's Horn
    }
    return corrections.get(class_name, class_name)

