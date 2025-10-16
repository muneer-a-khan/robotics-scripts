#!/usr/bin/env python3
"""
Component Name Mapper
Maps component names from model output to display names
"""

# Mapping from model class names to display names
COMPONENT_NAME_MAP = {
    'Photoresistor': 'Horn',  # Mislabeled during training, should be Horn
}

def map_component_name(model_name: str) -> str:
    """
    Map a model class name to its display name
    
    Args:
        model_name: Component name from model
        
    Returns:
        Mapped display name
    """
    return COMPONENT_NAME_MAP.get(model_name, model_name)

def get_all_mapped_names(model_names: dict) -> dict:
    """
    Get all model names with mappings applied
    
    Args:
        model_names: Dictionary of model class names {class_id: name}
        
    Returns:
        Dictionary with mapped names
    """
    return {class_id: map_component_name(name) for class_id, name in model_names.items()}

