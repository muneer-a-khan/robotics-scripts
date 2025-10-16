#!/usr/bin/env python3
"""
Component Name Mapper
Maps component names from model output to display names

NOTE: Photoresistor → Horn mapping has been moved to model_class_renamer.py
This now happens at model load time instead of during name mapping.
"""

# Mapping from model class names to display names
# (Currently empty - Photoresistor renaming now happens at model load time)
COMPONENT_NAME_MAP = {}

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

