#!/usr/bin/env python3
"""
Configuration reader for YOLO detection parameters.
This module provides a unified way to read configuration from config.json
for both Python implementations.
"""

import json
import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class YoloConfig:
    """Configuration data class for YOLO detection parameters."""
    
    # Detection thresholds
    confidence_threshold: float
    score_threshold: float
    nms_threshold: float
    
    # Input dimensions
    input_width: int
    input_height: int
    
    # Model settings
    default_model_name: str
    onnx_model_name: str
    
    # Paths
    input_dir: str
    output_dir: str
    model_dir: str
    
    # Processing options
    save_csv: bool
    save_visualizations: bool
    verbose_logging: bool


class ConfigReader:
    """Configuration reader class for parsing JSON config files."""
    
    def __init__(self, config_file_path: str = "config.json"):
        """
        Initialize the configuration reader.
        
        Args:
            config_file_path (str): Path to the JSON configuration file
        """
        self.config_file_path = config_file_path
        self._config = None
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from the JSON file."""
        try:
            with open(self.config_file_path, 'r') as file:
                data = json.load(file)
                
            yolo_config = data.get('yolo_config', {})
            
            # Extract configuration sections
            thresholds = yolo_config.get('detection_thresholds', {})
            dimensions = yolo_config.get('input_dimensions', {})
            model_settings = yolo_config.get('model_settings', {})
            paths = yolo_config.get('paths', {})
            processing = yolo_config.get('processing', {})
            
            # Create configuration object
            self._config = YoloConfig(
                # Detection thresholds
                confidence_threshold=thresholds.get('confidence_threshold', 0.25),
                score_threshold=thresholds.get('score_threshold', 0.25),
                nms_threshold=thresholds.get('nms_threshold', 0.45),
                
                # Input dimensions
                input_width=dimensions.get('width', 640),
                input_height=dimensions.get('height', 640),
                
                # Model settings
                default_model_name=model_settings.get('default_model_name', 'yolov8n.pt'),
                onnx_model_name=model_settings.get('onnx_model_name', 'yolov8n.onnx'),
                
                # Paths
                input_dir=paths.get('input_dir', 'input'),
                output_dir=paths.get('output_dir', 'output'),
                model_dir=paths.get('model_dir', 'YoloModel'),
                
                # Processing options
                save_csv=processing.get('save_csv', True),
                save_visualizations=processing.get('save_visualizations', True),
                verbose_logging=processing.get('verbose_logging', True)
            )
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def get_config(self) -> YoloConfig:
        """
        Get the configuration object.
        
        Returns:
            YoloConfig: The configuration data
        """
        if self._config is None:
            self.load_config()
        return self._config
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self.load_config()
    
    def print_config(self) -> None:
        """Print the current configuration values."""
        config = self.get_config()
        
        print("=== YOLO Configuration ===")
        print("Detection Thresholds:")
        print(f"  Confidence: {config.confidence_threshold}")
        print(f"  Score: {config.score_threshold}")
        print(f"  NMS: {config.nms_threshold}")
        
        print("Input Dimensions:")
        print(f"  Width: {config.input_width}")
        print(f"  Height: {config.input_height}")
        
        print("Model Settings:")
        print(f"  Default Model: {config.default_model_name}")
        print(f"  ONNX Model: {config.onnx_model_name}")
        
        print("Paths:")
        print(f"  Input Dir: {config.input_dir}")
        print(f"  Output Dir: {config.output_dir}")
        print(f"  Model Dir: {config.model_dir}")
        
        print("Processing Options:")
        print(f"  Save CSV: {config.save_csv}")
        print(f"  Save Visualizations: {config.save_visualizations}")
        print(f"  Verbose Logging: {config.verbose_logging}")
        print("===========================")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        config = self.get_config()
        return {
            'confidence_threshold': config.confidence_threshold,
            'score_threshold': config.score_threshold,
            'nms_threshold': config.nms_threshold,
            'input_width': config.input_width,
            'input_height': config.input_height,
            'default_model_name': config.default_model_name,
            'onnx_model_name': config.onnx_model_name,
            'input_dir': config.input_dir,
            'output_dir': config.output_dir,
            'model_dir': config.model_dir,
            'save_csv': config.save_csv,
            'save_visualizations': config.save_visualizations,
            'verbose_logging': config.verbose_logging
        }


# Global instance for easy access
_global_config_reader = None


def get_config(config_file_path: str = "config.json") -> YoloConfig:
    """
    Get the global configuration instance.
    
    Args:
        config_file_path (str): Path to config file (only used on first call)
        
    Returns:
        YoloConfig: The configuration data
    """
    global _global_config_reader
    if _global_config_reader is None:
        _global_config_reader = ConfigReader(config_file_path)
    return _global_config_reader.get_config()


def reload_config() -> None:
    """Reload the global configuration from file."""
    global _global_config_reader
    if _global_config_reader is not None:
        _global_config_reader.reload()


def print_config() -> None:
    """Print the current global configuration."""
    global _global_config_reader
    if _global_config_reader is not None:
        _global_config_reader.print_config()
    else:
        print("Configuration not loaded yet. Call get_config() first.")


# Convenience constants for backward compatibility
def get_constants():
    """Get configuration values as individual constants for backward compatibility."""
    config = get_config()
    return {
        'CONF_THRESHOLD': config.confidence_threshold,
        'SCORE_THRESHOLD': config.score_threshold,
        'NMS_THRESHOLD': config.nms_threshold,
        'NMS_IOU_THRESHOLD': config.nms_threshold,  # Alias for compatibility
        'INPUT_WIDTH': config.input_width,
        'INPUT_HEIGHT': config.input_height,
    }


if __name__ == "__main__":
    # Example usage
    try:
        config = get_config()
        print_config()
        
        # Show backward compatibility constants
        print("\nBackward Compatibility Constants:")
        constants = get_constants()
        for key, value in constants.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}") 