# Configuration System for YOLO Detection

This project now includes a unified configuration system that allows both Python and C++ implementations to read settings from a single `config.json` file.

## Overview

The configuration system provides:
- **Centralized configuration**: All settings in one JSON file
- **Cross-language support**: Both Python and C++ can read the same config
- **Easy modification**: Change settings without recompiling code
- **Backward compatibility**: Existing code continues to work
- **Type safety**: Structured configuration with proper data types

## Configuration File Structure

The `config.json` file contains all YOLO detection parameters:

```json
{
  "yolo_config": {
    "detection_thresholds": {
      "confidence_threshold": 0.25,
      "score_threshold": 0.25,
      "nms_threshold": 0.45
    },
    "input_dimensions": {
      "width": 640,
      "height": 640
    },
    "model_settings": {
      "default_model_name": "yolov8n.pt",
      "onnx_model_name": "yolov8n.onnx"
    },
    "paths": {
      "input_dir": "input",
      "output_dir": "output",
      "model_dir": "YoloModel"
    },
    "processing": {
      "save_csv": true,
      "save_visualizations": true,
      "verbose_logging": true
    }
  }
}
```

## Python Usage

### Basic Usage

```python
from config_reader import get_config

# Load configuration
config = get_config()

# Use configuration values
model_path = os.path.join(config.model_dir, config.default_model_name)
results = model(image_path, 
               conf=config.confidence_threshold, 
               iou=config.nms_threshold)
```

### Backward Compatibility

For existing code that uses constants, you can still access them:

```python
from config_reader import get_constants

constants = get_constants()
CONF_THRESHOLD = constants['CONF_THRESHOLD']
NMS_THRESHOLD = constants['NMS_THRESHOLD']
INPUT_WIDTH = constants['INPUT_WIDTH']
```

### Configuration Management

```python
from config_reader import get_config, print_config, reload_config

# Print current configuration
print_config()

# Reload configuration after external changes
reload_config()

# Access specific values
config = get_config()
print(f"Confidence threshold: {config.confidence_threshold}")
```

## C++ Usage

### Include Headers

```cpp
#include "Config.hpp"
```

### Basic Usage

```cpp
// Get configuration
const auto& config = Config::get();

// Use configuration values
float conf_threshold = config.confidence_threshold;
int input_width = config.input_width;
std::string model_path = config.model_dir + "/" + config.default_model_name;
```

### Configuration Management

```cpp
// Print current configuration
Config::print();

// Reload configuration
Config::reload();

// Access through singleton
ConfigReader& reader = ConfigReader::getInstance();
const YoloConfig& config = reader.getConfig();
```

## Modifying Configuration

### Runtime Modification (Python)

```python
import json
from config_reader import reload_config

# Read current config
with open("config.json", 'r') as f:
    config_data = json.load(f)

# Modify values
config_data['yolo_config']['detection_thresholds']['confidence_threshold'] = 0.5

# Write back to file
with open("config.json", 'w') as f:
    json.dump(config_data, f, indent=2)

# Reload in your application
reload_config()
```

### Direct File Editing

Simply edit the `config.json` file and reload the configuration in your application.

## Configuration Parameters

### Detection Thresholds
- **confidence_threshold**: Minimum confidence for detections (default: 0.25)
- **score_threshold**: Minimum score threshold (default: 0.25)
- **nms_threshold**: Non-Maximum Suppression IoU threshold (default: 0.45)

### Input Dimensions
- **width**: Input image width for the model (default: 640)
- **height**: Input image height for the model (default: 640)

### Model Settings
- **default_model_name**: Default PyTorch model filename (default: "yolov8n.pt")
- **onnx_model_name**: ONNX model filename (default: "yolov8n.onnx")

### Paths
- **input_dir**: Directory for input images (default: "input")
- **output_dir**: Directory for output files (default: "output")
- **model_dir**: Directory containing model files (default: "YoloModel")

### Processing Options
- **save_csv**: Whether to save detection results to CSV (default: true)
- **save_visualizations**: Whether to save visualization images (default: true)
- **verbose_logging**: Enable verbose logging output (default: true)

## Testing the Configuration System

Run the test script to verify the configuration system:

```bash
python test_config.py
```

This will:
1. Load and display the current configuration
2. Test configuration modification and reloading
3. Demonstrate usage examples

## Migration Guide

### For Existing Python Code

1. Replace hardcoded constants:
   ```python
   # Old way
   CONF_THRESHOLD = 0.25
   
   # New way
   from config_reader import get_config
   config = get_config()
   CONF_THRESHOLD = config.confidence_threshold
   ```

2. Update function calls:
   ```python
   # Old way
   results = model(image_path, conf=0.25, iou=0.45)
   
   # New way
   results = model(image_path, conf=config.confidence_threshold, iou=config.nms_threshold)
   ```

### For Existing C++ Code

1. Include the configuration header:
   ```cpp
   #include "Config.hpp"
   ```

2. Replace hardcoded constants:
   ```cpp
   // Old way
   const float CONF_THRESHOLD = 0.25f;
   
   // New way
   const auto& config = Config::get();
   float conf_threshold = config.confidence_threshold;
   ```

3. Update CMakeLists.txt to include the new source files:
   ```cmake
   add_executable(your_target 
       src/main.cpp 
       src/Config.cpp
       # ... other sources
   )
   ```

## Benefits

1. **Consistency**: Both Python and C++ use the same configuration values
2. **Flexibility**: Easy to experiment with different parameters
3. **Maintainability**: Single source of truth for all settings
4. **Deployment**: Different configurations for different environments
5. **Debugging**: Easy to trace configuration-related issues

## Error Handling

The configuration system includes proper error handling:

- **File not found**: Clear error message if config.json is missing
- **Invalid JSON**: Detailed error for malformed JSON
- **Missing values**: Default values are used for missing configuration keys
- **Type errors**: Graceful handling of incorrect data types

## Best Practices

1. **Version control**: Keep `config.json` in version control
2. **Environment-specific configs**: Use different config files for different environments
3. **Validation**: Validate configuration values in your application
4. **Documentation**: Document any custom configuration parameters
5. **Backup**: Keep backup of working configurations 