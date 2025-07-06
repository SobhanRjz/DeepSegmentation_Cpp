# 🎭 Segmentation Detector

A high-performance C++ implementation for YOLO-based instance segmentation using ONNX Runtime. This detector provides pixel-level object detection with mask generation, contour extraction, and comprehensive analysis utilities.

## 🚀 Features

- **Instance Segmentation**: Detect objects with pixel-perfect masks
- **Multiple Output Formats**: CSV, JSON, and individual PNG masks
- **Advanced Visualization**: Colored mask overlays with transparency control
- **Contour Analysis**: Extract and analyze object contours
- **Performance Optimized**: CUDA and CPU execution providers
- **YOLOv8 Compatible**: Supports YOLOv8 segmentation models with prototype masks
- **Comprehensive Analytics**: IoU calculation, area analysis, centroid detection

## 📁 Project Structure

```
├── include/
│   └── SegmentationDetector.hpp    # Main header file
├── src/
│   ├── SegmentationDetector.cpp    # Implementation
│   └── main_SegmentationDetector.cpp # Demo application
└── output/
    ├── masks/                      # Individual mask PNG files
    ├── segmentation_visualization.jpg
    ├── detected_segmentation_cpp.csv
    └── segmentation_results.json
```

## 🛠️ Building

### Prerequisites
- OpenCV 4.x
- ONNX Runtime (GPU or CPU)
- CMake 3.16+
- C++17 compiler

### Build Commands
```bash
mkdir build && cd build
cmake ..
make segmentation_detector

# Run demo
make run_segmentation_demo
```

## 🎯 Usage

### Basic Usage
```cpp
#include "SegmentationDetector.hpp"

// Create detector
SegmentationDetector detector("model.onnx", "output", "cuda");

// Configure detection parameters
SegmentationConfig config;
config.confidence_threshold = 0.25f;
config.mask_threshold = 0.5f;
config.mask_alpha = 0.5f;
detector.setConfig(config);

// Run detection
auto results = detector.detect("image.jpg");

// Save results
detector.saveSegmentationToCSV(results);
detector.saveMasksAsPNG(results);
detector.drawAndSaveSegmentation("image.jpg", results);
```

### Advanced Configuration
```cpp
SegmentationConfig config;

// Detection thresholds
config.confidence_threshold = 0.25f;
config.nms_threshold = 0.45f;
config.mask_threshold = 0.5f;

// Model-specific parameters
config.use_proto_masks = true;      // YOLOv8 style
config.num_prototypes = 32;         // Number of prototype masks
config.mask_width = 160;            // Prototype dimensions
config.mask_height = 160;

// Visualization options
config.fill_masks = true;           // Fill masks vs contours only
config.mask_alpha = 0.5f;           // Transparency (0.0-1.0)
config.show_contours = true;        // Show contour lines
config.contour_thickness = 2;       // Contour line thickness
```

## 📊 Output Formats

### CSV Output
```csv
x,y,width,height,confidence,class_id,class_name,mask_area,centroid_x,centroid_y
100.5,200.3,150.2,180.7,0.85,0,person,15420.0,175.6,290.5
```

### JSON Output
```json
{
  "detections": [
    {
      "bbox": {"x": 100.5, "y": 200.3, "width": 150.2, "height": 180.7},
      "confidence": 0.85,
      "class_id": 0,
      "class_name": "person",
      "segmentation": {
        "area": 15420.0,
        "centroid": {"x": 175.6, "y": 290.5},
        "contour_points": 342
      }
    }
  ]
}
```

### Individual Masks
- `mask_0_person.png` - Binary mask for first person detection
- `mask_1_car.png` - Binary mask for first car detection
- etc.

## 🔧 Model Compatibility

### Supported Models
- **YOLOv8 Segmentation**: `yolov8n-seg.onnx`, `yolov8s-seg.onnx`, `yolov8m-seg.onnx`, `yolov8l-seg.onnx`, `yolov8x-seg.onnx`
- **Custom Segmentation Models**: Any ONNX model with compatible output format

### Expected Model Outputs
1. **Detection Output**: `[batch, num_detections, 4+num_classes+mask_coeffs]`
   - First 4 values: bounding box (x, y, w, h)
   - Next N values: class probabilities
   - Last M values: mask coefficients

2. **Prototype Masks**: `[batch, num_prototypes, mask_height, mask_width]`
   - Prototype masks for generating final segmentation masks

## 📈 Analysis Utilities

### Mask Analysis
```cpp
// Calculate IoU between masks
double iou = SegmentationAnalysis::calculateIoU(mask1, mask2);

// Get mask properties
double area = SegmentationAnalysis::calculateMaskArea(mask);
cv::Point2f centroid = SegmentationAnalysis::calculateCentroid(mask);

// Contour analysis
auto contour = SegmentationAnalysis::extractLargestContour(mask);
double perimeter = SegmentationAnalysis::calculateContourPerimeter(contour);
auto simplified = SegmentationAnalysis::simplifyContour(contour, 2.0);
```

### Visualization Utilities
```cpp
// Create colored mask
cv::Mat colored = SegmentationAnalysis::createColoredMask(mask, cv::Scalar(255, 0, 0));

// Merge multiple masks
std::vector<cv::Mat> masks = {mask1, mask2, mask3};
std::vector<cv::Scalar> colors = {red, green, blue};
cv::Mat merged = SegmentationAnalysis::mergeMasks(masks, colors);
```

## ⚡ Performance Optimization

### CUDA Acceleration
```cpp
// Enable CUDA for faster inference
SegmentationDetector detector("model.onnx", "output", "cuda");
```

### Memory Optimization
- Automatic memory pooling for ONNX Runtime
- Efficient mask processing with OpenCV
- Optimized contour extraction algorithms

### Threading
- Multi-threaded ONNX Runtime execution
- Parallel mask processing for multiple detections

## 🎨 Visualization Features

### Mask Overlays
- Transparent colored masks over original image
- Configurable transparency (alpha blending)
- Distinct colors for each object class

### Contour Drawing
- Precise object boundaries
- Configurable line thickness
- Optional contour simplification

### Bounding Boxes
- Class labels with confidence scores
- Color-coded by object class
- Automatic text positioning

## 🔍 Model Analysis Tools

### Output Inspection
```cpp
// Analyze model outputs
detector.analyzeModelOutput("test_image.jpg");

// Print model information
detector.printModelInfo();
```

### Debug Information
- Model input/output shapes
- Execution provider details
- Processing time statistics

## 📝 Configuration Options

### Detection Parameters
- `confidence_threshold`: Minimum detection confidence (default: 0.25)
- `nms_threshold`: Non-maximum suppression threshold (default: 0.45)
- `mask_threshold`: Mask binarization threshold (default: 0.5)

### Model Parameters
- `use_proto_masks`: Enable prototype mask processing (default: true)
- `num_prototypes`: Number of prototype masks (default: 32)
- `mask_width/height`: Prototype mask dimensions (default: 160x160)

### Visualization Parameters
- `fill_masks`: Fill masks vs contours only (default: true)
- `mask_alpha`: Mask transparency (default: 0.5)
- `show_contours`: Show contour lines (default: true)
- `contour_thickness`: Contour line thickness (default: 2)

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Failed**
   ```
   ❌ Failed to load segmentation model
   ```
   - Check model path exists
   - Verify ONNX Runtime installation
   - Ensure model is compatible format

2. **No Detections Found**
   ```
   ⚠️ No detections after NMS
   ```
   - Lower confidence threshold
   - Check input image quality
   - Verify model is trained for your data

3. **Mask Processing Errors**
   ```
   ⚠️ Invalid prototype shape
   ```
   - Verify model outputs prototype masks
   - Check `num_prototypes` configuration
   - Ensure model is segmentation type

### Performance Issues
- Use CUDA execution provider for GPU acceleration
- Reduce input image resolution if needed
- Adjust `mask_threshold` for faster processing

## 📚 Examples

See `src/main_SegmentationDetector.cpp` for a complete example with:
- Model loading and configuration
- Detection processing
- Result analysis and statistics
- Multiple output format generation
- Visualization creation

## 🤝 Integration

The SegmentationDetector can be easily integrated into larger applications:

```cpp
// In your application
#include "SegmentationDetector.hpp"

class MyApplication {
private:
    SegmentationDetector detector_;
    
public:
    MyApplication() : detector_("model.onnx", "output", "cuda") {
        // Configure detector
        SegmentationConfig config;
        config.confidence_threshold = 0.3f;
        detector_.setConfig(config);
    }
    
    void processImage(const std::string& image_path) {
        auto results = detector_.detect(image_path);
        // Process results...
    }
};
```

## 📄 License

This segmentation detector is part of the DeepNetC++ project and follows the same licensing terms. 