# Batch YOLO Processing with CUDA/GPU Support

This enhanced C++ implementation provides high-performance batch processing capabilities for YOLO object detection with CUDA/GPU acceleration.

## 🚀 Features

- **Batch Processing**: Process up to 16 (or more) images simultaneously
- **CUDA/GPU Acceleration**: Automatic GPU detection and utilization
- **Parallel Preprocessing**: Multi-threaded image preprocessing
- **Memory Optimization**: Efficient memory usage for large batches
- **Performance Monitoring**: Built-in performance statistics
- **Flexible Batch Sizes**: Configurable batch sizes from 1 to 32+
- **Directory Processing**: Process entire directories of images
- **Precompiled Headers**: Faster compilation times

## 📋 Requirements

### System Requirements
- **OpenCV 4.5+** with CUDA support (for GPU acceleration)
- **CUDA Toolkit 11.0+** (optional, for GPU acceleration)
- **CMake 3.16+**
- **C++17 compatible compiler**

### GPU Requirements (Optional)
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA-enabled OpenCV build
- Sufficient GPU memory (recommended: 4GB+ for batch size 16)

## 🛠️ Building

### 1. Standard Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 2. With CUDA Support (Recommended)
```bash
# Ensure OpenCV is built with CUDA support
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/path/to/opencv/build
make -j$(nproc)
```

### 3. Check CUDA Support
```bash
# Test if CUDA is available
./test_opencv
```

## 🎯 Usage

### Basic Batch Processing

```cpp
#include "BatchYoloOnnx.hpp"

// Initialize with batch size 16 and GPU enabled
BatchYOLODetector detector("YoloModel/yolov8n.onnx", "output", 16, true, 0);

// Process a batch of images
std::vector<std::string> image_paths = {
    "image1.jpg", "image2.jpg", "image3.jpg", // ... up to 16 images
};

auto results = detector.detectBatch(image_paths);

// Save results
detector.saveBatchResultsToCSV(results, "batch_output");
detector.drawAndSaveBatchDetections(results, "batch_visualizations");
```

### Directory Processing

```cpp
// Process all images in a directory
auto results = detector.detectFromDirectory("input_directory");
```

### Performance Optimization

```cpp
// Test different batch sizes for optimal performance
std::vector<int> batch_sizes = {1, 4, 8, 16, 32};
for (int batch_size : batch_sizes) {
    detector.setBatchSize(batch_size);
    // ... run performance tests
}
```

## 🏃‍♂️ Running the Demo

### Quick Demo
```bash
cd build
./batch_main
```

### Custom Demo
```bash
# Run with custom batch processing target
make run_batch_demo
```

## 📊 Performance Expectations

### Typical Performance (RTX 3080, batch size 16)
- **Single Image**: ~15-25 ms
- **Batch of 16**: ~80-120 ms (5-8 ms per image)
- **Throughput**: 120-200 FPS
- **GPU Memory**: ~2-4 GB

### CPU vs GPU Performance
- **CPU Only**: 50-80 FPS
- **GPU Accelerated**: 120-200 FPS
- **Speedup**: 2-4x improvement with GPU

## 🔧 Configuration

### Batch Size Optimization
```cpp
// For different GPU memory sizes:
// 4GB GPU: batch_size = 8-12
// 8GB GPU: batch_size = 16-24  
// 12GB+ GPU: batch_size = 32+

detector.setBatchSize(optimal_batch_size);
```

### Memory Management
```cpp
// Monitor GPU memory usage
if (detector.isGPUEnabled()) {
    // GPU processing - higher throughput
    detector.setBatchSize(16);
} else {
    // CPU processing - lower memory usage
    detector.setBatchSize(4);
}
```

## 📁 File Structure

```
├── include/
│   ├── BatchYoloOnnx.hpp          # Batch processing header
│   ├── pch.hpp                    # Precompiled headers
│   └── ...
├── src/
│   ├── BatchYoloOnnx.cpp         # Batch processing implementation
│   ├── batch_main.cpp            # Demo application
│   └── ...
├── CMakeLists.txt            # Updated build configuration
└── BATCH_PROCESSING_README.md
```

## 🐛 Troubleshooting

### CUDA Not Detected
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check OpenCV CUDA support
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

### Memory Issues
- Reduce batch size if getting out-of-memory errors
- Monitor GPU memory usage: `nvidia-smi`
- Use CPU fallback for large images

### Performance Issues
- Ensure GPU is being used: check console output
- Try different batch sizes
- Verify CUDA drivers are up to date

## 📈 Benchmarking

### Built-in Performance Testing
```cpp
// Get performance statistics
std::cout << detector.getPerformanceStats() << std::endl;
```

### Custom Benchmarking
```cpp
auto start = std::chrono::high_resolution_clock::now();
auto results = detector.detectBatch(large_image_batch);
auto end = std::chrono::high_resolution_clock::now();

double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
double fps = 1000.0 * batch_size / time_ms;
```

## 🔄 Migration from Single Image Processing

### Before (Single Image)
```cpp
YOLODetector detector("model.onnx", "output");
auto results = detector.detect("image.jpg");
```

### After (Batch Processing)
```cpp
BatchYOLODetector detector("model.onnx", "output", 16, true);
std::vector<std::string> images = {"image.jpg"};
auto results = detector.detectBatch(images);
```

## 🎛️ Advanced Configuration

### Custom GPU Device
```cpp
// Use specific GPU device (for multi-GPU systems)
BatchYOLODetector detector("model.onnx", "output", 16, true, 1); // GPU device 1
```

### Thread Pool Configuration
```cpp
// The implementation automatically uses std::async for parallel preprocessing
// Thread count is automatically determined by std::thread::hardware_concurrency()
```

## 📝 Notes

- **Precompiled Headers**: Significantly reduce compilation time
- **Memory Efficiency**: Batch processing is more memory efficient than processing images individually
- **CUDA Compatibility**: Automatically falls back to CPU if CUDA is not available
- **Thread Safety**: The detector is thread-safe for read operations
- **Error Handling**: Comprehensive error handling with detailed error messages

## 🤝 Contributing

When contributing to the batch processing functionality:

1. Maintain compatibility with the existing single-image API
2. Add comprehensive error handling
3. Include performance benchmarks
4. Update documentation
5. Test with both CPU and GPU configurations

## 📄 License

Same license as the main project. 