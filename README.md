# ⚡ High-Performance Deep Learning Segmentation Framework

A high-performance C++ framework for running deep learning models with lightning-fast speed. Designed for real-time applications, this repo accelerates tasks such as football player segmentation (using YOLOv8 and CUDA), achieving 10ms per frame in C++ versus 15ms in Python. Batch processing further improves performance to 7ms per frame for 16 images.

Built for maximum performance across all modern GPU architectures with professional-grade C++ and Python implementations.

## 🏆 **Performance Highlights**

**Tested on RTX 4070 (16GB VRAM) with Football Player Segmentation:**
- **C++ Implementation**: 10ms single frame, 7ms batch processing (16 images)
- **Python Implementation**: 15ms single frame, 12ms batch processing  
- **C++ Speedup**: 1.7x faster than Python with 28% less memory usage
- **Throughput**: 142 FPS (C++) vs 83 FPS (Python)

*Performance scales with GPU specifications - RTX 4070 results shown as reference*

## 🎯 **Key Features**

### 🚀 **Universal GPU Support**
- **Multi-GPU Architecture**: CUDA, OpenCL, and CPU fallback support
- **Dynamic Memory Management**: Automatic optimization for any VRAM size
- **Scalable Batch Processing**: Adaptive batch sizes based on GPU capabilities
- **Cross-Platform**: Windows, Linux, macOS compatibility

### 🧠 **Deep Learning Model Support**
- **YOLO Family**: YOLOv8, YOLOv9, YOLOv10 segmentation models
- **Custom Models**: Easy integration of any ONNX/TorchScript model
- **Multiple Formats**: Support for .pt, .onnx, .engine (TensorRT) files
- **Dynamic Input**: Variable input sizes and batch dimensions

### ⚡ **Performance Optimizations**
- **C++ Core**: Optimized inference pipeline with minimal overhead
- **Memory Pooling**: Reduces allocation latency for batch processing
- **Mixed Precision**: FP16 support for 2x memory efficiency
- **Asynchronous Processing**: Overlapped CPU/GPU operations

### 🔧 **Professional Features**
- **Comprehensive Logging**: Production-ready error handling
- **Configuration System**: JSON-based parameter management
- **Batch Processing**: Efficient multi-image processing
- **Multiple Output Formats**: CSV, JSON, visualizations
- **Testing Suite**: Automated validation and benchmarking

## 📋 **Requirements**

### **Hardware Requirements**
- **GPU**: Any CUDA-compatible GPU (GTX 1060+ recommended)
- **VRAM**: 4GB minimum, 8GB+ recommended for batch processing
- **CPU**: Multi-core processor (Intel i5+ or AMD Ryzen 5+)
- **RAM**: 8GB minimum, 16GB+ recommended

### **Software Dependencies**
- **CUDA Toolkit**: 12.4 (latest stable version)
- **OpenCV**: 4.5+
- **PyTorch/LibTorch**: 2.6.0 with CUDA 12.4 support
- **ONNX Runtime**: 1.22.1 with GPU provider (latest version)
- **Python**: 3.8+ (for Python components, 3.10+ recommended)

## 🛠️ **Installation**

### **1. Clone Repository**
```bash
git clone https://github.com/SobhanRjz/DeepSegmentation_Cpp.git
cd DeepSegmentation_Cpp
```

### **2. Install System Dependencies**

The CMake build system automatically downloads and compiles all major dependencies from source! You only need to install basic system requirements.

#### **Ubuntu/Debian:**
```bash
# Install essential build tools and CUDA Toolkit
sudo apt update
sudo apt install cmake build-essential pkg-config git

# Install CUDA Toolkit 12.4 (required for GPU acceleration)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi
```

#### **CentOS/RHEL/Fedora:**
```bash
# Install build tools
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake git pkgconfig

# Install CUDA (adjust for your distribution)
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install cuda-toolkit-12-4
```

#### **macOS:**
```bash
# Install Xcode command line tools and CMake
xcode-select --install
brew install cmake git

# Note: CUDA not available on macOS, will build CPU-only versions
```

#### **Windows:**
```powershell
# Install Visual Studio 2019/2022 with C++ support
# Install CMake and Git
# Download and install CUDA Toolkit 12.4 from NVIDIA website

# Or use package manager:
choco install cmake git cuda
```

## 🚀 **Automatic Dependency Management**

### **What Gets Built Automatically:**
Our CMake system automatically downloads and builds the latest optimized versions:

✅ **Boost 1.85.0** - Essential C++ libraries  
✅ **OpenCV 4.10.0 + Contrib** - Computer vision with CUDA support  
✅ **ONNX Runtime 1.23.0** - High-performance inference engine with GPU  
✅ **LibTorch 2.6.0** - PyTorch C++ backend with CUDA 12.4 support  

**Build Time:** ~45-90 minutes (one-time setup)  
**Storage:** ~8GB for all dependencies  
**Result:** Fully optimized, CUDA-enabled libraries  

### **Smart Build Features:**
- 🔄 **Incremental Builds**: Only rebuilds what's changed
- 📝 **Build Markers**: Tracks completed builds to avoid rebuilds
- ⚡ **Parallel Compilation**: Uses all CPU cores for faster builds
- 🎯 **GPU-Optimized**: CUDA architecture-specific optimizations
- 🔒 **Version Locked**: Ensures compatibility between all components

## ⚙️ **Advanced Build Options**

### **Build Configuration Options:**
```bash
# Debug build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release with debug info (best for profiling)
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

# Specify CUDA architecture for your GPU
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89" ..  # RTX 4070
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86" ..  # RTX 3080
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75" ..  # GTX 1660

# Use specific compilers
cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 ..

# Custom CUDA toolkit location
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4 ..
```

### **Build Performance Optimization:**
```bash
# Use maximum parallel jobs
make -j$(nproc)

# On systems with limited RAM (reduce parallel jobs)
make -j4  # Use only 4 parallel jobs

# Use Ninja build system for faster builds
cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja

# Enable compiler cache (if ccache installed)
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
```

### **Selective Dependency Building:**
```bash
# Force rebuild specific dependency
rm libs/opencv_install/opencv_built.marker && make opencv_external

# Build only OpenCV
make opencv_external

# Build only ONNX Runtime  
make onnxruntime_external

# Build only LibTorch
make libtorch_external

# Skip tests and examples in dependencies (default)
# All dependencies are built with optimized, production-ready flags
```

### **Cross-Platform Build Notes:**

**For CPU-Only Builds (no CUDA):**
```bash
# The build system automatically detects CUDA availability
# If CUDA not found, will build CPU-only versions automatically
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**For Different Linux Distributions:**
```bash
# CentOS/RHEL: Use devtoolset for modern compilers
scl enable devtoolset-11 bash
cmake -DCMAKE_BUILD_TYPE=Release ..

# Alpine Linux: Install build dependencies
apk add cmake make g++ git
```

**Build Time Estimates by System:**
| System Type | CPU Cores | RAM | First Build Time | Subsequent Builds |
|-------------|-----------|-----|-----------------|-------------------|
| High-end Workstation | 16+ cores | 32GB+ | 25-35 minutes | 15-30 seconds |
| Mid-range Desktop | 8-12 cores | 16GB | 45-60 minutes | 30-45 seconds |
| Budget System | 4-6 cores | 8GB | 75-90 minutes | 45-60 seconds |
| Virtual Machine | 4 cores | 8GB | 90-120 minutes | 60-90 seconds |

### **3. Build the Project**

#### **One-Command Build (Recommended):**
```bash
# Create build directory and configure
mkdir build && cd build

# Configure with automatic dependency downloading
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build everything (includes automatic dependency compilation)
make -j$(nproc)  # Uses all CPU cores

# This will automatically:
# 1. Download and build Boost 1.85.0
# 2. Download and build OpenCV 4.10.0 with CUDA support
# 3. Download and build ONNX Runtime 1.23.0 with GPU support  
# 4. Download and build LibTorch 2.6.0 with CUDA 12.4 support
# 5. Compile your application executables
```

#### **Build Progress Tracking:**
```bash
# Monitor build progress (optional - run in another terminal)
watch -n 5 'ls -la libs/*/build.log 2>/dev/null | tail -10'

# Check available disk space (builds require ~8GB)
df -h .
```

#### **First Build vs. Subsequent Builds:**
```bash
# First build: ~45-90 minutes (downloads and compiles everything)
time make -j$(nproc)

# Subsequent builds: ~30 seconds (only rebuilds changed code)
# Dependencies are cached and won't be rebuilt unless forced
```

#### **Available Executables After Build:**
```bash
# Main applications (created in build/ directory)
./segmentation_batch          # Segmentation-specific batch processor
./point_batch                 # Point detection batch processor  
./unified_batch_processor     # Auto-detecting processor (recommended)
./test_opencv                 # OpenCV installation verification

# Demo targets
make run_unified_batch_demo   # Run with sample data
make run_opencv_test          # Test OpenCV functionality
```

### **4. Install Python Components (Optional)**
```bash
# Only needed if you want to use Python scripts
cd ../YoloPy
pip install -r requirements.txt

# Alternative: Install Python components with the same versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install onnxruntime-gpu==1.22.1 opencv-python ultralytics
```

## ⚙️ **Configuration**

### **Hardware-Specific Settings**
Edit `config.json` based on your GPU specifications:

```json
{
  "yolo_config": {
    "batch_processing": {
      "batch_size": 16,          // Adjust based on VRAM: 4GB=4, 8GB=8, 16GB=16, 24GB=24
      "execution_provider": "cuda"
    },
    "hardware_config": {
      "gpu_model": "YOUR_GPU_MODEL",
      "vram_gb": 16,             // Your GPU VRAM in GB
      "optimal_batch_size": 16,  // Optimal batch size for your GPU
      "memory_limit_mb": 14000   // VRAM limit (leave 2GB for system)
    },
    "performance_settings": {
      "enable_mixed_precision": true,  // Enable for better memory efficiency
      "memory_pooling": true,
      "gpu_preprocessing": true
    }
  }
}
```

### **GPU-Specific Batch Size Recommendations**
| GPU VRAM | Recommended Batch Size | Max Batch Size |
|----------|----------------------|----------------|
| 4GB      | 4                    | 6              |
| 6GB      | 6                    | 8              |
| 8GB      | 8                    | 12             |
| 10GB     | 10                   | 16             |
| 12GB     | 12                   | 18             |
| 16GB     | 16                   | 24             |
| 24GB     | 24                   | 32             |

## 🚀 **Usage**

### **Quick Start**
```bash
# Test your setup
./build/test_opencv

# Run segmentation on single image
./build/unified_batch_processor CUDA 8 0.25

# Run batch processing
./build/segmentation_batch_processor CUDA 16 0.25
```

### **Python API**
```python
from YoloPy.yolo_segment_detection import YOLOSegmentationDetector

# Initialize detector
detector = YOLOSegmentationDetector(
    model_path="YoloModel/your_model.pt",
    device="cuda"  # or "cpu"
)

# Process single image
results = detector.detect_and_segment("path/to/image.jpg")

# Process batch
detector.process_batch_images_native(image_paths, batch_size=16)
```

### **C++ API**
```cpp
#include "SegmentationDetector.hpp"

// Initialize detector
SegmentationDetector detector("YoloModel/your_model.pt", "output", "cuda");

// Process single image
auto results = detector.detect("path/to/image.jpg");

// Process batch
BatchSegmentationProcessor processor("YoloModel/your_model.pt", "output");
processor.processBatch(image_paths, 16);
```

## 📊 **Benchmarking**

### **Performance Testing**
```bash
# GPU benchmark
./build/unified_batch_processor benchmark --batch-size=16 --iterations=100

# Memory usage test
nvidia-smi dmon -s mu -c 60 &
./build/segmentation_batch_processor CUDA 16 0.25
```

### **Model Compatibility Test**
```bash
# Test your model
cd YoloPy
python yolo_segment_detection.py --model YoloModel/your_model.pt --test-image test.jpg
```

## 🔧 **Optimization Guide**

### **For High-End GPUs (RTX 4070+, RTX 4080+, RTX 4090)**
```bash
# Build with maximum optimizations
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 ..

# Use larger batch sizes
./build/unified_batch_processor CUDA 24 0.25

# Environment optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8
```

### **For Mid-Range GPUs (GTX 1660+, RTX 3060+)**
```bash
# Conservative batch sizes
./build/unified_batch_processor CUDA 8 0.25

# Enable mixed precision
# Edit config.json: "enable_mixed_precision": true
```

### **For Entry-Level GPUs (GTX 1060+)**
```bash
# Small batch sizes
./build/unified_batch_processor CUDA 4 0.25

# Reduce input resolution in config.json
"input_dimensions": {
  "width": 640,
  "height": 640
}
```

## 🧪 **Testing**

### **Run Test Suite**
```bash
cd tests
./install_and_test.sh
```

### **Individual Tests**
```bash
# Point detection test
./build/test_point_detection

# Segmentation test
./build/test_yolo_detector

# CSV comparison test
./build/test_csv_comparison
```

## 📁 **Project Structure**

```
DeepSegmentation_Cpp/
├── src/                          # C++ source files
│   ├── main_*.cpp               # Main executables
│   ├── *Detector.cpp            # Detection implementations
│   └── Batch*.cpp               # Batch processing
├── include/                     # C++ headers
├── YoloPy/                      # Python implementation
├── YoloModel/                   # Model files (.pt, .onnx)
├── tests/                       # Test suite
├── output/                      # Generated results
├── config.json                 # Configuration file
├── CMakeLists.txt              # Build system with auto-dependencies
├── build/                      # Build output directory
│   ├── segmentation_batch      # Main executables
│   ├── unified_batch_processor # (created after build)
│   └── test_opencv            
└── libs/                       # Auto-downloaded dependencies
    ├── boost/                  # Boost 1.85.0 source
    ├── opencv_src/             # OpenCV 4.10.0 source  
    ├── opencv_contrib_src/     # OpenCV contrib modules
    ├── opencv_install/         # OpenCV built libraries
    ├── pytorch/               # LibTorch 2.6.0 source
    ├── pytorch_install/       # LibTorch built libraries
    ├── onnxruntime_src/       # ONNX Runtime 1.23.0 source
    └── onnxruntime_install/   # ONNX Runtime built libraries
```

### **After First Build:**
```bash
# Your libs/ directory will contain (~8GB total):
libs/
├── opencv_install/lib/         # OpenCV shared libraries (.so files)
├── pytorch_install/lib/        # LibTorch shared libraries  
├── onnxruntime_install/lib/    # ONNX Runtime shared libraries
├── *_install/include/          # All header files
└── *.marker                    # Build completion markers

# Built executables in build/:
build/
├── segmentation_batch         # 🎯 Segmentation-focused processor
├── point_batch               # 🎯 Point detection processor
├── unified_batch_processor   # 🎯 Auto-detecting processor (recommended)
└── test_opencv              # 🔧 OpenCV verification utility
```

## 📈 **Performance Expectations**

### **Tested Configurations**
| GPU Model | VRAM | Batch Size | Single Frame | Batch Processing | Memory Usage |
|-----------|------|------------|--------------|------------------|--------------|
| RTX 4070  | 16GB | 16         | 10ms         | 7ms/frame        | 3.2GB        |
| RTX 3080  | 10GB | 12         | 12ms         | 8ms/frame        | 2.8GB        |
| RTX 3060  | 8GB  | 8          | 15ms         | 10ms/frame       | 2.2GB        |
| GTX 1660  | 6GB  | 6          | 20ms         | 15ms/frame       | 1.8GB        |

*Results may vary based on model complexity and input resolution*

## 🛡️ **Model Support**

### **Supported Model Types**
- **YOLOv8**: Segmentation models (.pt, .onnx)
- **YOLOv9**: Latest architecture support
- **YOLOv10**: Ultralytics format
- **Custom Models**: Any ONNX segmentation model
- **TensorRT**: .engine files for maximum performance

### **Model Formats**
- **PyTorch**: .pt files (native Ultralytics)
- **ONNX**: .onnx files (cross-platform)
- **TensorRT**: .engine files (NVIDIA optimization)
- **OpenVINO**: .xml files (Intel optimization)

## 🔧 **Troubleshooting**

### **Build System Issues**

**First Build Takes Too Long:**
```bash
# Normal first build time: 45-90 minutes
# Monitor progress with:
watch -n 10 'tail -20 CMakeFiles/CMakeOutput.log'

# Check current build status:
find libs/ -name "*.marker" 2>/dev/null  # Shows completed dependencies

# If build fails midway, clean and restart:
rm -rf build/ libs/
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Dependency Build Failures:**
```bash
# Force rebuild a specific dependency (if it failed):
rm libs/opencv_install/opencv_built.marker      # Force OpenCV rebuild
rm libs/onnxruntime_install/onnxruntime_built.marker  # Force ONNX Runtime rebuild  
rm libs/pytorch_install/libtorch_built.marker   # Force LibTorch rebuild

# Then rebuild:
make -j$(nproc)
```

**CUDA Toolkit Issues:**
```bash
# Verify CUDA installation before building
nvcc --version
nvidia-smi

# If CUDA not found during build:
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4

# Reconfigure and rebuild:
cd build && cmake -DCMAKE_BUILD_TYPE=Release ..
```

**CMake Configuration Issues:**
```bash
# If CMake can't find CUDA:
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4 ..

# If running out of disk space (need ~8GB):
df -h .
# Clean intermediate build files:
find build/ -name "*.o" -delete
find libs/ -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
```

**Runtime Issues:**
```bash
# If executables can't find libraries:
export LD_LIBRARY_PATH="$(pwd)/libs/opencv_install/lib:$(pwd)/libs/pytorch_install/lib:$(pwd)/libs/onnxruntime_install/lib:$LD_LIBRARY_PATH"

# If getting CUDA out of memory errors:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Monitor GPU usage during execution:
nvidia-smi -l 1
```

**Python Component Issues:**
```bash
# If Python can't find installed packages:
python -c "import torch; print(torch.__version__)"
python -c "import onnxruntime; print(onnxruntime.__version__)"

# Install matching versions:
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install onnxruntime-gpu==1.22.1
```

### **Version Compatibility Matrix**
| Component | Version | Compatible With |
|-----------|---------|----------------|
| CUDA Toolkit | 12.4 | RTX 40-series, RTX 30-series, GTX 16-series |
| LibTorch | 2.6.0+cu124 | CUDA 12.4 |
| ONNX Runtime | 1.23.0 | CUDA 12.4, Python 3.8-3.13 |
| OpenCV | 4.10.0 | All configurations |
| Boost | 1.85.0 | C++17 and above |

### **Dependency Cache Management**

**Check Build Status:**
```bash
# See which dependencies are built
find libs/ -name "*.marker" -exec basename {} \; 2>/dev/null

# Check dependency sizes
du -sh libs/*/

# View build logs
tail -50 libs/opencv_src/build.log      # OpenCV build log
tail -50 libs/pytorch/build.log         # LibTorch build log
```

**Clean Build Cache:**
```bash
# Clean only build artifacts (keep dependencies)
rm -rf build/
mkdir build && cd build

# Clean specific dependency (force rebuild)
rm -rf libs/opencv_src/ libs/opencv_install/
rm libs/opencv_install/opencv_built.marker

# Full clean (start from scratch - will re-download everything)
rm -rf build/ libs/
mkdir build && cd build

# Minimal clean (just remove object files)
find build/ -name "*.o" -delete
find build/ -name "CMakeCache.txt" -delete
```

**Backup/Restore Dependencies:**
```bash
# Backup built dependencies (useful for CI/development)
tar -czf dependencies_backup.tar.gz libs/

# Restore dependencies (skip long build times)
tar -xzf dependencies_backup.tar.gz

# Verify restoration
find libs/ -name "*.marker" | wc -l  # Should show 3-4 markers
```

**Development Workflow:**
```bash
# Daily development (fast rebuilds)
cd build && make -j$(nproc)  # 30 seconds

# After pulling updates (check for CMakeLists.txt changes)
cd build && cmake .. && make -j$(nproc)

# Major updates (force clean rebuild of project only)
rm -rf build/CMakeFiles build/Makefile
cd build && cmake .. && make -j$(nproc)

# Dependency update (rare - only when versions change)
rm -rf libs/ build/
mkdir build && cd build && cmake .. && make -j$(nproc)
```

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 **Support**

For issues and questions:
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and optimization tips
- **Wiki**: Detailed documentation and tutorials

## 🙏 **Acknowledgments**

- **Ultralytics**: YOLO model implementations
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework
- **ONNX Runtime**: Cross-platform inference
- **NVIDIA**: CUDA toolkit and optimization guides

---

**⚡ Built for Performance • 🧠 Designed for Flexibility • 🚀 Optimized for Production** 