#ifndef PCH_SEGMENTATION_BATCH_HPP
#define PCH_SEGMENTATION_BATCH_HPP

// ============================================================================
// STANDARD C++ HEADERS
// ============================================================================
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <limits>
#include <cmath>
#include <iomanip>
#include <memory>
#include <utility>
#include <chrono>
#include <mutex>
#include <thread>
#include <future>
#include <sstream>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <queue>
#include <stack>
#include <array>
#include <tuple>
#include <functional>
#include <numeric>
#include <random>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <locale>
#include <iterator>
#include <exception>
#include <cassert>

// ============================================================================
// OPENCV HEADERS
// ============================================================================
#ifdef OPENCV_AVAILABLE
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

// CUDA OpenCV headers (if available)
#ifdef OPENCV_HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#endif
#endif // OPENCV_AVAILABLE

// ============================================================================
// PYTORCH/LIBTORCH HEADERS
// ============================================================================
#include <torch/torch.h>
#include <torch/script.h>
#ifdef TORCH_CUDA_AVAILABLE
#include <torch/cuda.h>
#endif

// ============================================================================
// ONNX RUNTIME HEADERS
// ============================================================================
#include <onnxruntime_cxx_api.h>

// ============================================================================
// PROJECT HEADERS - Commented out to avoid circular dependencies
// These will be included by individual source files as needed
// ============================================================================
// #include "SegmentationDetector.hpp"
// #include "Config.hpp"
// #include "DetectionTypes.hpp"

// ============================================================================
// COMMON PREPROCESSOR DEFINITIONS
// ============================================================================

// Performance optimization flags
#ifndef NDEBUG
#define PCH_DEBUG_MODE 1
#else
#define PCH_DEBUG_MODE 0
#endif

// Memory alignment for SIMD operations
#define PCH_MEMORY_ALIGN 32

// Default batch processing constants
#define PCH_DEFAULT_CONF_THRESHOLD 0.25f
#define PCH_DEFAULT_IOU_THRESHOLD 0.45f
#define PCH_DEFAULT_INPUT_SIZE 1280
#define PCH_MAX_DETECTIONS 300

#endif // PCH_SEGMENTATION_BATCH_HPP 