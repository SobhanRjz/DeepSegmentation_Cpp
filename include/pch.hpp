#ifndef PCH_HPP
#define PCH_HPP

// Standard C++ headers
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

// OpenCV headers (only include if OpenCV is available)
// This prevents PCH compilation errors when OpenCV hasn't been built yet
#ifdef OPENCV_AVAILABLE
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// CUDA OpenCV headers (if available)
#ifdef OPENCV_HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#endif
#endif // OPENCV_AVAILABLE

#endif // PCH_HPP 