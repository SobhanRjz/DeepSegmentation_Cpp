# CMakeLists.txt for YOLO Tests
cmake_minimum_required(VERSION 3.16)
project(YOLOTests)

# Find required packages
find_package(GTest REQUIRED)
# Use OpenCV from parent project instead of finding it independently
# find_package(OpenCV REQUIRED)

# Use OpenCV from the main project build
set(OpenCV_DIR ${CMAKE_BINARY_DIR}/../install/lib/cmake/opencv4)
set(OpenCV_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/../install/include/opencv4)
set(OpenCV_LIBS ${CMAKE_BINARY_DIR}/../install/lib/libopencv_world.so)

# ONNX Runtime GPU setup (same as main CMakeLists.txt)
set(ONNXRUNTIME_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../libs/onnxruntime-linux-x64-gpu-1.22.0)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT_PATH}/lib)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

# Create test executable
add_executable(yolo_tests
    test_yolo_detector.cpp
    test_utils.cpp
    test_csv_comparison.cpp
    test_point_detection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/SingleImageYolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/Config.cpp
)

# Make sure tests depend on OpenCV being built
add_dependencies(yolo_tests opencv)

# Link libraries (including ONNX Runtime)
target_link_libraries(yolo_tests
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    GTest::GTest
    GTest::Main
    pthread
)

# Set RPATH so the test executable can find ONNX Runtime libraries
set_target_properties(yolo_tests PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Set C++ standard
set_property(TARGET yolo_tests PROPERTY CXX_STANDARD 17)

# Disable precompiled headers for now to avoid OpenCV dependency issues
# Use precompiled headers from main target (if available)
# if(TARGET single_image_yolo)
#     target_precompile_headers(yolo_tests REUSE_FROM single_image_yolo)
# else()
#     # Fallback: create our own PCH if main target not available
#     target_precompile_headers(yolo_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include/pch.hpp)
# endif()

# Add test discovery
include(GoogleTest)
gtest_discover_tests(yolo_tests) 