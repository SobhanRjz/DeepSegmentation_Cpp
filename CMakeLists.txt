cmake_minimum_required(VERSION 3.16)
project(DeepNetCPP)

# Set CUDA environment for LibTorch compatibility
set(ENV{CUDA_HOME} "/usr/local/cuda-12.4")
set(ENV{CUDA_ROOT} "/usr/local/cuda-12.4")

include(ExternalProject)
set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA SUPPORT
# ============================================================================

# Set policies for modern CMake support
cmake_policy(SET CMP0146 NEW)  # Use modern CUDA support
cmake_policy(SET CMP0135 NEW)  # Set DOWNLOAD_EXTRACT_TIMESTAMP

# Enable CUDA language support
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    add_definitions(-DWITH_CUDA)
    set(CUDA_FOUND TRUE)
else()
    message(STATUS "CUDA not found - GPU acceleration will be disabled")
    set(CUDA_FOUND FALSE)
endif()

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================

# Aggressive compiler optimizations for maximum speed
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -march=native -mtune=native -ffast-math")
message(STATUS "Project: ${PROJECT_NAME} ${PROJECT_VERSION}")

# Enable Link Time Optimization (LTO) for better performance
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Use Release build by default for performance
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

# Performance-oriented threading
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# ============================================================================
# DEPENDENCIES
# ============================================================================
ExternalProject_Add(boost
    #GIT_REPOSITORY https://github.com/boostorg/boost.git
    #GIT_TAG boost-1.85.0
    URL https://github.com/boostorg/boost/releases/download/boost-1.85.0/boost-1.85.0-cmake.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS "-DCMAKE_BUILD_TYPE=RELEASE" "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install" -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
)

ExternalProject_Add(opencv_contrib
    #GIT_REPOSITORY https://github.com/opencv/opencv_contrib.git
    #GIT_TAG 4.5.2
    #GIT_TAG 4.9.0
    #GIT_TAG 4.10.0
    URL https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.tar.gz
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/libs/opencv_contrib_src
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""      
)

# Set OpenCV to always build in Release mode and use a shared install path
# This prevents rebuilding OpenCV for both debug and release configurations
set(OPENCV_CMAKE_BUILD_TYPE "Release")
set(OPENCV_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/libs/opencv_install)
message(STATUS "OPENCV_INSTALL_PREFIX: ${OPENCV_INSTALL_PREFIX}")

# Create the install directory if it doesn't exist
file(MAKE_DIRECTORY ${OPENCV_INSTALL_PREFIX})

# Check if OpenCV is already built to avoid unnecessary rebuilds
set(OPENCV_MARKER_FILE ${OPENCV_INSTALL_PREFIX}/opencv_built.marker)

# Only build OpenCV if not already built
if(NOT EXISTS ${OPENCV_MARKER_FILE})
    message(STATUS "OpenCV not found, will build from source")
    
    ExternalProject_Add(opencv_contrib_external
        URL https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/libs/opencv_contrib_src
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""      
    )

    ExternalProject_Add(opencv_external
    #GIT_REPOSITORY https://github.com/opencv/opencv.git
    #GIT_TAG 4.5.2
    #GIT_TAG 4.9.0
    #GIT_TAG 4.10.0
    URL https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/libs/opencv_src
    CMAKE_ARGS
    "-D CMAKE_BUILD_TYPE=Release"
    "-D CMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_PREFIX}"
    "-D OPENCV_EXTRA_MODULES_PATH=${PROJECT_SOURCE_DIR}/libs/opencv_contrib_src/modules"
    "-D BUILD_OPENCV_WORLD=ON"
    "-D BUILD_OPENCV_DNN=ON"
    "-D WITH_GSTREAMER=ON"
    "-D WITH_OPENGL=ON"
    "-D WITH_V4L=ON"
    "-D WITH_CSTRIPES=ON"
    "-D WITH_OPENCL=ON"
    "-D OPENCV_ENABLE_NONFREE=ON"
    "-D WITH_CUDA=ON"
    "-D WITH_CUDNN=ON"
    "-D OPENCV_DNN_CUDA=ON"
    "-D CUDA_FAST_MATH=ON"
    "-D WITH_CUBLAS=ON"
    "-D ENABLE_FAST_MATH=ON"
    "-D WITH_TBB=ON"
    "-D WITH_OPENMP=ON"
    "-D WITH_IPP=ON"
    "-D WITH_EIGEN=ON"
    "-D BUILD_PERF_TESTS=OFF"
    "-D BUILD_TESTS=OFF"
    "-D BUILD_DOCS=OFF"
    "-D BUILD_EXAMPLES=OFF"
    "-D CUDA_ARCH_BIN=89"
    "-D CPU_BASELINE_REQUIRE=SSE2,SSE3,SSE4_2"
    "-D CPU_DISPATCH=FP16,FMA3,AVX,AVX2,AVX512_ICL"
    "-D BUILD_opencv_python3=ON"
    -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    DEPENDS opencv_contrib_external
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
                    COMMAND ${CMAKE_COMMAND} -E touch ${OPENCV_MARKER_FILE}
    )
    
    # Create aliases for the expected names using different approach
    set(OPENCV_TARGET_NAME opencv_external)
    set(OPENCV_CONTRIB_TARGET_NAME opencv_contrib_external)
else()
    message(STATUS "OpenCV already built, skipping build step")
    # Create empty targets for dependency management using different names
    add_custom_target(opencv_built_target)
    add_custom_target(opencv_contrib_built_target)
    set(OPENCV_TARGET_NAME opencv_built_target)
    set(OPENCV_CONTRIB_TARGET_NAME opencv_contrib_built_target)
endif()

# Set up OpenCV variables after it's built - use the already defined install path
set(OPENCV_INSTALL_PATH ${OPENCV_INSTALL_PREFIX})
set(OpenCV_DIR ${OPENCV_INSTALL_PATH}/lib/cmake/opencv4)
set(OpenCV_INCLUDE_DIRS ${OPENCV_INSTALL_PATH}/include/opencv4)

# Manually set OpenCV libraries since we're building as external project
set(OpenCV_LIBS 
    ${OPENCV_INSTALL_PATH}/lib/libopencv_core.so
    ${OPENCV_INSTALL_PATH}/lib/libopencv_imgproc.so
    ${OPENCV_INSTALL_PATH}/lib/libopencv_imgcodecs.so
    ${OPENCV_INSTALL_PATH}/lib/libopencv_highgui.so
    ${OPENCV_INSTALL_PATH}/lib/libopencv_dnn.so
)

include_directories(${OPENCV_INSTALL_PATH}/include/opencv4)
include_directories(${OPENCV_INSTALL_PATH}/include)

#include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Define OPENCV_AVAILABLE for precompiled headers
add_definitions(-DOPENCV_AVAILABLE)

# Check if OpenCV has CUDA support
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    
    # Try to detect CUDA modules in OpenCV
    foreach(lib ${OpenCV_LIBS})
        if(${lib} MATCHES ".*cuda.*")
            message(STATUS "Found OpenCV CUDA module: ${lib}")
            set(OPENCV_HAS_CUDA TRUE)
        endif()
    endforeach()
    
    if(OPENCV_HAS_CUDA)
        message(STATUS "OpenCV compiled with CUDA support")
    else()
        message(STATUS "OpenCV compiled without CUDA support - GPU preprocessing will fallback to CPU")
    endif()
endif()

# ============================================================================
# ONNX RUNTIME SETUP
# ============================================================================

# Set ONNX Runtime to always build in Release mode and use a shared install path
# This prevents rebuilding ONNX Runtime for both debug and release configurations
set(ONNXRUNTIME_CMAKE_BUILD_TYPE "Release")
set(ONNXRUNTIME_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/libs/onnxruntime_install)
set(ONNXRUNTIME_SOURCE_PATH ${PROJECT_SOURCE_DIR}/libs/onnxruntime_src)
set(ONNXRUNTIME_BUILD_PATH ${ONNXRUNTIME_SOURCE_PATH}/build/Linux/Release)
message(STATUS "ONNXRUNTIME_INSTALL_PREFIX: ${ONNXRUNTIME_INSTALL_PREFIX}")

# Create the install directory if it doesn't exist
file(MAKE_DIRECTORY ${ONNXRUNTIME_INSTALL_PREFIX})

# Check if ONNX Runtime is already built to avoid unnecessary rebuilds
set(ONNXRUNTIME_MARKER_FILE ${ONNXRUNTIME_INSTALL_PREFIX}/onnxruntime_built.marker)

# Only build ONNX Runtime if not already built
if(NOT EXISTS ${ONNXRUNTIME_MARKER_FILE})
    message(STATUS "ONNX Runtime not found, will build from source with optimized flags")
    
    # Check if ONNX Runtime source exists
    if(NOT EXISTS ${ONNXRUNTIME_SOURCE_PATH})
        message(STATUS "ONNX Runtime source not found, cloning...")
        ExternalProject_Add(onnxruntime_clone
            GIT_REPOSITORY https://github.com/microsoft/onnxruntime.git
            GIT_TAG v1.23.0  # Use stable version
            GIT_SHALLOW FALSE  # Need full history for submodules
            GIT_SUBMODULES_RECURSE TRUE
            SOURCE_DIR ${ONNXRUNTIME_SOURCE_PATH}
            UPDATE_COMMAND ""
            PATCH_COMMAND ""
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
        set(ONNXRUNTIME_CLONE_TARGET onnxruntime_clone)
    else()
        message(STATUS "ONNX Runtime source found at ${ONNXRUNTIME_SOURCE_PATH}")
        # Create empty target for dependency management
        add_custom_target(onnxruntime_source_ready)
        set(ONNXRUNTIME_CLONE_TARGET onnxruntime_source_ready)
    endif()

    # Build ONNX Runtime with your optimized flags
    ExternalProject_Add(onnxruntime_external
        SOURCE_DIR ${ONNXRUNTIME_SOURCE_PATH}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND cd ${ONNXRUNTIME_SOURCE_PATH} && 
                      ./build.sh --config Release 
                                 --build_shared_lib 
                                 --parallel 2 
                                 --use_cuda 
                                 --cuda_home /usr/local/cuda-12.4 
                                 --cudnn_home /usr 
                                 --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=89 
                                 --cmake_extra_defines onnxruntime_USE_CUDA_NHWC_OPS=ON 
                                 --skip_tests
        UPDATE_COMMAND ""
        DEPENDS ${ONNXRUNTIME_CLONE_TARGET}
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${ONNXRUNTIME_BUILD_PATH} ${ONNXRUNTIME_INSTALL_PREFIX}/lib
                        COMMAND ${CMAKE_COMMAND} -E copy_directory ${ONNXRUNTIME_SOURCE_PATH}/include ${ONNXRUNTIME_INSTALL_PREFIX}/include
                        COMMAND ${CMAKE_COMMAND} -E touch ${ONNXRUNTIME_MARKER_FILE}
    )
    
    set(ONNXRUNTIME_TARGET_NAME onnxruntime_external)
else()
    message(STATUS "ONNX Runtime already built, skipping build step")
    # Create empty target for dependency management
    add_custom_target(onnxruntime_built_target)
    set(ONNXRUNTIME_TARGET_NAME onnxruntime_built_target)
endif()

# Set up ONNX Runtime variables after it's built
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_INSTALL_PREFIX}/lib)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INSTALL_PREFIX}/include)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)

message(STATUS "ONNX Runtime will be built at: ${ONNXRUNTIME_INSTALL_PREFIX}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")
message(STATUS "ONNX Runtime headers: ${ONNXRUNTIME_INCLUDE_DIRS}")

# Set up ONNX Runtime include directories
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime/core/session)

# ============================================================================
# LIBTORCH SETUP
# ============================================================================

# Set LibTorch to always build in Release mode and use a shared install path
# This prevents rebuilding LibTorch for both debug and release configurations
set(LIBTORCH_CMAKE_BUILD_TYPE "Release")
set(LIBTORCH_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/libs/pytorch_install)
message(STATUS "LIBTORCH_INSTALL_PREFIX: ${LIBTORCH_INSTALL_PREFIX}")

# Create the install directory if it doesn't exist
file(MAKE_DIRECTORY ${LIBTORCH_INSTALL_PREFIX})

# Check if LibTorch is already built to avoid unnecessary rebuilds
set(LIBTORCH_MARKER_FILE ${LIBTORCH_INSTALL_PREFIX}/libtorch_built.marker)

# Only build LibTorch if not already built
if(NOT EXISTS ${LIBTORCH_MARKER_FILE})
    message(STATUS "LibTorch not found, will build from source")
    
    # Check if PyTorch source exists
    if(NOT EXISTS ${PROJECT_SOURCE_DIR}/libs/pytorch)
        message(STATUS "PyTorch source not found, cloning...")
        ExternalProject_Add(pytorch_clone
            GIT_REPOSITORY https://github.com/pytorch/pytorch.git
            GIT_TAG v2.6.0  # Use stable version
            GIT_SHALLOW TRUE
            GIT_SUBMODULES_RECURSE TRUE
            SOURCE_DIR ${PROJECT_SOURCE_DIR}/libs/pytorch
            UPDATE_COMMAND ""
            PATCH_COMMAND ""
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
        set(PYTORCH_CLONE_TARGET pytorch_clone)
    else()
        message(STATUS "PyTorch source found at ${PROJECT_SOURCE_DIR}/libs/pytorch")
        # Create empty target for dependency management
        add_custom_target(pytorch_source_ready)
        set(PYTORCH_CLONE_TARGET pytorch_source_ready)
    endif()

    ExternalProject_Add(libtorch_external
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/libs/pytorch
        CMAKE_ARGS
        "-D CMAKE_BUILD_TYPE=${LIBTORCH_CMAKE_BUILD_TYPE}"
        "-D CMAKE_INSTALL_PREFIX=${LIBTORCH_INSTALL_PREFIX}"
        "-D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-D BUILD_SHARED_LIBS=ON"
        "-D USE_CUDA=ON"
        "-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4"
        "-D TORCH_CUDA_ARCH_LIST=8.9"                    # Only your GPU architecture
        "-D CMAKE_CUDA_ARCHITECTURES=89"                 # CMake-style CUDA arch
        "-D USE_SYSTEM_NCCL=ON"                          # Multi-GPU support
        "-D BUILD_TEST=OFF"                              # Skip unit tests
        "-D BUILD_BENCHMARK=OFF"                         # Skip benchmarks  
        "-D BUILD_PYTHON=OFF"                            # Pure C++ build
        "-D USE_DISTRIBUTED=OFF"                         # Cut out ProcessGroup/TensorPipe
        "-D USE_TENSORPIPE=OFF"                          # More distributed plumbing
        "-D BUILD_CAFFE2_OPS=OFF"                        # Legacy ops not needed
        "-D USE_XNNPACK=OFF"                             # Mobile/quantization CPU paths
        "-D USE_QNNPACK=OFF"                             # Mobile/quantization CPU paths
        "-D USE_NNPACK=OFF"                              # Mobile/quantization CPU paths
        "-D USE_PYTORCH_QNNPACK=OFF"                     # Mobile/quantization CPU paths
        "-D USE_MKLDNN=OFF"                              # Skip heavy oneDNN for GPU-focused build
        "-D USE_FBGEMM=ON"                               # Small, helpful for CPU fallback GEMM
        "-D USE_KINETO=ON"                               # Profiling support
        "-D USE_ITT=ON"                                  # Intel profiling support
        UPDATE_COMMAND ""
        DEPENDS ${PYTORCH_CLONE_TARGET}
        INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
                        COMMAND ${CMAKE_COMMAND} -E touch ${LIBTORCH_MARKER_FILE}
    )
    
    set(LIBTORCH_TARGET_NAME libtorch_external)
else()
    message(STATUS "LibTorch already built, skipping build step")
    # Create empty target for dependency management
    add_custom_target(libtorch_built_target)
    set(LIBTORCH_TARGET_NAME libtorch_built_target)
endif()

# Set up LibTorch variables after it's built - use the already defined install path
set(TORCH_ROOT_PATH ${LIBTORCH_INSTALL_PREFIX})
set(CMAKE_PREFIX_PATH ${TORCH_ROOT_PATH})

# Manually set LibTorch paths since we're using external project
set(Torch_DIR ${TORCH_ROOT_PATH}/share/cmake/Torch)
set(TORCH_INCLUDE_DIRS ${TORCH_ROOT_PATH}/include ${TORCH_ROOT_PATH}/include/torch/csrc/api/include)

# Manually set LibTorch libraries
set(TORCH_LIBRARIES 
    ${TORCH_ROOT_PATH}/lib/libtorch.so
    ${TORCH_ROOT_PATH}/lib/libtorch_cpu.so
    ${TORCH_ROOT_PATH}/lib/libtorch_cuda.so
    ${TORCH_ROOT_PATH}/lib/libc10.so
    ${TORCH_ROOT_PATH}/lib/libc10_cuda.so
    ${TORCH_ROOT_PATH}/lib/libtorch_global_deps.so
)

# Try to find LibTorch, but use manual settings if not found
find_package(Torch QUIET PATHS ${TORCH_ROOT_PATH})
if(NOT Torch_FOUND)
    message(STATUS "Using manually configured LibTorch paths")
    # Define torch_library target manually
    add_library(torch_library SHARED IMPORTED)
    set_target_properties(torch_library PROPERTIES
        IMPORTED_LOCATION ${TORCH_ROOT_PATH}/lib/libtorch.so
        INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
    )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message(STATUS "LibTorch found at: ${TORCH_ROOT_PATH}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")

# Check if LibTorch has CUDA support
if(TORCH_CUDA_VERSION)
    message(STATUS "LibTorch compiled with CUDA ${TORCH_CUDA_VERSION}")
    add_definitions(-DTORCH_CUDA_AVAILABLE)
else()
    message(STATUS "LibTorch compiled without CUDA support")
endif()

# Add cuBLAS and CUDA libraries explicitly for LibTorch
if(CUDA_FOUND)
    # Find CUDA toolkit
    find_package(CUDAToolkit REQUIRED)
    
    if(CUDAToolkit_FOUND)
        message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_ROOT}")
        
        # Link CUDA libraries
        list(APPEND TORCH_LIBRARIES CUDA::cublas)
        list(APPEND TORCH_LIBRARIES CUDA::cusolver) 
        list(APPEND TORCH_LIBRARIES CUDA::curand)
        list(APPEND TORCH_LIBRARIES CUDA::cudart)
        
        message(STATUS "Added CUDA libraries to LibTorch")
    endif()
endif()

# Include LibTorch headers
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${TORCH_ROOT_PATH}/include)
include_directories(${TORCH_ROOT_PATH}/include/torch/csrc/api/include)

# ============================================================================
# EXECUTABLES
# ============================================================================

# Segmentation Batch Processing application with ONNX Runtime
add_executable(segmentation_batch
    src/main_SegmentationBatch.cpp
    src/BatchSegmentationProcessor.cpp
    src/SegmentationDetector.cpp
    src/Config.cpp
)
add_dependencies(segmentation_batch ${OPENCV_TARGET_NAME} ${LIBTORCH_TARGET_NAME} ${ONNXRUNTIME_TARGET_NAME})

# Link libraries including CUDA if available
set(SEGMENTATION_BATCH_LIBS ${OpenCV_LIBS} ${ONNXRUNTIME_LIB} ${TORCH_LIBRARIES})
if(CUDA_FOUND AND CUDAToolkit_FOUND)
    list(APPEND SEGMENTATION_BATCH_LIBS CUDA::cudart)
endif()

target_link_libraries(segmentation_batch ${SEGMENTATION_BATCH_LIBS})

# Set RPATH so the executable can find ONNX Runtime, LibTorch and OpenCV libraries
set_target_properties(segmentation_batch PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}:${TORCH_ROOT_PATH}/lib:${OPENCV_INSTALL_PATH}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Use specialized precompiled header for segmentation batch processing
target_precompile_headers(segmentation_batch PRIVATE include/pch_segmentation_batch.hpp)

# Point Batch Processing application with ONNX Runtime
add_executable(point_batch
    src/main_PointBatch.cpp
    src/BatchPointProcessor.cpp
    src/PointDetector.cpp
    src/Config.cpp
)
add_dependencies(point_batch ${OPENCV_TARGET_NAME} ${LIBTORCH_TARGET_NAME} ${ONNXRUNTIME_TARGET_NAME})

# Link libraries including CUDA if available
set(POINT_BATCH_LIBS ${OpenCV_LIBS} ${ONNXRUNTIME_LIB} ${TORCH_LIBRARIES})
if(CUDA_FOUND AND CUDAToolkit_FOUND)
    list(APPEND POINT_BATCH_LIBS CUDA::cudart)
endif()

target_link_libraries(point_batch ${POINT_BATCH_LIBS})

# Set RPATH so the executable can find ONNX Runtime, LibTorch and OpenCV libraries
set_target_properties(point_batch PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}:${TORCH_ROOT_PATH}/lib:${OPENCV_INSTALL_PATH}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Use specialized precompiled header for point batch processing
target_precompile_headers(point_batch PRIVATE include/pch_segmentation_batch.hpp)

# Unified Batch Processor (automatically detects model type)
add_executable(unified_batch_processor
    src/main_UnifiedBatchProcessor.cpp
    src/BatchSegmentationProcessor.cpp
    src/BatchPointProcessor.cpp
    src/SegmentationDetector.cpp
    src/PointDetector.cpp
    src/Config.cpp
)
add_dependencies(unified_batch_processor ${OPENCV_TARGET_NAME} ${LIBTORCH_TARGET_NAME} ${ONNXRUNTIME_TARGET_NAME})

# Link libraries including CUDA if available
set(UNIFIED_BATCH_LIBS ${OpenCV_LIBS} ${ONNXRUNTIME_LIB} ${TORCH_LIBRARIES})
if(CUDA_FOUND AND CUDAToolkit_FOUND)
    list(APPEND UNIFIED_BATCH_LIBS CUDA::cudart)
endif()

target_link_libraries(unified_batch_processor ${UNIFIED_BATCH_LIBS})

# Set RPATH so the executable can find ONNX Runtime, LibTorch and OpenCV libraries
set_target_properties(unified_batch_processor PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}:${TORCH_ROOT_PATH}/lib:${OPENCV_INSTALL_PATH}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Use specialized precompiled header for unified batch processing
target_precompile_headers(unified_batch_processor PRIVATE include/pch_segmentation_batch.hpp)

# Test OpenCV utility (separate executable)
add_executable(test_opencv src/testOpenCV.cpp)
add_dependencies(test_opencv ${OPENCV_TARGET_NAME})
target_link_libraries(test_opencv ${OpenCV_LIBS})

# ============================================================================
# TESTING
# ============================================================================

# Add testing support
enable_testing()

# Add tests subdirectory if it exists - temporarily disabled due to OpenCV dependency issues
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
#     add_subdirectory(tests)
# endif()

# ============================================================================
# DEMO TARGETS
# ============================================================================

# Add custom target for segmentation batch processing demo
add_custom_target(run_segmentation_batch_demo
    COMMAND LD_LIBRARY_PATH=${ONNXRUNTIME_LIB_DIR}:${TORCH_ROOT_PATH}/lib:${OPENCV_INSTALL_PATH}/lib:$$LD_LIBRARY_PATH ./segmentation_batch
    DEPENDS segmentation_batch
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running segmentation batch processing demonstration with ONNX Runtime"
)

# Add custom target for point batch processing demo
add_custom_target(run_point_batch_demo
    COMMAND LD_LIBRARY_PATH=${ONNXRUNTIME_LIB_DIR}:${TORCH_ROOT_PATH}/lib:${OPENCV_INSTALL_PATH}/lib:$$LD_LIBRARY_PATH ./point_batch
    DEPENDS point_batch
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running point batch processing demonstration with ONNX Runtime"
)

# Add custom target for unified batch processing demo
add_custom_target(run_unified_batch_demo
    COMMAND LD_LIBRARY_PATH=${ONNXRUNTIME_LIB_DIR}:${TORCH_ROOT_PATH}/lib:${OPENCV_INSTALL_PATH}/lib:$$LD_LIBRARY_PATH ./unified_batch_processor
    DEPENDS unified_batch_processor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running unified batch processing demonstration with automatic model detection"
)

# Add custom target for OpenCV test
add_custom_target(run_opencv_test
    COMMAND LD_LIBRARY_PATH=${OPENCV_INSTALL_PATH}/lib:$$LD_LIBRARY_PATH ./test_opencv
    DEPENDS test_opencv
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running OpenCV installation test"
)
