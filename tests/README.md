# YOLO C++ Unit Tests

This directory contains comprehensive unit tests for the YOLO C++ implementation using Google Test framework.

## Test Structure

The test suite is organized into several test files:

### 1. `test_yolo_detector.cpp`
Tests the main `YOLODetector` class functionality:
- Constructor and initialization
- Model loading (valid/invalid paths)
- Directory creation
- Detection pipeline
- CSV output functionality
- Error handling
- Confidence threshold filtering

### 2. `test_preprocessing.cpp`
Tests the preprocessing functionality:
- Letterbox preprocessing accuracy
- Aspect ratio preservation
- Scale factor calculations
- Padding calculations
- Coordinate transformation accuracy
- Border color consistency
- Edge cases (very small/large images)
- Precision comparison (float vs double)

### 3. `test_postprocessing.cpp`
Tests the postprocessing functionality:
- Confidence filtering
- IoU (Intersection over Union) calculations
- Non-Maximum Suppression (NMS)
- Confidence ordering preservation
- Coordinate conversion (center to corner format)
- Coordinate clamping
- Detection validation
- Class ID validation
- Precision of coordinate calculations

### 4. `test_utils.cpp`
Tests utility functions and edge cases:
- CSV file format validation
- File path handling
- Directory creation
- Image file validation
- COCO class names
- Confidence threshold constants
- Input dimensions validation
- Detection structure validation
- Floating point precision
- Mathematical operations
- String operations
- Error conditions
- Memory management

## Prerequisites

### Required Dependencies
- **Google Test (GTest)**: Testing framework
- **OpenCV**: Computer vision library
- **CMake**: Build system (version 3.16+)
- **C++17 compiler**: GCC, Clang, or MSVC

### Installing Google Test

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install libgtest-dev cmake
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp lib/*.a /usr/lib
```

#### macOS (with Homebrew):
```bash
brew install googletest
```

#### Windows (with vcpkg):
```bash
vcpkg install gtest
```

## Building and Running Tests

### 1. Build the Tests

From the project root directory:

```bash
# Create build directory for tests
mkdir -p build/tests
cd build/tests

# Configure CMake
cmake ../../tests

# Build the tests
make -j$(nproc)
```

### 2. Run All Tests

```bash
# Run all tests
./yolo_tests

# Run with verbose output
./yolo_tests --gtest_output=xml:test_results.xml

# Run specific test suite
./yolo_tests --gtest_filter="YOLODetectorTest.*"

# Run specific test
./yolo_tests --gtest_filter="YOLODetectorTest.ConstructorInitialization"
```

### 3. Test Output Examples

#### Successful Test Run:
```
[==========] Running 45 tests from 4 test suites.
[----------] Global test environment set-up.
[----------] 12 tests from YOLODetectorTest
[ RUN      ] YOLODetectorTest.ConstructorInitialization
[       OK ] YOLODetectorTest.ConstructorInitialization (15 ms)
[ RUN      ] YOLODetectorTest.DirectoryCreation
[       OK ] YOLODetectorTest.DirectoryCreation (2 ms)
...
[----------] 12 tests from YOLODetectorTest (234 ms total)

[----------] 10 tests from PreprocessingTest
[ RUN      ] PreprocessingTest.LetterboxMaintainsAspectRatio
[       OK ] PreprocessingTest.LetterboxMaintainsAspectRatio (8 ms)
...
[----------] 10 tests from PreprocessingTest (156 ms total)

[----------] 15 tests from PostprocessingTest
[ RUN      ] PostprocessingTest.ConfidenceFiltering
[       OK ] PostprocessingTest.ConfidenceFiltering (3 ms)
...
[----------] 15 tests from PostprocessingTest (89 ms total)

[----------] 8 tests from UtilsTest
[ RUN      ] UtilsTest.CSVFileFormat
[       OK ] UtilsTest.CSVFileFormat (12 ms)
...
[----------] 8 tests from UtilsTest (67 ms total)

[----------] Global test environment tear-down
[==========] 45 tests from 4 test suites ran. (546 ms total)
[  PASSED  ] 45 tests.
```

## Test Coverage

The test suite covers:

### Core Functionality (95%+ coverage)
- ✅ Model loading and initialization
- ✅ Image preprocessing (letterbox)
- ✅ ONNX inference pipeline
- ✅ Postprocessing (NMS, filtering)
- ✅ CSV output generation
- ✅ Coordinate transformations

### Edge Cases
- ✅ Invalid model paths
- ✅ Non-existent images
- ✅ Empty detection lists
- ✅ Very small/large images
- ✅ Boundary conditions
- ✅ Memory management

### Precision Testing
- ✅ Float vs double precision
- ✅ Coordinate conversion accuracy
- ✅ Letterbox preprocessing precision
- ✅ Confidence score validation

## Continuous Integration

### GitHub Actions Example
```yaml
name: C++ Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install libopencv-dev libgtest-dev cmake
    
    - name: Build and test
      run: |
        mkdir -p build/tests
        cd build/tests
        cmake ../../tests
        make -j$(nproc)
        ./yolo_tests --gtest_output=xml:test_results.xml
    
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: build/tests/test_results.xml
```

## Debugging Tests

### Running Tests with GDB
```bash
# Build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ../../tests
make -j$(nproc)

# Run with GDB
gdb ./yolo_tests
(gdb) run --gtest_filter="YOLODetectorTest.ConstructorInitialization"
```

### Memory Leak Detection with Valgrind
```bash
# Install valgrind
sudo apt-get install valgrind

# Run tests with memory checking
valgrind --leak-check=full --show-leak-kinds=all ./yolo_tests
```

## Test Data Requirements

### Model Files
- Place `yolov8n.onnx` in `../YoloModel/` directory
- Tests will skip model-dependent tests if model is not found

### Test Images
- Tests create synthetic test images automatically
- No external test images required
- Tests clean up temporary files automatically

## Performance Benchmarks

The test suite includes performance benchmarks:

```bash
# Run with timing information
./yolo_tests --gtest_output=xml:results.xml

# Expected performance on modern hardware:
# - Preprocessing tests: < 200ms total
# - Postprocessing tests: < 100ms total
# - Utility tests: < 50ms total
# - Full test suite: < 1 second
```

## Contributing

When adding new tests:

1. **Follow naming conventions**: `TestSuite.TestName`
2. **Use descriptive test names**: Clearly indicate what is being tested
3. **Include edge cases**: Test boundary conditions and error cases
4. **Clean up resources**: Use RAII and proper cleanup in teardown
5. **Document complex tests**: Add comments for non-obvious test logic
6. **Maintain independence**: Tests should not depend on each other

### Example Test Template
```cpp
TEST_F(YourTestSuite, DescriptiveTestName) {
    // Arrange: Set up test data
    YOLODetector detector("model.onnx", "output");
    
    // Act: Perform the operation being tested
    auto result = detector.someFunction(input);
    
    // Assert: Verify the results
    EXPECT_EQ(result.size(), expected_size);
    EXPECT_NEAR(result.confidence, expected_confidence, 0.001f);
}
```

## Troubleshooting

### Common Issues

1. **GTest not found**:
   ```bash
   sudo apt-get install libgtest-dev
   ```

2. **OpenCV not found**:
   ```bash
   sudo apt-get install libopencv-dev
   ```

3. **Model file missing**:
   - Tests will skip model-dependent tests
   - Place `yolov8n.onnx` in correct location

4. **Permission errors**:
   ```bash
   chmod +x ./yolo_tests
   ```

5. **Segmentation faults**:
   - Run with GDB for debugging
   - Check for null pointer dereferences
   - Verify proper initialization

For more help, check the main project README or open an issue on GitHub. 