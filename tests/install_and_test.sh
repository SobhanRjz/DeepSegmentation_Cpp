#!/bin/bash

# YOLO C++ Unit Tests Installation and Execution Script
# This script installs Google Test and runs the unit tests

set -e  # Exit on any error

echo "🚀 YOLO C++ Unit Tests Setup and Execution"
echo "=========================================="

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            echo "ubuntu"
        elif command -v yum &> /dev/null; then
            echo "centos"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Install Google Test based on OS
install_gtest() {
    local os=$(detect_os)
    echo "📦 Installing Google Test for $os..."
    
    case $os in
        "ubuntu")
            sudo apt-get update
            sudo apt-get install -y libgtest-dev cmake build-essential
            
            # Build and install Google Test
            if [ ! -f /usr/lib/libgtest.a ]; then
                echo "Building Google Test from source..."
                cd /usr/src/gtest
                sudo cmake CMakeLists.txt
                sudo make
                sudo cp lib/*.a /usr/lib/ 2>/dev/null || sudo cp *.a /usr/lib/
            fi
            ;;
        "macos")
            if command -v brew &> /dev/null; then
                brew install googletest cmake
            else
                echo "❌ Homebrew not found. Please install Homebrew first."
                exit 1
            fi
            ;;
        "centos")
            sudo yum install -y cmake gcc-c++ make
            # Install Google Test from source
            install_gtest_from_source
            ;;
        *)
            echo "⚠️  Unknown OS. Attempting to install Google Test from source..."
            install_gtest_from_source
            ;;
    esac
}

# Install Google Test from source
install_gtest_from_source() {
    echo "📥 Installing Google Test from source..."
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd $TEMP_DIR
    
    # Download and build Google Test
    git clone https://github.com/google/googletest.git
    cd googletest
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    
    # Clean up
    cd /
    rm -rf $TEMP_DIR
}

# Check if Google Test is installed
check_gtest() {
    echo "🔍 Checking Google Test installation..."
    
    if pkg-config --exists gtest; then
        echo "✅ Google Test found via pkg-config"
        return 0
    elif [ -f /usr/lib/libgtest.a ] || [ -f /usr/local/lib/libgtest.a ]; then
        echo "✅ Google Test libraries found"
        return 0
    elif command -v gtest-config &> /dev/null; then
        echo "✅ Google Test config found"
        return 0
    else
        echo "❌ Google Test not found"
        return 1
    fi
}

# Build the tests
build_tests() {
    echo "🔨 Building unit tests..."
    
    # Go to project root
    cd "$(dirname "$0")/.."
    
    # Create build directory for tests
    mkdir -p build/tests
    cd build/tests
    
    # Configure CMake
    echo "Configuring CMake..."
    cmake ../../tests -DCMAKE_BUILD_TYPE=Release
    
    # Build
    echo "Building tests..."
    make -j$(nproc)
    
    if [ $? -eq 0 ]; then
        echo "✅ Tests built successfully"
    else
        echo "❌ Failed to build tests"
        exit 1
    fi
}

# Run the tests
run_tests() {
    echo "🧪 Running unit tests..."
    
    if [ -f "./yolo_tests" ]; then
        echo "Executing test suite..."
        ./yolo_tests --gtest_output=xml:test_results.xml
        
        if [ $? -eq 0 ]; then
            echo "✅ All tests passed!"
        else
            echo "❌ Some tests failed"
            exit 1
        fi
    else
        echo "❌ Test executable not found"
        exit 1
    fi
}

# Generate test report
generate_report() {
    echo "📊 Generating test report..."
    
    if [ -f "test_results.xml" ]; then
        echo "Test results saved to: $(pwd)/test_results.xml"
        
        # Extract basic statistics
        if command -v xmllint &> /dev/null; then
            local total_tests=$(xmllint --xpath "count(//testcase)" test_results.xml 2>/dev/null || echo "N/A")
            local failed_tests=$(xmllint --xpath "count(//failure)" test_results.xml 2>/dev/null || echo "0")
            local passed_tests=$((total_tests - failed_tests))
            
            echo "📈 Test Summary:"
            echo "   Total tests: $total_tests"
            echo "   Passed: $passed_tests"
            echo "   Failed: $failed_tests"
        fi
    fi
}

# Main execution
main() {
    echo "Starting YOLO C++ unit test setup..."
    
    # Check if Google Test is already installed
    if ! check_gtest; then
        echo "Google Test not found. Installing..."
        install_gtest
        
        # Verify installation
        if ! check_gtest; then
            echo "❌ Failed to install Google Test"
            exit 1
        fi
    fi
    
    # Build tests
    build_tests
    
    # Run tests
    run_tests
    
    # Generate report
    generate_report
    
    echo ""
    echo "🎉 Unit test execution completed successfully!"
    echo "📁 Test results are available in: $(pwd)/test_results.xml"
}

# Help function
show_help() {
    echo "YOLO C++ Unit Tests Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --install-only Install Google Test only"
    echo "  --build-only   Build tests only"
    echo "  --run-only     Run tests only (assumes already built)"
    echo "  --clean        Clean build directory"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full setup and test execution"
    echo "  $0 --install-only     # Install Google Test only"
    echo "  $0 --build-only       # Build tests only"
    echo "  $0 --run-only         # Run tests only"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --install-only)
        check_gtest || install_gtest
        ;;
    --build-only)
        build_tests
        ;;
    --run-only)
        cd "$(dirname "$0")/../build/tests"
        run_tests
        generate_report
        ;;
    --clean)
        echo "🧹 Cleaning build directory..."
        rm -rf "$(dirname "$0")/../build/tests"
        echo "✅ Build directory cleaned"
        ;;
    "")
        main
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_help
        exit 1
        ;;
esac 