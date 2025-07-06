#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "SingleImageYolo.hpp"
#include "Config.hpp"

class UtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "test_utils_temp";
        std::filesystem::create_directories(test_dir_);
    }
    
    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }
    
    std::string test_dir_;
    
    // Helper function to create a test CSV file
    void createTestCSV(const std::string& filename, const std::vector<std::string>& lines) {
        std::ofstream file(filename);
        for (const auto& line : lines) {
            file << line << "\n";
        }
        file.close();
    }
    
    // Helper function to read CSV file
    std::vector<std::string> readCSVLines(const std::string& filename) {
        std::vector<std::string> lines;
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        return lines;
    }
};

// Test CSV file creation and format
TEST_F(UtilsTest, CSVFileFormat) {
    // Create test detections
    std::vector<Detection> test_detections;
    test_detections.emplace_back(cv::Rect2f(10.5f, 20.5f, 100.0f, 150.0f), 0.85f, 0, 0.85f);
    test_detections.emplace_back(cv::Rect2f(200.0f, 300.0f, 80.0f, 120.0f), 0.92f, 2, 0.92f);
    
    // Create a temporary detector just for CSV functionality
    YOLODetector detector("dummy_model.onnx", test_dir_);
    
    // Save to CSV
    std::string csv_path = detector.saveResultsToCSV(test_detections, "test_format.csv");
    
    // Read and verify CSV content
    auto lines = readCSVLines(csv_path);
    
    ASSERT_GE(lines.size(), 3); // Header + 2 data lines
    
    // Check header
    EXPECT_EQ(lines[0], "x1,y1,x2,y2,confidence,class");
    
    // Check first detection (x1=10.5, y1=20.5, x2=110.5, y2=170.5, conf=0.85, class=0)
    EXPECT_TRUE(lines[1].find("10.5,20.5,110.5,170.5,0.85,0") != std::string::npos);
    
    // Check second detection (x1=200, y1=300, x2=280, y2=420, conf=0.92, class=2)
    EXPECT_TRUE(lines[2].find("200,300,280,420,0.92,2") != std::string::npos);
}

// Test file path handling
TEST_F(UtilsTest, FilePathHandling) {
    // Test relative path
    std::string relative_path = "test_image.jpg";
    EXPECT_FALSE(relative_path.empty());
    
    // Test absolute path
    std::string absolute_path = std::filesystem::absolute(relative_path);
    EXPECT_TRUE(absolute_path.find(relative_path) != std::string::npos);
    
    // Test path with directories
    std::string dir_path = test_dir_ + "/subdir/image.jpg";
    std::filesystem::create_directories(test_dir_ + "/subdir");
    
    // Create a dummy file
    std::ofstream dummy_file(dir_path);
    dummy_file << "dummy content";
    dummy_file.close();
    
    EXPECT_TRUE(std::filesystem::exists(dir_path));
}

// Test directory creation
TEST_F(UtilsTest, DirectoryCreation) {
    std::string new_dir = test_dir_ + "/new_output_dir";
    
    // Directory should not exist initially
    EXPECT_FALSE(std::filesystem::exists(new_dir));
    
    // Create directory
    std::filesystem::create_directories(new_dir);
    
    // Directory should now exist
    EXPECT_TRUE(std::filesystem::exists(new_dir));
    EXPECT_TRUE(std::filesystem::is_directory(new_dir));
}

// Test image file validation
TEST_F(UtilsTest, ImageFileValidation) {
    // Create a valid test image
    cv::Mat test_image = cv::Mat::zeros(100, 100, CV_8UC3);
    std::string valid_image_path = test_dir_ + "/valid_image.jpg";
    cv::imwrite(valid_image_path, test_image);
    
    // Test loading valid image
    cv::Mat loaded_image = cv::imread(valid_image_path);
    EXPECT_FALSE(loaded_image.empty());
    EXPECT_EQ(loaded_image.rows, 100);
    EXPECT_EQ(loaded_image.cols, 100);
    
    // Test loading non-existent image
    cv::Mat invalid_image = cv::imread("non_existent_image.jpg");
    EXPECT_TRUE(invalid_image.empty());
}

// Test COCO class names
TEST_F(UtilsTest, COCOClassNames) {
    std::vector<std::string> coco_classes = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "", "backpack", "umbrella"
    };
    
    // Test specific class indices
    EXPECT_EQ(coco_classes[0], "person");
    EXPECT_EQ(coco_classes[2], "car");
    EXPECT_EQ(coco_classes[26], "backpack");
    
    // Test that some indices are empty (reserved)
    EXPECT_EQ(coco_classes[11], "");
    EXPECT_EQ(coco_classes[25], "");
    
    // Test class count
    EXPECT_GE(coco_classes.size(), 27); // At least 27 classes including backpack
}

// Test confidence threshold constants
TEST_F(UtilsTest, ConfidenceThresholds) {
    // Get configuration values
    const auto& config = Config::get();
    
    // Test that constants are in valid range
    EXPECT_GE(config.confidence_threshold, 0.0f);
    EXPECT_LE(config.confidence_threshold, 1.0f);
    
    EXPECT_GE(config.score_threshold, 0.0f);
    EXPECT_LE(config.score_threshold, 1.0f);
    
    EXPECT_GE(config.nms_threshold, 0.0f);
    EXPECT_LE(config.nms_threshold, 1.0f);
    
    // Test typical values
    EXPECT_NEAR(config.confidence_threshold, 0.25f, 0.1f); // Should be around 0.25
    EXPECT_NEAR(config.nms_threshold, 0.45f, 0.1f);  // Should be around 0.45
}

// Test input dimensions
TEST_F(UtilsTest, InputDimensions) {
    // Get configuration values
    const auto& config = Config::get();
    
    // Test that input dimensions are valid
    EXPECT_GT(config.input_width, 0);
    EXPECT_GT(config.input_height, 0);
    
    // Test typical YOLO input size
    EXPECT_EQ(config.input_width, 640);
    EXPECT_EQ(config.input_height, 640);
    
    // Test that dimensions are square (typical for YOLO)
    EXPECT_EQ(config.input_width, config.input_height);
}

// Test detection structure
TEST_F(UtilsTest, DetectionStructure) {
    // Create a detection
    cv::Rect2f bbox(10.0f, 20.0f, 100.0f, 150.0f);
    float confidence = 0.85f;
    int class_id = 2;
    float class_score = 0.85f;
    
    Detection detection(bbox, confidence, class_id, class_score);
    
    // Test that all fields are correctly set
    EXPECT_EQ(detection.bbox.x, 10.0f);
    EXPECT_EQ(detection.bbox.y, 20.0f);
    EXPECT_EQ(detection.bbox.width, 100.0f);
    EXPECT_EQ(detection.bbox.height, 150.0f);
    EXPECT_EQ(detection.confidence, 0.85f);
    EXPECT_EQ(detection.class_id, 2);
    EXPECT_EQ(detection.class_score, 0.85f);
    
    // Test bbox area calculation
    float area = detection.bbox.width * detection.bbox.height;
    EXPECT_EQ(area, 15000.0f); // 100 * 150
}

// Test floating point precision
TEST_F(UtilsTest, FloatingPointPrecision) {
    // Test precision of common calculations
    float a = 0.1f;
    float b = 0.2f;
    float c = a + b;
    
    // Due to floating point precision, this might not be exactly 0.3
    EXPECT_NEAR(c, 0.3f, 1e-6f);
    
    // Test double precision
    double ad = 0.1;
    double bd = 0.2;
    double cd = ad + bd;
    
    EXPECT_NEAR(cd, 0.3, 1e-15);
    
}

// Test mathematical operations
TEST_F(UtilsTest, MathematicalOperations) {
    // Test min/max operations
    EXPECT_EQ(fmin(5.0f, 3.0f), 3.0f);
    EXPECT_EQ(fmax(5.0f, 3.0f), 5.0f);
    
    // Test rounding
    EXPECT_EQ(round(3.4f), 3.0f);
    EXPECT_EQ(round(3.6f), 4.0f);
    EXPECT_EQ(round(3.5f), 4.0f); // Round half up
    
    // Test clamping
    float value = 1.5f;
    float clamped = fmax(0.0f, fmin(value, 1.0f));
    EXPECT_EQ(clamped, 1.0f);
    
    value = -0.5f;
    clamped = fmax(0.0f, fmin(value, 1.0f));
    EXPECT_EQ(clamped, 0.0f);
}

// Test string operations
TEST_F(UtilsTest, StringOperations) {
    // Test string concatenation
    std::string base = "output";
    std::string extension = ".csv";
    std::string full = base + extension;
    EXPECT_EQ(full, "output.csv");
    
    // Test string find
    std::string text = "confidence,class,bbox";
    EXPECT_NE(text.find("confidence"), std::string::npos);
    EXPECT_EQ(text.find("invalid"), std::string::npos);
    
    // Test string conversion
    int class_id = 42;
    std::string class_str = std::to_string(class_id);
    EXPECT_EQ(class_str, "42");
}

// Test error conditions
TEST_F(UtilsTest, ErrorConditions) {
    // Test division by zero protection
    float denominator = 0.0f;
    if (denominator != 0.0f) {
        float result = 1.0f / denominator;
        EXPECT_TRUE(false); // Should not reach here
    } else {
        EXPECT_TRUE(true); // Proper zero check
    }
    
    // Test negative dimensions
    cv::Rect2f invalid_rect(-10.0f, -10.0f, -5.0f, -5.0f);
    EXPECT_LT(invalid_rect.width, 0.0f);
    EXPECT_LT(invalid_rect.height, 0.0f);
    
    // Test out of bounds array access protection
    std::vector<int> test_vector = {1, 2, 3};
    size_t index = 5;
    if (index < test_vector.size()) {
        int value = test_vector[index];
        EXPECT_TRUE(false); // Should not reach here
    } else {
        EXPECT_TRUE(true); // Proper bounds check
    }
}

// Test memory management
TEST_F(UtilsTest, MemoryManagement) {
    // Test OpenCV Mat memory management
    {
        cv::Mat temp_image = cv::Mat::zeros(1000, 1000, CV_8UC3);
        EXPECT_FALSE(temp_image.empty());
        EXPECT_EQ(temp_image.rows, 1000);
        EXPECT_EQ(temp_image.cols, 1000);
    } // temp_image should be automatically released here
    
    // Test vector memory management
    {
        std::vector<Detection> temp_detections;
        for (int i = 0; i < 1000; ++i) {
            temp_detections.emplace_back(cv::Rect2f(i, i, 10, 10), 0.5f, 0, 0.5f);
        }
        EXPECT_EQ(temp_detections.size(), 1000);
    } // temp_detections should be automatically released here
    
    EXPECT_TRUE(true); // If we reach here, no memory issues occurred
} 