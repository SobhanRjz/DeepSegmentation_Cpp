#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include "SingleImageYolo.hpp"

class YOLODetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directories
        std::filesystem::create_directories("test_output");
        std::filesystem::create_directories("test_input");
        
        // Create a simple test image (640x480, 3 channels)
        test_image_ = cv::Mat::zeros(480, 640, CV_8UC3);
        // Add some simple shapes for testing
        cv::rectangle(test_image_, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(255, 0, 0), -1);
        cv::circle(test_image_, cv::Point(400, 300), 50, cv::Scalar(0, 255, 0), -1);
        
        // Save test image
        test_image_path_ = "test_input/test_image.jpg";
        cv::imwrite(test_image_path_, test_image_);
        
        // Model path (assuming it exists)
        model_path_ = "../YoloModel/yolov8n.onnx";
    }
    
    void TearDown() override {
        // Clean up test files
        std::filesystem::remove_all("test_output");
        std::filesystem::remove_all("test_input");
    }
    
    cv::Mat test_image_;
    std::string test_image_path_;
    std::string model_path_;
};

// Test constructor and basic initialization
TEST_F(YOLODetectorTest, ConstructorInitialization) {
    // Test with valid model path
    if (std::filesystem::exists(model_path_)) {
        YOLODetector detector(model_path_, "test_output");
        EXPECT_TRUE(detector.isModelLoaded());
    }
    
    // Test with invalid model path
    YOLODetector invalid_detector("invalid_path.onnx", "test_output");
    EXPECT_FALSE(invalid_detector.isModelLoaded());
}

// Test directory creation
TEST_F(YOLODetectorTest, DirectoryCreation) {
    std::string output_dir = "test_output/new_dir";
    YOLODetector detector(model_path_, output_dir);
    
    EXPECT_TRUE(std::filesystem::exists(output_dir));
}

// Test detection with valid image
TEST_F(YOLODetectorTest, DetectionWithValidImage) {
    if (!std::filesystem::exists(model_path_)) {
        GTEST_SKIP() << "Model file not found, skipping detection test";
    }
    
    YOLODetector detector(model_path_, "test_output");
    ASSERT_TRUE(detector.isModelLoaded());
    
    // Test detection
    auto detections = detector.detect(test_image_path_);
    
    // Should not crash and return a vector (may be empty for simple test image)
    EXPECT_GE(detections.size(), 0);
}

// Test detection with invalid image path
TEST_F(YOLODetectorTest, DetectionWithInvalidImage) {
    if (!std::filesystem::exists(model_path_)) {
        GTEST_SKIP() << "Model file not found, skipping detection test";
    }
    
    YOLODetector detector(model_path_, "test_output");
    ASSERT_TRUE(detector.isModelLoaded());
    
    // Test with non-existent image
    EXPECT_THROW(detector.detect("non_existent_image.jpg"), std::runtime_error);
}

// Test CSV saving functionality
TEST_F(YOLODetectorTest, CSVSaving) {
    YOLODetector detector(model_path_, "test_output");
    
    // Create mock detections
    std::vector<Detection> mock_detections;
    mock_detections.emplace_back(cv::Rect2f(10.5f, 20.5f, 100.0f, 150.0f), 0.85f, 0, 0.85f);
    mock_detections.emplace_back(cv::Rect2f(200.0f, 300.0f, 80.0f, 120.0f), 0.92f, 2, 0.92f);
    
    // Save to CSV
    std::string csv_path = detector.saveResultsToCSV(mock_detections, "test_detections.csv");
    
    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(csv_path));
    
    // Verify CSV content
    std::ifstream file(csv_path);
    std::string line;
    
    // Check header
    std::getline(file, line);
    EXPECT_EQ(line, "x1,y1,x2,y2,confidence,class");
    
    // Check first detection
    std::getline(file, line);
    EXPECT_TRUE(line.find("10.5,20.5,110.5,170.5,0.85,0") != std::string::npos);
    
    // Check second detection
    std::getline(file, line);
    EXPECT_TRUE(line.find("200,300,280,420,0.92,2") != std::string::npos);
}

// Test image processing pipeline
TEST_F(YOLODetectorTest, ProcessImagePipeline) {
    if (!std::filesystem::exists(model_path_)) {
        GTEST_SKIP() << "Model file not found, skipping pipeline test";
    }
    
    YOLODetector detector(model_path_, "test_output");
    ASSERT_TRUE(detector.isModelLoaded());
    
    // Test complete pipeline
    auto [detections, csv_path] = detector.processImage(test_image_path_, true);
    
    // Verify results
    EXPECT_GE(detections.size(), 0);
    EXPECT_FALSE(csv_path.empty());
    EXPECT_TRUE(std::filesystem::exists(csv_path));
}

// Test confidence threshold filtering
TEST_F(YOLODetectorTest, ConfidenceThresholdFiltering) {
    if (!std::filesystem::exists(model_path_)) {
        GTEST_SKIP() << "Model file not found, skipping threshold test";
    }
    
    YOLODetector detector(model_path_, "test_output");
    ASSERT_TRUE(detector.isModelLoaded());
    
    // Test with different confidence thresholds
    auto detections_low = detector.detect(test_image_path_, 0.1f);
    auto detections_high = detector.detect(test_image_path_, 0.8f);
    
    // Higher threshold should result in fewer or equal detections
    EXPECT_GE(detections_low.size(), detections_high.size());
}

// Test detection structure validity
TEST_F(YOLODetectorTest, DetectionStructureValidity) {
    // Create a detection and verify its structure
    cv::Rect2f bbox(10.0f, 20.0f, 100.0f, 150.0f);
    float confidence = 0.85f;
    int class_id = 2;
    float class_score = 0.85f;
    
    Detection detection(bbox, confidence, class_id, class_score);
    
    EXPECT_EQ(detection.bbox.x, 10.0f);
    EXPECT_EQ(detection.bbox.y, 20.0f);
    EXPECT_EQ(detection.bbox.width, 100.0f);
    EXPECT_EQ(detection.bbox.height, 150.0f);
    EXPECT_EQ(detection.confidence, 0.85f);
    EXPECT_EQ(detection.class_id, 2);
    EXPECT_EQ(detection.class_score, 0.85f);
}

// Test model loading with different paths
TEST_F(YOLODetectorTest, ModelLoadingVariations) {
    // Test with relative path
    if (std::filesystem::exists("../YoloModel/yolov8n.onnx")) {
        YOLODetector detector1("../YoloModel/yolov8n.onnx", "test_output");
        EXPECT_TRUE(detector1.isModelLoaded());
    }
    
    // Test with absolute path
    std::string abs_path = std::filesystem::absolute("../YoloModel/yolov8n.onnx");
    if (std::filesystem::exists(abs_path)) {
        YOLODetector detector2(abs_path, "test_output");
        EXPECT_TRUE(detector2.isModelLoaded());
    }
    
    // Test with non-existent path
    YOLODetector detector3("non_existent_model.onnx", "test_output");
    EXPECT_FALSE(detector3.isModelLoaded());
}

// Test error handling
TEST_F(YOLODetectorTest, ErrorHandling) {
    YOLODetector detector("invalid_model.onnx", "test_output");
    
    // Should not crash when model is not loaded
    EXPECT_FALSE(detector.isModelLoaded());
    
    // Detection should throw when model not loaded
    EXPECT_THROW(detector.detect(test_image_path_), std::runtime_error);
} 