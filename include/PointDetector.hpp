#ifndef POINT_DETECTOR_HPP
#define POINT_DETECTOR_HPP

#include "pch.hpp"
#include "Config.hpp"
#include <onnxruntime_cxx_api.h>

// Additional includes for linter compatibility
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <variant>

// ============================================================================
// Point Detection Structures
// ============================================================================

/**
 * @brief Structure to hold a single detected point
 */
struct DetectedPoint {
    cv::Point2f position;    // Point coordinates (x, y)
    float confidence;        // Detection confidence
    int point_id;           // Point identifier/class
    float visibility;       // Visibility score (optional)
    
    DetectedPoint(const cv::Point2f& pos, float conf, int id = 0, float vis = 1.0f)
        : position(pos), confidence(conf), point_id(id), visibility(vis) {}
};

/**
 * @brief Structure to hold point detection results for an object/instance
 */
struct PointDetectionResult {
    cv::Rect2f bbox;                        // Bounding box around the detected object
    std::vector<DetectedPoint> points;      // Detected points
    float overall_confidence;               // Overall detection confidence
    int class_id;                          // Object class ID
    
    PointDetectionResult(const cv::Rect2f& box, float conf, int cls_id = 0)
        : bbox(box), overall_confidence(conf), class_id(cls_id) {}
};

// ============================================================================
// Point Detection Configuration
// ============================================================================

struct PointDetectionConfig {
    // Detection thresholds
    float confidence_threshold;
    float point_confidence_threshold;
    float nms_threshold;
    
    // Model-specific parameters
    int num_points_per_object;      // Expected number of points per detected object
    bool has_bounding_box;          // Whether model outputs bounding boxes
    bool has_point_classes;         // Whether points have different classes/types
    bool has_visibility_scores;     // Whether model outputs visibility scores
    
    // Output format configuration
    int bbox_output_size;           // Size of bbox output (4 for x,y,w,h)
    int point_output_size;          // Size per point (2 for x,y, 3 for x,y,conf, etc.)
    
    PointDetectionConfig()
        : confidence_threshold(0.25f)
        , point_confidence_threshold(0.5f)
        , nms_threshold(0.45f)
        , num_points_per_object(1)
        , has_bounding_box(true)
        , has_point_classes(false)
        , has_visibility_scores(false)
        , bbox_output_size(4)
        , point_output_size(2) {}
};

// ============================================================================
// ONNX Model Wrapper for Point Detection
// ============================================================================

class PointDetectionOnnxWrapper {
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
    cv::Size input_size_;
    
    void initializeSession(const std::string& model_path, const std::string& provider);
    void configureExecutionProvider(Ort::SessionOptions& session_options, const std::string& provider);
    void extractModelInfo();

public:
    PointDetectionOnnxWrapper(const std::string& model_path, const std::string& provider = "cpu");
    cv::Size getInputSize() const { return input_size_; }
    std::vector<Ort::Value> runInference(const std::vector<float>& input_data, 
                                        const std::vector<int64_t>& input_shape);
};

// ============================================================================
// Point Detector Class
// ============================================================================

class PointDetector {
private:
    std::string model_path_;
    std::string output_dir_;
    std::unique_ptr<PointDetectionOnnxWrapper> model_;
    bool model_loaded_;
    PointDetectionConfig config_;
    
    // Preprocessing parameters (same as SingleImageYolo for consistency)
    int original_width_;
    int original_height_;
    double exact_scale_ratio_;
    int processed_width_;
    int processed_height_;

    // Private methods
    void setupDirectories();
    void loadModel();
    
    // Preprocessing (reuse from SingleImageYolo)
    std::vector<float> preprocess(const cv::Mat& image, cv::Rect2d& scale_info);
    
    // Postprocessing for point detection
    std::vector<PointDetectionResult> postprocess(const std::vector<float>& output_data, 
                                                  const std::vector<int64_t>& output_shape,
                                                  const cv::Rect2d& scale_info);
    
    // NMS on raw model output (before coordinate transformation)
    std::vector<std::vector<std::vector<float>>> non_max_suppression(
        const std::vector<float>& prediction,
        const std::vector<int64_t>& shape,  // [batch, features, boxes]
        float conf_thres,
        float iou_thres,
        const std::vector<int>& classes,
        bool agnostic,
        bool multi_label,
        int max_det,
        int nc,  // number of classes (optional)
        int max_nms,
        int max_wh,
        bool in_place
    );
    
    // EXACT ULTRALYTICS scale_boxes implementation
    std::vector<cv::Rect2f> scaleBoxesUltralytics(
        const cv::Size& img1_shape,
        const std::vector<cv::Rect2f>& boxes,
        const cv::Size& img0_shape,
        const cv::Rect2d& scale_info);

public:
    PointDetector(const std::string& model_path, const std::string& output_dir = "output", 
                 const std::string& provider = "cpu");
    
    ~PointDetector() = default;
    
    // Main detection methods
    std::vector<PointDetectionResult> detect(const std::string& image_path, float conf = -1.0f);
    std::vector<PointDetectionResult> detect(const cv::Mat& image, float conf = -1.0f);
    
    // Visualization methods
    std::string drawAndSavePoints(const std::string& image_path, 
                                 const std::vector<PointDetectionResult>& detections, 
                                 const std::string& output_filename = "points.jpg");
    
    // Export methods
    std::string savePointsToCSV(const std::vector<PointDetectionResult>& results, 
                               const std::string& output_filename = "detected_point_cpp.csv");
    
    std::string savePointsToJSON(const std::vector<PointDetectionResult>& results, 
                                const std::string& output_filename = "points.json");
    
    // Configuration
    bool isModelLoaded() const { return model_loaded_; }
    void setConfig(const PointDetectionConfig& config) { config_ = config; }
    const PointDetectionConfig& getConfig() const { return config_; }
    
    // Model introspection helpers
    void analyzeModelOutput(const std::string& test_image_path);
    void printModelInfo() const;
    
    // Complete processing pipeline
    std::pair<std::vector<PointDetectionResult>, std::string> processImage(
        const std::string& image_name = "input/TestImage.jpeg",
        bool save_csv = true,
        float conf = -1.0f
    );
};

// ============================================================================
// Point Analysis Utilities
// ============================================================================

namespace PointAnalysis {
    /**
     * @brief Calculate distance between two points
     */
    double calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);
    
    /**
     * @brief Calculate angle between three points
     */
    double calculateAngle(const cv::Point2f& p1, const cv::Point2f& center, const cv::Point2f& p3);
    
    /**
     * @brief Find closest point to a reference point
     */
    int findClosestPoint(const cv::Point2f& reference, const std::vector<DetectedPoint>& points);
    
    /**
     * @brief Calculate centroid of multiple points
     */
    cv::Point2f calculateCentroid(const std::vector<DetectedPoint>& points);
    
    /**
     * @brief Filter points by confidence threshold
     */
    std::vector<DetectedPoint> filterByConfidence(const std::vector<DetectedPoint>& points, 
                                                  float threshold);
}

#endif // POINT_DETECTOR_HPP 