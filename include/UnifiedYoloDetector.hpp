#ifndef UNIFIED_YOLO_DETECTOR_HPP
#define UNIFIED_YOLO_DETECTOR_HPP

#include "pch.hpp"
#include "DetectionTypes.hpp"
#include "Config.hpp"
#include <onnxruntime_cxx_api.h>

// Forward declaration for backward compatibility
struct Detection {
    cv::Rect2f bbox;
    float confidence;
    int class_id;
    float class_score;
    
    Detection(const cv::Rect2f& box, float conf, int cls_id, float cls_score)
        : bbox(box), confidence(conf), class_id(cls_id), class_score(cls_score) {}
};

// ============================================================================
// Enhanced ONNX Model Wrapper for Multiple Detection Types
// ============================================================================

class UnifiedOnnxWrapper {
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
    cv::Size input_size_;
    ModelType model_type_;
    
    void initializeSession(const std::string& model_path, const std::string& provider);
    void configureExecutionProvider(Ort::SessionOptions& session_options, const std::string& provider);
    void extractModelInfo();
    void detectModelType(const std::string& model_path);

public:
    UnifiedOnnxWrapper(const std::string& model_path, const std::string& provider = "cpu");
    cv::Size getInputSize() const { return input_size_; }
    ModelType getModelType() const { return model_type_; }
    std::vector<Ort::Value> runInference(const std::vector<float>& input_data, 
                                        const std::vector<int64_t>& input_shape);
};

// ============================================================================
// Unified YOLO Detector Class
// ============================================================================

class UnifiedYoloDetector {
private:
    std::string model_path_;
    std::string output_dir_;
    std::unique_ptr<UnifiedOnnxWrapper> model_;
    bool model_loaded_;
    DetectionConfig detection_config_;
    
    // Preprocessing parameters
    int original_width_;
    int original_height_;
    double exact_scale_ratio_;
    int processed_width_;
    int processed_height_;

    // Private methods
    void setupDirectories();
    void loadModel();
    void configureDetectionType();
    
    // Preprocessing (same as SingleImageYolo)
    std::vector<float> preprocess(const cv::Mat& image, cv::Rect2d& scale_info);
    
    // Postprocessing methods for different detection types
    std::vector<UnifiedDetection> postprocessBoundingBox(const std::vector<double>& output_data, 
                                                        const std::vector<int64_t>& output_shape,
                                                        const cv::Rect2d& scale_info);
    
    std::vector<UnifiedDetection> postprocessKeypoints(const std::vector<double>& output_data, 
                                                      const std::vector<int64_t>& output_shape,
                                                      const cv::Rect2d& scale_info);
    
    std::vector<UnifiedDetection> postprocessSegmentation(const std::vector<double>& output_data, 
                                                         const std::vector<int64_t>& output_shape,
                                                         const cv::Rect2d& scale_info);
    
    // NMS for different detection types
    std::vector<UnifiedDetection> applyNMS(const std::vector<UnifiedDetection>& detections, float nms_threshold);
    
    // Coordinate transformation utilities
    cv::Point2f transformPoint(const cv::Point2f& point, const cv::Rect2d& scale_info) const;
    cv::Rect2f transformBoundingBox(const cv::Rect2f& bbox, const cv::Rect2d& scale_info) const;

public:
    UnifiedYoloDetector(const std::string& model_path, const std::string& output_dir = "output", 
                       const std::string& provider = "cpu");
    
    ~UnifiedYoloDetector() = default;
    
    // Main detection methods
    std::vector<UnifiedDetection> detect(const std::string& image_path, float conf = -1.0f);
    std::vector<UnifiedDetection> detect(const cv::Mat& image, float conf = -1.0f);
    
    // Backward compatibility methods
    std::vector<Detection> detectLegacy(const std::string& image_path, float conf = -1.0f);
    
    // Specialized detection methods
    std::vector<KeypointDetection> detectKeypoints(const std::string& image_path, float conf = -1.0f);
    std::vector<BoundingBoxDetection> detectBoundingBoxes(const std::string& image_path, float conf = -1.0f);
    std::vector<SegmentationDetection> detectSegmentation(const std::string& image_path, float conf = -1.0f);
    
    // Visualization methods
    std::string drawAndSaveDetections(const std::string& image_path, 
                                     const std::vector<UnifiedDetection>& detections, 
                                     const std::string& output_filename = "detections.jpg");
    
    std::string drawAndSaveKeypoints(const std::string& image_path, 
                                    const std::vector<KeypointDetection>& keypoint_detections, 
                                    const std::string& output_filename = "keypoints.jpg");
    
    std::string drawAndSaveSegmentation(const std::string& image_path, 
                                       const std::vector<SegmentationDetection>& seg_detections, 
                                       const std::string& output_filename = "segmentation.jpg");
    
    // Export methods
    std::string saveResultsToCSV(const std::vector<UnifiedDetection>& results, 
                                 const std::string& output_filename = "output_unified.csv");
    
    std::string saveKeypointsToCSV(const std::vector<KeypointDetection>& keypoint_detections, 
                                  const std::string& output_filename = "keypoints.csv");
    
    std::string saveKeypointsToJSON(const std::vector<KeypointDetection>& keypoint_detections, 
                                   const std::string& output_filename = "keypoints.json");
    
    // Configuration and status
    bool isModelLoaded() const { return model_loaded_; }
    DetectionType getDetectionType() const { return detection_config_.detection_type; }
    ModelType getModelType() const { return detection_config_.model_type; }
    void setDetectionConfig(const DetectionConfig& config) { detection_config_ = config; }
    const DetectionConfig& getDetectionConfig() const { return detection_config_; }
    
    // Complete processing pipeline
    std::pair<std::vector<UnifiedDetection>, std::string> processImage(
        const std::string& image_name = "input/TestImage.jpeg",
        bool save_csv = true,
        float conf = -1.0f
    );
};

// ============================================================================
// Utility Functions for Keypoint Analysis
// ============================================================================

namespace KeypointAnalysis {
    /**
     * @brief Calculate angle between three keypoints
     */
    double calculateAngle(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3);
    
    /**
     * @brief Calculate distance between two keypoints
     */
    double calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);
    
    /**
     * @brief Check if a pose is in a specific position (e.g., arms raised)
     */
    bool isArmsRaised(const KeypointDetection& detection);
    
    /**
     * @brief Get pose confidence score based on visible keypoints
     */
    float getPoseConfidence(const KeypointDetection& detection);
    
    /**
     * @brief Extract pose features for analysis
     */
    std::vector<double> extractPoseFeatures(const KeypointDetection& detection);
}

#endif // UNIFIED_YOLO_DETECTOR_HPP 