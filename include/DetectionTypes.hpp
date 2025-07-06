#ifndef DETECTION_TYPES_HPP
#define DETECTION_TYPES_HPP

#include "pch.hpp"
#include <variant>

// ============================================================================
// Detection Type Enums
// ============================================================================

enum class DetectionType {
    BOUNDING_BOX,
    KEYPOINTS,
    SEGMENTATION
};

enum class ModelType {
    YOLO_DETECT,      // Standard object detection
    YOLO_POSE,        // Pose estimation with keypoints
    YOLO_SEGMENT      // Instance segmentation
};

// ============================================================================
// Base Detection Result Structures
// ============================================================================

/**
 * @brief Base class for all detection results
 */
struct BaseDetection {
    float confidence;
    int class_id;
    float class_score;
    DetectionType type;
    
    BaseDetection(float conf, int cls_id, float cls_score, DetectionType det_type)
        : confidence(conf), class_id(cls_id), class_score(cls_score), type(det_type) {}
    
    virtual ~BaseDetection() = default;
};

/**
 * @brief Bounding box detection result
 */
struct BoundingBoxDetection : public BaseDetection {
    cv::Rect2f bbox;
    
    BoundingBoxDetection(const cv::Rect2f& box, float conf, int cls_id, float cls_score)
        : BaseDetection(conf, cls_id, cls_score, DetectionType::BOUNDING_BOX), bbox(box) {}
};

/**
 * @brief Keypoint detection result
 */
struct KeypointDetection : public BaseDetection {
    cv::Rect2f bbox;                    // Bounding box around the object
    std::vector<cv::Point2f> keypoints; // Keypoint coordinates
    std::vector<float> keypoint_scores; // Confidence for each keypoint
    std::vector<bool> keypoint_visible; // Visibility flag for each keypoint
    
    KeypointDetection(const cv::Rect2f& box, float conf, int cls_id, float cls_score,
                     const std::vector<cv::Point2f>& kpts, 
                     const std::vector<float>& kpt_scores,
                     const std::vector<bool>& kpt_visible)
        : BaseDetection(conf, cls_id, cls_score, DetectionType::KEYPOINTS), 
          bbox(box), keypoints(kpts), keypoint_scores(kpt_scores), keypoint_visible(kpt_visible) {}
};

/**
 * @brief Segmentation detection result
 */
struct SegmentationDetection : public BaseDetection {
    cv::Rect2f bbox;           // Bounding box
    cv::Mat mask;              // Segmentation mask
    std::vector<cv::Point> contour; // Object contour
    
    SegmentationDetection(const cv::Rect2f& box, float conf, int cls_id, float cls_score,
                         const cv::Mat& seg_mask, const std::vector<cv::Point>& obj_contour)
        : BaseDetection(conf, cls_id, cls_score, DetectionType::SEGMENTATION), 
          bbox(box), mask(seg_mask), contour(obj_contour) {}
};

// ============================================================================
// Unified Detection Result
// ============================================================================

/**
 * @brief Unified detection result that can hold any detection type
 */
class UnifiedDetection {
private:
    std::variant<BoundingBoxDetection, KeypointDetection, SegmentationDetection> detection_;

public:
    template<typename T>
    UnifiedDetection(T&& detection) : detection_(std::forward<T>(detection)) {}
    
    DetectionType getType() const {
        return std::visit([](const auto& det) { return det.type; }, detection_);
    }
    
    float getConfidence() const {
        return std::visit([](const auto& det) { return det.confidence; }, detection_);
    }
    
    int getClassId() const {
        return std::visit([](const auto& det) { return det.class_id; }, detection_);
    }
    
    cv::Rect2f getBoundingBox() const {
        return std::visit([](const auto& det) { return det.bbox; }, detection_);
    }
    
    template<typename T>
    const T& get() const {
        return std::get<T>(detection_);
    }
    
    template<typename T>
    bool is() const {
        return std::holds_alternative<T>(detection_);
    }
};

// ============================================================================
// Detection Configuration
// ============================================================================

struct DetectionConfig {
    ModelType model_type;
    DetectionType detection_type;
    
    // Thresholds
    float confidence_threshold;
    float nms_threshold;
    float keypoint_threshold;  // For keypoint visibility
    
    // Model-specific parameters
    int num_keypoints;         // For pose models
    bool include_bbox;         // Whether to include bounding box for keypoints/segmentation
    
    DetectionConfig() 
        : model_type(ModelType::YOLO_DETECT)
        , detection_type(DetectionType::BOUNDING_BOX)
        , confidence_threshold(0.25f)
        , nms_threshold(0.45f)
        , keypoint_threshold(0.5f)
        , num_keypoints(17)  // COCO pose format
        , include_bbox(true) {}
};

// ============================================================================
// Human Pose Keypoint Definitions (COCO format)
// ============================================================================

enum class COCOKeypoint {
    NOSE = 0,
    LEFT_EYE = 1,
    RIGHT_EYE = 2,
    LEFT_EAR = 3,
    RIGHT_EAR = 4,
    LEFT_SHOULDER = 5,
    RIGHT_SHOULDER = 6,
    LEFT_ELBOW = 7,
    RIGHT_ELBOW = 8,
    LEFT_WRIST = 9,
    RIGHT_WRIST = 10,
    LEFT_HIP = 11,
    RIGHT_HIP = 12,
    LEFT_KNEE = 13,
    RIGHT_KNEE = 14,
    LEFT_ANKLE = 15,
    RIGHT_ANKLE = 16
};

// Keypoint connections for skeleton drawing
const std::vector<std::pair<int, int>> COCO_SKELETON = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},           // Head
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, // Arms
    {5, 11}, {6, 12}, {11, 12},               // Torso
    {11, 13}, {13, 15}, {12, 14}, {14, 16}    // Legs
};

// ============================================================================
// Utility Functions
// ============================================================================

namespace DetectionUtils {
    /**
     * @brief Get model type from model filename
     */
    ModelType getModelTypeFromFilename(const std::string& model_path);
    
    /**
     * @brief Get expected detection type for a model type
     */
    DetectionType getDetectionTypeForModel(ModelType model_type);
    
    /**
     * @brief Get keypoint name from COCO keypoint enum
     */
    std::string getKeypointName(COCOKeypoint keypoint);
    
    /**
     * @brief Convert unified detections to legacy Detection format for backward compatibility
     */
    std::vector<struct Detection> toLegacyDetections(const std::vector<UnifiedDetection>& unified_detections);
}

#endif // DETECTION_TYPES_HPP 