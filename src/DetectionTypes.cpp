#include "pch.hpp"
#include "DetectionTypes.hpp"
#include <variant>

// ============================================================================
// DetectionUtils Implementation
// ============================================================================

namespace DetectionUtils {

ModelType getModelTypeFromFilename(const std::string& model_path) {
    std::string filename = model_path;
    std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
    
    if (filename.find("pose") != std::string::npos) {
        return ModelType::YOLO_POSE;
    } else if (filename.find("seg") != std::string::npos || filename.find("segment") != std::string::npos) {
        return ModelType::YOLO_SEGMENT;
    } else {
        return ModelType::YOLO_DETECT;
    }
}

DetectionType getDetectionTypeForModel(ModelType model_type) {
    switch (model_type) {
        case ModelType::YOLO_POSE:
            return DetectionType::KEYPOINTS;
        case ModelType::YOLO_SEGMENT:
            return DetectionType::SEGMENTATION;
        case ModelType::YOLO_DETECT:
        default:
            return DetectionType::BOUNDING_BOX;
    }
}

std::string getKeypointName(COCOKeypoint keypoint) {
    static const std::vector<std::string> keypoint_names = {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    };
    
    int idx = static_cast<int>(keypoint);
    if (idx >= 0 && idx < static_cast<int>(keypoint_names.size())) {
        return keypoint_names[idx];
    }
    return "unknown";
}

// Forward declaration of Detection struct for backward compatibility
struct Detection {
    cv::Rect2f bbox;
    float confidence;
    int class_id;
    float class_score;
    
    Detection(const cv::Rect2f& box, float conf, int cls_id, float cls_score)
        : bbox(box), confidence(conf), class_id(cls_id), class_score(cls_score) {}
};

std::vector<Detection> toLegacyDetections(const std::vector<UnifiedDetection>& unified_detections) {
    std::vector<Detection> legacy_detections;
    legacy_detections.reserve(unified_detections.size());
    
    for (const auto& unified : unified_detections) {
        cv::Rect2f bbox = unified.getBoundingBox();
        float confidence = unified.getConfidence();
        int class_id = unified.getClassId();
        
        // For legacy compatibility, use confidence as class_score
        legacy_detections.emplace_back(bbox, confidence, class_id, confidence);
    }
    
    return legacy_detections;
}

} // namespace DetectionUtils 