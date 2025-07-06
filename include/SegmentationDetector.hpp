#ifndef SEGMENTATION_DETECTOR_HPP
#define SEGMENTATION_DETECTOR_HPP

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
#include <torch/torch.h>
#include <torch/script.h>

// ============================================================================
// Segmentation Detection Structures
// ============================================================================

/**
 * @brief Structure to hold a single segmentation mask
 */
struct SegmentationMask {
    cv::Mat mask;                    // Binary mask (CV_8UC1)
    cv::Mat colored_mask;            // Colored visualization mask (CV_8UC3)
    std::vector<cv::Point> contour;  // Object contour points
    double area;                     // Mask area in pixels
    cv::Point2f centroid;            // Mask centroid
    
    SegmentationMask() : area(0.0), centroid(0.0f, 0.0f) {}
    
    SegmentationMask(const cv::Mat& seg_mask, const std::vector<cv::Point>& obj_contour)
        : mask(seg_mask), contour(obj_contour) {
        calculateProperties();
    }
    
    void calculateProperties();
};

/**
 * @brief Structure to hold segmentation detection results for an object/instance
 */
struct SegmentationResult {
    cv::Rect2f bbox;                    // Bounding box around the detected object
    SegmentationMask segmentation;      // Segmentation mask and properties
    float confidence;                   // Detection confidence
    int class_id;                      // Object class ID
    std::string class_name;            // Object class name
    
    SegmentationResult(const cv::Rect2f& box, float conf, int cls_id = 0, const std::string& cls_name = "")
        : bbox(box), confidence(conf), class_id(cls_id), class_name(cls_name) {}
};

// ============================================================================
// Segmentation Detection Configuration
// ============================================================================

struct SegmentationConfig {
    // Detection thresholds
    float confidence_threshold;
    float nms_threshold;
    float mask_threshold;               // Threshold for mask binarization
    
    // Model-specific parameters
    bool has_bounding_box;              // Whether model outputs bounding boxes
    bool use_proto_masks;               // Whether model uses prototype masks (YOLOv8 style)
    int mask_width;                     // Mask output width
    int mask_height;                    // Mask output height
    int num_prototypes;                 // Number of prototype masks (for YOLOv8)
    
    // Output format configuration
    int bbox_output_size;               // Size of bbox output (4 for x,y,w,h)
    int mask_output_size;               // Size of mask coefficients per detection
    
    // Visualization options
    bool fill_masks;                    // Fill masks or just contours
    float mask_alpha;                   // Transparency for mask overlay
    bool show_contours;                 // Show contour lines
    int contour_thickness;              // Thickness of contour lines
    
    SegmentationConfig()
        : confidence_threshold(0.25f)
        , nms_threshold(0.45f)
        , mask_threshold(0.5f)
        , has_bounding_box(true)
        , use_proto_masks(true)
        , mask_width(160)
        , mask_height(160)
        , num_prototypes(32)
        , bbox_output_size(4)
        , mask_output_size(32)
        , fill_masks(true)
        , mask_alpha(0.5f)
        , show_contours(true)
        , contour_thickness(2) {}
};

// ============================================================================
// ONNX Model Wrapper for Segmentation Detection
// ============================================================================

class SegmentationOnnxWrapper {
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
    SegmentationOnnxWrapper(const std::string& model_path, const std::string& provider = "cpu");
    cv::Size getInputSize() const { return input_size_; }
    std::vector<Ort::Value> runInference(const std::vector<float>& input_data, 
                                        const std::vector<int64_t>& input_shape);
    const std::vector<std::string>& getOutputNames() const { return output_names_; }
};

// ============================================================================
// Segmentation Detector Class
// ============================================================================

class SegmentationDetector {
private:
    std::string model_path_;
    std::string output_dir_;
    std::unique_ptr<SegmentationOnnxWrapper> model_;
    bool model_loaded_;
    SegmentationConfig config_;
    
    // Class names for visualization
    std::vector<std::string> class_names_;
    std::vector<cv::Scalar> class_colors_;
    
    // Preprocessing parameters (same as SingleImageYolo for consistency)
    int original_width_;
    int original_height_;
    double exact_scale_ratio_;
    int processed_width_;
    int processed_height_;

    // Private methods
    void setupDirectories();
    void loadModel();
    void initializeClassNames();
    void generateClassColors();
    
public:
    // Preprocessing (reuse from SingleImageYolo)
    std::vector<float> preprocess(const cv::Mat& image, cv::Rect2d& scale_info, double& pimgwidth, double& pimgheight);
    
    // Postprocessing for segmentation detection
    void postprocess(
        const std::vector<std::vector<float>>& batch_outputs,
        const std::vector<std::vector<int64_t>>& batch_output_shapes,
        const int batch_size,
        std::vector<std::vector<SegmentationResult>>& all_results,
        torch::Tensor& mask_tensor, torch::Tensor& bboxes_tensor,
        std::vector<float>& detection_confidences, std::vector<int>& detection_class_ids, std::vector<std::string>& detection_class_names,
        const std::vector<cv::Mat>& original_images, int height, int width);
    
    // Simple postprocess for detect function
    std::vector<SegmentationResult> postprocess(
        const std::vector<std::vector<float>>& output_data,
        const std::vector<std::vector<int64_t>>& output_shapes,
        const cv::Rect2d& scale_info,
        int height, int width);

    // Mask processing methods
    cv::Mat processPrototypeMasks(const std::vector<float>& mask_coeffs,
                                 const std::vector<float>& prototypes,
                                 const std::vector<int64_t>& proto_shape,
                                 const cv::Rect2f& bbox);
    
    // EXACT ULTRALYTICS implementations
    std::vector<cv::Mat> processMaskUltralytics(
        const std::vector<float>& protos,
        const std::vector<int64_t>& proto_shape,
        const std::vector<std::vector<float>>& masks_in,
        const std::vector<cv::Rect2f>& bboxes,
        const cv::Size& img_shape,
        bool upsample = false);
    
    cv::Mat cropMaskUltralytics(const cv::Mat& mask, const cv::Rect2f& bbox);
    
    std::vector<cv::Rect2f> scaleBoxesUltralytics(
        const cv::Size& img1_shape,
        const std::vector<cv::Rect2f>& boxes,
        const cv::Size& img0_shape,
        const cv::Rect2d& scale_info);
    
    cv::Mat cropMask(const cv::Mat& mask, const cv::Rect2f& bbox);
    std::vector<cv::Point> extractContour(const cv::Mat& mask);
    
    // Helper function to calculate IoU between two bounding boxes
    float calculateIoU(const cv::Rect2f& box1, const cv::Rect2f& box2);
    
    // Helper function to scale contour coordinates from processed to original image space
    std::vector<cv::Point> scaleContourUltralytics(
        const std::vector<cv::Point>& contour,
        const cv::Size& img1_shape,     // processed image size
        const cv::Size& img0_shape,     // original image size
        const cv::Rect2d& scale_info);
    
    // NMS for segmentation results
    std::vector<SegmentationResult> applyNMS(const std::vector<SegmentationResult>& detections, 
                                            float nms_threshold);
    
    // EXACT ULTRALYTICS non_max_suppression implementation
    std::vector<std::vector<std::vector<float>>> non_max_suppression(
        const std::vector<float>& prediction_data,
        const std::vector<int64_t>& prediction_shape,
        float conf_thres = 0.25f,
        float iou_thres = 0.45f,
        const std::vector<int>& classes = {},
        bool agnostic = false,
        bool multi_label = false,
        int max_det = 300,
        int nc = 0,
        int max_nms = 30000,
        int max_wh = 7680,
        bool in_place = true,
        bool rotated = false,
        bool end2end = false,
        bool return_idxs = false
    );

    std::vector<std::vector<std::vector<float>>> fast_batch_nms(
        const std::vector<float>& preds,
        const std::vector<int64_t>& shape,  // [batch, features, boxes]
        float conf_thresh, float iou_thresh,
        int max_det, int max_wh,
        const std::vector<int>& classes,
        bool agnostic,
        int nc
    );

    
    // NMS on raw model output (before coordinate transformation) - DEPRECATED
    std::vector<std::vector<std::vector<double>>> non_max_suppression_old(
        const std::vector<double>& prediction_data,
        const std::vector<int64_t>& prediction_shape,
        float conf_thres = 0.25f,
        float iou_thres = 0.45f,
        const std::vector<int>& classes = {},
        bool agnostic = false,
        bool multi_label = false,
        int max_det = 300,
        int nc = 0,
        int max_nms = 30000,
        int max_wh = 7680,
        bool in_place = true
    );

public:
    SegmentationDetector(const std::string& model_path, const std::string& output_dir = "output", 
                        const std::string& provider = "cpu");
    
    ~SegmentationDetector() = default;
    
    // Main detection methods
    std::vector<SegmentationResult> detect(const std::string& image_path, float conf = -1.0f, int height = 0, int width = 0);
    std::vector<SegmentationResult> detect(const cv::Mat& image, float conf = -1.0f, int height = 0, int width = 0);
    
    // Visualization methods
    std::string drawAndSaveSegmentation(const std::string& image_path, 
                                       const std::vector<SegmentationResult>& detections, 
                                       const std::string& output_filename = "segmentation.jpg");
    
    cv::Mat drawSegmentationOverlay(const cv::Mat& image, 
                                   const std::vector<SegmentationResult>& detections);
    
    // Export methods
    std::string saveSegmentationToCSV(const std::vector<SegmentationResult>& results, 
                                     const std::string& output_filename = "detected_segmentation_cpp.csv");
    
    std::string saveSegmentationToJSON(const std::vector<SegmentationResult>& results, 
                                      const std::string& output_filename = "segmentation.json");
    
    std::string saveMasksAsPNG(const std::vector<SegmentationResult>& results,
                              const std::string& base_filename = "mask");
    
    // Configuration
    bool isModelLoaded() const { return model_loaded_; }
    void setConfig(const SegmentationConfig& config) { config_ = config; }
    const SegmentationConfig& getConfig() const { return config_; }
    void setClassNames(const std::vector<std::string>& names);
    
    // Model introspection helpers
    void analyzeModelOutput(const std::string& test_image_path);
    void printModelInfo() const;
    
    // Complete processing pipeline
    std::pair<std::vector<SegmentationResult>, std::string> processImage(
        const std::string& image_name = "input/TestImage.jpeg",
        bool save_csv = true,
        bool save_masks = true,
        float conf = -1.0f
    );
};

// ============================================================================
// Segmentation Analysis Utilities
// ============================================================================

namespace SegmentationAnalysis {
    /**
     * @brief Calculate IoU (Intersection over Union) between two masks
     */
    double calculateIoU(const cv::Mat& mask1, const cv::Mat& mask2);
    
    /**
     * @brief Calculate mask area in pixels
     */
    double calculateMaskArea(const cv::Mat& mask);
    
    /**
     * @brief Calculate mask centroid
     */
    cv::Point2f calculateCentroid(const cv::Mat& mask);
    
    /**
     * @brief Extract largest contour from mask
     */
    std::vector<cv::Point> extractLargestContour(const cv::Mat& mask);
    
    /**
     * @brief Calculate contour area
     */
    double calculateContourArea(const std::vector<cv::Point>& contour);
    
    /**
     * @brief Calculate contour perimeter
     */
    double calculateContourPerimeter(const std::vector<cv::Point>& contour);
    
    /**
     * @brief Simplify contour using Douglas-Peucker algorithm
     */
    std::vector<cv::Point> simplifyContour(const std::vector<cv::Point>& contour, double epsilon);
    
    /**
     * @brief Check if point is inside mask
     */
    bool isPointInMask(const cv::Point& point, const cv::Mat& mask);
    
    /**
     * @brief Create colored mask for visualization
     */
    cv::Mat createColoredMask(const cv::Mat& mask, const cv::Scalar& color);
    
    /**
     * @brief Merge multiple masks into single visualization
     */
    cv::Mat mergeMasks(const std::vector<cv::Mat>& masks, const std::vector<cv::Scalar>& colors);
}

#endif // SEGMENTATION_DETECTOR_HPP 