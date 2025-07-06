#ifndef BATCH_POINT_PROCESSOR_HPP
#define BATCH_POINT_PROCESSOR_HPP

// Using precompiled header - all common headers are included
// Only include project-specific headers that are not in PCH
#include "PointDetector.hpp"
#include "Config.hpp"

// Forward declarations
struct PreprocessInfo;

struct PreprocessInfo {
    double r;
    int top, left;
    int new_w, new_h;
    int final_w, final_h;
    
    // Reset method for reuse
    void reset() {
        r = 0.0;
        top = left = 0;
        new_w = new_h = 0;
        final_w = final_h = 0;
    }
};

class BatchLogger {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point last_checkpoint_;
    std::vector<std::pair<std::string, double>> timing_log_;

public:
    BatchLogger();

    void checkpoint(const std::string& operation);
    void logImageTiming(int image_idx, double time_ms, int detections = -1);
    void printSummary(int total_images);
};

void xywh2xyxy(std::vector<float>& box);

std::vector<std::vector<std::vector<float>>> non_max_suppression_cpp(
    const std::vector<float>& prediction,
    const std::vector<int64_t>& shape,  // [batch, features, boxes]
    float conf_thres,
    float iou_thres,
    const std::vector<int>& classes,
    bool agnostic,
    bool multi_label,
    int max_det,
    int nc,
    int max_nms,
    int max_wh,
    bool in_place
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

// Convert boxes from processed image coordinates to original image coordinates
torch::Tensor scale_boxes(const std::vector<int64_t>& img1_shape, torch::Tensor boxes, 
                         const std::vector<int64_t>& img0_shape, 
                         const std::pair<std::pair<float, float>, std::pair<float, float>>* ratio_pad = nullptr,
                         bool padding = true, bool xywh = false);

// Clip bounding boxes to image boundaries
torch::Tensor clip_boxes(torch::Tensor boxes, const std::vector<int64_t>& shape);

// Construct results from NMS output and scale boxes to original image coordinates (LibTorch version)
std::vector<PointDetectionResult> construct_results(
    const std::vector<std::vector<std::vector<float>>>& nms_output,
    const std::vector<int64_t>& processed_img_shape,  // [height, width] of processed image
    const std::vector<cv::Mat>& orig_images,
    const std::vector<std::string>& image_paths);

// Alternative OpenCV-only implementation for better performance (no LibTorch dependency)
std::vector<PointDetectionResult> construct_results_opencv(
    const std::vector<std::vector<std::vector<float>>>& nms_output,
    const cv::Size& processed_img_size,  // Size of processed image
    const std::vector<cv::Mat>& orig_images,
    const std::vector<std::string>& image_paths);

// Helper: Clip boxes to image boundaries
torch::Tensor clip_boxesTorch(torch::Tensor boxes, const std::vector<int64_t>& shape);

// Helper: Rescale boxes from input to original image shape
torch::Tensor scale_boxesTorch(
    const std::vector<int64_t>& img1_shape, // input shape, {h, w}
    torch::Tensor boxes,
    const std::vector<int64_t>& img0_shape); // original shape, {h, w}

// Point-specific processing functions using LibTorch
std::pair<torch::Tensor, torch::Tensor> processPointsUltralyticsTorch(
    const torch::Tensor& detection_output,  // [N, 6] - boxes + conf + class
    const std::vector<int64_t>& shape_before_upsample, // {ih, iw}
    const std::vector<int64_t>& shape_after_upsample, // {ih, iw}
    bool scale_coords = true);

class TrueBatchPointProcessor {
public:
    // Nested BoundingBox struct
    struct BoundingBox {
        int x{0};
        int y{0};
        int width{0};
        int height{0};

        BoundingBox() = default;
        BoundingBox(int _x, int _y, int w, int h);
        float area() const;
        BoundingBox intersect(const BoundingBox &other) const;
    };

private:
    PointDetector detector_;
    std::string model_path_;
    std::string output_dir_;
    Ort::Env env_;
    Ort::Session session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
    BatchLogger logger_;

    // Helper methods
    Ort::SessionOptions createSessionOptions(const std::string& provider);
    void configureProvider(Ort::SessionOptions& session_options, const std::string& provider);
    void extractModelInfo();

public:
    // Constructor
    TrueBatchPointProcessor(const std::string& model_path, 
                           const std::string& output_dir,
                           const std::string& provider);

    // Main processing methods
    std::vector<cv::Mat> loadAllImages(const std::vector<std::string>& image_paths);
    void processTrueBatch(const std::vector<std::string>& image_paths, int batch_size, float conf_threshold);
    
    // ULTRA-OPTIMIZED preprocess function with advanced memory management (same as segmentation)
    inline void preprocess(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& letterboxed_imgs, 
                            torch::Tensor& im, int batch_size, bool cuda_available, int& input_h, int& input_w);
        
    // Inference method (remove inline, implement in .cpp)
    inline void runBatchCUDA(const torch::Tensor& input_tensor, 
    std::vector<std::vector<float>>& output_data,
    std::vector<std::vector<int64_t>>& output_shapes, bool warmup);

    // Postprocessing (fix signature to match implementation)
    void postprocess(
        const std::vector<std::vector<float>>& batch_outputs,
        const std::vector<std::vector<int64_t>>& batch_output_shapes,
        const int batch_size,
        std::vector<std::vector<PointDetectionResult>>& all_results,
        torch::Tensor& point_tensor, torch::Tensor& bboxes_tensor,
        std::vector<float>& detection_confidences, std::vector<int>& detection_class_ids, std::vector<std::string>& detection_class_names,
        const std::vector<cv::Mat>& original_images, int height, int width);
        
    // Result saving methods (fix signatures to match implementation)
    void saveBatchResults(const std::vector<std::string>& image_paths,
                         const std::vector<std::vector<PointDetectionResult>>& batch_results);
    void saveBatchResults(const std::vector<std::string>& image_paths,
                         const torch::Tensor& point_tensor, 
                         const torch::Tensor& bboxes_tensor,
                         const std::vector<float>& detection_confidences,
                         const std::vector<int>& detection_class_ids,
                         const std::vector<std::string>& detection_class_names,
                         int height, int width);

    // CSV comparison and parsing methods
    bool compareCSVFiles(const std::string& cpp_csv, const std::string& python_csv, float tolerance);
    std::vector<std::string> parseCSVLine(const std::string& line);
};

// Collect test images function declaration
std::vector<std::string> collectTestImages(int batch_size);

// 🚀 GPU-accelerated functions for Python-level performance
torch::Tensor gpu_accelerated_nms(torch::Tensor prediction, float conf_thres, float iou_thres, int max_det);
torch::Tensor gpu_nms_impl(torch::Tensor boxes, torch::Tensor scores, float iou_threshold);

#endif // BATCH_POINT_PROCESSOR_HPP 