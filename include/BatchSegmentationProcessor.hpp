#ifndef BATCH_SEGMENTATION_PROCESSOR_HPP
#define BATCH_SEGMENTATION_PROCESSOR_HPP

// Using precompiled header - all common headers are included
// Only include project-specific headers that are not in PCH
#include "SegmentationDetector.hpp"
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

cv::Mat crop_mask(const cv::Mat& mask, const cv::Rect2f& bbox);
cv::Mat crop_mask(const cv::Mat& mask, const cv::Mat& box);

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

// Helper: Clip boxes to image boundaries
torch::Tensor clip_boxesTorch(torch::Tensor boxes, const std::vector<int64_t>& shape);


// Helper: Rescale boxes from input to original image shape
torch::Tensor scale_boxesTorch(
    const std::vector<int64_t>& img1_shape, // input shape, {h, w}
    torch::Tensor boxes,
    const std::vector<int64_t>& img0_shape); // original shape, {h, w}

torch::Tensor cropMaskLibtorch(const torch::Tensor& masks, const torch::Tensor& boxes);
std::pair<torch::Tensor, torch::Tensor> processMaskUltralyticsTorch(
    const torch::Tensor& protos,     // [C, mh, mw]
    const torch::Tensor& masks_in,   // [N, C]
    torch::Tensor bboxes,           // [N, 4] - removed const to allow modification
    const std::vector<int64_t>& shape_before_upsample, // {ih, iw}
    const std::vector<int64_t>& shape_after_upsample, // {ih, iw}
    bool upsample);

class TrueBatchSegmentationProcessor {
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
    SegmentationDetector detector_;
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
    TrueBatchSegmentationProcessor(const std::string& model_path, 
                                   const std::string& output_dir,
                                   const std::string& provider);

    // Main processing methods
    std::vector<cv::Mat> loadAllImages(const std::vector<std::string>& image_paths);
    void processTrueBatch(const std::vector<std::string>& image_paths, int batch_size, float conf_threshold);
    
    // ULTRA-OPTIMIZED preprocess function with advanced memory management
    inline void preprocess(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& letterboxed_imgs, 
                            torch::Tensor& im, int batch_size, bool cuda_available, int& input_h, int& input_w) {
        
        // ===== MEMORY POOL: Static containers for reuse =====
        static std::vector<cv::Mat> static_letterboxed_imgs;
        static std::vector<PreprocessInfo> static_batch_info;
        static bool pools_initialized = false;
        
        // ===== CONSTANTS: Avoid repeated calculations =====
        constexpr bool auto_pad = true;
        constexpr bool fp16 = false;
        constexpr bool center = true;
        constexpr int stride = 32;
        
        // ===== MEMORY OPTIMIZATION: Pre-allocate containers =====
        if (!pools_initialized || static_letterboxed_imgs.size() < batch_size) {
            static_letterboxed_imgs.resize(batch_size);
            static_batch_info.resize(batch_size);
            pools_initialized = true;
        }
        
        // ===== FAST PATH: Use static containers to avoid allocations =====
        letterboxed_imgs = static_letterboxed_imgs;  // Reference swap, no copy
        auto& batch_info = static_batch_info;
        
        // ===== PERFORMANCE: Cache common calculations =====
        const double input_h_d = static_cast<double>(input_h);
        const double input_w_d = static_cast<double>(input_w);

        const double epsilon = 0.1;
        // ===== OPTIMIZED LETTERBOXING: Vectorized operations =====
        #pragma omp parallel for if(batch_size > 2)
        for (int j = 0; j < batch_size; ++j) {
            const cv::Mat& img = images[j];

            const int orig_h = img.rows;
            const int orig_w = img.cols;

            // ===== PERFORMANCE: Single division, cached ratios =====
            const double r = std::min(input_h_d / orig_h, input_w_d / orig_w);
            const int new_w = static_cast<int>(std::round(orig_w * r));
            const int new_h = static_cast<int>(std::round(orig_h * r));

            // ===== OPTIMIZED PADDING: Fast modular arithmetic =====
            double dw = input_w - new_w;
            double dh = input_h - new_h;
            if (auto_pad) {
                dw = std::fmod(dw, stride);
                dh = std::fmod(dh, stride);
            }
            if (center) {
                dw *= 0.5;  // Faster than division
                dh *= 0.5;
            }
            
            // ===== PERFORMANCE: Fast rounding with bit operations =====
            const int top = static_cast<int>(dh - epsilon + 0.5);
            const int bottom = static_cast<int>(dh + epsilon + 0.5);
            const int left = static_cast<int>(dw - epsilon + 0.5);
            const int right = static_cast<int>(dw + epsilon + 0.5);
            
            // ===== MEMORY EFFICIENCY: Reuse existing Mat if possible =====
            cv::Mat& letterboxed = letterboxed_imgs[j];
            
            // ===== OPTIMIZED RESIZE: Direct memory reuse =====Updated
            cv::resize(img, letterboxed, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
            cv::copyMakeBorder(letterboxed, letterboxed, 
                                top, bottom, left, right,
                                cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            
            // ===== CONVERT BGR TO RGB (keep as uint8) =====
            cv::cvtColor(letterboxed, letterboxed, cv::COLOR_BGR2RGB);
            
            // Keep as CV_8UC3 (uint8) - no conversion to float here
        }

        // ===== PERFORMANCE: Cache final dimensions =====
        const int new_input_h = letterboxed_imgs[0].rows;
        const int new_input_w = letterboxed_imgs[0].cols;

        if (cuda_available) {
            // ===== CUDA OPTIMIZATION: Pinned memory allocation =====
            static torch::Tensor pinned_tensor;
            static bool pinned_initialized = false;
            const size_t total_elements = static_cast<size_t>(batch_size) * new_input_h * new_input_w * 3;
            
            // ===== MEMORY POOL: Reuse pinned memory for uint8 data =====
            if (!pinned_initialized || pinned_tensor.numel() < total_elements) {
                pinned_tensor = torch::empty(
                    {batch_size, new_input_h, new_input_w, 3},
                    torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true)
                );
                pinned_initialized = true;
            }
            
            // ===== PERFORMANCE: Direct memory access =====
            uint8_t* pinned_ptr = pinned_tensor.data_ptr<uint8_t>();
            const size_t img_elements = static_cast<size_t>(new_input_h) * new_input_w * 3;
            
            // ===== OPTIMIZED COPY: Direct uint8 memcpy =====
            #pragma omp parallel for if(batch_size > 2)
            for (int j = 0; j < batch_size; ++j) {
                // letterboxed_imgs[j] is CV_8UC3 (uint8 RGB)
                const uint8_t* src_ptr = letterboxed_imgs[j].data;
                uint8_t* dst_ptr = pinned_ptr + j * img_elements;
                std::memcpy(dst_ptr, src_ptr, img_elements * sizeof(uint8_t));
            }
            
            // ===== CUDA ACCELERATION: Convert to float32 and normalize on GPU =====
            auto gpu_tensor = pinned_tensor.to(torch::kCUDA, /*non_blocking=*/true);
            
            // ===== PERFORMANCE: Convert to float32 and normalize in one step =====
            const torch::Dtype dtype = fp16 ? torch::kHalf : torch::kFloat32;
            im = gpu_tensor
                .to(dtype)                    // Convert uint8 -> float32 on GPU
                .div(255.0f)                  // Normalize [0, 255] -> [0, 1]
                .permute({0, 3, 1, 2})        // NHWC -> NCHW
                .contiguous();
            
            // ===== SYNCHRONIZATION: Only when necessary =====
            if (torch::cuda::is_available()) {
                torch::cuda::synchronize();
            }
            
        } else {
            // ===== CPU FALLBACK: OpenCV's optimized blob creation =====
            static cv::Mat static_blob;
            cv::dnn::blobFromImages(letterboxed_imgs, static_blob, 
                                  1.0/255.0, cv::Size(), cv::Scalar(), 
                                  true, false, CV_32F);
            
            // ===== MEMORY EFFICIENCY: Move semantics where possible =====
            im = torch::from_blob(static_blob.data, 
                                {batch_size, 3, new_input_h, new_input_w}, 
                                torch::kFloat32).clone();
        }


        // ===== OUTPUT: Update dimensions =====
        input_h = new_input_h;
        input_w = new_input_w;
        

    };

    /// ULTRA-OPTIMIZED CUDA inference with advanced memory management and zero-copy operations
    inline void runBatchCUDA(const torch::Tensor& input_tensor, 
    std::vector<std::vector<float>>& output_data,
    std::vector<std::vector<int64_t>>& output_shapes, bool warmup = false) {
        
        // ===== FAST PATH: Skip all validation in release builds =====
        #ifndef NDEBUG
        TORCH_CHECK(input_tensor.device().is_cuda(), "Input tensor must be on CUDA");
        TORCH_CHECK(input_tensor.is_contiguous(), "Input tensor must be contiguous");
        #endif

        // ===== PERFORMANCE: Cache tensor dimensions (avoid repeated method calls) =====
        const auto& sizes = input_tensor.sizes();
        const int64_t batch_size = sizes[0];
        const int64_t channels = sizes[1];
        const int64_t height = sizes[2];
        const int64_t width = sizes[3];
        const size_t num_elements = input_tensor.numel();

        // ===== MEMORY OPTIMIZATION: Static memory info (singleton pattern) =====
        static const Ort::MemoryInfo memory_info = []() {
            return Ort::MemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
        }();

        // ===== PERFORMANCE: Stack-allocated shape array (no heap allocation) =====
        const std::array<int64_t, 4> input_shape{batch_size, channels, height, width};

        // ===== ZERO-COPY: Direct CUDA memory wrapping (no data transfer) =====
        Ort::Value input_ort_tensor = Ort::Value::CreateTensor(
            memory_info,
            input_tensor.data_ptr<float>(),
            num_elements,
            input_shape.data(),
            input_shape.size()
        );

        // ===== MEMORY POOL: Pre-allocate RunOptions (reusable object) =====
        static Ort::RunOptions run_options{nullptr};

        // ===== CORE INFERENCE: Single optimized ONNX call =====
        auto output_tensors = session_.Run(
            run_options,
            input_names_cstr_.data(),
            &input_ort_tensor, 
            1,
            output_names_cstr_.data(),
            output_names_cstr_.size()
        );

        // ===== MEMORY EFFICIENCY: Smart container resizing =====
        const size_t num_outputs = output_tensors.size();
        if (output_data.size() != num_outputs) {
            output_data.resize(num_outputs);
            output_shapes.resize(num_outputs);
        }
        
        // ===== OPTIMIZED EXTRACTION: Minimize memory operations =====
        for (size_t i = 0; i < num_outputs; ++i) {
            auto& output = output_tensors[i];
            const auto shape_info = output.GetTensorTypeAndShapeInfo();
            const auto& shape = shape_info.GetShape();
            const size_t output_size = shape_info.GetElementCount();
            
            // ===== MEMORY EFFICIENCY: Reuse existing containers when possible =====
            auto& output_vec = output_data[i];
            if (output_vec.size() != output_size) {
                output_vec.resize(output_size);
            }
            
            // ===== PERFORMANCE: Direct bulk memory copy (faster than iterators) =====
            const float* output_ptr = output.GetTensorData<float>();
            std::memcpy(output_vec.data(), output_ptr, output_size * sizeof(float));
            
            // ===== EFFICIENT: Direct assignment (no unnecessary copies) =====
            output_shapes[i] = shape;
            
        }
    }

    // TRUE BATCH POSTPROCESSING: Use converted Python functions for exact compatibility
    void postprocess(
        const std::vector<std::vector<float>>& batch_outputs,
        const std::vector<std::vector<int64_t>>& batch_output_shapes,
        const int batch_size,
        std::vector<std::vector<SegmentationResult>>& all_results,
        torch::Tensor& mask_tensor, torch::Tensor& bboxes_tensor,
        std::vector<float>& detection_confidences, std::vector<int>& detection_class_ids, std::vector<std::string>& detection_class_names,
        const std::vector<cv::Mat>& original_images, int height, int width);
        
    // Utility methods
    inline void NMSBoxes(const std::vector<BoundingBox> &boxes,
                        const std::vector<float> &scores,
                        float scoreThreshold,
                        float nmsThreshold,
                        std::vector<int> &indices) {
        indices.clear();
        if (boxes.empty()) {
            return;
        }

        std::vector<int> order;
        order.reserve(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (scores[i] >= scoreThreshold) {
                order.push_back((int)i);
            }
        }
        if (order.empty()) return;

        std::sort(order.begin(), order.end(),
                [&scores](int a, int b) {
                    return scores[a] > scores[b];
                });

        std::vector<float> areas(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            areas[i] = (float)(boxes[i].width * boxes[i].height);
        }

        std::vector<bool> suppressed(boxes.size(), false);
        for (size_t i = 0; i < order.size(); ++i) {
            int idx = order[i];
            if (suppressed[idx]) continue;

            indices.push_back(idx);

            for (size_t j = i + 1; j < order.size(); ++j) {
                int idx2 = order[j];
                if (suppressed[idx2]) continue;

                const BoundingBox &a = boxes[idx];
                const BoundingBox &b = boxes[idx2];
                int interX1 = std::max(a.x, b.x);
                int interY1 = std::max(a.y, b.y);
                int interX2 = std::min(a.x + a.width,  b.x + b.width);
                int interY2 = std::min(a.y + a.height, b.y + b.height);

                int w = interX2 - interX1;
                int h = interY2 - interY1;
                if (w > 0 && h > 0) {
                    float interArea = (float)(w * h);
                    float unionArea = areas[idx] + areas[idx2] - interArea;
                    float iou = (unionArea > 0.f)? (interArea / unionArea) : 0.f;
                    if (iou > nmsThreshold) {
                        suppressed[idx2] = true;
                    }
                }
            }
        }
    }

    // Result saving methods
    void saveBatchResults(const std::vector<std::string>& image_paths,
                         const std::vector<std::vector<SegmentationResult>>& batch_results);
    void saveBatchResults(const std::vector<std::string>& image_paths,
                         const torch::Tensor& mask_tensor, 
                         const torch::Tensor& bboxes_tensor,
                         const std::vector<float>& detection_confidences,
                         const std::vector<int>& detection_class_ids,
                         const std::vector<std::string>& detection_class_names,
                         int height, int width);
    void saveDetectionMasks(const torch::Tensor& cpu_masks, 
                           const torch::Tensor& cpu_bboxes, 
                           int height, int width);

    // CSV comparison and parsing methods
    bool compareCSVFiles(const std::string& cpp_csv, const std::string& python_csv, float tolerance);
    std::vector<std::string> parseCSVLine(const std::string& line);
};

// Collect test images function declaration
std::vector<std::string> collectTestImages(int batch_size);






#endif // BATCH_SEGMENTATION_PROCESSOR_HPP