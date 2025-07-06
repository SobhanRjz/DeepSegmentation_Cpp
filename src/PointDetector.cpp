#include "pch.hpp"
#include "PointDetector.hpp"
#include "Config.hpp"
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <iomanip>

// ============================================================================
// ONNX Model Wrapper Implementation for Point Detection
// ============================================================================

// Utility: Convert [center_x, center_y, w, h] -> [x1, y1, x2, y2]
inline void xywh2xyxy(std::vector<float>& box) {
    float x = box[0], y = box[1], w = box[2], h = box[3];
    box[0] = x - w/2.0f;
    box[1] = y - h/2.0f;
    box[2] = x + w/2.0f;
    box[3] = y + h/2.0f;
}

PointDetectionOnnxWrapper::PointDetectionOnnxWrapper(const std::string& model_path, const std::string& provider) {
    initializeSession(model_path, provider);
}

void PointDetectionOnnxWrapper::initializeSession(const std::string& model_path, const std::string& provider) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PointDetector");
    
    Ort::SessionOptions session_options;
    configureExecutionProvider(session_options, provider);
    
    #ifdef _WIN32
        std::wstring model_path_w(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, model_path_w.c_str(), session_options);
    #else
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    #endif
    
    extractModelInfo();
}

void PointDetectionOnnxWrapper::configureExecutionProvider(Ort::SessionOptions& session_options, const std::string& provider) {
    std::vector<std::string> available_providers = Ort::GetAvailableProviders();
    
    std::cout << "🔧 Available execution providers: ";
    for (const auto& p : available_providers) {
        std::cout << p << " ";
    }
    std::cout << "\n";
    
    // Enable optimizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    
    // Set optimal thread counts
    int num_threads = std::thread::hardware_concurrency();
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetInterOpNumThreads(num_threads / 2);
    
    // Enable memory optimizations
    session_options.EnableMemPattern();
    session_options.EnableCpuMemArena();
    
    // Disable unnecessary features for performance
    session_options.DisableProfiling();
    session_options.SetLogSeverityLevel(3); // Only errors
    
    if (provider == "cuda") {
        auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
        
        if (cuda_available != available_providers.end()) {
            try {
                // Use the older, more reliable static API for CUDA provider
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                cuda_options.arena_extend_strategy = 1;
                cuda_options.gpu_mem_limit = 0;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.do_copy_in_default_stream = 1;
                
                // Use the statically linked CUDA provider
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "✅ CUDA execution provider enabled (static provider)" << "\n";
                return;
            } catch (const std::exception& e) {
                // Try simple CUDA provider without options
                try {
                    session_options.AppendExecutionProvider("CUDA");
                    std::cout << "✅ CUDA execution provider enabled (simple provider)" << "\n";
                    return;
                } catch (const std::exception& e2) {
                std::cout << "⚠️  CUDA initialization failed: " << e.what() << "\n";
                    std::cout << "⚠️  Simple CUDA provider failed: " << e2.what() << "\n";
                }
            }
        } else {
            std::cout << "❌ CUDA provider not available" << "\n";
        }
    }
    
    std::cout << "🖥️  Using optimized CPU execution provider" << "\n";
}

void PointDetectionOnnxWrapper::extractModelInfo() {
    // Extract input names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t input_count = session_->GetInputCount();
    
    for (size_t i = 0; i < input_count; i++) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_names_.push_back(input_name.get());
    }
    
    // Extract output names
    size_t output_count = session_->GetOutputCount();
    for (size_t i = 0; i < output_count; i++) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.push_back(output_name.get());
    }
    
    // Create C-style string arrays
    for (const auto& name : input_names_) {
        input_names_cstr_.push_back(name.c_str());
    }
    for (const auto& name : output_names_) {
        output_names_cstr_.push_back(name.c_str());
    }
    
    // Set input size from config
    const auto& config = Config::get();
    input_size_ = cv::Size(config.input_width, config.input_height);
    
    std::cout << "✅ Point Detection Model initialized:" << "\n";
    std::cout << "   Input: " << input_names_[0] << " (" << input_size_.width << "x" << input_size_.height << ")" << "\n";
    std::cout << "   Outputs: ";
    for (const auto& name : output_names_) {
        std::cout << name << " ";
    }
    std::cout << "\n";
}

std::vector<Ort::Value> PointDetectionOnnxWrapper::runInference(const std::vector<float>& input_data, 
                                                               const std::vector<int64_t>& input_shape) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input_data.data()), input_data.size(),
        input_shape.data(), input_shape.size());
    
    return session_->Run(Ort::RunOptions{nullptr}, 
                       input_names_cstr_.data(), &input_tensor, 1,
                       output_names_cstr_.data(), output_names_cstr_.size());
}

// ============================================================================
// PointDetector Implementation
// ============================================================================

PointDetector::PointDetector(const std::string& model_path, const std::string& output_dir, const std::string& provider)
    : model_path_(model_path), output_dir_(output_dir), model_loaded_(false) {
    setupDirectories();
    loadModel();
}

void PointDetector::setupDirectories() {
    std::filesystem::create_directories(output_dir_);
}

void PointDetector::loadModel() {
    try {
        // Get execution provider from config if not specified
        const auto& config = Config::get();
        std::string provider = config.execution_provider;
        
        model_ = std::make_unique<PointDetectionOnnxWrapper>(model_path_, provider);
        model_loaded_ = true;
        std::cout << "✅ Point Detection Model loaded successfully from: " << model_path_ << "\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to load point detection model: " << e.what() << "\n";
        model_loaded_ = false;
    }
}

// EXACT ULTRALYTICS scale_boxes implementation
std::vector<cv::Rect2f> PointDetector::scaleBoxesUltralytics(
    const cv::Size& img1_shape,
    const std::vector<cv::Rect2f>& boxes,
    const cv::Size& img0_shape,
    const cv::Rect2d& scale_info) {
    
    std::vector<cv::Rect2f> scaled_boxes;
    
    // EXACT ULTRALYTICS: scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False)
    // if ratio_pad is None:  # calculate from img0_shape
    //     gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    //     pad = (
    //         round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
    //         round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
    //     )  # wh padding
    
    float gain = std::min(
        static_cast<float>(img1_shape.height) / static_cast<float>(img0_shape.height),
        static_cast<float>(img1_shape.width) / static_cast<float>(img0_shape.width)
    );  // gain = old / new
    
    float pad_w = std::round((static_cast<float>(img1_shape.width) - static_cast<float>(img0_shape.width) * gain) / 2.0f - 0.1f);
    float pad_h = std::round((static_cast<float>(img1_shape.height) - static_cast<float>(img0_shape.height) * gain) / 2.0f - 0.1f);
    
    // std::cout << "🔧 ULTRALYTICS scale_boxes: gain=" << gain << ", pad=(" << pad_w << "," << pad_h << ")" << "\n";
    // std::cout << "   img1_shape: " << img1_shape.width << "x" << img1_shape.height << "\n";
    // std::cout << "   img0_shape: " << img0_shape.width << "x" << img0_shape.height << "\n";
    
    for (const auto& box : boxes) {
        // Convert cv::Rect2f to xyxy format for processing
        float x1 = box.x;
        float y1 = box.y;
        float x2 = box.x + box.width;
        float y2 = box.y + box.height;
        
        // EXACT ULTRALYTICS: if padding: (padding=True by default)
        //     boxes[..., 0] -= pad[0]  # x padding
        //     boxes[..., 1] -= pad[1]  # y padding
        //     if not xywh:  # xywh=False by default
        //         boxes[..., 2] -= pad[0]  # x padding
        //         boxes[..., 3] -= pad[1]  # y padding
        x1 -= pad_w;  // x1 -= pad[0]
        y1 -= pad_h;  // y1 -= pad[1]
        x2 -= pad_w;  // x2 -= pad[0] (since xywh=False)
        y2 -= pad_h;  // y2 -= pad[1] (since xywh=False)
        
        // EXACT ULTRALYTICS: boxes[..., :4] /= gain
        x1 /= gain;
        y1 /= gain;
        x2 /= gain;
        y2 /= gain;
        
        // EXACT ULTRALYTICS: return clip_boxes(boxes, img0_shape)
        // clip_boxes implementation:
        //     boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        //     boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        //     boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        //     boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img0_shape.width)));   // clamp(0, shape[1])
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img0_shape.height)));  // clamp(0, shape[0])
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img0_shape.width)));   // clamp(0, shape[1])
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img0_shape.height)));  // clamp(0, shape[0])
        
        // Convert back to cv::Rect2f format
        cv::Rect2f scaled_box;
        scaled_box.x = x1;
        scaled_box.y = y1;
        scaled_box.width = x2 - x1;
        scaled_box.height = y2 - y1;
        
        scaled_boxes.push_back(scaled_box);
    }
    
    return scaled_boxes;
}

// Reuse the exact preprocessing from SingleImageYolo.cpp
std::vector<float> PointDetector::preprocess(const cv::Mat& image, cv::Rect2d& scale_info) {
    // Get configuration values
    const auto& config = Config::get();
    
    // Store original dimensions
    original_width_ = image.cols;
    original_height_ = image.rows;
    
    cv::Mat img = image.clone();
    
    // EXACT Ultralytics LetterBox implementation
    // Current shape [height, width]
    std::vector<int> shape = {img.rows, img.cols};
    std::vector<int> new_shape = {config.input_height, config.input_width};
    
    // LetterBox parameters (matching Python defaults)
    // CRITICAL FIX: Changed auto_pad from false to true to match Ultralytics default behavior
    // This resolves the ~2.34 pixel Y-coordinate difference observed in comparisons
    bool center = true;  // This should match your Python LetterBox(center=?) parameter
    bool auto_pad = true;  // auto parameter in Python - CHANGED FROM FALSE TO TRUE
    bool scale_fill = false;  // scale_fill parameter in Python
    bool scaleup = true;  // scaleup parameter in Python
    int stride = 32;  // stride parameter in Python
    
    // Scale ratio (new / old) - EXACT Ultralytics calculation
    double r = std::min(static_cast<double>(new_shape[0]) / static_cast<double>(shape[0]), 
                       static_cast<double>(new_shape[1]) / static_cast<double>(shape[1]));
    
    if (!scaleup) {  // only scale down, do not scale up (for better val mAP)
        r = std::min(r, 1.0);
    }
    
    // Compute padding - EXACT Ultralytics method
    std::vector<double> ratio = {r, r};  // width, height ratios
    std::vector<int> new_unpad = {
        static_cast<int>(std::round(static_cast<double>(shape[1]) * r)), 
        static_cast<int>(std::round(static_cast<double>(shape[0]) * r))
    };
    
    double dw = static_cast<double>(new_shape[1]) - static_cast<double>(new_unpad[0]);  // wh padding
    double dh = static_cast<double>(new_shape[0]) - static_cast<double>(new_unpad[1]);  // wh padding
    
    if (auto_pad) {  // minimum rectangle
        dw = fmod(dw, stride);
        dh = fmod(dh, stride);
    } else if (scale_fill) {  // stretch
        dw = 0.0;
        dh = 0.0;
        new_unpad[0] = new_shape[1];
        new_unpad[1] = new_shape[0];
        ratio[0] = static_cast<double>(new_shape[1]) / static_cast<double>(shape[1]);  // width ratio
        ratio[1] = static_cast<double>(new_shape[0]) / static_cast<double>(shape[0]);  // height ratio
    }
    
    if (center) {
        dw /= 2.0;  // divide padding into 2 sides
        dh /= 2.0;
    }
    
    // Resize if needed - EXACT Ultralytics method
    if (shape[1] != new_unpad[0] || shape[0] != new_unpad[1]) {  // resize
        cv::resize(img, img, cv::Size(new_unpad[0], new_unpad[1]), 0, 0, cv::INTER_LINEAR);
    }
    
    // EXACT Ultralytics padding calculation - CRITICAL FIX for y-coordinate differences
    int top, bottom, left, right;
    if (center) {
        // EXACT Python Ultralytics calculation: round(dh - 0.1) and round(dh + 0.1)
        // For dh = 106.5: round(106.5 - 0.1) = round(106.4) = 106, round(106.5 + 0.1) = round(106.6) = 107
        top = static_cast<int>(std::round(dh - 0.1));
        bottom = static_cast<int>(std::round(dh + 0.1));
        left = static_cast<int>(std::round(dw - 0.1));
        right = static_cast<int>(std::round(dw + 0.1));
    } else {
        top = 0;
        left = 0;
        bottom = static_cast<int>(std::round(dh));
        right = static_cast<int>(std::round(dw));
    }
    
    if (config_.verbose_logging) {
        std::cout << "LetterBox preprocessing: " << shape[1] << "x" << shape[0] 
                  << " -> " << new_shape[1] << "x" << new_shape[0] << ", ratio=" << r << "\n";
    }
    
    // Apply padding with EXACT Ultralytics method
    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // Store original padding values for debug
    int original_top = top;
    int original_left = left;
    
    // CRITICAL FIX: DO NOT force image to match ONNX input size when auto=True
    // Ultralytics Python behavior: when auto=True, the final image size can be smaller than target
    // Only add additional padding if auto=False (scale_fill mode)
    bool need_additional_padding = false;
    
    if (!auto_pad) {  // Only for auto=False case
        if (img.rows != new_shape[0] || img.cols != new_shape[1]) {
            need_additional_padding = true;
        }
    }
    
    if (need_additional_padding) {
        int additional_top = (new_shape[0] - img.rows) / 2;
        int additional_bottom = new_shape[0] - img.rows - additional_top;
        int additional_left = (new_shape[1] - img.cols) / 2;
        int additional_right = new_shape[1] - img.cols - additional_left;
        
        cv::copyMakeBorder(img, img, additional_top, additional_bottom, additional_left, additional_right, 
                          cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        
        // Update padding values to include additional padding
        left += additional_left;
        top += additional_top;
        
        if (config_.verbose_logging) {
            std::cout << "Additional padding applied: " << additional_top << "," << additional_left << "\n";
        }
    }
    
    if (config_.verbose_logging) {
        std::cout << "Final size: " << img.cols << "x" << img.rows 
                  << ", padding: " << left << "," << top << ", ratio: " << r << "\n";
    }
    
    // Store EXACT scale info for postprocessing - USE DOUBLE PRECISION
    scale_info.x = left;
    scale_info.y = top;
    scale_info.width = r;  // Keep as double precision - NO CONVERSION TO FLOAT
    scale_info.height = r;
    
    // Store exact double precision ratio in class member to avoid precision loss
    exact_scale_ratio_ = r;
    
    // Store actual processed image dimensions for input tensor creation
    processed_width_ = img.cols;
    processed_height_ = img.rows;
    
    // Convert BGR to RGB (Ultralytics expects RGB)
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    
    // Convert to double and normalize [0,1] (EXACT Ultralytics normalization)
    rgb_img.convertTo(rgb_img, CV_64F, 1.0/255.0);

    // Convert from HWC to CHW format (ONNX expects CHW)
    std::vector<cv::Mat> channels(3);
    cv::split(rgb_img, channels);
    
    // Prepare output vector - use ACTUAL image size, not target size
    int actual_height = img.rows;
    int actual_width = img.cols;
    int input_size = actual_height * actual_width * 3;
    std::vector<float> input_tensor_data(input_size);
    
    if (config_.verbose_logging) {
        std::cout << "Creating input tensor: " << actual_width << "x" << actual_height << " (size=" << input_size << ")\n";
    }
    
    // Fill tensor data in CHW format (channels first)
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < actual_height; ++h) {
            for (int w = 0; w < actual_width; ++w) {
                input_tensor_data[c * actual_height * actual_width + h * actual_width + w] = 
                    channels[c].at<double>(h, w);
            }
        }
    }
    
    return input_tensor_data;
}


std::vector<PointDetectionResult> PointDetector::postprocess(const std::vector<float>& output_data, 
                                                             const std::vector<int64_t>& output_shape,
                                                             const cv::Rect2d& scale_info) {
    std::vector<PointDetectionResult> results;
    
    if (config_.verbose_logging) {
        std::cout << "Analyzing point detection output: shape=[";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ",";
        }
        std::cout << "], size=" << output_data.size() << "\n";
    }
    
    // Use EXACT same coordinate transformation as SingleImageYolo.cpp
    double gain = std::min(static_cast<double>(processed_height_) / static_cast<double>(original_height_), 
                           static_cast<double>(processed_width_) / static_cast<double>(original_width_));
    double pad_w = std::round((static_cast<double>(processed_width_) - static_cast<double>(original_width_) * gain) / 2 - 0.1);
    double pad_h = std::round((static_cast<double>(processed_height_) - static_cast<double>(original_height_) * gain) / 2 - 0.1);
    
    if (config_.verbose_logging) {
        std::cout << "Coordinate transform: " << original_width_ << "x" << original_height_ 
                  << " -> " << processed_width_ << "x" << processed_height_ << ", gain=" << gain << "\n";
    }

    // Step 1: Apply Ultralytics NMS on raw model output first
    auto nms_output = non_max_suppression(
        output_data, 
        output_shape,
        config_.confidence_threshold,  // conf_thres
        config_.nms_threshold,         // iou_thres
        {},                           // classes (empty = all classes)
        false,                        // agnostic
        false,                        // multi_label
        300,                          // max_det
        0,                            // nc (auto-calculate)
        30000,                        // max_nms
        7680,                         // max_wh
        true                          // in_place
    );
    
    // Step 2: Process NMS output and transform coordinates
    if (!nms_output.empty() && !nms_output[0].empty()) {
        const auto& detections = nms_output[0];  // First (and only) batch
        
        if (config_.verbose_logging) {
            std::cout << "Processing " << detections.size() << " detections after NMS\n";
        }
        
        for (const auto& detection : detections) {
            if (detection.size() < 6) continue;  // Need at least [x1, y1, x2, y2, conf, class]
            
            // Extract detection data (already in xyxy format from NMS)
            double x1 = detection[0];
            double y1 = detection[1];
            double x2 = detection[2];
            double y2 = detection[3];
            float confidence = static_cast<float>(detection[4]);
            int class_id = static_cast<int>(detection[5]);
            
            // Calculate center point from bounding box
            double px = (x1 + x2) / 2.0;  // Point x = bbox center x
            double py = (y1 + y2) / 2.0;  // Point y = bbox center y
            
            // EXACT Ultralytics coordinate transformation for point
            // Step 1: Remove padding (subtract padding offsets)
            x1 -= pad_w;
            y1 -= pad_h;
            x2 -= pad_w;
            y2 -= pad_h;

            // Step 2: Scale back to original image coordinates (inverse of ratio)
            x1 /= gain;
            y1 /= gain;
            x2 /= gain;
            y2 /= gain;
            
            // Step 3: Clamp coordinates to original image bounds (EXACT Ultralytics method)
            double orig_width = static_cast<double>(original_width_);
            double orig_height = static_cast<double>(original_height_);
            x1 = std::max(0.0, std::min(orig_width, x1));
            y1 = std::max(0.0, std::min(orig_height, y1));
            x2 = std::max(0.0, std::min(orig_width, x2));
            y2 = std::max(0.0, std::min(orig_height, y2));

            px = (x1 + x2) / 2.0;  // Point x = bbox center x
            py = (y1 + y2) / 2.0;  // Point y = bbox center y
            
            
            // Step 2: Scale back to original image coordinates (inverse of ratio)
            
            
            // Create detection result with minimal bounding box (just for compatibility)
            float box_size = 10.0f; // Small box around point for visualization
            cv::Rect2f bbox(static_cast<float>(px - box_size/2), static_cast<float>(py - box_size/2), box_size, box_size);
            cv::Point2f point(static_cast<float>(px), static_cast<float>(py));
            
            PointDetectionResult detection_result(bbox, confidence, class_id);
            detection_result.points.emplace_back(point, confidence, class_id);
            
            results.push_back(detection_result);
            
            if (results.size() <= 10) {  // Only show first 10 for debugging
                std::cout << "   → Point " << results.size() << ": (" << px << "," << py 
                          << ") conf=" << confidence << " class=" << class_id << "\n";
            }
        }
    }
    
    std::cout << "   Final result: " << results.size() << " point detections" << "\n";
    return results;
}

// Exact implementation of Ultralytics non_max_suppression function
// Fast C++ implementation of non_max_suppression based on Python YOLOv8 code
std::vector<std::vector<std::vector<float>>> PointDetector::non_max_suppression(
    const std::vector<float>& prediction,
    const std::vector<int64_t>& shape,  // [batch, features, boxes]
    float conf_thres = 0.25f,
    float iou_thres = 0.45f,
    const std::vector<int>& classes = {},
    bool agnostic = false,
    bool multi_label = false,
    int max_det = 300,
    int nc = 0,  // number of classes (optional)
    int max_nms = 30000,
    int max_wh = 7680,
    bool in_place = true
) {
    // Input validation
    assert(conf_thres >= 0.0f && conf_thres <= 1.0f && "Invalid Confidence threshold");
    assert(iou_thres >= 0.0f && iou_thres <= 1.0f && "Invalid IoU threshold");
    
    int batch_size = int(shape[0]);
    int features = int(shape[1]);
    int num_boxes = int(shape[2]);
    
    // Auto-detect number of classes if not provided
    if (nc == 0) nc = features - 4;
    int nm = features - nc - 4;  // number of masks
    int mi = 4 + nc;  // mask start index
    
    // Output: [batch][num_dets][vector: box + conf + class + ...mask...]
    std::vector<std::vector<std::vector<float>>> output(batch_size);
    
    // Process each image in batch
    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        std::vector<std::vector<float>> detections;
        
        // Step 1: Extract candidates that pass confidence threshold
        std::vector<bool> candidates(num_boxes, false);
        int candidate_count = 0;
        
        for (int box_idx = 0; box_idx < num_boxes; ++box_idx) {
            // Find max class confidence for this box
            float max_conf = 0.0f;
            for (int class_idx = 0; class_idx < nc; ++class_idx) {
                int pred_idx = batch_idx * features * num_boxes + (4 + class_idx) * num_boxes + box_idx;
                float conf = prediction[pred_idx];
                if (conf > max_conf) {
                    max_conf = conf;
                }
            }
            
            if (max_conf > conf_thres) {
                candidates[box_idx] = true;
                candidate_count++;
            }
        }
        
        if (candidate_count == 0) {
            output[batch_idx] = std::vector<std::vector<float>>();
            continue;
        }
        
        // Step 2: Extract box coordinates and convert xywh to xyxy
        std::vector<std::vector<float>> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<std::vector<float>> masks;
        
        boxes.reserve(candidate_count);
        scores.reserve(candidate_count);
        class_ids.reserve(candidate_count);
        masks.reserve(candidate_count);
        
        for (int box_idx = 0; box_idx < num_boxes; ++box_idx) {
            if (!candidates[box_idx]) continue;
            
            // Extract box coordinates [center_x, center_y, w, h]
            std::vector<float> box(4);
            for (int coord = 0; coord < 4; ++coord) {
                int pred_idx = batch_idx * features * num_boxes + coord * num_boxes + box_idx;
                box[coord] = prediction[pred_idx];
            }
            
            // Convert xywh to xyxy
            xywh2xyxy(box);
            
            // Find best class and confidence
            float best_conf = 0.0f;
            int best_class = -1;
            
            if (multi_label) {
                // Multi-label: collect all classes above threshold
                for (int class_idx = 0; class_idx < nc; ++class_idx) {
                    int pred_idx = batch_idx * features * num_boxes + (4 + class_idx) * num_boxes + box_idx;
                    float conf = prediction[pred_idx];
                    if (conf > conf_thres) {
                        if (conf > best_conf) {
                            best_conf = conf;
                            best_class = class_idx;
                        }
                    }
                }
            } else {
                // Single-label: find best class
                for (int class_idx = 0; class_idx < nc; ++class_idx) {
                    int pred_idx = batch_idx * features * num_boxes + (4 + class_idx) * num_boxes + box_idx;
                    float conf = prediction[pred_idx];
                    if (conf > best_conf) {
                        best_conf = conf;
                        best_class = class_idx;
                    }
                }
            }
            
            if (best_conf <= conf_thres || best_class == -1) continue;
            
            // Class filter
            if (!classes.empty()) {
                bool class_allowed = false;
                for (int allowed_class : classes) {
                    if (best_class == allowed_class) {
                        class_allowed = true;
                        break;
                    }
                }
                if (!class_allowed) continue;
            }
            
            // Extract mask coefficients
            std::vector<float> mask_coeffs;
            for (int mask_idx = mi; mask_idx < features; ++mask_idx) {
                int pred_idx = batch_idx * features * num_boxes + mask_idx * num_boxes + box_idx;
                mask_coeffs.push_back(prediction[pred_idx]);
            }
            
            boxes.push_back(box);
            scores.push_back(best_conf);
            class_ids.push_back(best_class);
            masks.push_back(mask_coeffs);
        }
        
        if (boxes.empty()) {
            output[batch_idx] = std::vector<std::vector<float>>();
            continue;
        }
        
        // Step 3: Sort by confidence (descending)
        std::vector<int> indices(boxes.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
            return scores[a] > scores[b];
        });
        
        // Limit to max_nms boxes
        if (indices.size() > max_nms) {
            indices.resize(max_nms);
        }
        
        // Step 4: Apply NMS using OpenCV's optimized implementation
        std::vector<cv::Rect2d> cv_boxes;
        std::vector<float> cv_scores;
        std::vector<int> final_indices;
        
        cv_boxes.reserve(indices.size());
        cv_scores.reserve(indices.size());
        
        for (int idx : indices) {
            const auto& box = boxes[idx];
            
            // Add class offset for class-specific NMS (unless agnostic)
            float class_offset = agnostic ? 0.0f : class_ids[idx] * max_wh;
            
            cv_boxes.emplace_back(
                box[0] + class_offset,  // x1
                box[1] + class_offset,  // y1
                box[2] - box[0],        // width
                box[3] - box[1]         // height
            );
            cv_scores.push_back(scores[idx]);
        }
        
        // Apply OpenCV NMS
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(cv_boxes, cv_scores, conf_thres, iou_thres, nms_indices, 1.0f, max_det);
        
        // Step 5: Gather final results
        std::vector<std::vector<float>> final_detections;
        final_detections.reserve(nms_indices.size());
        
        for (int nms_idx : nms_indices) {
            if (nms_idx >= 0 && nms_idx < indices.size()) {
                int original_idx = indices[nms_idx];
                
                // Create detection: [x1, y1, x2, y2, conf, cls, ...mask...]
                std::vector<float> detection;
                detection.reserve(6 + masks[original_idx].size());
                
                // Add box coordinates
                detection.insert(detection.end(), boxes[original_idx].begin(), boxes[original_idx].end());
                
                // Add confidence and class
                detection.push_back(scores[original_idx]);
                detection.push_back(static_cast<float>(class_ids[original_idx]));
                
                // Add mask coefficients
                detection.insert(detection.end(), masks[original_idx].begin(), masks[original_idx].end());
                
                final_detections.push_back(std::move(detection));
            }
        }
        
        output[batch_idx] = std::move(final_detections);
    }
    
    // Print summary
    int total_detections = 0;
    for (const auto& batch_result : output) {
        total_detections += batch_result.size();
    }
    
    return output;
}

std::vector<PointDetectionResult> PointDetector::detect(const std::string& image_path, float conf) {
    if (!model_loaded_) {
        throw std::runtime_error("Point detection model not loaded");
    }

    float confidence_threshold = (conf < 0.0f) ? config_.confidence_threshold : conf;

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + image_path);
    }

    return detect(image, confidence_threshold);
}

std::vector<PointDetectionResult> PointDetector::detect(const cv::Mat& image, float conf) {
    if (!model_loaded_) {
        throw std::runtime_error("Point detection model not loaded");
    }

    float confidence_threshold = (conf < 0.0f) ? config_.confidence_threshold : conf;
    config_.confidence_threshold = confidence_threshold;

    std::cout << "🎯 Running point detection on image: " << image.cols << "x" << image.rows << "\n";
    std::cout << "   Using confidence threshold: " << confidence_threshold << "\n";

    // Preprocess image
    cv::Rect2d scale_info;
    std::vector<float> input_data = preprocess(image, scale_info);
    
    // Create input shape
    std::vector<int64_t> input_shape = {1, 3, 
                                       static_cast<int64_t>(processed_height_), 
                                       static_cast<int64_t>(processed_width_)};
    
    // Run inference
    auto output_tensors = model_->runInference(input_data, input_shape);
    
    // Process outputs
    auto& output_tensor = output_tensors[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    
    const float* output_data = output_tensor.GetTensorData<float>();
    size_t output_size = output_info.GetElementCount();
    
    // Convert to vector for processing (keep as float)
    std::vector<float> output_vector(output_size);
    for (size_t i = 0; i < output_size; ++i) {
        output_vector[i] = output_data[i];
    }
    
    // Postprocess
    auto detections = postprocess(output_vector, output_shape, scale_info);
    
    std::cout << "✅ Point detection completed: " << detections.size() << " points found" << "\n";
    return detections;
}

void PointDetector::analyzeModelOutput(const std::string& test_image_path) {
    if (!model_loaded_) {
        std::cout << "❌ Model not loaded, cannot analyze output" << "\n";
        return;
    }
    
    std::cout << "🔍 Analyzing model output format with test image..." << "\n";
    
    try {
        cv::Mat test_image = cv::imread(test_image_path);
        if (test_image.empty()) {
            std::cout << "❌ Could not load test image: " << test_image_path << "\n";
            return;
        }
        
        // Run a test inference
        cv::Rect2d scale_info;
        std::vector<float> input_data = preprocess(test_image, scale_info);
        
        std::vector<int64_t> input_shape = {1, 3, 
                                           static_cast<int64_t>(processed_height_), 
                                           static_cast<int64_t>(processed_width_)};
        
        auto output_tensors = model_->runInference(input_data, input_shape);
        
        std::cout << "📊 Model Output Analysis:" << "\n";
        std::cout << "   Number of output tensors: " << output_tensors.size() << "\n";
        
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto& tensor = output_tensors[i];
            auto info = tensor.GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            
            std::cout << "   Output " << i << " shape: [";
            for (size_t j = 0; j < shape.size(); ++j) {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << "\n";
            std::cout << "   Output " << i << " size: " << info.GetElementCount() << "\n";
            
            // Show first few values
            const float* data = tensor.GetTensorData<float>();
            std::cout << "   First 10 values: ";
            for (size_t j = 0; j < std::min(static_cast<size_t>(10), info.GetElementCount()); ++j) {
                std::cout << std::fixed << std::setprecision(4) << data[j] << " ";
            }
            std::cout << "\n";
        }
        
        std::cout << "💡 Based on this output, you may need to adjust the postprocess() function" << "\n";
        std::cout << "   to match your specific model's output format." << "\n";
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error during model analysis: " << e.what() << "\n";
    }
}

void PointDetector::printModelInfo() const {
    if (!model_loaded_) {
        std::cout << "❌ Model not loaded" << "\n";
        return;
    }
    
    std::cout << "📋 Point Detection Model Information:" << "\n";
    std::cout << "   Model path: " << model_path_ << "\n";
    std::cout << "   Input size: " << model_->getInputSize().width << "x" << model_->getInputSize().height << "\n";
    std::cout << "   Configuration:" << "\n";
    std::cout << "     - Confidence threshold: " << config_.confidence_threshold << "\n";
    std::cout << "     - Point confidence threshold: " << config_.point_confidence_threshold << "\n";
    std::cout << "     - NMS threshold: " << config_.nms_threshold << "\n";
    std::cout << "     - Expected points per object: " << config_.num_points_per_object << "\n";
    std::cout << "     - Has bounding boxes: " << (config_.has_bounding_box ? "Yes" : "No") << "\n";
    std::cout << "     - Has point classes: " << (config_.has_point_classes ? "Yes" : "No") << "\n";
    std::cout << "     - Has visibility scores: " << (config_.has_visibility_scores ? "Yes" : "No") << "\n";
}

std::string PointDetector::savePointsToCSV(const std::vector<PointDetectionResult>& results, 
                                          const std::string& output_filename) {
    std::string csv_path = "/home/rajabzade/DeepNetC++/output/" + output_filename;
    std::ofstream file(csv_path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + csv_path);
    }

    // Write CSV header in Python format
    file << "x,y,confidence,class_id,class_name\n";

    // Write point data in simple format
    for (const auto& detection : results) {
        for (const auto& point : detection.points) {
            // Simple class name mapping (you can customize this)
            std::string class_name = "object_" + std::to_string(point.point_id);
            
            file << point.position.x << ","
                 << point.position.y << ","
                 << point.confidence << ","
                 << point.point_id << ","
                 << class_name << "\n";
        }
    }

    file.close();
    std::cout << "💾 Point detection results saved to CSV: " << csv_path << "\n";
    return csv_path;
}

std::string PointDetector::drawAndSavePoints(const std::string& image_path, 
                                           const std::vector<PointDetectionResult>& detections, 
                                           const std::string& output_filename) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image for visualization: " + image_path);
    }
    
    // Colors for different detection instances
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),    // Green
        cv::Scalar(255, 0, 0),    // Blue
        cv::Scalar(0, 0, 255),    // Red
        cv::Scalar(255, 255, 0),  // Cyan
        cv::Scalar(255, 0, 255),  // Magenta
        cv::Scalar(0, 255, 255),  // Yellow
        cv::Scalar(128, 0, 128),  // Purple
        cv::Scalar(255, 165, 0),  // Orange
    };
    
    for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
        const auto& detection = detections[det_idx];
        cv::Scalar color = colors[det_idx % colors.size()];
        
        // Draw points only (no bounding boxes for point detection)
        for (size_t pt_idx = 0; pt_idx < detection.points.size(); ++pt_idx) {
            const auto& point = detection.points[pt_idx];
            
            // Draw point as circle
            int radius = 6;
            cv::circle(image, point.position, radius, color, -1);
            cv::circle(image, point.position, radius + 1, cv::Scalar(0, 0, 0), 2); // Black border
            
            // Draw point label with coordinates and confidence
            std::string point_label = "(" + std::to_string(static_cast<int>(point.position.x)) + 
                                     "," + std::to_string(static_cast<int>(point.position.y)) + 
                                     ") " + std::to_string(static_cast<int>(point.confidence * 100)) + "%";
            cv::putText(image, point_label, 
                       cv::Point(point.position.x + 10, point.position.y - 10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
    }
    
    // Save the image
    std::string output_path = output_dir_ + "/" + output_filename;
    cv::imwrite(output_path, image);
    std::cout << "🎨 Point detection visualization saved to: " << output_path << "\n";
    
    return output_path;
}

std::pair<std::vector<PointDetectionResult>, std::string> PointDetector::processImage(
    const std::string& image_name,
    bool save_csv,
    float conf) {
    
    float confidence_threshold = (conf < 0.0f) ? config_.confidence_threshold : conf;
    std::string csv_path;
    
    try {
        // Run detection
        std::vector<PointDetectionResult> results = detect(image_name, confidence_threshold);
        
        // Save results to CSV if requested
        if (save_csv) {
            csv_path = savePointsToCSV(results);
        }
        
        std::cout << "✅ Point detection processing completed successfully" << "\n";
        return std::make_pair(results, csv_path);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Point detection processing failed: " << e.what() << "\n";
        throw;
    }
}

// ============================================================================
// Point Analysis Utilities Implementation
// ============================================================================

namespace PointAnalysis {

double calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

double calculateAngle(const cv::Point2f& p1, const cv::Point2f& center, const cv::Point2f& p3) {
    cv::Point2f v1 = p1 - center;
    cv::Point2f v2 = p3 - center;
    
    double dot = v1.x * v2.x + v1.y * v2.y;
    double mag1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
    double mag2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
    
    if (mag1 == 0 || mag2 == 0) return 0.0;
    
    double cos_angle = dot / (mag1 * mag2);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle)); // Clamp to [-1, 1]
    
    return std::acos(cos_angle) * 180.0 / M_PI; // Convert to degrees
}

int findClosestPoint(const cv::Point2f& reference, const std::vector<DetectedPoint>& points) {
    if (points.empty()) return -1;
    
    int closest_idx = 0;
    double min_distance = calculateDistance(reference, points[0].position);
    
    for (size_t i = 1; i < points.size(); ++i) {
        double distance = calculateDistance(reference, points[i].position);
        if (distance < min_distance) {
            min_distance = distance;
            closest_idx = static_cast<int>(i);
        }
    }
    
    return closest_idx;
}

cv::Point2f calculateCentroid(const std::vector<DetectedPoint>& points) {
    if (points.empty()) return cv::Point2f(0, 0);
    
    float sum_x = 0, sum_y = 0;
    for (const auto& point : points) {
        sum_x += point.position.x;
        sum_y += point.position.y;
    }
    
    return cv::Point2f(sum_x / points.size(), sum_y / points.size());
}

std::vector<DetectedPoint> filterByConfidence(const std::vector<DetectedPoint>& points, float threshold) {
    std::vector<DetectedPoint> filtered;
    for (const auto& point : points) {
        if (point.confidence >= threshold) {
            filtered.push_back(point);
        }
    }
    return filtered;
}

} // namespace PointAnalysis 