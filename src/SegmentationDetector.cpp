// Note: For segmentation_batch target, this uses pch_segmentation_batch.hpp as PCH
// For other targets, this uses the regular pch.hpp
#include "SegmentationDetector.hpp"
#include "Config.hpp"
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <iomanip>
#include <numeric>  // For std::iota
#include <vector>   // For std::vector
#include <string>   // For std::string

// ============================================================================
// SegmentationMask Implementation
// ============================================================================

void SegmentationMask::calculateProperties() {
    if (mask.empty()) return;
    
    // Calculate area
    area = cv::countNonZero(mask);
    
    // Calculate centroid - prefer contour-based calculation if contour exists
    if (!contour.empty()) {
        // Calculate centroid from contour points (which are in correct coordinate space)
        double sum_x = 0.0, sum_y = 0.0;
        for (const auto& point : contour) {
            sum_x += point.x;
            sum_y += point.y;
        }
        centroid.x = static_cast<float>(sum_x / contour.size());
        centroid.y = static_cast<float>(sum_y / contour.size());
    } else {
        // Fallback to mask-based calculation
        cv::Moments moments = cv::moments(mask);
        if (moments.m00 > 0) {
            centroid.x = static_cast<float>(moments.m10 / moments.m00);
            centroid.y = static_cast<float>(moments.m01 / moments.m00);
        }
    }
    
    // Extract contour if not provided
    if (contour.empty()) {
        std::vector<std::vector<cv::Point>> contours;
        // Use CHAIN_APPROX_NONE to preserve all contour points for detailed segmentation
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (!contours.empty()) {
            // Find largest contour
            size_t largest_idx = 0;
            double largest_area = 0;
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = cv::contourArea(contours[i]);
                if (area > largest_area) {
                    largest_area = area;
                    largest_idx = i;
                }
            }
            contour = contours[largest_idx];
        }
    }
}

// ============================================================================
// ONNX Model Wrapper Implementation for Segmentation Detection
// ============================================================================

SegmentationOnnxWrapper::SegmentationOnnxWrapper(const std::string& model_path, const std::string& provider) {
    initializeSession(model_path, provider);
}

void SegmentationOnnxWrapper::initializeSession(const std::string& model_path, const std::string& provider) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SegmentationDetector");
    
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

void SegmentationOnnxWrapper::configureExecutionProvider(Ort::SessionOptions& session_options, const std::string& provider) {
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
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                cuda_options.arena_extend_strategy = 1;
                cuda_options.gpu_mem_limit = 0;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.do_copy_in_default_stream = 1;
                
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "✅ CUDA execution provider enabled" << "\n";
                return;
            } catch (const std::exception& e) {
                std::cout << "⚠️  CUDA initialization failed: " << e.what() << "\n";
            }
        } else {
            std::cout << "❌ CUDA provider not available" << "\n";
        }
    }
    
    std::cout << "🖥️  Using optimized CPU execution provider" << "\n";
}

void SegmentationOnnxWrapper::extractModelInfo() {
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
    input_size_ = cv::Size(1280, 1280);  // Force correct input size
    
    std::cout << "✅ Segmentation Model initialized:" << "\n";
    std::cout << "   Input: " << input_names_[0] << " (" << input_size_.width << "x" << input_size_.height << ")" << "\n";
    std::cout << "   Outputs: ";
    for (const auto& name : output_names_) {
        std::cout << name << " ";
    }
    std::cout << "\n";
}

std::vector<Ort::Value> SegmentationOnnxWrapper::runInference(const std::vector<float>& input_data, 
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
// SegmentationDetector Implementation
// ============================================================================

SegmentationDetector::SegmentationDetector(const std::string& model_path, const std::string& output_dir, const std::string& provider)
    : model_path_(model_path), output_dir_(output_dir), model_loaded_(false) {
    setupDirectories();
    loadModel();
    initializeClassNames();
    generateClassColors();
}

void SegmentationDetector::setupDirectories() {
    std::filesystem::create_directories(output_dir_);
    std::filesystem::create_directories(output_dir_ + "/masks");
}

void SegmentationDetector::loadModel() {
    try {
        // Get execution provider from config if not specified
        const auto& config = Config::get();
        std::string provider = config.execution_provider;
        
        model_ = std::make_unique<SegmentationOnnxWrapper>(model_path_, provider);
        model_loaded_ = true;
        std::cout << "✅ Segmentation Model loaded successfully from: " << model_path_ << "\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to load segmentation model: " << e.what() << "\n";
        model_loaded_ = false;
    }
}

void SegmentationDetector::initializeClassNames() {
    // Match the Python model output format - single class with ID 0 and name "0"
    class_names_ = {"0", "1"};
}

void SegmentationDetector::generateClassColors() {
    class_colors_.clear();
    for (size_t i = 0; i < class_names_.size(); ++i) {
        // Generate brighter, more distinct colors for each class
        cv::Scalar color;
        if (i == 0) {
            color = cv::Scalar(0, 255, 0);    // Bright green for class 0
        } else if (i == 1) {
            color = cv::Scalar(255, 0, 0);    // Bright blue for class 1
        } else {
            // Generate bright colors for additional classes
            color = cv::Scalar(
                (i * 80 + 150) % 255,         // Ensure minimum brightness
                (i * 120 + 150) % 255,
                (i * 200 + 150) % 255
            );
        }
        class_colors_.push_back(color);
    }
}

void SegmentationDetector::setClassNames(const std::vector<std::string>& names) {
    class_names_ = names;
    generateClassColors();
}

// Reuse the exact preprocessing from PointDetector.cpp
std::vector<float> SegmentationDetector::preprocess(const cv::Mat& image, cv::Rect2d& scale_info, double& pimgwidth, double& pimgheight) {
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
    
    // std::cout << "EXACT Ultralytics LetterBox:" << "\n";
    // std::cout << "  Parameters: center=" << center << ", auto=" << auto_pad << ", scale_fill=" << scale_fill << ", scaleup=" << scaleup << "\n";
    // std::cout << "  Original shape: [" << shape[0] << ", " << shape[1] << "]" << "\n";
    // std::cout << "  New shape: [" << new_shape[0] << ", " << new_shape[1] << "]" << "\n";
    // std::cout << "  Scale ratio (r): " << std::fixed << std::setprecision(16) << r << "\n";
    // std::cout << "  New unpadded: [" << new_unpad[0] << ", " << new_unpad[1] << "]" << "\n";
    // std::cout << "  dw, dh: " << std::fixed << std::setprecision(10) << dw << ", " << dh << "\n";
    

    // Apply padding with EXACT Ultralytics method
    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // Store original padding values for coordinate transformation
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
    }
    
    // Production mode: minimal logging
    if (config_.verbose_logging) {
        std::cout << "Final image size: " << img.cols << "x" << img.rows 
                  << ", padding: left=" << left << ", top=" << top << "\n";
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
    
    // The dynamic ONNX model can handle variable input sizes like 1280x736
    if (config_.verbose_logging) {
        std::cout << "Using natural processed size: " << img.cols << "x" << img.rows << "\n";
    }
    
    // Convert BGR to RGB (Ultralytics expects RGB)
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    
    // Convert to double and normalize [0,1] (EXACT Ultralytics normalization)
    rgb_img.convertTo(rgb_img, CV_32F, 1.0/255.0);

    // Convert from HWC to CHW format (ONNX expects CHW)
    std::vector<cv::Mat> channels(3);
    cv::split(rgb_img, channels);
    
    // Use ACTUAL processed dimensions for tensor creation (natural size)
    pimgheight = img.rows;  // Use actual height (e.g., 736)
    pimgwidth = img.cols;   // Use actual width (e.g., 1280)
    int input_size = pimgheight * pimgwidth * 3;
    std::vector<float> input_tensor_data(input_size);
    
    if (config_.verbose_logging) {
        std::cout << "Creating input tensor: " << pimgwidth << "x" << pimgheight << " (size=" << input_size << ")\n";
    }
    
    // Fill tensor data in CHW format (channels first)
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < pimgheight; ++h) {
            for (int w = 0; w < pimgwidth; ++w) {
                input_tensor_data[c * pimgheight * pimgwidth + h * pimgwidth + w] = 
                    channels[c].at<float>(h, w);
            }
        }
    }
    return input_tensor_data;
}

// EXACT ULTRALYTICS non_max_suppression implementation
std::vector<std::vector<std::vector<float>>> SegmentationDetector::non_max_suppression(
    const std::vector<float>& prediction_data,
    const std::vector<int64_t>& prediction_shape,
    float conf_thres,
    float iou_thres,
    const std::vector<int>& classes,
    bool agnostic,
    bool multi_label,
    int max_det,
    int nc,
    int max_nms,
    int max_wh,
    bool in_place,
    bool rotated,
    bool end2end,
    bool return_idxs) {
    
    // Checks
    if (!(0 <= conf_thres && conf_thres <= 1)) {
        std::cerr << "Invalid Confidence threshold " << conf_thres << ", valid values are between 0.0 and 1.0" << "\n";
        return {};
    }
    if (!(0 <= iou_thres && iou_thres <= 1)) {
        std::cerr << "Invalid IoU " << iou_thres << ", valid values are between 0.0 and 1.0" << "\n";
        return {};
    }
    
    if (prediction_shape.size() != 3) {
        return {};
    }
    
    int bs = static_cast<int>(prediction_shape[0]);  // batch size (BCN, i.e. 1,84,6300)
    int prediction_features = static_cast<int>(prediction_shape[1]);  // features (84)
    int prediction_boxes = static_cast<int>(prediction_shape[2]);  // boxes (6300)
    
    // if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
    if (prediction_features == 6 || end2end) {
        std::vector<std::vector<std::vector<float>>> output(bs);
        for (int b = 0; b < bs; ++b) {
            std::vector<std::vector<float>> batch_output;
            for (int i = 0; i < prediction_boxes && static_cast<int>(batch_output.size()) < max_det; ++i) {
                int base_idx = b * prediction_boxes * prediction_features + i * prediction_features;
                if (base_idx + 4 < prediction_data.size()) {
                    float conf = prediction_data[base_idx + 4];
                    if (conf > conf_thres) {
                        std::vector<float> pred;
                        for (int j = 0; j < prediction_features; ++j) {
                            pred.push_back(prediction_data[base_idx + j]);
                        }
                        
                        // Filter by classes if specified
                        if (!classes.empty()) {
                            int cls = static_cast<int>(prediction_data[base_idx + 5]);
                            bool class_match = false;
                            for (int target_cls : classes) {
                                if (cls == target_cls) {
                                    class_match = true;
                                    break;
                                }
                            }
                            if (class_match) {
                                batch_output.push_back(pred);
                            }
                        } else {
                            batch_output.push_back(pred);
                        }
                    }
                }
            }
            output[b] = batch_output;
        }
        return output;
    }
    
    // nc = nc or (prediction.shape[1] - 4)  # number of classes
    if (nc == 0) {
        nc = prediction_features - 4;
    }
    int nm = prediction_features - nc - 4;  // number of masks
    int mi = 4 + nc;  // mask start index
    
    // xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    std::vector<std::vector<bool>> xc(bs);
    for (int b = 0; b < bs; ++b) {
        xc[b].resize(prediction_boxes);
        for (int i = 0; i < prediction_boxes; ++i) {
            float max_class_conf = 0.0f;
            for (int c = 4; c < mi; ++c) {
                int idx = b * prediction_boxes * prediction_features + c * prediction_boxes + i;
                if (idx < prediction_data.size()) {
                    max_class_conf = std::max(max_class_conf, prediction_data[idx]);
                }
            }
            xc[b][i] = max_class_conf > conf_thres;
        }
    }
    
    // multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    multi_label = multi_label && (nc > 1);
    
    // prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    std::vector<std::vector<std::vector<float>>> prediction(bs);
    for (int b = 0; b < bs; ++b) {
        prediction[b].resize(prediction_boxes);
        for (int i = 0; i < prediction_boxes; ++i) {
            prediction[b][i].resize(prediction_features);
            for (int j = 0; j < prediction_features; ++j) {
                int original_idx = b * prediction_boxes * prediction_features + j * prediction_boxes + i;
                if (original_idx < prediction_data.size()) {
                    prediction[b][i][j] = prediction_data[original_idx];
                }
            }
        }
    }
    
    // if not rotated: prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    if (!rotated) {
        for (int b = 0; b < bs; ++b) {
            for (int i = 0; i < prediction_boxes; ++i) {
                if (prediction[b][i].size() >= 4) {
                    // xywh2xyxy conversion
                    float x_center = prediction[b][i][0];
                    float y_center = prediction[b][i][1];
                    float width = prediction[b][i][2];
                    float height = prediction[b][i][3];
                    
                    prediction[b][i][0] = x_center - width / 2.0f;   // x1
                    prediction[b][i][1] = y_center - height / 2.0f;  // y1
                    prediction[b][i][2] = x_center + width / 2.0f;   // x2
                    prediction[b][i][3] = y_center + height / 2.0f;  // y2
                }
            }
        }
    }
    
    // output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    std::vector<std::vector<std::vector<float>>> output(bs);
    
    // for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
    for (int xi = 0; xi < bs; ++xi) {
        // filt = xc[xi]  # confidence
        // x, xk = x[filt], xk[filt]
        std::vector<std::vector<float>> x;
        for (int i = 0; i < prediction_boxes; ++i) {
            if (xc[xi][i]) {
                x.push_back(prediction[xi][i]);
            }
        }
        
        // If none remain process next image
        if (x.empty()) {
            continue;
        }
        
        // Detections matrix nx6 (xyxy, conf, cls)
        // box, cls, mask = x.split((4, nc, nm), 1)
        std::vector<std::vector<float>> box, cls, mask;
        for (const auto& detection : x) {
            if (detection.size() >= mi) {
                // box (first 4 elements)
                std::vector<float> b(detection.begin(), detection.begin() + 4);
                box.push_back(b);
                
                // cls (elements 4 to mi)
                std::vector<float> c(detection.begin() + 4, detection.begin() + mi);
                cls.push_back(c);
                
                // mask (elements mi to end)
                if (detection.size() > mi) {
                    std::vector<float> m(detection.begin() + mi, detection.end());
                    mask.push_back(m);
                } else {
                    mask.push_back(std::vector<float>(nm, 0.0f));
                }
            }
        }
        
        std::vector<std::vector<float>> final_x;
        
        if (multi_label) {
            // i, j = torch.where(cls > conf_thres)
            // x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            for (size_t i = 0; i < cls.size(); ++i) {
                for (size_t j = 0; j < cls[i].size(); ++j) {
                    if (cls[i][j] > conf_thres) {
                        std::vector<float> detection;
                        // Add box
                        detection.insert(detection.end(), box[i].begin(), box[i].end());
                        // Add confidence
                        detection.push_back(cls[i][j]);
                        // Add class index
                        detection.push_back(static_cast<float>(j));
                        // Add mask
                        detection.insert(detection.end(), mask[i].begin(), mask[i].end());
                        final_x.push_back(detection);
                    }
                }
            }
        } else {
            // best class only
            // conf, j = cls.max(1, keepdim=True)
            // x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            for (size_t i = 0; i < cls.size(); ++i) {
                float conf = 0.0f;
                int j = 0;
                for (size_t k = 0; k < cls[i].size(); ++k) {
                    if (cls[i][k] > conf) {
                        conf = cls[i][k];
                        j = static_cast<int>(k);
                    }
                }
                
                if (conf > conf_thres) {
                    std::vector<float> detection;
                    // Add box
                    detection.insert(detection.end(), box[i].begin(), box[i].end());
                    // Add confidence
                    detection.push_back(conf);
                    // Add class index
                    detection.push_back(static_cast<float>(j));
                    // Add mask
                    detection.insert(detection.end(), mask[i].begin(), mask[i].end());
                    final_x.push_back(detection);
                }
            }
        }
        
        x = final_x;
        
        // Filter by class
        if (!classes.empty()) {
            std::vector<std::vector<float>> filtered_x;
            for (const auto& detection : x) {
                if (detection.size() >= 6) {
                    int cls_id = static_cast<int>(detection[5]);
                    for (int target_cls : classes) {
                        if (cls_id == target_cls) {
                            filtered_x.push_back(detection);
                            break;
                        }
                    }
                }
            }
            x = filtered_x;
        }
        
        // Check shape
        int n = static_cast<int>(x.size());  // number of boxes
        if (n == 0) {  // no boxes
            continue;
        }
        
        if (n > max_nms) {  // excess boxes
            // sort by confidence and remove excess boxes
            std::sort(x.begin(), x.end(), [](const std::vector<float>& a, const std::vector<float>& b) {
                return a.size() >= 5 && b.size() >= 5 && a[4] > b[4];
            });
            x.resize(max_nms);
        }
        
        // Batched NMS
        // c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        // boxes = x[:, :4] + c  # boxes (offset by class)
        // i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        
        for (const auto& detection : x) {
            if (detection.size() >= 6) {
                float x1 = detection[0];
                float y1 = detection[1];
                float x2 = detection[2];
                float y2 = detection[3];
                float conf = detection[4];
                float cls = detection[5];
                
                // c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
                float c = agnostic ? 0.0f : cls * max_wh;
                
                // boxes = x[:, :4] + c  # boxes (offset by class)
                cv::Rect box(
                    static_cast<int>(x1 + c),
                    static_cast<int>(y1 + c),
                    static_cast<int>(x2 - x1),
                    static_cast<int>(y2 - y1)
                );
                boxes.push_back(box);
                scores.push_back(conf);
            }
        }
        
        std::vector<int> i;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres, iou_thres, i);
        
        // i = i[:max_det]  # limit detections
        if (static_cast<int>(i.size()) > max_det) {
            i.resize(max_det);
        }
        
        // output[xi] = x[i]
        std::vector<std::vector<float>> batch_output;
        for (int idx : i) {
            if (idx >= 0 && idx < static_cast<int>(x.size())) {
                batch_output.push_back(x[idx]);
            }
        }
        output[xi] = batch_output;
    }
    
    return output;
}
std::vector<cv::Mat> processMaskUltralyticsFast(
    const std::vector<float>& protos,
    const std::vector<int64_t>& proto_shape, // [B, C, H, W]
    const std::vector<std::vector<float>>& masks_in,
    const std::vector<cv::Rect2f>& bboxes,
    const cv::Size& img_shape,
    float mask_threshold = 0.5f,
    bool upsample = true)
{
    std::vector<cv::Mat> result_masks;
    if (proto_shape.size() != 4 || masks_in.empty()) return result_masks;

    int c = static_cast<int>(proto_shape[1]);
    int mh = static_cast<int>(proto_shape[2]);
    int mw = static_cast<int>(proto_shape[3]);
    int ih = img_shape.height, iw = img_shape.width;

    const size_t proto_size = mh * mw;

    // Precompute meshgrid for cropping (row/col index matrices)
    cv::Mat col_mat(mh, mw, CV_32F), row_mat(mh, mw, CV_32F);
    for (int y = 0; y < mh; ++y)
        for (int x = 0; x < mw; ++x) {
            col_mat.at<float>(y, x) = static_cast<float>(x);
            row_mat.at<float>(y, x) = static_cast<float>(y);
        }

    for (size_t mask_idx = 0; mask_idx < masks_in.size() && mask_idx < bboxes.size(); ++mask_idx) {
        const auto& mask_coeffs = masks_in[mask_idx];
        const auto& bbox = bboxes[mask_idx];
        if (mask_coeffs.size() != c) {
            result_masks.emplace_back();
            continue;
        }

        // 1. Fast linear combination, direct pointer manipulation
        cv::Mat mask(mh, mw, CV_32F, cv::Scalar(0));
        float* mask_ptr = mask.ptr<float>();
        std::fill(mask_ptr, mask_ptr + proto_size, 0.f);

        for (int i = 0; i < c; ++i) {
            const float coeff = mask_coeffs[i];
            if (std::abs(coeff) < 1e-6f) continue;
            const float* proto_ptr = protos.data() + i * proto_size;
            for (size_t j = 0; j < proto_size; ++j)
                mask_ptr[j] += coeff * proto_ptr[j];
        }

        // 2. Sigmoid activation, vectorized
        for (size_t j = 0; j < proto_size; ++j)
            mask_ptr[j] = 1.0f / (1.0f + std::exp(-mask_ptr[j]));

        // 3. Downsample bounding box
        float width_ratio  = static_cast<float>(mw) / iw;
        float height_ratio = static_cast<float>(mh) / ih;
        float x1 = bbox.x * width_ratio;
        float y1 = bbox.y * height_ratio;
        float x2 = (bbox.x + bbox.width) * width_ratio;
        float y2 = (bbox.y + bbox.height) * height_ratio;

        // 4. Crop mask: vectorized, no mat copies
        cv::Mat mask_keep(mh, mw, CV_8U, cv::Scalar(0));
        for (int y = 0; y < mh; ++y) {
            if (y < y1 || y >= y2) continue;
            for (int x = 0; x < mw; ++x) {
                if (x >= x1 && x < x2) mask_keep.at<uchar>(y, x) = 1;
            }
        }
        // Multiply mask by mask_keep in place (in float)
        for (size_t j = 0; j < proto_size; ++j)
            mask_ptr[j] *= ((uchar*)mask_keep.data)[j];

        // 5. Threshold and upsample
        cv::Mat binary_mask;
        cv::threshold(mask, binary_mask, mask_threshold, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U);
        if (upsample) {
            cv::Mat upsampled;
            cv::resize(binary_mask, upsampled, cv::Size(iw, ih), 0, 0, cv::INTER_NEAREST);
            result_masks.push_back(upsampled);
        } else {
            result_masks.push_back(binary_mask);
        }
    }
    return result_masks;
}

// Legacy function for backward compatibility - now calls the optimized version
std::vector<std::vector<std::vector<float>>> SegmentationDetector::fast_batch_nms(
    const std::vector<float>& preds,
    const std::vector<int64_t>& shape,  // [batch, features, boxes]
    float conf_thresh, float iou_thresh,
    int max_det, int max_wh,
    const std::vector<int>& classes = {},
    bool agnostic = false,
    int nc = 0
) {
    return non_max_suppression(
        preds, shape, conf_thresh, iou_thresh, classes, agnostic, false,
        max_det, nc, 30000, max_wh, true
    );
}


void SegmentationDetector::postprocess(
        const std::vector<std::vector<float>>& batch_outputs,
        const std::vector<std::vector<int64_t>>& batch_output_shapes,
        const int batch_size,
        std::vector<std::vector<SegmentationResult>>& all_results,
        torch::Tensor& mask_tensor, torch::Tensor& bboxes_tensor,
        std::vector<float>& detection_confidences, std::vector<int>& detection_class_ids, std::vector<std::string>& detection_class_names,
        const std::vector<cv::Mat>& original_images, int height, int width) {

        auto time_start = std::chrono::high_resolution_clock::now();
        std::vector<std::string> class_names = {"class0", "class1"};
        
        const std::vector<float>& detection_output = batch_outputs[0];
        const std::vector<float>& prototype_output = batch_outputs[1];
        const std::vector<int64_t>& detection_shape = batch_output_shapes[0];
        const std::vector<int64_t>& prototype_shape = batch_output_shapes[1];

        // Parse shapes
        const int num_protos = prototype_shape[1];
        const int proto_h = prototype_shape[2];
        const int proto_w = prototype_shape[3];


        // ===== 1. BATCH NMS (same as before) =====
        auto start_time_nms = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<std::vector<float>>> nms_results = fast_batch_nms(
            detection_output,
            detection_shape,
            0.25,
            0.45,
            300,   // max_det
            7680,  // max_wh
            {},    // classes
            false, // agnostic
            2      // nc (number of classes) - hardcoded for segmentation
        );

 
        // ===== 2. BATCH MASK PROCESSING =====
        // Get the first batch results (we only process single images)
        const auto& batch_detections = nms_results[0];
        

        // EXACT ULTRALYTICS WORKFLOW: Process masks for NMS survivors
        // Extract bboxes and mask coefficients from NMS results
        std::vector<cv::Rect2f> survivor_bboxes;
        std::vector<std::vector<float>> survivor_mask_coeffs;
        std::vector<SegmentationResult> pre_mask_results;
        std::vector<SegmentationResult> results;
        
        // Clear the output vectors
        detection_confidences.clear();
        detection_class_ids.clear();
        detection_class_names.clear();
        
        for (const auto& detection : batch_detections) {
            if (detection.size() >= 6 + num_protos) {
                // Extract bbox (xyxy format from NMS)
                float x1 = detection[0];
                float y1 = detection[1];
                float x2 = detection[2];
                float y2 = detection[3];
                float confidence = detection[4];
                int class_id = static_cast<int>(detection[5]);
                
                cv::Rect2f bbox(x1, y1, x2 - x1, y2 - y1);
                survivor_bboxes.push_back(bbox);
                
                // Extract mask coefficients (last 32 values)
                std::vector<float> mask_coeffs;
                for (int i = 6; i < 6 + num_protos; ++i) {
                    if (i < detection.size()) {
                        mask_coeffs.push_back(detection[i]);
                    }
                }
                survivor_mask_coeffs.push_back(mask_coeffs);
                
                // Store the REAL detection data for CSV output
                detection_confidences.push_back(confidence);
                detection_class_ids.push_back(class_id);
                std::string class_name = (class_id < class_names.size()) ? class_names[class_id] : std::to_string(class_id);
                detection_class_names.push_back(class_name);
                
                // Create preliminary result
                SegmentationResult result(bbox, confidence, class_id, class_name);
                pre_mask_results.push_back(result);
            }
        }


        cv::Size img_shape(width, height);  // Current processed image size (img.shape[2:])    

        // Convert survivor_mask_coeffs to torch::Tensor [num_masks, num_protos]
        torch::Tensor masks_in = torch::empty({(int)survivor_mask_coeffs.size(), num_protos}, torch::kFloat32);
        for (size_t i = 0; i < survivor_mask_coeffs.size(); ++i)
            std::memcpy(masks_in[i].data_ptr(), survivor_mask_coeffs[i].data(), num_protos * sizeof(float));

        // Convert prototype_output to torch::Tensor [num_protos, proto_h, proto_w]
        torch::Tensor protos = torch::from_blob((float*)prototype_output.data(),
                                                {num_protos, proto_h, proto_w}, torch::kFloat32).clone();

        // Convert survivor_bboxes to torch::Tensor [num_masks, 4] (xyxy)
        torch::Tensor bboxes = torch::empty({(int)survivor_bboxes.size(), 4}, torch::kFloat32);
        for (size_t i = 0; i < survivor_bboxes.size(); ++i) {
            bboxes[i][0] = survivor_bboxes[i].x;
            bboxes[i][1] = survivor_bboxes[i].y;
            bboxes[i][2] = survivor_bboxes[i].x + survivor_bboxes[i].width;
            bboxes[i][3] = survivor_bboxes[i].y + survivor_bboxes[i].height;
        }

        // Shape for upsample
        std::vector<int64_t> shape_before_upsample = {height, width}; 
        std::vector<int64_t> shape_after_upsample = {original_images[0].rows, original_images[0].cols};
        // Convert tensor data to cv::Mat format for existing function
        std::vector<float> protos_vec(prototype_output.begin(), prototype_output.end());
        auto result_masks_cv = processMaskUltralytics(
            protos_vec,
            prototype_shape,
            survivor_mask_coeffs,
            survivor_bboxes,
            img_shape,
            true              // upsample if needed
        );
        
        // Convert cv::Mat results back to tensors if needed
        torch::Tensor result_masks, result_bboxes;


        // We need to ensure our detection arrays match the final tensor outputs
        int final_detections = result_masks.size(0);
        
        if (final_detections != detection_confidences.size()) {
            // Resize vectors to match the final filtered results
            detection_confidences.resize(final_detections);
            detection_class_ids.resize(final_detections);
            detection_class_names.resize(final_detections);
        }

        // Assign to the output parameters
        mask_tensor = result_masks;
        bboxes_tensor = result_bboxes;
        
        std::cout << "✅ Total Predictions: " << mask_tensor.size(0) << std::endl;
    }

// Simple postprocess function for detect method
std::vector<SegmentationResult> SegmentationDetector::postprocess(
    const std::vector<std::vector<float>>& output_data,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const cv::Rect2d& scale_info,
    int height, int width) {
    
    // For now, return empty results - this is a placeholder implementation
    // The actual implementation would need the full batch processing logic
    std::vector<SegmentationResult> results;
    std::cerr << "⚠️ Simple postprocess not fully implemented - use batch processing version\n";
    return results;
}


// EXACT ULTRALYTICS process_mask implementation
std::vector<cv::Mat> SegmentationDetector::processMaskUltralytics(
    const std::vector<float>& protos,
    const std::vector<int64_t>& proto_shape,
    const std::vector<std::vector<float>>& masks_in,
    const std::vector<cv::Rect2f>& bboxes,
    const cv::Size& img_shape,
    bool upsample) {
    
    std::vector<cv::Mat> result_masks;
    
    if (proto_shape.size() != 4 || masks_in.empty()) {
        return result_masks;
    }
    
    int c = static_cast<int>(proto_shape[1]);    // mask_dim (32)
    int mh = static_cast<int>(proto_shape[2]);   // mask_h (320)
    int mw = static_cast<int>(proto_shape[3]);   // mask_w (320)
    int ih = img_shape.height;                   // input height
    int iw = img_shape.width;                    // input width
    
    // std::cout << "🔧 ULTRALYTICS process_mask: protos(" << c << "," << mh << "," << mw 
    //           << ") -> img(" << iw << "," << ih << ")" << "\n";
    
    // Process each mask
    for (size_t mask_idx = 0; mask_idx < masks_in.size() && mask_idx < bboxes.size(); ++mask_idx) {
        const auto& mask_coeffs = masks_in[mask_idx];
        const auto& bbox = bboxes[mask_idx];
        
        if (mask_coeffs.size() != c) {
            continue;
        }
        
        // Check if coefficients contain meaningful values
        bool has_non_zero = false;
        for (float coeff : mask_coeffs) {
            if (std::abs(coeff) > 1e-6) {
                has_non_zero = true;
                break;
            }
        }
        
        if (!has_non_zero) {
            result_masks.push_back(cv::Mat());  // Add empty mask to maintain index alignment
            continue;
        }
        
        // EXACT ULTRALYTICS: masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
        cv::Mat mask = cv::Mat::zeros(mh, mw, CV_32F);
        
        for (int i = 0; i < c; ++i) {
            int proto_start = i * mh * mw;
            if (proto_start + mh * mw > protos.size()) continue;
            
            cv::Mat proto(mh, mw, CV_32F, (void*)(protos.data() + proto_start));
            cv::Mat proto_copy;
            proto.copyTo(proto_copy);
            
            // Linear combination: mask += coeff * proto
            proto_copy *= mask_coeffs[i];
            mask += proto_copy;
        }
        
        // Apply sigmoid activation
        cv::Mat sigmoid_mask;
        cv::exp(-mask, sigmoid_mask);
        sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);
        
        // EXACT ULTRALYTICS: Calculate ratios
        float width_ratio = static_cast<float>(mw) / static_cast<float>(iw);
        float height_ratio = static_cast<float>(mh) / static_cast<float>(ih);
        
        // EXACT ULTRALYTICS: downsampled_bboxes = bboxes.clone()
        cv::Rect2f downsampled_bbox = bbox;
        downsampled_bbox.x *= width_ratio;           // x1 *= width_ratio
        downsampled_bbox.width *= width_ratio;       // x2 *= width_ratio (width = x2 - x1)
        downsampled_bbox.y *= height_ratio;          // y1 *= height_ratio  
        downsampled_bbox.height *= height_ratio;     // y2 *= height_ratio (height = y2 - y1)
        
        // EXACT ULTRALYTICS: masks = crop_mask(masks, downsampled_bboxes)
        cv::Mat cropped_mask = cropMaskUltralytics(sigmoid_mask, downsampled_bbox);
        
        // Apply proper thresholding and convert to CV_8UC1 for contour detection
        cv::Mat binary_mask;
        cv::threshold(cropped_mask, binary_mask, config_.mask_threshold, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8UC1);
        
        // Resize to input image size if upsampling is requested
        if (upsample) {
            cv::Mat upsampled_mask;
            cv::resize(binary_mask, upsampled_mask, cv::Size(iw, ih), 0, 0, cv::INTER_NEAREST);
            result_masks.push_back(upsampled_mask);
        } else {
            result_masks.push_back(binary_mask);
        }
    }
    return result_masks;
}

// EXACT ULTRALYTICS crop_mask implementation
cv::Mat SegmentationDetector::cropMaskUltralytics(const cv::Mat& mask, const cv::Rect2f& bbox) {
    // _, h, w = masks.shape
    int h = mask.rows;
    int w = mask.cols;
    
    // x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    float x1 = bbox.x;
    float y1 = bbox.y;
    float x2 = bbox.x + bbox.width;
    float y2 = bbox.y + bbox.height;
    
    // r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    // c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
    // return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    
    cv::Mat result = cv::Mat::zeros(h, w, CV_32F);
    
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            // EXACT ULTRALYTICS: (r >= x1) * (r < x2) * (c >= y1) * (c < y2)
            bool r_condition = (col >= x1) && (col < x2);  // r (cols) condition
            bool c_condition = (row >= y1) && (row < y2);  // c (rows) condition
            
            if (r_condition && c_condition) {
                result.at<float>(row, col) = mask.at<float>(row, col);
            }
        }
    }
    
    return result;
}

// EXACT ULTRALYTICS scale_boxes implementation
std::vector<cv::Rect2f> SegmentationDetector::scaleBoxesUltralytics(
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

std::vector<SegmentationResult> SegmentationDetector::detect(const std::string& image_path, float conf, int height, int width) {
    if (!model_loaded_) {
        std::cerr << "❌ Model not loaded!" << "\n";
        return {};
    }
    
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "❌ Failed to load image: " << image_path << "\n";
        return {};
    }
    
    return detect(image, conf);
}

std::vector<SegmentationResult> SegmentationDetector::detect(const cv::Mat& image, float conf, int height, int width) {
    if (!model_loaded_) {
        std::cerr << "❌ Model not loaded!" << "\n";
        return {};
    }
    
    if (conf > 0) {
        config_.confidence_threshold = conf;
    }
    
    cv::Rect2d scale_info;
    double pimgwidth, pimgheight;
    auto input_data = preprocess(image, scale_info, pimgwidth, pimgheight);
    
    // DYNAMIC INPUT SIZE: Use actual processed image dimensions
    // The dynamic ONNX model can handle variable input sizes like 1280x736
    std::vector<int64_t> input_shape = {1, 3, static_cast<int64_t>(pimgheight), static_cast<int64_t>(pimgwidth)};
    
    // Calculate expected tensor size based on actual processed dimensions
    size_t expected_size = 1 * 3 * pimgheight * pimgwidth;
    
    // Verify tensor size matches (should be exact now)
    if (input_data.size() != expected_size) {
        std::cerr << "❌ Tensor size mismatch! Expected: " << expected_size 
                  << ", Got: " << input_data.size() << "\n";
        return {};
    }
    
    // std::cout << "✅ Tensor size matches perfectly - using natural image dimensions" << "\n";
    
    try {
        auto outputs = model_->runInference(input_data, input_shape);
        
        std::vector<std::vector<float>> output_data;
        std::vector<std::vector<int64_t>> output_shapes;
        
        for (auto& output : outputs) {
            auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
            output_shapes.push_back(shape);
            
            float* output_ptr = output.GetTensorMutableData<float>();
            size_t output_size = 1;
            for (auto dim : shape) output_size *= dim;
            
            std::vector<float> output_vector(output_ptr, output_ptr + output_size);
            output_data.push_back(output_vector);
        }
        
        return postprocess(output_data, output_shapes, scale_info, height, width);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Inference failed: " << e.what() << "\n";
        return {};
    }
}

std::string SegmentationDetector::drawAndSaveSegmentation(
    const std::string& image_path, 
    const std::vector<SegmentationResult>& detections, 
    const std::string& output_filename) {
    
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "❌ Failed to load image for visualization" << "\n";
        return "";
    }
    
    cv::Mat result_image = drawSegmentationOverlay(image, detections);
    
    std::string output_path = output_dir_ + "/" + output_filename;
    cv::imwrite(output_path, result_image);
    
    std::cout << "🎨 Segmentation visualization saved: " << output_path << "\n";
    return output_path;
}

cv::Mat SegmentationDetector::drawSegmentationOverlay(
    const cv::Mat& image, 
    const std::vector<SegmentationResult>& detections) {
    
    cv::Mat result = image.clone();
    if (config_.verbose_logging) {
        std::cout << "Drawing " << detections.size() << " detections\n";
    }
    
    for (size_t idx = 0; idx < detections.size(); ++idx) {
        const auto& detection = detections[idx];
        cv::Scalar color = (detection.class_id < class_colors_.size()) ? 
                          class_colors_[detection.class_id] : cv::Scalar(0, 255, 0);
        
        // Prepare label for this detection
        std::string label = detection.class_name + " " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
        
        // ===== SEGMENTATION MASK DRAWING =====

        // Check if we have a valid contour
        if (!detection.segmentation.contour.empty()) {
            // Generate random color for this detection
            cv::RNG rng(idx * 12345); // Use detection index as seed for reproducible colors
            cv::Scalar random_color(
                rng.uniform(0, 256),
                rng.uniform(0, 256),
                rng.uniform(0, 256)
            );
            
            std::vector<std::vector<cv::Point>> contours = {detection.segmentation.contour};
            
            // Create a mask for the filled contour
            cv::Mat mask = cv::Mat::zeros(result.size(), CV_8UC1);
            cv::fillPoly(mask, contours, cv::Scalar(255));
            
            // Create a colored overlay
            cv::Mat colored_overlay = cv::Mat::zeros(result.size(), CV_8UC3);
            colored_overlay.setTo(random_color, mask);
            
            // Apply the overlay with alpha blending
            cv::addWeighted(result, 1.0, colored_overlay, 0.6, 0, result);
            
            // Draw contour lines over the filled area
            cv::drawContours(result, contours, -1, color, 2);  // 2-pixel thick contour lines
            
            // Add class label at the top of bounding box
            std::string label = detection.class_name + " " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
            int baseline;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point text_origin(static_cast<int>(detection.bbox.x), static_cast<int>(detection.bbox.y) - 5);
            
            // Draw label background and text
            cv::rectangle(result, text_origin + cv::Point(0, baseline), 
                         text_origin + cv::Point(text_size.width, -text_size.height), color, -1);
            cv::putText(result, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    return result;
}

std::string SegmentationDetector::saveSegmentationToCSV(
    const std::vector<SegmentationResult>& results, 
    const std::string& output_filename) {
    
    std::string csv_path = output_dir_ + "/" + output_filename;
    std::ofstream file(csv_path);
    
    if (!file.is_open()) {
        std::cerr << "❌ Failed to create CSV file: " << csv_path << "\n";
        return "";
    }
    
    // Write header matching Python format
    file << "box_area,x1,y1,x2,y2,class_id,class_name,confidence,center_x,center_y,width,height,has_mask,mask_area,mask_shape\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        
        // Calculate box area
        float box_area = result.bbox.width * result.bbox.height;
        
        // Calculate box coordinates (x1, y1, x2, y2)
        float x1 = result.bbox.x;
        float y1 = result.bbox.y;
        float x2 = result.bbox.x + result.bbox.width;
        float y2 = result.bbox.y + result.bbox.height;
        
        // Calculate center coordinates
        float center_x = result.bbox.x + result.bbox.width / 2.0f;
        float center_y = result.bbox.y + result.bbox.height / 2.0f;
        
        // Check if mask exists
        bool has_mask = !result.segmentation.mask.empty();
        
        // Get mask info
        std::string mask_shape = "";
        if (has_mask) {
            mask_shape = std::to_string(result.segmentation.mask.rows) + "x" + std::to_string(result.segmentation.mask.cols);
        }
        
        file << std::fixed << std::setprecision(6)
             << box_area << ","
             << x1 << ","
             << y1 << ","
             << x2 << ","
             << y2 << ","
             << result.class_id << ","
             << result.class_name << ","
             << result.confidence << ","
             << center_x << ","
             << center_y << ","
             << result.bbox.width << ","
             << result.bbox.height << ","
             << (has_mask ? "True" : "False") << ","
             << result.segmentation.area << ","
             << mask_shape << "\n";
    }
    
    file.close();
    std::cout << "📄 Segmentation results saved to CSV: " << csv_path << "\n";
    return csv_path;
}

std::string SegmentationDetector::saveSegmentationToJSON(
    const std::vector<SegmentationResult>& results, 
    const std::string& output_filename) {
    
    std::string json_path = output_dir_ + "/" + output_filename;
    std::ofstream file(json_path);
    
    if (!file.is_open()) {
        std::cerr << "❌ Failed to create JSON file: " << json_path << "\n";
        return "";
    }
    
    file << "{\n  \"detections\": [\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        
        file << "    {\n";
        file << "      \"bbox\": {\"x\": " << result.bbox.x << ", \"y\": " << result.bbox.y 
             << ", \"width\": " << result.bbox.width << ", \"height\": " << result.bbox.height << "},\n";
        file << "      \"confidence\": " << result.confidence << ",\n";
        file << "      \"class_id\": " << result.class_id << ",\n";
        file << "      \"class_name\": \"" << result.class_name << "\",\n";
        file << "      \"segmentation\": {\n";
        file << "        \"area\": " << result.segmentation.area << ",\n";
        file << "        \"centroid\": {\"x\": " << result.segmentation.centroid.x 
             << ", \"y\": " << result.segmentation.centroid.y << "},\n";
        file << "        \"contour_points\": " << result.segmentation.contour.size() << "\n";
        file << "      }\n";
        file << "    }";
        
        if (i < results.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ]\n}\n";
    file.close();
    
    std::cout << "📄 Segmentation results saved to JSON: " << json_path << "\n";
    return json_path;
}

std::string SegmentationDetector::saveMasksAsPNG(
    const std::vector<SegmentationResult>& results,
    const std::string& base_filename) {
    
    std::string masks_dir = output_dir_ + "/masks";
    
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].segmentation.mask.empty()) {
            std::string mask_filename = masks_dir + "/" + base_filename + "_" + 
                                      std::to_string(i) + "_" + results[i].class_name + ".png";
            cv::imwrite(mask_filename, results[i].segmentation.mask);
        }
    }
    
    std::cout << "🎭 Individual masks saved to: " << masks_dir << "\n";
    return masks_dir;
}

void SegmentationDetector::analyzeModelOutput(const std::string& test_image_path) {
    if (!model_loaded_) {
        std::cout << "❌ Model not loaded for analysis" << "\n";
        return;
    }
    
    cv::Mat image = cv::imread(test_image_path);
    if (image.empty()) {
        std::cout << "❌ Failed to load test image: " << test_image_path << "\n";
        return;
    }
    
    cv::Rect2d scale_info;
    double pimgwidth, pimgheight;
    auto input_data = preprocess(image, scale_info, pimgwidth, pimgheight);
    
    const auto& config = Config::get();
    std::vector<int64_t> input_shape = {1, 3, static_cast<int64_t>(pimgheight), static_cast<int64_t>(pimgwidth)};
    
    try {
        auto outputs = model_->runInference(input_data, input_shape);
        
        std::cout << "\n🔍 Model Output Analysis:" << "\n";
        std::cout << "   Number of outputs: " << outputs.size() << "\n";
        
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "   Output " << i << " shape: ";
            for (auto dim : shape) std::cout << dim << " ";
            std::cout << "\n";
            
            if (i < model_->getOutputNames().size()) {
                std::cout << "   Output " << i << " name: " << model_->getOutputNames()[i] << "\n";
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Analysis failed: " << e.what() << "\n";
    }
}

void SegmentationDetector::printModelInfo() const {
    if (!model_loaded_) {
        std::cout << "❌ Model not loaded" << "\n";
        return;
    }
    
    std::cout << "\n📋 Segmentation Model Information:" << "\n";
    std::cout << "   Model path: " << model_path_ << "\n";
    std::cout << "   Input size: " << model_->getInputSize() << "\n";
    std::cout << "   Output directory: " << output_dir_ << "\n";
    std::cout << "   Number of classes: " << class_names_.size() << "\n";
    std::cout << "   Confidence threshold: " << config_.confidence_threshold << "\n";
    std::cout << "   NMS threshold: " << config_.nms_threshold << "\n";
    std::cout << "   Mask threshold: " << config_.mask_threshold << "\n";
}

std::pair<std::vector<SegmentationResult>, std::string> SegmentationDetector::processImage(
    const std::string& image_name,
    bool save_csv,
    bool save_masks,
    float conf) {
    
    auto results = detect(image_name, conf);
    
    std::string output_info = "Processed " + std::to_string(results.size()) + " segmentations";
    
    if (save_csv) {
        // Extract base filename from image path
        std::string base_name = image_name;
        size_t last_slash = base_name.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            base_name = base_name.substr(last_slash + 1);
        }
        size_t last_dot = base_name.find_last_of(".");
        if (last_dot != std::string::npos) {
            base_name = base_name.substr(0, last_dot);
        }
        
        std::string csv_filename = base_name + "_SegmentationDetection_cpp.csv";
        saveSegmentationToCSV(results, csv_filename);
    }
    
    if (save_masks) {
        saveMasksAsPNG(results);
    }
    
    drawAndSaveSegmentation(image_name, results);
    
    return {results, output_info};
}

// ============================================================================
// Segmentation Analysis Utilities Implementation
// ============================================================================

namespace SegmentationAnalysis {

double calculateIoU(const cv::Mat& mask1, const cv::Mat& mask2) {
    if (mask1.size() != mask2.size()) {
        return 0.0;
    }
    
    cv::Mat intersection, union_mat;
    cv::bitwise_and(mask1, mask2, intersection);
    cv::bitwise_or(mask1, mask2, union_mat);
    
    double intersection_area = cv::countNonZero(intersection);
    double union_area = cv::countNonZero(union_mat);
    
    return (union_area > 0) ? intersection_area / union_area : 0.0;
}

double calculateMaskArea(const cv::Mat& mask) {
    return cv::countNonZero(mask);
}

cv::Point2f calculateCentroid(const cv::Mat& mask) {
    cv::Moments moments = cv::moments(mask);
    if (moments.m00 > 0) {
        return cv::Point2f(
            static_cast<float>(moments.m10 / moments.m00),
            static_cast<float>(moments.m01 / moments.m00)
        );
    }
    return cv::Point2f(0, 0);
}

std::vector<cv::Point> extractLargestContour(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    // Use CHAIN_APPROX_NONE to preserve all contour points for detailed segmentation
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    
    if (contours.empty()) {
        return std::vector<cv::Point>();
    }
    
    size_t largest_idx = 0;
    double largest_area = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > largest_area) {
            largest_area = area;
            largest_idx = i;
        }
    }
    
    return contours[largest_idx];
}

double calculateContourArea(const std::vector<cv::Point>& contour) {
    return cv::contourArea(contour);
}

double calculateContourPerimeter(const std::vector<cv::Point>& contour) {
    return cv::arcLength(contour, true);
}

std::vector<cv::Point> simplifyContour(const std::vector<cv::Point>& contour, double epsilon) {
    std::vector<cv::Point> simplified;
    cv::approxPolyDP(contour, simplified, epsilon, true);
    return simplified;
}

bool isPointInMask(const cv::Point& point, const cv::Mat& mask) {
    if (point.x < 0 || point.y < 0 || point.x >= mask.cols || point.y >= mask.rows) {
        return false;
    }
    return mask.at<uchar>(point.y, point.x) > 0;
}

cv::Mat createColoredMask(const cv::Mat& mask, const cv::Scalar& color) {
    cv::Mat colored_mask = cv::Mat::zeros(mask.size(), CV_8UC3);
    colored_mask.setTo(color, mask);
    return colored_mask;
}

cv::Mat mergeMasks(const std::vector<cv::Mat>& masks, const std::vector<cv::Scalar>& colors) {
    if (masks.empty()) {
        return cv::Mat();
    }
    
    cv::Mat merged = cv::Mat::zeros(masks[0].size(), CV_8UC3);
    
    for (size_t i = 0; i < masks.size() && i < colors.size(); ++i) {
        cv::Mat colored = createColoredMask(masks[i], colors[i]);
        cv::add(merged, colored, merged, masks[i]);
    }
    
    return merged;
}

} // namespace SegmentationAnalysis 

// Add NMS function before the postprocess function
std::vector<SegmentationResult> SegmentationDetector::applyNMS(
    const std::vector<SegmentationResult>& detections, 
    float nms_threshold) {
    
    if (detections.empty()) return {};
    
    // Sort detections by confidence (highest first)
    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&detections](size_t a, size_t b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<SegmentationResult> nms_results;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (suppressed[idx]) continue;
        
        nms_results.push_back(detections[idx]);
        
        // Suppress overlapping detections
        for (size_t j = i + 1; j < indices.size(); ++j) {
            size_t other_idx = indices[j];
            if (suppressed[other_idx]) continue;
            
            float iou = calculateIoU(detections[idx].bbox, detections[other_idx].bbox);
            if (iou > nms_threshold) {
                suppressed[other_idx] = true;
            }
        }
    }
    
    return nms_results;
}

// Helper function to calculate IoU between two bounding boxes
float SegmentationDetector::calculateIoU(const cv::Rect2f& box1, const cv::Rect2f& box2) {
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection;
    
    return (union_area > 0) ? intersection / union_area : 0.0f;
}

cv::Mat SegmentationDetector::cropMask(const cv::Mat& mask, const cv::Rect2f& bbox) {
    // Convert bbox to mask coordinates
    cv::Rect crop_rect(
        static_cast<int>(std::max(0.0f, bbox.x)),
        static_cast<int>(std::max(0.0f, bbox.y)),
        static_cast<int>(std::min(static_cast<float>(mask.cols - bbox.x), bbox.width)),
        static_cast<int>(std::min(static_cast<float>(mask.rows - bbox.y), bbox.height))
    );
    
    if (crop_rect.width <= 0 || crop_rect.height <= 0) {
        return cv::Mat();
    }
    
    return mask(crop_rect).clone();
}

std::vector<cv::Point> SegmentationDetector::extractContour(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    // Use CHAIN_APPROX_NONE to preserve all contour points for detailed segmentation
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    
    if (contours.empty()) {
        return std::vector<cv::Point>();
    }
    
    // Return largest contour (most detailed)
    size_t largest_idx = 0;
    double largest_area = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > largest_area) {
            largest_area = area;
            largest_idx = i;
        }
    }

    return contours[largest_idx];
} 

// Add this function before postprocess function (around line 820)

// Helper function to scale contour coordinates from processed image space to original image space
std::vector<cv::Point> SegmentationDetector::scaleContourUltralytics(
    const std::vector<cv::Point>& contour,
    const cv::Size& img1_shape,     // processed image size
    const cv::Size& img0_shape,     // original image size
    const cv::Rect2d& scale_info) {
    
    if (contour.empty()) {
        return contour;
    }
    
    // Calculate same transformation parameters as scaleBoxesUltralytics
    float gain = std::min(
        static_cast<float>(img1_shape.height) / static_cast<float>(img0_shape.height),
        static_cast<float>(img1_shape.width) / static_cast<float>(img0_shape.width)
    );
    
    float pad_w = std::round((static_cast<float>(img1_shape.width) - static_cast<float>(img0_shape.width) * gain) / 2.0f - 0.1f);
    float pad_h = std::round((static_cast<float>(img1_shape.height) - static_cast<float>(img0_shape.height) * gain) / 2.0f - 0.1f);
    

    std::vector<cv::Point> scaled_contour;
    scaled_contour.reserve(contour.size());
    
    for (const auto& point : contour) {
        // Apply inverse transformation: same as bounding box scaling but for points
        float x = static_cast<float>(point.x);
        float y = static_cast<float>(point.y);
        
        // Remove padding
        x -= pad_w;
        y -= pad_h;
        
        // Scale back to original size
        x /= gain;
        y /= gain;
        
        // Clip to original image bounds
        x = std::max(0.0f, std::min(x, static_cast<float>(img0_shape.width)));
        y = std::max(0.0f, std::min(y, static_cast<float>(img0_shape.height)));
        
        scaled_contour.emplace_back(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
    }
    
    // std::cout << "(" << scaled_contour[0].x << "," << scaled_contour[0].y << ")" << "\n";
    
    return scaled_contour;
}