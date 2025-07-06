// Using precompiled header - all common headers are included
#include "BatchSegmentationProcessor.hpp"

// Additional specific includes not in PCH
#include <torch/types.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <opencv2/core/ocl.hpp>


// Helper: Crop mask to bounding box
cv::Mat crop_mask(const cv::Mat& mask, const cv::Rect2f& bbox)
{
    // mask: (mh, mw), bbox: [x, y, width, height] (float)
    int mh = mask.rows, mw = mask.cols;
    float x1 = bbox.x, y1 = bbox.y, x2 = bbox.x + bbox.width, y2 = bbox.y + bbox.height;

    // Create meshgrid for column and row indices
    cv::Mat col_mat(mh, mw, CV_32F);
    cv::Mat row_mat(mh, mw, CV_32F);
    for (int y = 0; y < mh; ++y) {
        for (int x = 0; x < mw; ++x) {
            col_mat.at<float>(y, x) = static_cast<float>(x);
            row_mat.at<float>(y, x) = static_cast<float>(y);
        }
    }

    // Build the boolean mask as in numpy
    cv::Mat mask1, mask2, mask3, mask4;
    cv::compare(col_mat, cv::Scalar(x1), mask1, cv::CMP_GE);
    cv::compare(col_mat, cv::Scalar(x2), mask2, cv::CMP_LT);
    cv::compare(row_mat, cv::Scalar(y1), mask3, cv::CMP_GE);
    cv::compare(row_mat, cv::Scalar(y2), mask4, cv::CMP_LT);
    
    cv::Mat mask_keep;
    cv::bitwise_and(mask1, mask2, mask_keep);
    cv::bitwise_and(mask_keep, mask3, mask_keep);
    cv::bitwise_and(mask_keep, mask4, mask_keep);

    // Convert mask_keep to float for multiplication
    cv::Mat mask_keep_float;
    mask_keep.convertTo(mask_keep_float, mask.type(), 1.0/255.0);

    cv::Mat cropped;
    cv::multiply(mask, mask_keep_float, cropped);

    return cropped;
}

// Overload for cv::Mat bbox (for backward compatibility)
cv::Mat crop_mask(const cv::Mat& mask, const cv::Mat& box)
{
    float x1 = box.at<float>(0,0), y1 = box.at<float>(0,1), x2 = box.at<float>(0,2), y2 = box.at<float>(0,3);
    cv::Rect2f bbox(x1, y1, x2 - x1, y2 - y1);
    return crop_mask(mask, bbox);
}

// Utility: Convert [center_x, center_y, w, h] -> [x1, y1, x2, y2]
inline void xywh2xyxy(std::vector<float>& box) {
    float x = box[0], y = box[1], w = box[2], h = box[3];
    box[0] = x - w/2.0f;
    box[1] = y - h/2.0f;
    box[2] = x + w/2.0f;
    box[3] = y + h/2.0f;
}

// Fast C++ implementation of non_max_suppression based on Python YOLOv8 code
std::vector<std::vector<std::vector<float>>> non_max_suppression_cpp(
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


// Legacy function for backward compatibility - now calls the optimized version
std::vector<std::vector<std::vector<float>>> fast_batch_nms(
    const std::vector<float>& preds,
    const std::vector<int64_t>& shape,  // [batch, features, boxes]
    float conf_thresh, float iou_thresh,
    int max_det, int max_wh,
    const std::vector<int>& classes = {},
    bool agnostic = false,
    int nc = 0
) {
    return non_max_suppression_cpp(
        preds, shape, conf_thresh, iou_thresh, classes, agnostic, false,
        max_det, nc, 30000, max_wh, true
    );
}

// Enhanced Logger class for detailed timing
BatchLogger::BatchLogger() {
    start_time_ = std::chrono::high_resolution_clock::now();
    last_checkpoint_ = start_time_;
}

void BatchLogger::checkpoint(const std::string& operation) {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - last_checkpoint_);
    double ms = duration.count() / 1000.0;
    timing_log_.emplace_back(operation, ms);

    std::cout << "⏱️  " << operation << ": " << std::fixed << std::setprecision(3) 
              << ms << "ms" << std::endl;

    last_checkpoint_ = now;
}

void BatchLogger::logImageTiming(int image_idx, double time_ms, int detections) {
    std::cout << "   📷 Image " << std::setw(3) << (image_idx + 1) << ": " 
              << std::fixed << std::setprecision(3) << time_ms << "ms";
    if (detections >= 0) {
        std::cout << " (" << detections << " detections)";
    }
    std::cout << std::endl;
}

void BatchLogger::printSummary(int total_images) {
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - start_time_);
    double total_ms = total_duration.count() / 1000.0;

    std::cout << "\n📊 DETAILED TIMING SUMMARY\n";
    std::cout << "========================================\n";

    double cumulative = 0;
    for (const auto& [operation, time] : timing_log_) {
        cumulative += time;
        double percentage = (time / total_ms) * 100.0;
        std::cout << "   " << std::setw(25) << std::left << operation << ": " 
                  << std::setw(8) << std::right << std::fixed << std::setprecision(3) << time << "ms "
                  << "(" << std::setprecision(1) << percentage << "%)" << std::endl;
    }

    std::cout << "----------------------------------------\n";
    std::cout << "   " << std::setw(25) << std::left << "TOTAL TIME" << ": " 
              << std::setw(8) << std::right << std::fixed << std::setprecision(3) << total_ms << "ms\n";
    std::cout << "   " << std::setw(25) << std::left << "AVERAGE PER IMAGE" << ": " 
              << std::setw(8) << std::right << std::fixed << std::setprecision(3) << (total_ms / total_images) << "ms\n";
    std::cout << "   " << std::setw(25) << std::left << "THROUGHPUT (FPS)" << ": " 
              << std::setw(8) << std::right << std::fixed << std::setprecision(2) << (1000.0 * total_images / total_ms) << "\n";
    std::cout << "   " << std::setw(25) << std::left << "IMAGES PER MINUTE" << ": " 
              << std::setw(8) << std::right << std::fixed << std::setprecision(1) << (60000.0 * total_images / total_ms) << "\n";
    std::cout << "========================================\n";
}
// Helper: Clip boxes to image boundaries
torch::Tensor clip_boxesTorch(torch::Tensor boxes, const std::vector<int64_t>& shape) {
    boxes.select(1, 0) = boxes.select(1, 0).clamp(0, shape[1]); // x1
    boxes.select(1, 1) = boxes.select(1, 1).clamp(0, shape[0]); // y1
    boxes.select(1, 2) = boxes.select(1, 2).clamp(0, shape[1]); // x2
    boxes.select(1, 3) = boxes.select(1, 3).clamp(0, shape[0]); // y2
    return boxes;
}

// Helper: Scale boxes to original image shape
torch::Tensor scale_boxesTorch(
    const std::vector<int64_t>& img1_shape, // input shape, {h, w}
    torch::Tensor boxes,
    const std::vector<int64_t>& img0_shape) // original shape, {h, w}
{
    float gain = std::min(
        float(img1_shape[0]) / img0_shape[0],
        float(img1_shape[1]) / img0_shape[1]);

    float pad_x = std::round((img1_shape[1] - img0_shape[1] * gain) / 2.0f - 0.1f);
    float pad_y = std::round((img1_shape[0] - img0_shape[0] * gain) / 2.0f - 0.1f);

    boxes.select(1, 0) -= pad_x;
    boxes.select(1, 1) -= pad_y;
    boxes.select(1, 2) -= pad_x;
    boxes.select(1, 3) -= pad_y;
    boxes.slice(1, 0, 4) /= gain;
    return clip_boxesTorch(boxes, img0_shape);
}


std::pair<torch::Tensor, torch::Tensor> processMaskUltralyticsTorch(
    const torch::Tensor& protos,     // [C, mh, mw]
    const torch::Tensor& masks_in,   // [N, C]
    torch::Tensor bboxes,           // [N, 4] - removed const to allow modification
    const std::vector<int64_t>& shape_before_upsample, // {ih, iw}
    const std::vector<int64_t>& shape_after_upsample, // {ih, iw}
    bool upsample)
{
    // Check if CUDA is available and force GPU usage
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, 0);
    } else {
        std::cout << "⚠️  CUDA not available, using CPU" << std::endl;
    }

    // Move all tensors to GPU
    auto protos_gpu = protos.to(device);
    auto masks_in_gpu = masks_in.to(device);
    auto bboxes_gpu = bboxes.to(device);

    int64_t c  = protos_gpu.size(0);
    int64_t mh = protos_gpu.size(1);
    int64_t mw = protos_gpu.size(2);
    int64_t ih = shape_before_upsample[0];
    int64_t iw = shape_before_upsample[1];
    int64_t num_masks = masks_in_gpu.size(0);
    std::vector<int64_t> input_shape = {ih, iw};
    std::vector<int64_t> orig_shape = {shape_after_upsample[0], shape_after_upsample[1]};

    // 1. Mask decode: (N, C) @ (C, mh*mw) --> (N, mh, mw) - ALL ON GPU
    auto protos_flat = protos_gpu.view({c, -1});                  // [C, mh*mw]
    auto masks = torch::matmul(masks_in_gpu, protos_flat)         // [N, mh*mw] - GPU matmul
                    .view({num_masks, mh, mw});                   // [N, mh, mw]

    // 2. Ratios
    float width_ratio = static_cast<float>(mw) / iw;
    float height_ratio = static_cast<float>(mh) / ih;

    // 3. Downsample bboxes for mask crop (clone so original not modified) - ON GPU
    auto downsampled_bboxes = bboxes_gpu.clone();
    downsampled_bboxes.select(1, 0).mul_(width_ratio);  // x1
    downsampled_bboxes.select(1, 2).mul_(width_ratio);  // x2
    downsampled_bboxes.select(1, 1).mul_(height_ratio); // y1
    downsampled_bboxes.select(1, 3).mul_(height_ratio); // y2

    // 4. Crop mask to bbox - GPU operations
    masks = cropMaskLibtorch(masks, downsampled_bboxes);      // [N, mh, mw]

    // 5. Optional upsample to input size - GPU interpolation
    if (upsample) {
        masks = torch::nn::functional::interpolate(
            masks.unsqueeze(1),                               // [N,1,mh,mw]
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{ih, iw})
                .mode(torch::kBilinear)
                .align_corners(false)
        ).squeeze(1);                                         // [N,ih,iw]
    }

    // 6. Binarize (threshold at 0.0) - GPU operation
    masks = masks.gt(0.0f);    

    // 7. Mask filter: keep predictions with nonzero masks - GPU operations
    if (masks.numel() > 0) {
        torch::Tensor keep = masks.sum({1, 2}) > 0;  // [N] - GPU sum
        auto keep_idx = torch::nonzero(keep).squeeze();
        // Filter masks and bboxes - GPU indexing
        masks = masks.index_select(0, keep_idx);
        bboxes_gpu = bboxes_gpu.index_select(0, keep_idx);
    }

    // 8. Scale boxes to original image shape - GPU operations
    if (bboxes_gpu.numel() > 0)
        bboxes_gpu = scale_boxesTorch(input_shape, bboxes_gpu, orig_shape);


    return {masks, bboxes_gpu};
}


torch::Tensor cropMaskLibtorch(const torch::Tensor& masks, const torch::Tensor& boxes) {
    // masks: [N, H, W] - GPU tensor
    // boxes: [N, 4] - GPU tensor
    auto sizes = masks.sizes();
    int64_t N = sizes[0], H = sizes[1], W = sizes[2];

    // Move operations to GPU device
    auto device = masks.device();
    
    // x1, y1, x2, y2: [N, 1, 1] - all on GPU
    auto chunks = boxes.split(1, 1);
    auto x1 = chunks[0].unsqueeze(-1);  // [N,1,1]
    auto y1 = chunks[1].unsqueeze(-1);
    auto x2 = chunks[2].unsqueeze(-1);
    auto y2 = chunks[3].unsqueeze(-1);

    // Create coordinate grids on GPU
    auto r = torch::arange(W, torch::TensorOptions().device(device).dtype(masks.dtype())).view({1, 1, W}); // [1,1,W]
    auto c = torch::arange(H, torch::TensorOptions().device(device).dtype(masks.dtype())).view({1, H, 1}); // [1,H,1]

    // Logical mask: (r >= x1) & (r < x2) & (c >= y1) & (c < y2) - all GPU operations
    auto mask = ((r >= x1) & (r < x2) & (c >= y1) & (c < y2)).to(masks.dtype());

    // Elementwise multiply on GPU
    return masks * mask;
}


TrueBatchSegmentationProcessor::TrueBatchSegmentationProcessor(
    const std::string& model_path, 
    const std::string& output_dir,
    const std::string& provider)
    : detector_(model_path, output_dir, provider),
      model_path_(model_path), output_dir_(output_dir),
      env_(ORT_LOGGING_LEVEL_WARNING, "BatchSegmentation"),
      session_(env_, model_path.c_str(), createSessionOptions(provider)),
      logger_() 
{
    std::filesystem::create_directories(output_dir_);
    extractModelInfo();
    std::cout << "✅ Batch ONNX session initialized for: " << model_path_ << "\n";
    logger_.checkpoint("Model Initialization");
}

Ort::SessionOptions TrueBatchSegmentationProcessor::createSessionOptions(const std::string& provider) {
    Ort::SessionOptions session_options;
    configureProvider(session_options, provider);
    return session_options;
}

void TrueBatchSegmentationProcessor::configureProvider(Ort::SessionOptions& session_options, const std::string& provider) {
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    int num_threads = std::max(1u, std::thread::hardware_concurrency());
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetInterOpNumThreads(std::max(1, int(num_threads / 2)));

    session_options.EnableMemPattern();
    session_options.EnableCpuMemArena();
    session_options.DisableProfiling();
    session_options.SetLogSeverityLevel(3);

    std::string provider_lower = provider;
    std::transform(provider_lower.begin(), provider_lower.end(), provider_lower.begin(), ::tolower);

    if (provider_lower == "cuda" || provider_lower == "cudaexecutionprovider") {
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
        if (cuda_available != available_providers.end()) {
            try {
                // Method 1: Try with minimal CUDA options to avoid shared library loading
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                // Don't set advanced options that might trigger shared library loading
                
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "✅ CUDA batch processing enabled (minimal options)\n";
                return;
            } catch (const std::exception& e) {
                // Method 2: Try the CUDAExecutionProvider string directly
                try {
                    session_options.AppendExecutionProvider("CUDAExecutionProvider");
                    std::cout << "✅ CUDA batch processing enabled (CUDAExecutionProvider)\n";
                    return;
                } catch (const std::exception& e2) {
                    // Method 3: Force disable provider validation and use CUDA directly
                    try {
                        session_options.SetLogSeverityLevel(4); // Suppress warnings
                        OrtCUDAProviderOptions simple_cuda{};
                        simple_cuda.device_id = 0;
                        session_options.AppendExecutionProvider_CUDA(simple_cuda);
                        std::cout << "✅ CUDA batch processing enabled (forced)\n";
                        return;
                    } catch (const std::exception& e3) {
                        std::cout << "⚠️  All CUDA initialization methods failed:\n";
                        std::cout << "    Method 1: " << e.what() << "\n";
                        std::cout << "    Method 2: " << e2.what() << "\n";
                        std::cout << "    Method 3: " << e3.what() << "\n";
                    }
                }
            }
        } else {
            std::cout << "⚠️  CUDAExecutionProvider is not available in this ONNX Runtime build.\n";
        }
    }
    std::cout << "🖥️  Using CPU batch processing\n";
}

void TrueBatchSegmentationProcessor::extractModelInfo() {
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Extract input names
    size_t input_count = session_.GetInputCount();
    for (size_t i = 0; i < input_count; i++) {
        auto input_name = session_.GetInputNameAllocated(i, allocator);
        input_names_.push_back(input_name.get());
    }
    
    // Extract output names
    size_t output_count = session_.GetOutputCount();
    for (size_t i = 0; i < output_count; i++) {
        auto output_name = session_.GetOutputNameAllocated(i, allocator);
        output_names_.push_back(output_name.get());
    }
    
    // Create C-style string arrays
    for (const auto& name : input_names_) {
        input_names_cstr_.push_back(name.c_str());
    }
    for (const auto& name : output_names_) {
        output_names_cstr_.push_back(name.c_str());
    }
}

// Load all images into memory efficiently
std::vector<cv::Mat> TrueBatchSegmentationProcessor::loadAllImages(const std::vector<std::string>& image_paths) {
    std::cout << "📂 Loading " << image_paths.size() << " images into memory...\n";
    
    std::vector<cv::Mat> images;
    images.reserve(image_paths.size());
    
    int loaded_count = 0;
    for (size_t i = 0; i < image_paths.size(); ++i) {
        auto load_start = std::chrono::high_resolution_clock::now();
        
        cv::Mat img = cv::imread(image_paths[i]);
        if (img.empty()) {
            std::cerr << "⚠️  Failed to read " << image_paths[i] << "\n";
            continue;
        }
        images.push_back(img);
        loaded_count++;
        
        auto load_end = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start);
        double load_ms = load_time.count() / 1000.0;
        
        if (i < 10 || i % 10 == 0) {  // Log first 10 and every 10th image
            logger_.logImageTiming(i, load_ms);
        }
    }
    
    std::cout << "✅ Loaded " << loaded_count << "/" << image_paths.size() << " images successfully\n";
    logger_.checkpoint("Image Loading");
    return images;
}


// TRUE BATCH POSTPROCESSING: Use converted Python functions for exact compatibility
void TrueBatchSegmentationProcessor::postprocess(
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
    auto [result_masks, result_bboxes] = processMaskUltralyticsTorch(
        protos,           // [C, mh, mw]
        masks_in,         // [N, C]
        bboxes,           // [N, 4]
        shape_before_upsample,            // {ih, iw}
        shape_after_upsample,            // {ih, iw}
        true              // upsample if needed
    ); // returns [N, ih, iw] (bool mask for each detection)


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


void TrueBatchSegmentationProcessor::processTrueBatch(const std::vector<std::string>& image_paths, int batch_size, float conf_threshold) {
    // Main true batch processing workflow
    std::cout << "\n🎯 STARTING TRUE BATCH PROCESSING WITH DETAILED LOGGING\n";
    std::cout << "=========================================================\n";
    std::cout << "   Total images: " << image_paths.size() << "\n";
    std::cout << "   Confidence threshold: " << conf_threshold << "\n";
    std::cout << "   Warmup enabled: YES (3 runs)\n";
    std::cout << "=========================================================\n";
    
    // Optimize OpenCV settings
    cv::setUseOptimized(true);
    cv::ocl::setUseOpenCL(false);  // Disable OpenCL to avoid conflicts with CUDA

    // Check device availability once
    const bool cuda_available = torch::cuda::is_available();
    const torch::Device device = cuda_available ? torch::kCUDA : torch::kCPU;
    std::cout << "🔧 Using device: " << (cuda_available ? "CUDA" : "CPU") << "\n";

    // Load images once and reuse
    const auto images = loadAllImages(image_paths);
    
    // Constants - avoid recalculation
    int input_h = 1280, input_w = 1280;
    constexpr int stride = 32;
    constexpr bool auto_pad = true;
    constexpr bool center = true;
    constexpr bool fp16 = false;
    // Pre-allocate letterboxed images container
    std::vector<cv::Mat> letterboxed_imgs;
    letterboxed_imgs.reserve(batch_size);
    std::vector<std::vector<float>> batch_outputs;
    std::vector<std::vector<int64_t>> batch_shapes;
    std::vector<std::vector<SegmentationResult>> all_results;
    torch::Tensor im;
    torch::Tensor mask_tensor;
    torch::Tensor bboxes_tensor;
    

    // Postprocessing - add detection data vectors
    std::vector<float> detection_confidences;
    std::vector<int> detection_class_ids;
    std::vector<std::string> detection_class_names;
        
    // Main processing loop
    for (int i = 0; i < 10; ++i) {
        auto start_time_preprocess = std::chrono::high_resolution_clock::now();

        // Clear detection data vectors
        detection_confidences.clear();
        detection_class_ids.clear();
        detection_class_names.clear();

        // Preprocess
        preprocess(images, letterboxed_imgs, im, batch_size, cuda_available, input_h, input_w);

        // Run batch
        runBatchCUDA(im, batch_outputs, batch_shapes, false);

        // Postprocess
        postprocess(batch_outputs, batch_shapes, batch_size, all_results, mask_tensor, bboxes_tensor, 
                    detection_confidences, detection_class_ids, detection_class_names, images, input_h, input_w);


        // Save results and compare with Python version (outside the loop)
        #ifndef NDEBUG
            std::cout << "\n📊 SAVING RESULTS AND COMPARING WITH PYTHON...\n";
            std::cout << "================================================\n";
            saveBatchResults(image_paths, mask_tensor, bboxes_tensor, detection_confidences, detection_class_ids, detection_class_names, input_h, input_w);
        #endif

        auto end_time_preprocess = std::chrono::high_resolution_clock::now();
        auto duration_preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time_preprocess - start_time_preprocess);
        std::cout << "🎯 ***** Preprocess ALL time *****: " << duration_preprocess.count() << "ms\n";
    }


    // Print comprehensive timing summary
    logger_.printSummary(image_paths.size());
}

// CSV comparison function to check if C++ and Python results are equal
bool TrueBatchSegmentationProcessor::compareCSVFiles(const std::string& cpp_csv, const std::string& python_csv, float tolerance) {
    std::ifstream cpp_file(cpp_csv);
    std::ifstream python_file(python_csv);
    
    if (!cpp_file.is_open()) {
        std::cout << "⚠️  Could not open C++ CSV file: " << cpp_csv << std::endl;
        return false;
    }
    
    if (!python_file.is_open()) {
        std::cout << "⚠️  Could not open Python CSV file: " << python_csv << std::endl;
        cpp_file.close();
        return false;
    }
    
    std::string cpp_line, python_line;
    int line_number = 0;
    bool files_equal = true;
    
    std::cout << "📊 Comparing CSV files:\n";
    std::cout << "   C++:    " << cpp_csv << "\n";
    std::cout << "   Python: " << python_csv << "\n";
    
    while (std::getline(cpp_file, cpp_line) && std::getline(python_file, python_line)) {
        line_number++;
        
        // Skip header line
        if (line_number == 1) {
            if (cpp_line != python_line) {
                std::cout << "⚠️  Header mismatch at line " << line_number << ":\n";
                std::cout << "   C++:    " << cpp_line << "\n";
                std::cout << "   Python: " << python_line << "\n";
            }
            continue;
        }
        
        // Parse CSV lines and compare values
        std::vector<std::string> cpp_values = parseCSVLine(cpp_line);
        std::vector<std::string> python_values = parseCSVLine(python_line);
        
        if (cpp_values.size() != python_values.size()) {
            std::cout << "❌ Column count mismatch at line " << line_number << ":\n";
            std::cout << "   C++ columns: " << cpp_values.size() << "\n";
            std::cout << "   Python columns: " << python_values.size() << "\n";
            files_equal = false;
            continue;
        }
        
        // Compare each value
        for (size_t i = 0; i < cpp_values.size(); ++i) {
            if (i < 2) {  // String columns (filename, class_name)
                if (cpp_values[i] != python_values[i]) {
                    std::cout << "❌ String mismatch at line " << line_number << ", column " << i << ":\n";
                    std::cout << "   C++:    '" << cpp_values[i] << "'\n";
                    std::cout << "   Python: '" << python_values[i] << "'\n";
                    files_equal = false;
                }
            } else {  // Numeric columns (confidence, bbox coordinates)
                try {
                    float cpp_val = std::stof(cpp_values[i]);
                    float python_val = std::stof(python_values[i]);
                    float diff = std::abs(cpp_val - python_val);
                    
                    if (diff > tolerance) {
                        std::cout << "❌ Numeric mismatch at line " << line_number << ", column " << i << ":\n";
                        std::cout << "   C++:    " << std::fixed << std::setprecision(6) << cpp_val << "\n";
                        std::cout << "   Python: " << std::fixed << std::setprecision(6) << python_val << "\n";
                        std::cout << "   Diff:   " << std::fixed << std::setprecision(6) << diff << " (tolerance: " << tolerance << ")\n";
                        files_equal = false;
                    }
                } catch (const std::exception& e) {
                    std::cout << "❌ Parse error at line " << line_number << ", column " << i << ":\n";
                    std::cout << "   C++:    '" << cpp_values[i] << "'\n";
                    std::cout << "   Python: '" << python_values[i] << "'\n";
                    std::cout << "   Error:  " << e.what() << "\n";
                    files_equal = false;
                }
            }
        }
    }
    
    // Check if one file has more lines than the other
    bool cpp_has_more = static_cast<bool>(std::getline(cpp_file, cpp_line));
    bool python_has_more = static_cast<bool>(std::getline(python_file, python_line));
    
    if (cpp_has_more || python_has_more) {
        std::cout << "❌ File length mismatch:\n";
        if (cpp_has_more) std::cout << "   C++ file has more lines\n";
        if (python_has_more) std::cout << "   Python file has more lines\n";
        files_equal = false;
    }
    
    cpp_file.close();
    python_file.close();
    
    if (files_equal) {
        std::cout << "✅ CSV files are identical (within tolerance " << tolerance << ")\n";
        std::cout << "   Total lines compared: " << line_number << "\n";
        } else {
        std::cout << "❌ CSV files have differences\n";
    }
    
    return files_equal;
}

// Save all batch results
void TrueBatchSegmentationProcessor::saveBatchResults(const std::vector<std::string>& image_paths,
                        const std::vector<std::vector<SegmentationResult>>& batch_results) {
    
    std::cout << "\n💾 Saving batch results...\n";
    auto start_save = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < std::min(image_paths.size(), batch_results.size()); ++i) {
        if (!batch_results[i].empty()) {
            std::string base_name = "batch_" + std::to_string(i);
            
            // Save CSV and visualization using SegmentationDetector methods
            std::string cpp_csv = base_name + "_segmentation_CPP.csv";
            
            detector_.saveSegmentationToCSV(batch_results[i], cpp_csv);
            //detector_.drawAndSaveSegmentation(image_paths[i], batch_results[i], base_name + "_segmentation_CPP.jpg");
            
            std::cout << "\n📋 Saved C++ results for batch " << i << ":\n";
            std::cout << "   CSV: " << cpp_csv << "\n";
        }
    }
    
    auto end_save = std::chrono::high_resolution_clock::now();
    auto save_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save);
    
    std::cout << "✅ All batch results saved in " << save_time.count() << "ms\n";
}

// Save batch results using tensors from postprocessing
void TrueBatchSegmentationProcessor::saveBatchResults(const std::vector<std::string>& image_paths,
                     const torch::Tensor& mask_tensor, 
                     const torch::Tensor& bboxes_tensor,
                     const std::vector<float>& detection_confidences,
                     const std::vector<int>& detection_class_ids,
                     const std::vector<std::string>& detection_class_names,
                     int height, int width) {
    
    std::cout << "\n💾 Saving batch results from tensors...\n";
    auto start_save = std::chrono::high_resolution_clock::now();
    
    if (mask_tensor.numel() == 0 || bboxes_tensor.numel() == 0) {
        std::cout << "⚠️  No detections to save (empty tensors)\n";
        return;
    }
    
    // Move tensors to CPU for processing
    auto cpu_masks = mask_tensor.to(torch::kCPU);
    auto cpu_bboxes = bboxes_tensor.to(torch::kCPU);
    
    int num_detections = cpu_masks.size(0);
    std::cout << "📊 Processing " << num_detections << " detections\n";
    
    // Validate that we have detection data for all detections
    if (detection_confidences.size() != num_detections || 
        detection_class_ids.size() != num_detections || 
        detection_class_names.size() != num_detections) {
        std::cout << "❌ Mismatch in detection data sizes!\n";
        std::cout << "   Tensors: " << num_detections << " detections\n";
        std::cout << "   Confidences: " << detection_confidences.size() << "\n";
        std::cout << "   Class IDs: " << detection_class_ids.size() << "\n";
        std::cout << "   Class Names: " << detection_class_names.size() << "\n";
        return;
    }
    
    // Create CSV file with proper header
    std::string csv_filename = "output/batch_0_segmentation_CPP.csv";
    std::ofstream csv_file(csv_filename);
    
    if (!csv_file.is_open()) {
        std::cout << "❌ Failed to create CSV file: " << csv_filename << "\n";
        return;
    }
    
    // Write header matching SegmentationDetector format
    csv_file << "box_area,x1,y1,x2,y2,class_id,class_name,confidence,center_x,center_y,width,height,has_mask,mask_area,mask_shape\n";
    
    // Process each detection
    for (int i = 0; i < num_detections; ++i) {
        // Extract bounding box [x1, y1, x2, y2] - assuming bboxes_tensor is [N, 4]
        auto bbox_accessor = cpu_bboxes.accessor<float, 2>();
        float x1 = bbox_accessor[i][0];
        float y1 = bbox_accessor[i][1]; 
        float x2 = bbox_accessor[i][2];
        float y2 = bbox_accessor[i][3];
        
        // Calculate bbox properties
        float width_bbox = x2 - x1;
        float height_bbox = y2 - y1;
        float box_area = width_bbox * height_bbox;
        float center_x = x1 + width_bbox / 2.0f;
        float center_y = y1 + height_bbox / 2.0f;
        
        // Extract mask for this detection [H, W]
        auto mask_slice = cpu_masks[i];  // [H, W]
        auto mask_accessor = mask_slice.accessor<bool, 2>();
        
        // Calculate mask area (number of True pixels)
        int mask_area = 0;
        int mask_h = mask_slice.size(0);
        int mask_w = mask_slice.size(1);
        
        for (int h = 0; h < mask_h; ++h) {
            for (int w = 0; w < mask_w; ++w) {
                if (mask_accessor[h][w]) {
                    mask_area++;
                }
            }
        }
        
        // Use REAL detection data instead of hardcoded values
        float confidence = detection_confidences[i];
        int class_id = detection_class_ids[i];
        std::string class_name = detection_class_names[i];
        bool has_mask = true;
        std::string mask_shape = std::to_string(mask_h) + "x" + std::to_string(mask_w);  // Format like Python: "736x1280"
        
        // Write CSV line with proper formatting and precision
        csv_file << std::fixed << std::setprecision(2);
        csv_file << box_area << ","
                 << x1 << ","
                 << y1 << ","
                 << x2 << ","
                 << y2 << ","
                 << class_id << ","
                 << class_name << ","
                 << std::setprecision(6) << confidence << ","
                 << std::setprecision(2) << center_x << ","
                 << center_y << ","
                 << width_bbox << ","
                 << height_bbox << ","
                 << (has_mask ? "True" : "False") << ","
                 << mask_area << ","
                 << mask_shape << "\n";
    }
    
    csv_file.close();
    std::cout << "✅ Saved CSV: " << csv_filename << " (" << num_detections << " detections)\n";
}

// Helper function to parse CSV line
std::vector<std::string> TrueBatchSegmentationProcessor::parseCSVLine(const std::string& line) {
    std::vector<std::string> values;
    std::stringstream ss(line);
    std::string value;
    
    while (std::getline(ss, value, ',')) {
        // Trim whitespace
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        values.push_back(value);
    }
    
    return values;
}

// Helper function to save detection masks as images
void TrueBatchSegmentationProcessor::saveDetectionMasks(const torch::Tensor& cpu_masks, 
                        const torch::Tensor& cpu_bboxes, 
                        int height, int width) {
    
    int num_detections = cpu_masks.size(0);
    std::cout << "🖼️  Saving " << num_detections << " detection masks...\n";
    
    for (int i = 0; i < std::min(num_detections, 5); ++i) {  // Save first 5 masks only
        // Extract mask [H, W]
        auto mask_slice = cpu_masks[i];
        auto mask_accessor = mask_slice.accessor<bool, 2>();
        
        // Convert to OpenCV Mat
        cv::Mat mask_img(height, width, CV_8UC1, cv::Scalar(0));
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                if (h < mask_slice.size(0) && w < mask_slice.size(1)) {
                    mask_img.at<uint8_t>(h, w) = mask_accessor[h][w] ? 255 : 0;
                }
            }
        }
        
        // Save mask image
        std::string mask_filename = "detection_mask_" + std::to_string(i) + "_CPP.png";
        cv::imwrite(mask_filename, mask_img);
        std::cout << "   Saved: " << mask_filename << "\n";
    }
}


// Collect test images for batch processing
std::vector<std::string> collectTestImages(int batch_size = 32) {
    std::vector<std::string> image_paths;
    // Push back the specific test image 10 times
    for (int i = 0; i < batch_size; ++i) {
        image_paths.push_back("Dataset/input_SegmentDetection/TestImage.png");
    }
    return image_paths;
}