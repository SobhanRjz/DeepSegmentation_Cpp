#ifndef FAST_BATCH_PROCESSOR_HPP
#define FAST_BATCH_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <vector>
#include <memory>

class FastBatchProcessor {
private:
    // Pre-allocated buffers to avoid repeated allocations
    cv::Mat batch_buffer_;
    std::vector<cv::Mat> preprocessed_images_;
    std::vector<float> tensor_data_;
    
    // Thread pool for parallel processing
    size_t num_threads_;
    
    // Configuration
    cv::Size input_size_;
    size_t batch_size_;
    
    // Performance optimizations
    bool use_gpu_;
    cv::cuda::GpuMat gpu_buffer_;
    cv::cuda::Stream stream_;
    
public:
    FastBatchProcessor(const cv::Size& input_size, size_t batch_size, bool use_gpu = true)
        : input_size_(input_size), batch_size_(batch_size), use_gpu_(use_gpu) {
        preprocessed_images_.reserve(batch_size);
        
        num_threads_ = std::thread::hardware_concurrency();
        
        // Calculate tensor size and pre-allocate
        size_t tensor_size = batch_size_ * 3 * input_size_.width * input_size_.height;
        tensor_data_.reserve(tensor_size);
        
        // Initialize GPU resources if available
        if (use_gpu_ && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                gpu_buffer_.create(input_size_.height * batch_size_, input_size_.width, CV_8UC3);
                stream_ = cv::cuda::Stream();
            } catch (const std::exception& e) {
                use_gpu_ = false;
                std::cout << "GPU preprocessing disabled: " << e.what() << std::endl;
            }
        }
    }
    
    cv::Mat preprocessBatchOptimized(const std::vector<cv::Mat>& images) {
        if (use_gpu_ && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            return preprocessBatchGPU(images);
        } else {
            return preprocessBatchCPU(images);
        }
    }
    
private:
    cv::Mat preprocessBatchCPU(const std::vector<cv::Mat>& images) {
        // CPU-optimized preprocessing with parallel processing
        preprocessed_images_.clear();
        preprocessed_images_.reserve(images.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < images.size(); ++i) {
            cv::Mat resized;
            cv::resize(images[i], resized, input_size_, 0, 0, cv::INTER_LINEAR);
            
            #pragma omp critical
            {
                preprocessed_images_.push_back(resized);
            }
        }
        
        cv::Mat batch_blob;
        cv::dnn::blobFromImages(preprocessed_images_, batch_blob, 
                              1.0/255.0, input_size_, cv::Scalar(), true, false, CV_32F);
        return batch_blob;
    }
    
    cv::Mat preprocessBatchGPU(const std::vector<cv::Mat>& images) {
        // GPU-accelerated preprocessing
        std::vector<cv::cuda::GpuMat> gpu_images;
        gpu_images.reserve(images.size());
        
        // Upload to GPU and resize
        for (const auto& img : images) {
            cv::cuda::GpuMat gpu_img;
            gpu_img.upload(img, stream_);
            
            // Resize on GPU using correct CUDA function
            cv::cuda::GpuMat resized;
            cv::cuda::resize(gpu_img, resized, input_size_, 0, 0, cv::INTER_LINEAR, stream_);
            
            gpu_images.push_back(resized);
        }
        
        // Download and create blob
        preprocessed_images_.clear();
        for (auto& gpu_img : gpu_images) {
            cv::Mat cpu_img;
            gpu_img.download(cpu_img, stream_);
            preprocessed_images_.push_back(cpu_img);
        }
        
        stream_.waitForCompletion();
        
        cv::Mat batch_blob;
        cv::dnn::blobFromImages(preprocessed_images_, batch_blob, 
                              1.0/255.0, input_size_, cv::Scalar(), true, false, CV_32F);
        return batch_blob;
    }
    
    cv::Mat preprocessSingleImageOptimized(const cv::Mat& image) {
        if (image.empty()) return cv::Mat();
        
        // Calculate scale to maintain aspect ratio
        double scale = std::min(
            static_cast<double>(input_size_.width) / image.cols,
            static_cast<double>(input_size_.height) / image.rows
        );
        
        // Calculate new dimensions
        int new_width = static_cast<int>(image.cols * scale);
        int new_height = static_cast<int>(image.rows * scale);
        
        // Resize with optimized interpolation
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
        
        // Letterbox padding
        cv::Mat letterboxed = cv::Mat::zeros(input_size_, CV_8UC3);
        int pad_x = (input_size_.width - new_width) / 2;
        int pad_y = (input_size_.height - new_height) / 2;
        
        cv::Rect roi(pad_x, pad_y, new_width, new_height);
        resized.copyTo(letterboxed(roi));
        
        return letterboxed;
    }
};

#endif // FAST_BATCH_PROCESSOR_HPP 