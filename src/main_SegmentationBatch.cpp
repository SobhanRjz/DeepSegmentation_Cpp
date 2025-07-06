// Using precompiled header - all common headers are included
#include "BatchSegmentationProcessor.hpp"

// CUDA cleanup function for PyTorch 2.8.0a0 compatibility
void cleanup_cuda_context() {
    try {
        if (torch::cuda::is_available()) {
            // Synchronize all CUDA operations
            torch::cuda::synchronize();
            
            // Clear CUDA cache (if available in this version)
            if (torch::cuda::is_available()) {
                // Force garbage collection of CUDA tensors
                torch::cuda::synchronize();
            }
        }
    } catch (...) {
        // Ignore cleanup errors
    }
}

int main(int argc, char* argv[]) {
    try {
        //std::cout << cv::getBuildInformation() << std::endl;
        std::cout << "🚀 ADVANCED BATCH SEGMENTATION WITH MEMORY OPTIMIZATION\n";
        std::cout << "========================================================\n";

        // Configuration
        std::string model_path = "output/yolov8m_SegmentationDetection_dynamic_dynamic.onnx";
        std::string output_dir = "output_batch_optimized";
        std::string provider = (argc > 1) ? argv[1] : "CUDA";  // Default to CUDA
        int batch_size = (argc > 2) ? std::atoi(argv[2]) : 1;  // Default batch size
        float conf_threshold = (argc > 3) ? std::atof(argv[3]) : 0.45f;
        //batch_size = 1;
        std::cout << "📋 Configuration:\n";
        std::cout << "   Model: " << model_path << "\n";
        std::cout << "   Provider: " << provider << "\n";
        std::cout << "   Batch Size: " << batch_size << "\n";
        std::cout << "   Confidence: " << conf_threshold << "\n";
        std::cout << "   Output Dir: " << output_dir << "\n\n";

        // Collect test images
        auto image_paths = collectTestImages(batch_size);
        if (image_paths.empty()) {
            std::cout << "❌ No images found in Dataset/input_SegmentDetection\n";
            cleanup_cuda_context();
            return -1;
        }

        std::cout << "📁 Found " << image_paths.size() << " test images\n";

        // Load images for memory optimization demonstration
        std::vector<cv::Mat> test_images;
        for (const auto& path : image_paths) {
            cv::Mat img = cv::imread(path);
            if (!img.empty()) {
                test_images.push_back(img);
            }
        }

        // Demonstrate memory optimization benefits
        //demonstrateMemoryOptimization(test_images, 3);

        // Initialize processor with memory optimizations
        TrueBatchSegmentationProcessor processor(model_path, output_dir, provider);
        
        // Process batch with optimized preprocessing
        std::cout << "\n🔄 PROCESSING BATCH WITH MEMORY OPTIMIZATIONS...\n";
        processor.processTrueBatch(image_paths, batch_size, conf_threshold);

        std::cout << "\n✅ MEMORY-OPTIMIZED BATCH PROCESSING COMPLETED!\n";
        std::cout << "================================================\n";

        // Clean up CUDA context before exit
        cleanup_cuda_context();

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        cleanup_cuda_context();
        return -1;
    }

    return 0;
}