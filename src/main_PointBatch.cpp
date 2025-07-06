#include "pch.hpp"
#include "BatchPointProcessor.hpp"
#include "Config.hpp"

#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

int main(int argc, char* argv[]) {
    try {
        std::cout << "🎯 POINT DETECTION BATCH PROCESSOR\n";
        std::cout << "==================================\n";
        std::cout << "High-performance batch processing for YOLO point detection models\n\n";
        
        // Parse command line arguments or use defaults
        std::string model_path;
        std::string output_dir = "output_point_batch";
        std::string provider = "cuda";
        float conf_threshold = 0.25f;
        int batch_size = 1;
        
        if (argc >= 2) {
            model_path = argv[1];
        } else {
            
            // Try to find point detection model automatically
            std::vector<std::string> possible_models = {
                "output/yolov8m_PointDetection_dynamic_dynamic.onnx",
                "YoloModel/yolov8n_point.onnx",
                "YoloModel/yolov8s_point.onnx",
                "YoloModel/yolov8m_point.onnx",
                "YoloModel/yolov8l_point.onnx",
                "YoloModel/yolov8x_point.onnx"
            };
            
            for (const auto& model : possible_models) {
                if (std::filesystem::exists(model)) {
                    model_path = model;
                    std::cout << "📁 Found point detection model: " << model_path << "\n";
                    break;
                }
            }
            
            if (model_path.empty()) {
                std::cout << "❌ No point detection model found. Please specify model path as first argument.\n";
                std::cout << "Usage: " << argv[0] << " <model_path> [output_dir] [provider] [conf_threshold] [batch_size]\n";
                std::cout << "\nExample models to use:\n";
                std::cout << "  YoloModel/yolov8m_PointDetection_dynamic.onnx\n";
                std::cout << "  YoloModel/yolov8n_point.onnx\n";
                return -1;
            }
        }
        
        if (argc >= 3) output_dir = argv[2];
        if (argc >= 4) provider = argv[3];
        if (argc >= 5) conf_threshold = std::stof(argv[4]);
        if (argc >= 6) batch_size = std::stoi(argv[5]);
        
        std::cout << "📋 Point Detection Configuration:\n";
        std::cout << "   Model path: " << model_path << "\n";
        std::cout << "   Output directory: " << output_dir << "\n";
        std::cout << "   Execution provider: " << provider << "\n";
        std::cout << "   Confidence threshold: " << conf_threshold << "\n";
        std::cout << "   Batch size: " << batch_size << "\n\n";
        
        // Check if model file exists
        if (!std::filesystem::exists(model_path)) {
            std::cout << "❌ Model file not found: " << model_path << "\n";
            return -1;
        }
        
        // Create output directory
        std::filesystem::create_directories(output_dir);
        
        // Collect test images for point detection
        auto image_paths = collectTestImages(batch_size);

        
        std::cout << "🎯 STARTING POINT DETECTION BATCH PROCESSING\n";
        std::cout << "============================================\n";
        
        // Create and run point batch processor
        TrueBatchPointProcessor processor(model_path, output_dir, provider);
        processor.processTrueBatch(image_paths, batch_size, conf_threshold);
        
        std::cout << "\n🎉 POINT DETECTION BATCH PROCESSING COMPLETED!\n";
        std::cout << "==============================================\n";
        std::cout << "   Images processed: " << std::min(static_cast<int>(image_paths.size()), batch_size) << "\n";
        std::cout << "   Results saved to: " << output_dir << "\n";
        std::cout << "   Check CSV files for detected points\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cout << "❌ Unknown error occurred in point detection batch processing\n";
        return -1;
    }
}

// Additional utility functions for point detection batch processing

namespace PointBatchUtils {

// Function to process a specific directory of images
void processPointDetectionDirectory(const std::string& model_path,
                                   const std::string& input_dir,
                                   const std::string& output_dir = "output_point_batch",
                                   const std::string& provider = "cuda",
                                   float conf_threshold = 0.25f) {
    
    std::cout << "🎯 Processing Point Detection Directory\n";
    std::cout << "Input: " << input_dir << "\n";
    std::cout << "Model: " << model_path << "\n";
    
    // Collect all images from directory
    std::vector<std::string> image_paths;
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".jpg" || 
            entry.path().extension() == ".png" ||
            entry.path().extension() == ".jpeg") {
            image_paths.push_back(entry.path().string());
        }
    }
    
    if (image_paths.empty()) {
        std::cout << "❌ No images found in directory: " << input_dir << "\n";
        return;
    }
    
    TrueBatchPointProcessor processor(model_path, output_dir, provider);
    processor.processTrueBatch(image_paths, image_paths.size(), conf_threshold);
}

// Function to compare point detection results with Python implementation
void compareWithPython(const std::string& cpp_csv, const std::string& python_csv, float tolerance = 2.0f) {
    std::cout << "🔍 Comparing C++ and Python point detection results\n";
    std::cout << "C++ CSV: " << cpp_csv << "\n";
    std::cout << "Python CSV: " << python_csv << "\n";
    std::cout << "Tolerance: " << tolerance << " pixels\n";
    
    // This would implement detailed comparison logic
    // For now, just placeholder
    if (std::filesystem::exists(cpp_csv) && std::filesystem::exists(python_csv)) {
        std::cout << "✅ Both files exist, comparison would proceed\n";
    } else {
        std::cout << "❌ One or both comparison files missing\n";
    }
}

} // namespace PointBatchUtils 