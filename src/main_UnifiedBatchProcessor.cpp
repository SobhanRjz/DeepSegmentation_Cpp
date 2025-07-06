#include "pch.hpp"
#include "Config.hpp"
#include "BatchSegmentationProcessor.hpp"
#include "BatchPointProcessor.hpp"

#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

// Helper function to detect model type based on file name
std::string detectModelType(const std::string& model_path) {
    std::string filename = std::filesystem::path(model_path).filename().string();
    std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
    
    if (filename.find("seg") != std::string::npos || 
        filename.find("segment") != std::string::npos) {
        return "segmentation";
    } else if (filename.find("point") != std::string::npos ||
               filename.find("pose") != std::string::npos ||
               filename.find("keypoint") != std::string::npos) {
        return "point";
    } else {
        // Default fallback - could also check model outputs
        std::cout << "⚠️  Could not detect model type from filename. Please specify manually.\n";
        return "unknown";
    }
}

// Function to prompt user for model type if detection fails
std::string promptModelType() {
    std::cout << "\n🤔 Could not automatically detect model type.\n";
    std::cout << "Please specify the model type:\n";
    std::cout << "1. segmentation (for YOLOv8 segmentation models)\n";
    std::cout << "2. point (for YOLOv8 point/pose detection models)\n";
    std::cout << "Enter choice (1 or 2): ";
    
    std::string choice;
    std::getline(std::cin, choice);
    
    if (choice == "1" || choice == "segmentation") {
        return "segmentation";
    } else if (choice == "2" || choice == "point") {
        return "point";
    } else {
        std::cout << "❌ Invalid choice. Defaulting to segmentation.\n";
        return "segmentation";
    }
}

// Function to collect test images from various directories
std::vector<std::string> collectAllTestImages(const std::string& model_type, int max_images = 10) {
    std::vector<std::string> image_paths;
    
    // Define input directories based on model type
    std::vector<std::string> input_dirs;
    if (model_type == "point") {
        input_dirs = {
            "Dataset/input_PointDetection",
            "Dataset/inputRectangle",
            "YoloPy/input",
            "input"
        };
    } else if (model_type == "segmentation") {
        input_dirs = {
            "Dataset/input_SegmentDetection", 
            "Dataset/inputRectangle",
            "YoloPy/input",
            "input"
        };
    } else {
        input_dirs = {
            "Dataset/inputRectangle",
            "YoloPy/input", 
            "input"
        };
    }
    
    // Search for images in input directories
    for (const auto& dir : input_dirs) {
        if (std::filesystem::exists(dir)) {
            std::cout << "🔍 Searching for images in: " << dir << "\n";
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (entry.path().extension() == ".jpg" || 
                    entry.path().extension() == ".png" ||
                    entry.path().extension() == ".jpeg") {
                    image_paths.push_back(entry.path().string());
                    std::cout << "   Found: " << entry.path().filename() << "\n";
                    if (static_cast<int>(image_paths.size()) >= max_images) {
                        break;
                    }
                }
            }
            if (!image_paths.empty()) break;  // Use first directory with images
        }
    }
    
    if (image_paths.empty()) {
        std::cout << "⚠️  No test images found in any directory\n";
        std::cout << "   Searched directories: ";
        for (const auto& dir : input_dirs) {
            std::cout << dir << " ";
        }
        std::cout << "\n";
    } else {
        std::cout << "✅ Found " << image_paths.size() << " test images for " << model_type << " processing\n";
    }
    
    return image_paths;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "🚀 UNIFIED BATCH PROCESSOR FOR YOLO MODELS\n";
        std::cout << "===========================================\n";
        std::cout << "Automatically detects and processes both segmentation and point detection models\n\n";
        
        // Parse command line arguments or use defaults
        std::string model_path;
        std::string output_dir = "output_batch_optimized";
        std::string provider = "cuda";
        float conf_threshold = 0.45f;
        int batch_size = 1;
        std::string model_type = "auto";
        
        if (argc >= 2) {
            model_path = argv[1];
        } else {
            // Try to find model automatically
            std::vector<std::string> possible_models = {
                "YoloModel/yolov8m_SegmentationDetection_dynamic.torchscript",
                "YoloModel/yolov8m_SegmentationDetection_dynamic.onnx",
                "YoloModel/yolov8n-seg.onnx",
                "YoloModel/yolov8m_PointDetection_dynamic.onnx",
                "YoloModel/yolov8n.onnx"
            };
            
            for (const auto& model : possible_models) {
                if (std::filesystem::exists(model)) {
                    model_path = model;
                    std::cout << "📁 Found model: " << model_path << "\n";
                    break;
                }
            }
            
            if (model_path.empty()) {
                std::cout << "❌ No model found. Please specify model path as first argument.\n";
                std::cout << "Usage: " << argv[0] << " <model_path> [output_dir] [provider] [conf_threshold] [batch_size]\n";
                return -1;
            }
        }
        
        if (argc >= 3) output_dir = argv[2];
        if (argc >= 4) provider = argv[3];
        if (argc >= 5) conf_threshold = std::stof(argv[4]);
        if (argc >= 6) batch_size = std::stoi(argv[5]);
        if (argc >= 7) model_type = argv[6];
        
        // Auto-detect model type if not specified
        if (model_type == "auto") {
            model_type = detectModelType(model_path);
            if (model_type == "unknown") {
                model_type = promptModelType();
            }
        }
        
        std::cout << "📋 Configuration:\n";
        std::cout << "   Model path: " << model_path << "\n";
        std::cout << "   Model type: " << model_type << "\n";
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
        
        // Collect test images based on model type
        auto image_paths = collectAllTestImages(model_type, batch_size);
        if (image_paths.empty()) {
            std::cout << "❌ No test images found. Please add images to input directories.\n";
            return -1;
        }
        
        // Process based on model type
        if (model_type == "segmentation") {
            std::cout << "\n🎯 PROCESSING WITH SEGMENTATION BATCH PROCESSOR\n";
            std::cout << "==============================================\n";
            
            TrueBatchSegmentationProcessor processor(model_path, output_dir, provider);
            processor.processTrueBatch(image_paths, batch_size, conf_threshold);
            
        } else if (model_type == "point") {
            std::cout << "\n🎯 PROCESSING WITH POINT DETECTION BATCH PROCESSOR\n";
            std::cout << "=================================================\n";
            
            TrueBatchPointProcessor processor(model_path, output_dir, provider);
            processor.processTrueBatch(image_paths, batch_size, conf_threshold);
            
        } else {
            std::cout << "❌ Unsupported model type: " << model_type << "\n";
            std::cout << "   Supported types: segmentation, point\n";
            return -1;
        }
        
        std::cout << "\n🎉 UNIFIED BATCH PROCESSING COMPLETED SUCCESSFULLY!\n";
        std::cout << "==================================================\n";
        std::cout << "   Model type: " << model_type << "\n";
        std::cout << "   Images processed: " << std::min(static_cast<int>(image_paths.size()), batch_size) << "\n";
        std::cout << "   Results saved to: " << output_dir << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cout << "❌ Unknown error occurred\n";
        return -1;
    }
}

// Additional helper functions for specific use cases

namespace BatchProcessorUtils {

// Function to run segmentation batch processing programmatically
void runSegmentationBatch(const std::string& model_path, 
                         const std::vector<std::string>& image_paths,
                         const std::string& output_dir = "output_segmentation",
                         const std::string& provider = "cuda",
                         float conf_threshold = 0.25f) {
    
    std::cout << "🎯 Running Segmentation Batch Processing\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Images: " << image_paths.size() << "\n";
    
    TrueBatchSegmentationProcessor processor(model_path, output_dir, provider);
    processor.processTrueBatch(image_paths, image_paths.size(), conf_threshold);
}

// Function to run point detection batch processing programmatically  
void runPointDetectionBatch(const std::string& model_path,
                           const std::vector<std::string>& image_paths, 
                           const std::string& output_dir = "output_point_detection",
                           const std::string& provider = "cuda",
                           float conf_threshold = 0.25f) {
    
    std::cout << "🎯 Running Point Detection Batch Processing\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Images: " << image_paths.size() << "\n";
    
    TrueBatchPointProcessor processor(model_path, output_dir, provider);
    processor.processTrueBatch(image_paths, image_paths.size(), conf_threshold);
}

// Function to auto-detect and run appropriate batch processor
void runAutoBatch(const std::string& model_path,
                 const std::vector<std::string>& image_paths,
                 const std::string& output_dir = "output_auto",
                 const std::string& provider = "cuda", 
                 float conf_threshold = 0.25f) {
    
    std::string model_type = detectModelType(model_path);
    
    if (model_type == "segmentation") {
        runSegmentationBatch(model_path, image_paths, output_dir + "_segmentation", provider, conf_threshold);
    } else if (model_type == "point") {
        runPointDetectionBatch(model_path, image_paths, output_dir + "_point", provider, conf_threshold);
    } else {
        std::cout << "❌ Could not auto-detect model type for: " << model_path << "\n";
        std::cout << "   Please use runSegmentationBatch() or runPointDetectionBatch() directly\n";
    }
}

} // namespace BatchProcessorUtils 