#include "pch.hpp"
#include "Config.hpp"
#include <sstream>

ConfigReader::ConfigReader(const std::string& config_file_path) 
    : config_file_path_(config_file_path) {
    parseConfig();
}

std::string ConfigReader::cleanValue(const std::string& value) {
    std::string cleaned = value;
    
    // Remove leading and trailing whitespace
    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r"));
    cleaned.erase(cleaned.find_last_not_of(" \t\n\r") + 1);
    
    // Remove quotes
    if (cleaned.front() == '"' && cleaned.back() == '"') {
        cleaned = cleaned.substr(1, cleaned.length() - 2);
    }
    
    // Remove comma if present
    if (!cleaned.empty() && cleaned.back() == ',') {
        cleaned.pop_back();
    }
    
    return cleaned;
}

std::string ConfigReader::extractStringValue(const std::string& line) {
    size_t colon_pos = line.find(':');
    if (colon_pos == std::string::npos) return "";
    
    std::string value = line.substr(colon_pos + 1);
    return cleanValue(value);
}

float ConfigReader::extractFloatValue(const std::string& line) {
    std::string value_str = extractStringValue(line);
    try {
        return std::stof(value_str);
    } catch (const std::exception&) {
        return 0.0f;
    }
}

int ConfigReader::extractIntValue(const std::string& line) {
    std::string value_str = extractStringValue(line);
    try {
        return std::stoi(value_str);
    } catch (const std::exception&) {
        return 0;
    }
}

bool ConfigReader::extractBoolValue(const std::string& line) {
    std::string value_str = extractStringValue(line);
    std::transform(value_str.begin(), value_str.end(), value_str.begin(), ::tolower);
    return value_str == "true";
}

void ConfigReader::parseConfig() {
    std::ifstream file(config_file_path_);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_file_path_);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Detection thresholds
        if (line.find("confidence_threshold") != std::string::npos) {
            config_.confidence_threshold = extractFloatValue(line);
        } else if (line.find("score_threshold") != std::string::npos) {
            config_.score_threshold = extractFloatValue(line);
        } else if (line.find("nms_threshold") != std::string::npos) {
            config_.nms_threshold = extractFloatValue(line);
        }
        // Input dimensions
        else if (line.find("\"width\"") != std::string::npos) {
            config_.input_width = extractIntValue(line);
        } else if (line.find("\"height\"") != std::string::npos) {
            config_.input_height = extractIntValue(line);
        }
        // Model settings
        else if (line.find("default_model_name") != std::string::npos) {
            config_.default_model_name = extractStringValue(line);
        } else if (line.find("onnx_model_name") != std::string::npos) {
            config_.onnx_model_name = extractStringValue(line);
        }
        // Paths
        else if (line.find("input_dir") != std::string::npos) {
            config_.input_dir = extractStringValue(line);
        } else if (line.find("output_dir") != std::string::npos) {
            config_.output_dir = extractStringValue(line);
        } else if (line.find("model_dir") != std::string::npos) {
            config_.model_dir = extractStringValue(line);
        }
        // Processing options
        else if (line.find("save_csv") != std::string::npos) {
            config_.save_csv = extractBoolValue(line);
        } else if (line.find("save_visualizations") != std::string::npos) {
            config_.save_visualizations = extractBoolValue(line);
        } else if (line.find("verbose_logging") != std::string::npos) {
            config_.verbose_logging = extractBoolValue(line);
        }
        // Batch processing settings
        else if (line.find("batch_size") != std::string::npos) {
            config_.batch_size = extractIntValue(line);
        } else if (line.find("max_test_images") != std::string::npos) {
            config_.max_test_images = extractIntValue(line);
        } else if (line.find("execution_provider") != std::string::npos) {
            config_.execution_provider = extractStringValue(line);
        }
    }
    
    file.close();
}

void ConfigReader::reload() {
    parseConfig();
}

void ConfigReader::printConfig() const {
    std::cout << "=== YOLO Configuration ===" << std::endl;
    std::cout << "Detection Thresholds:" << std::endl;
    std::cout << "  Confidence: " << config_.confidence_threshold << std::endl;
    std::cout << "  Score: " << config_.score_threshold << std::endl;
    std::cout << "  NMS: " << config_.nms_threshold << std::endl;
    
    std::cout << "Input Dimensions:" << std::endl;
    std::cout << "  Width: " << config_.input_width << std::endl;
    std::cout << "  Height: " << config_.input_height << std::endl;
    
    std::cout << "Model Settings:" << std::endl;
    std::cout << "  Default Model: " << config_.default_model_name << std::endl;
    std::cout << "  ONNX Model: " << config_.onnx_model_name << std::endl;
    
    std::cout << "Paths:" << std::endl;
    std::cout << "  Input Dir: " << config_.input_dir << std::endl;
    std::cout << "  Output Dir: " << config_.output_dir << std::endl;
    std::cout << "  Model Dir: " << config_.model_dir << std::endl;
    
    std::cout << "Processing Options:" << std::endl;
    std::cout << "  Save CSV: " << (config_.save_csv ? "true" : "false") << std::endl;
    std::cout << "  Save Visualizations: " << (config_.save_visualizations ? "true" : "false") << std::endl;
    std::cout << "  Verbose Logging: " << (config_.verbose_logging ? "true" : "false") << std::endl;
    
    std::cout << "Batch Processing:" << std::endl;
    std::cout << "  Batch Size: " << config_.batch_size << std::endl;
    std::cout << "  Max Test Images: " << config_.max_test_images << std::endl;
    std::cout << "  Execution Provider: " << config_.execution_provider << std::endl;
    std::cout << "=========================" << std::endl;
}

ConfigReader& ConfigReader::getInstance(const std::string& config_file_path) {
    static ConfigReader instance(config_file_path);
    return instance;
}

// Global convenience functions implementation
namespace Config {
    const YoloConfig& get() {
        return ConfigReader::getInstance().getConfig();
    }
    
    void reload() {
        ConfigReader::getInstance().reload();
    }
    
    void print() {
        ConfigReader::getInstance().printConfig();
    }
} 