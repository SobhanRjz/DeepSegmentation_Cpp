#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

/**
 * @brief Configuration structure to hold YOLO detection parameters
 */
struct YoloConfig {
    // Detection thresholds
    float confidence_threshold;
    float score_threshold;
    float nms_threshold;
    
    // Input dimensions
    int input_width;
    int input_height;
    
    // Model settings
    std::string default_model_name;
    std::string onnx_model_name;
    
    // Paths
    std::string input_dir;
    std::string output_dir;
    std::string model_dir;
    
    // Processing options
    bool save_csv;
    bool save_visualizations;
    bool verbose_logging;
    
    // Batch processing settings
    int batch_size;
    int max_test_images;
    std::string execution_provider;
};

/**
 * @brief Simple JSON parser for configuration files
 * 
 * This is a lightweight JSON parser specifically designed for our config file.
 * It doesn't require external dependencies like nlohmann/json.
 */
class ConfigReader {
private:
    std::string config_file_path_;
    YoloConfig config_;
    
    /**
     * @brief Remove whitespace and quotes from a string value
     */
    std::string cleanValue(const std::string& value);
    
    /**
     * @brief Extract string value from JSON line
     */
    std::string extractStringValue(const std::string& line);
    
    /**
     * @brief Extract float value from JSON line
     */
    float extractFloatValue(const std::string& line);
    
    /**
     * @brief Extract integer value from JSON line
     */
    int extractIntValue(const std::string& line);
    
    /**
     * @brief Extract boolean value from JSON line
     */
    bool extractBoolValue(const std::string& line);
    
    /**
     * @brief Parse the configuration file
     */
    void parseConfig();

public:
    /**
     * @brief Constructor
     * @param config_file_path Path to the JSON configuration file
     */
    explicit ConfigReader(const std::string& config_file_path = "config.json");
    
    /**
     * @brief Get the configuration structure
     * @return Reference to the configuration
     */
    const YoloConfig& getConfig() const { return config_; }
    
    /**
     * @brief Reload configuration from file
     */
    void reload();
    
    /**
     * @brief Print current configuration values
     */
    void printConfig() const;
    
    /**
     * @brief Get singleton instance
     */
    static ConfigReader& getInstance(const std::string& config_file_path = "config.json");
};

// Global convenience functions
namespace Config {
    /**
     * @brief Get the global configuration instance
     */
    const YoloConfig& get();
    
    /**
     * @brief Reload the global configuration
     */
    void reload();
    
    /**
     * @brief Print the current configuration
     */
    void print();
}

#endif // CONFIG_HPP 