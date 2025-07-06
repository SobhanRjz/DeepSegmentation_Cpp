#include "pch.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "testOpenCV.hpp"

void test_opencv_installation() {
    std::cout << "✅ OpenCV version: " << CV_VERSION << std::endl;

    // Check for CUDA devices
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_devices > 0) {
        std::cout << "✅ CUDA is available. Found " << cuda_devices << " device(s)." << std::endl;
        cv::cuda::printShortCudaDeviceInfo(0);
    } else {
        std::cout << "⚠️  No CUDA devices found or CUDA not enabled in OpenCV." << std::endl;
    }

    // Create a simple black image
    cv::Mat image(400, 400, CV_8UC3, cv::Scalar(0, 255, 0));
    
    cv::putText(image, "OpenCV Works!", cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);

    // Save the image
    cv::imwrite("test_output.jpg", image);
    std::cout << "✅ Image saved as test_output.jpg" << std::endl;

    // Show the image (if GUI available)
    cv::imshow("OpenCV Test", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
int main() { 
    test_opencv_installation();
}