#include <fmt/core.h>
#include <iostream>
#include <string>
#include <filesystem>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/stitching.hpp"

namespace fs = std::filesystem;

// 全局变量
fs::path file_path = fs::current_path().parent_path();

// 读取图像函数
cv::Mat read_image(const std::string& img_path) {
    std::string path = file_path.string() + "/" + img_path;
    fmt::print("Reading image from: {}\n", path);
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        fmt::print("Error: Could not read the image from {}\n", path);
        return {};
    }
    fmt::print("Image read successfully. Size: {}x{}\n", img.cols, img.rows);
    return img;
}

// 显示图像函数
void show_image(const std::string& window_name, const cv::Mat& img, const int width = 800) {
    if (img.empty()) return;
    
    double aspect_ratio = static_cast<double>(img.cols) / img.rows;
    int height = static_cast<int>(width / aspect_ratio);
    
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(width, height));
    
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, resized_img);
    cv::waitKey(100); // 短暂等待以确保窗口更新
}

// 检测和显示图像特征点
bool detect_and_show_features(const cv::Mat& img1, const cv::Mat& img2) {
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(5000); // 增加特征点数量
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    // 检测特征点和计算描述符
    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
    
    fmt::print("Detected {} keypoints in image 1\n", keypoints1.size());
    fmt::print("Detected {} keypoints in image 2\n", keypoints2.size());
    
    // 如果特征点太少，直接返回失败
    if (keypoints1.size() < 50 || keypoints2.size() < 50) {
        fmt::print("Error: Not enough features detected in images\n");
        return false;
    }
    
    // 使用BFMatcher进行特征匹配
    const cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    fmt::print("Found {} matches between images\n", matches.size());
    
    // 如果匹配点太少，直接返回失败
    if (matches.size() < 10) {
        fmt::print("Error: Not enough matches between images\n");
        return false;
    }
    
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    show_image("Feature Matches", img_matches);
    return true;
}

// 图像拼接函数
bool stitch_images(const std::vector<cv::Mat>& images, cv::Mat& result, cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA) {
    try {
        // 创建拼接器
        const cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
        
        // 设置自定义特征查找器（增加特征点数量）
        const cv::Ptr<cv::ORB> orb = cv::ORB::create(5000);
        stitcher->setFeaturesFinder(orb);
        
        // 执行拼接
        // 处理拼接状态
        if (cv::Stitcher::Status status = stitcher->stitch(images, result); status == cv::Stitcher::OK) {
            fmt::print("Stitching completed successfully\n");
            return true;
        } else {
            fmt::print("Stitching failed with status code: {}\n", status);
            
            // 根据状态码提供具体建议
            switch (status) {
                case cv::Stitcher::ERR_NEED_MORE_IMGS:
                    fmt::print("Error: Need more images or failed to find features\n");
                    break;
                case cv::Stitcher::ERR_HOMOGRAPHY_EST_FAIL:
                    fmt::print("Error: Homography estimation failed\n");
                    break;
                case cv::Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL:
                    fmt::print("Error: Camera parameters adjustment failed\n");
                    break;
                default:
                    fmt::print("Error: Unknown stitching error\n");
                    break;
            }
            return false;
        }
    } catch (const cv::Exception& e) {
        fmt::print("OpenCV Exception caught during stitching: {}\n", e.what());
        return false;
    }
}

int main(int argc, char* argv[]) {
    fmt::print("OpenCV Image Stitching Console\n");
    fmt::print("OpenCV version: {}.{}.{}\n", CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION);
    fmt::print("Working directory: {}\n", file_path.string());
    fmt::print("Type 'exit' to quit or 'help' for commands\n");
    
    // 预加载图像
    cv::Mat img_one, img_two;
    bool images_loaded = false;
    
    std::string command;
    while (true) {
        fmt::print("\n> ");
        std::getline(std::cin, command);
        
        if (command == "exit") {
            fmt::print("Exiting console. Goodbye!\n");
            break;
        }
        else if (command == "help") {
            fmt::print("Available commands:\n");
            fmt::print("  load          - Load images (1.jpg and 2.jpg)\n");
            fmt::print("  features      - Detect and show features\n");
            fmt::print("  stitch_scans  - Stitch images in SCANS mode\n");
            fmt::print("  exit          - Exit the program\n");
        }
        else if (command == "load") {
            img_one = read_image("1.png");
            img_two = read_image("2.png");
            
            if (!img_one.empty() && !img_two.empty()) {
                images_loaded = true;
                fmt::print("Images loaded successfully\n");
            } else {
                images_loaded = false;
                fmt::print("Failed to load one or both images\n");
            }
        }
        else if (command == "features") {
            if (images_loaded) {
                detect_and_show_features(img_one, img_two);
            } else {
                fmt::print("Images not loaded. Use 'load' command first.\n");
            }
        }
        else if (command == "stitch_scans") {
            if (images_loaded) {
                std::vector<cv::Mat> images = {img_one, img_two};

                if (cv::Mat result; stitch_images(images, result, cv::Stitcher::SCANS)) {
                    cv::imwrite("scans_result.jpg", result);
                    fmt::print("Result saved as scans_result.jpg\n");
                }
            } else {
                fmt::print("Images not loaded. Use 'load' command first.\n");
            }
        }
        else {
            fmt::print("Unknown command: {}. Type 'help' for available commands.\n", command);
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}