#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <opencv2/dnn.hpp>
#include <fstream>

static std::vector<std::string> loadLabels(const std::string& path){
    std::vector<std::string> L;
    std::ifstream f(path);
    for(std::string line; std::getline(f, line);) if(!line.empty()){
        L.push_back(line);
    }
    return L;

}
    

void saveSample(const cv::Mat& roi, const std::string& dir){
    if(roi.empty()){
        std::cerr << "ERROR: ROI is empty, cannot save sample.\n";
        return;
    }
    std::filesystem::create_directories(dir);
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    std::string path = dir + "/img_" + std::to_string(timestamp) + ".png";
    cv::imwrite(path, roi);
    std::cout << "Saved sample to " << path << "\n";
}

cv::Mat preprocessROI(const cv::Mat& roi){
    cv::Mat gray;
    cv::Mat resized;
    cv::Mat floating;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(224, 224));
    resized.convertTo(floating, CV_32F, 1.0 / 255);
    return floating;
}

int main() {
    std::cout << "OpenCV: " << CV_VERSION << "\n";
    std::cout << "TensorFlow C: " << TF_Version() << "\n";

    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "ERROR: Could not open camera.\n";
        return 1;
    }

    cv::Mat frame;
    cv::Mat roi;
    cv::Mat roiPreview;
    cv::Mat mask;
    cv::Mat cleaned;
    cv::Mat hsv;
    
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("ROI", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Mask", cv::WINDOW_AUTOSIZE);

    cv::Scalar low(0, 30, 60);
    cv::Scalar high(25, 200, 255);

    cv::dnn::Net net = cv::dnn::readNetFromONNX("gestures.onnx");
    auto labels = loadLabels("labels.txt");
    int NUM_CLASSES = labels.size();

    while (true) {
        capture.read(frame);
        if (frame.empty()) break;

        // We first convert to HSV color space and then create a mask based on the defined color range for skin
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, low, high, mask );

        // We then clean the mask (morphollgy)
        cv::erode(mask, cleaned, cv::Mat(), cv::Point(-1,-1), 1);
        cv::dilate(cleaned, cleaned, cv::Mat(), cv::Point(-1,-1), 2);
        cv::GaussianBlur(cleaned, cleaned, cv::Size(5, 5), 0);

        // Assume largest contour is the hand
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cleaned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        cv::Rect bestBox;

        double bestArea = 0.0;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > bestArea) {
                bestArea = area;
                bestBox = cv::boundingRect(contour);
            }
        }

        // Draw bounding box and extract ROI
        if (bestArea > 5000) {
            cv::rectangle(frame, bestBox, cv::Scalar(0, 255, 0), 2);


            int pad = 10;
            cv::Rect padded(
                std::max(bestBox.x - pad, 0),
                std::max(bestBox.y - pad, 0),
                std::min(bestBox.width + 2 * pad, frame.cols - bestBox.x + pad),
                std::min(bestBox.height + 2 * pad, frame.rows - bestBox.y + pad)
            );


            roi = frame(padded).clone();
            cv::Mat pre = preprocessROI(roi);

            


            pre.convertTo(roiPreview, CV_8U, 255.0);
            cv::imshow("ROI", roiPreview);
        } else{
            cv::Mat blank(64, 64, CV_8UC3, cv::Scalar(0,0,0));
            cv::imshow("ROI", blank);
        }

        // HUD
        cv::putText(frame, "q: quit | s: save ROI", {20, 20},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, {255,255,255}, 2);

        cv::imshow("Webcam", frame);
        cv::imshow("Mask", cleaned);
        int k = cv::waitKey(1) & 0xFF;
        if (k == 'q') break;
        if (k == '0' | k == '1' | k == '2'){
            if(!roiPreview.empty()){
                std::string dir = (k == '0') ? "data/0_palm" : (k == '1') ? "data/1_fist" : "data/2_peace";
                saveSample(roiPreview, dir);
            }
    
        }
    }
    return 0;
}