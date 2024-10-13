#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>


class ObjectDetector
{
private:
    cv::dnn::Net m_net; // Open CV neural network object
    std::vector<std::string> m_classes; // Classes vector
    float m_confidence_threshold; // Minimun confidence to make accurate predictions
    float m_nms_threshold; // Non-Maximum Suppression algorithm threshold 

    void post_process(cv::Mat& frame, const std::vector<cv::Mat>& outs);
    void draw_predictions(int classId, float confidence, int left, int top, int right, int bottom, cv::Mat& frame);

public:
    // Class constructor
    ObjectDetector(const std::string& model_config, const std::string& model_weights, 
                   const std::string& classes_file, float confidence_threshold = 0.5, float nms_threshold = 0.4);

    // Make detection on received frame
    void detect(cv::Mat& frame);
};