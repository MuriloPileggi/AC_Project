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
    // Enumerator to keep track of traffic light status
    enum TrafficLightStatus {
        NONE,
        RED,
        GREEN,
        YELLOW
    };

    cv::dnn::Net m_net; // Open CV neural network object
    std::vector<std::string> m_classes; // Classes vector
    float m_confidence_threshold; // Minimun confidence to make accurate predictions
    float m_nms_threshold; // Non-Maximum Suppression algorithm threshold 
    bool m_stop_sign_detected = false;
    TrafficLightStatus m_traffic_light_status = NONE;

    void draw_predictions(int classId, float confidence, int left, int top, int right, int bottom, cv::Mat& frame);
    void post_process(cv::Mat& frame, const std::vector<cv::Mat>& outs);
    TrafficLightStatus set_traffic_light_status(const cv::Mat& roi);

public:
    // Class constructor
    ObjectDetector(const std::string& model_config, const std::string& model_weights, 
                   const std::string& classes_file, float confidence_threshold = 0.5, float nms_threshold = 0.4);

    
    // Make detection on received frame
    void detect(cv::Mat& frame);

    // Method that returns traffic light status as a int_8t, so we can publish results
    // with ROS to our bridge node
    int8_t get_traffic_light_status();

    // Method to return member boolean variable stop sign detected
    // Will return true if a stop sign is detected in frame, if not always false
    bool stop_sign_detected();
    
};