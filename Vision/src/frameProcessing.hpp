#pragma once

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include "opencv2/opencv.hpp"

class FrameProcessing
{
private:
    // Use low pass filter to remove noise, removes high frequency stuff like edges
    const int kernel_size = 9; // bigger kernel = more smoothing
    // Define values for edge detection
	const int min_val = 60;
	const int max_val = 150;
    // Define values for hough lines calculations
    const float rho = 2;
	const float pi = 3.14159265358979323846;
	const float theta = pi/180;
	const float threshold = 25;
	const int min_line_length = 25;
	const int max_line_gap = 50;

public:
    // Edge detection method
    // Finds gradient in x, y directions, gradient direction is perpendicular to edges
    void apply_edge_detection_to_frame(cv::Mat* frame);

    // Method for converting the current frame to grayscale
    void apply_gray_scale_to_frame(cv::Mat* frame);

    // Method for creating and applying mask to frame
    void apply_mask_to_frame(cv::Mat* frame);

    // Method to apply gaussian smoothing 
    void apply_smoothing_to_frame(cv::Mat* frame);

    // Calculate steering angle from processed frame
    double calc_steering_angle(const std::vector<cv::Vec4i>& h_lines, const int& car_center);

    // Find Hough Lines
    std::vector<cv::Vec4i> find_hough_lines(const cv::Mat& frame);

    // Method for finding median between lanes
    double median(std::vector<double> vector);

    // Method to resize images
    void resize_frame(cv::Mat* frame, const size_t& x, const size_t& y);

};
