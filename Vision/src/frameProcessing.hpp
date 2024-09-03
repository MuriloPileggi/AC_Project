#pragma once

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include "opencv2/opencv.hpp"

class frameProcessing
{
private:
    // Use low pass filter to remove noise, removes high frequency stuff like edges
    const int kernelSize = 5; // bigger kernel = more smoothing
    // Define values for edge detection
	const int minVal = 50;
	const int maxVal = 250;
    // Define values for hough lines calculations
    const float rho = 2;
	const float pi = 3.14159265358979323846;
	const float theta = pi/180;
	const float threshold = 45;
	const int minLineLength = 40;
	const int maxLineGap = 100;

public:
    // Method for finding median between lanes
    double median(std::vector<double> vector);

    // Method for converting the current frame to grayscale
    void apply_gray_scale_to_frame(cv::Mat* framePtr);

    // Method to apply gaussian smoothing 
    void apply_smoothing_to_frame(cv::Mat* grayFramePtr);

    // Edge detection method
    // Finds gradient in x, y directions, gradient direction is perpendicular to edges
    void apply_edge_detection_to_frame(cv::Mat* smoothedFramePtr);

    // Method for creating and applying mask to frame
    void maskedFrame(cv::Mat* edgesFramePtr);

    // Method to resize images
    void resize_frame(cv::Mat* resized_frame, const size_t& x, const size_t& y);
};
