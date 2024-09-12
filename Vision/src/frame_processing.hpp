#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "../ai_libraries/mlpp.hpp"

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
	const float threshold = 45;
	const int min_line_length = 40;
	const int max_line_gap = 100;

public:

    // Struct to hold line parameters
    struct LineParameters {
        float slope;
        double x1, x2, y2;
        int y1;
    };
    
    // Edge detection method
    // Finds gradient in x, y directions, gradient direction is perpendicular to edges
    void apply_edge_detection_to_frame(cv::Mat& frame);

    // Method for converting the current frame to grayscale
    void apply_gray_scale_to_frame(cv::Mat& frame);

    // Method for creating and applying mask to frame
    void apply_mask_to_frame(cv::Mat& frame);

    // Method to apply gaussian smoothing 
    void apply_smoothing_to_frame(cv::Mat& frame);

    // Method to call all pre processing methods
    void apply_pre_processing_to_frame(cv::Mat& frame);

    // Display only method that blends image and allows us to see how well lane detection is going
    cv::Mat blend_frame(const cv::Mat& frame, const LineParameters& positive_line_parameters, const LineParameters& negative_line_parameters);

    // Method that classifies extracted hough lines found into left and right lines
    // Classification is based on line slope
    MLPP::Mat3d<double> classify_hough_lines(const cv::Mat& frame, const std::vector<cv::Vec4i>& h_lines);

    // Method to extract hough lines from masked frame
    std::vector<cv::Vec4i> extract_hough_lines(const cv::Mat& frame);

    // Method that finds positive and negative slope lines average
    // Average the position of lines and extrapolate to the top and bottom of the lane.
    std::vector<float>  get_average_positive_and_negative_slopes_lines(const MLPP::Mat3d<double>& slope_lines);

    // Method that finds X coordinate for positive and negative average slope lines
    // I.E., we find X coordinate for average positive slope line when Y = 0
    // The same for average negative slope line
    std::vector<double>  get_x_intercept_coordinates(const cv::Mat& frame, const MLPP::Mat3d<double>& slope_lines);

    // Method for finding median between lanes
    double median(std::vector<double> vector);

    // Method for plotting lane lines
    // Need end points of line to draw in. Have x1,y1
    // (xIntercept,im.shape[1]) where im.shape[1] is the bottom of the
    // image. take y2 as some num (min/max y in the good lines?) then
    // find corresponding x
    void plot_lane_lines(const cv::Mat& frame, const std::vector<float>& avrg_slope_lines, const std::vector<double>& x_intercept_coord, 
                            LineParameters& positive_slope_parameters, LineParameters& negative_slope_parameters);

    // Method to resize images
    void resize_frame(cv::Mat& frame, const size_t& x, const size_t& y);
};
