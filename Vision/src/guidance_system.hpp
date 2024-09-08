#pragma once

#include "frame_processing.hpp"


class GuidanceSystem
{
private:
   
public:
    // Method that calculates steering angle based on detected lanes from frame processing class
    // Calculate angle between car center line and lane center line
    double calc_steering_angle(const cv::Mat& frame, FrameProcessing::LineParameters& right_line_parameters, 
                                FrameProcessing::LineParameters& left_line_parameters);
};

