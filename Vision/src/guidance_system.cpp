#include "guidance_system.hpp"

// Implementation for calculating steering angle method
double GuidanceSystem::calc_steering_angle(const cv::Mat& frame, FrameProcessing::LineParameters& right_line_parameters, 
                                FrameProcessing::LineParameters& left_line_parameters)
{
    // Let's Build ABC Triangle so we can get steering angle
    // Point A (Car center X coordinate) = (frame.cols/2, frame.rows)
    // Point B (Detected Lanes Center X coordinate) = ((left_line.x1 + right_line.x1)/2, frame.rows)
    // Point C (Lanes Center Intersection with ROI bounds) = (Bx, frame.rows - frame.rows/3)
    int a_x = frame.cols/2;
    int b_x = (left_line_parameters.x1 + right_line_parameters.x1)/2;
    int b_y = frame.rows;
    int c_y = frame.rows/3;

    // With points set, we calculate the cathets, AB and BC
    // Distances are calculated in pixels
    int cathet_AB = b_x - a_x;
    int cathet_BC = b_y - c_y;

    // With cathets calculated we can now calculate steering angle
    double theta_angle_radians = std::atan2(cathet_BC, cathet_AB);

    // Convert result into degrees
    double theta_angle_degrees = theta_angle_radians * 180.0 / CV_PI;

    return theta_angle_degrees;
}
