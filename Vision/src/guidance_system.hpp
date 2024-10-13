#pragma once

#include <chrono>
#include "frame_processing.hpp"

class GuidanceSystem
{
public:
    // Method that calculates steering angle based on detected lanes from frame processing class
    // Calculate angle between car center line and lane center line
    double calc_steering_angle(const cv::Mat& frame, FrameProcessing::LineParameters& right_line_parameters,
                                FrameProcessing::LineParameters& left_line_parameters);
};


class PIDController
{
private:
    const double m_set_point = 90; // Desired angle, will always be this value, servo center position
    const double m_output_min = -30; // Member variable to store maximun output, this way we can prevent extreme actions, set to servo motor left limit
    const double m_output_max = 30; // Member variable to store maximun output, this way we can prevent extreme actions, set to servo motor right limit
    double m_kp, m_ki, m_kd; // Member variables to hold proportional, integral and derivative gains respectively 
    double m_integral; // Member variable to hold accumulated error over time
    double m_previous_error; // Member variable to hold the previous computed error
    std::chrono::high_resolution_clock::time_point m_last_time; // Memeber variable to store last time between iterations

public:
    // Class constructor
    PIDController(double p, double i, double d):
        m_kp(p),
        m_ki(i),
        m_kd(d)
    {
        m_last_time = std::chrono::high_resolution_clock::now();
    }
    
    // Method for computing correction value to send to servo motor
    // Computes the error between the car's current position and desired position
    double compute_error(double& steering_angle, double& dt);

    // Method to keep track of delta time between computations
    double get_dt();

    // Method to get angle correction based on error
    int8_t get_angle_correction(const double& error);

    // Method to change controller parameters after initialization
    void set_parameters(double& p, double& i, double& d); 
};


