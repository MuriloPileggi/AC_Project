#include "guidance_system.hpp"

// Implementation for calculating steering angle method
double GuidanceSystem::calc_steering_angle(const cv::Mat& frame, FrameProcessing::LineParameters& right_line_parameters, 
                                FrameProcessing::LineParameters& left_line_parameters)
{
    /* // Let's Build ABC Triangle so we can get steering angle
    // Point A (Car center X coordinate) = (frame.cols/2, frame.rows)
    // Point B (Detected Lanes Center X coordinate) = ((left_line.x1 + right_line.x1)/2, frame.rows)
    // Point C (Lanes Center Intersection with ROI bounds) = (Bx, frame.rows - frame.rows/3)
    int a_x = frame.cols/2;
    int b_x = (left_line_parameters.x2 + right_line_parameters.x2)/2;
    int b_y = frame.rows;
    int c_y = frame.rows - left_line_parameters.y2;

    std::cout << b_x << std::endl;

    // With points set, we calculate the cathets, AB and BC
    // Distances are calculated in pixels
    int cathet_AB = b_x - a_x;
    int cathet_BC = std::abs(c_y - b_y);

    // With cathets calculated we can now calculate steering angle
    double theta_angle_radians = std::atan2(cathet_AB, cathet_BC);

    // Convert result into degrees
    double theta_angle_degrees = theta_angle_radians * (180.0 / CV_PI); 

    return theta_angle_degrees;
    */

   // Bottom Mask coordinates
   //int a_x = frame.cols/2;
   //int a_y = frame.rows;
   //int b_x = frame.cols/2;
   //int c_x = (left_line_parameters.x2 + right_line_parameters.x2)/2;
   //int c_y = frame.rows - left_line_parameters.y2;
   //int b_y = c_y;

   // Top Mask Coordinate
   int a_x = frame.cols/2;
   int a_y = frame.rows/2;
   int b_x = frame.cols/2;
   int c_x = (left_line_parameters.x2 + right_line_parameters.x2)/2;
   int c_y = frame.rows/2 - left_line_parameters.y2;
   int b_y = c_y;

   int adjacent_cathet = a_y - b_y;
   int opposite_cathet = c_x - b_x;

   double angle_radian = atan2(adjacent_cathet, opposite_cathet);

   double angle_degrees = angle_radian * (180.0 / CV_PI);

   return angle_degrees;
}


#pragma region PID

// PID Controller1 compute error method implementation
double PIDController::compute_error(double& steering_angle, double& dt)
{
    // Get error 
    double error = m_set_point - steering_angle;

    // Get proportional term
    double p_out = m_kp * error;

    // Get integral term
    m_integral += error * dt;
    double i_out = m_ki * m_integral;

    // Get derivative term
    double derivative = (error - m_previous_error) / dt;
    double d_out = m_kd * derivative;

    // Calculate total output
    double output = p_out + i_out + d_out;

    // Restrict value to max/min
    if (output > m_output_max) {
        output = m_output_max;
    }
    else if (output < m_output_min) {
        output = m_output_min;
    }

    // Save error to previous error
    m_previous_error = error;

    return output;
}

// PID Controller get dt method implementation
double PIDController::get_dt()
{
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - m_last_time);
    m_last_time = current_time;
    return dt.count();
}

// PID Controller set parameters method implementation
void PIDController::set_parameters(double& p, double& i, double& d)
{
    m_kp = p;
    m_ki = i;
    m_kd = d;
}

#pragma endregion