#include "frameProcessing.hpp"

// Edge detection method
// Finds gradient in x, y directions, gradient direction is perpendicular to edges
void FrameProcessing::apply_edge_detection_to_frame(cv::Mat* frame) 
{
    cv::Canny(*frame, *frame, min_val, max_val);
}

// Method for converting the current frame to grayscale
void FrameProcessing::apply_gray_scale_to_frame(cv::Mat*frame) 
{
    cv::cvtColor(*frame, *frame, cv::COLOR_BGR2GRAY);
}

// Method for creating and applying mask to frame
void FrameProcessing::apply_mask_to_frame(cv::Mat* frame) 
{
    // Create mask to only keep area define by four corners
    // And black out every area outside

    // Define masked frame
    // Create all black frame with same dimensions as original
    // 3rd arg is CV_<bit-depth>{U|S|F}C(<number_of_channels>), so this is 8bit,
    // unsigned int, channels: 1
    cv::Mat mask(frame->rows, frame->cols, CV_8UC1, cv::Scalar(0)); // CV_8UC3 to make it a 3 channel

    // Define the points for the mask
    cv::Point topLeft(0,frame->rows/3);  // Top-left of polygon
    cv::Point topRight(frame->cols, frame->rows/3);  // Top-right of polygon
    cv::Point bottomLeft(0, frame->rows/3 + frame->rows/3);  // Bottom-left of polygon
    cv::Point bottomRight(frame->cols, frame->rows/3 + frame->rows/3);  // Bottom-right of polygon

    // Create vector from array with defined points
    cv::Point vertices1[] = {topLeft, topRight, bottomRight, bottomLeft};
    std::vector<cv::Point> vertices(vertices1, vertices1+sizeof(vertices1)/sizeof(cv::Point));

    // Create vector of vectors, add the vertices defined above
    std::vector<std::vector<cv::Point>> verticesToFill;
    verticesToFill.push_back(vertices);

    // Fill in the vertices on the blank frame
    cv::fillPoly(mask, verticesToFill, cv::Scalar(255, 255, 255));

    // Create frame only where mask and edge detection frame are the same
    // Create masked frame, which takes 2 inputs and outputs only where the two
    // frames overlap
    cv::Mat apply_mask_to_frame = frame->clone();
    cv::bitwise_and(*frame, mask, *frame);
}

// Method to apply gaussian smoothing
void FrameProcessing::apply_smoothing_to_frame(cv::Mat* frame) 
{
    cv::GaussianBlur(*frame, *frame, cv::Size(kernel_size, kernel_size), 0, 0);
}

// Calculate steering angle method
// Calculates the angle needed to pass to servo motor for car's guiding system
double FrameProcessing::calc_steering_angle(const std::vector<cv::Vec4i>& h_lines, const int& car_center)
{
    if (h_lines.empty()) throw std::runtime_error("Error: Hough lines vector is empty");
    
    // Separate left and right lanes   
    std::vector<cv::Vec4i> left_lines;
    std::vector<cv::Vec4i> right_lines;

    for (const auto& line : h_lines) {
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        double slope = (y2 - y1) / static_cast<double>(x2 - x1);
        if (slope < 0) {
            left_lines.push_back(line);
        } else {
            right_lines.push_back(line);
        }
    }

    // Average the lines (you might want to use a more sophisticated method)
    cv::Vec4i left_line = cv::Vec4i(0, 0, 0, 0);
    cv::Vec4i right_line = cv::Vec4i(0, 0, 0, 0);
    
    if (!left_lines.empty()) {
        for (const auto& line : left_lines) {
            left_line += line;
        }
        left_line /= static_cast<int>(left_lines.size());
    }
    
    if (!right_lines.empty()) {
        for (const auto& line : right_lines) {
            right_line += line;
        }
        right_line /= static_cast<int>(right_lines.size());
    }

    // Calculate vanishing point
    double left_m = static_cast<double>(left_line[3] - left_line[1]) / (left_line[2] - left_line[0]);
    double left_b = left_line[1] - left_m * left_line[0];
    double right_m = static_cast<double>(right_line[3] - right_line[1]) / (right_line[2] - right_line[0]);
    double right_b = right_line[1] - right_m * right_line[0];
    
    double vanishing_x = (right_b - left_b) / (left_m - right_m);
    double vanishing_y = left_m * vanishing_x + left_b;

    // Calculate steering angle
    double steeringAngle = std::atan((vanishing_x - car_center) / vanishing_y);

    return std::atan(steeringAngle) * (180/pi);
}

// Find all Hough lines
std::vector<cv::Vec4i> FrameProcessing::find_hough_lines(const cv::Mat& frame)
{
    cv::Mat masked_edges = frame;

    // Hough line transform 
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(masked_edges, lines, rho, theta, threshold, min_line_length, max_line_gap);

    if (lines.empty()) {
        std::cout << "No lines detected" << std::endl;
        return std::vector<cv::Vec4i>();
    }

    return lines;
}

// Calculate median method
double FrameProcessing::median(std::vector<double> vector)
{
    // Get size of vector
    int vectorSize = vector.size();

    // If vector is empty throw error
    if (vectorSize == 0) {
        throw std::domain_error("Vector is empty!");
    }

    // Sort Vector
    std::sort(vector.begin(), vector.end());

    // Define middle and median
    int middle;
    double median;

    // If vector size is even, take average of two middle values
    if (vectorSize % 2 == 0) {
        // A value representing the middle of the vector
        middle = vectorSize / 2;

        // Take average of middle values, assume vector [1,2,3,4], take average
        // of 2 and 3
        median = (vector[middle - 1] + vector[middle]) / 2;
    }

    // Odd numbered vector size
    else {
        // Result will be lowest int, so if vectorSize = 5, middle will be 2
        middle = vectorSize / 2;
        median = vector[middle];
    }

    return median;
}

// Method for reisizing frames implementation
void FrameProcessing::resize_frame(cv::Mat* frame, const size_t& x, const size_t& y)
{
    cv::resize(*frame, *frame, cv::Size(x,y));
}
