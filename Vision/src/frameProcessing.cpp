#include "frameProcessing.hpp"

double frameProcessing::median(std::vector<double> vector)
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

 // Method for converting the current frame to grayscale
void frameProcessing::apply_gray_scale_to_frame(cv::Mat *resizedFramePtr) 
{
    cv::cvtColor(*resizedFramePtr, *resizedFramePtr, cv::COLOR_BGR2GRAY);
}

// Method to apply gaussian smoothing
void frameProcessing::apply_smoothing_to_frame(cv::Mat *grayFramePtr) 
{
    cv::GaussianBlur(*grayFramePtr, *grayFramePtr, cv::Size(kernelSize, kernelSize), 0, 0);
}

// Edge detection method
// Finds gradient in x, y directions, gradient direction is perpendicular to edges
void frameProcessing::apply_edge_detection_to_frame(cv::Mat *smoothedFramePtr) 
{
    cv::Canny(*smoothedFramePtr, *smoothedFramePtr, minVal, maxVal);
}

// Method for creating and applying mask to frame
void frameProcessing::maskedFrame(cv::Mat *edgesFramePtr) 
{
    // Create mask to only keep area define by four corners
    // And black out every area outside

    // Define masked frame
    // Create all black frame with same dimensions as original
    // 3rd arg is CV_<bit-depth>{U|S|F}C(<number_of_channels>), so this is 8bit,
    // unsigned int, channels: 1
    cv::Mat mask(edgesFramePtr->size().height, edgesFramePtr->size().width, CV_8UC1, cv::Scalar(0)); // CV_8UC3 to make it a 3 channel

    // Define the points for the mask
    cv::Point p1 = cv::Point(0, edgesFramePtr->size().height);
    cv::Point p2 = cv::Point(455, 320);
    cv::Point p3 = cv::Point(505, 320);
    cv::Point p4 = cv::Point(edgesFramePtr->size().width, edgesFramePtr->size().height);

    // Create vector from array with defined points
    cv::Point vertices1[] = {p1, p2, p3, p4};
    std::vector<cv::Point> vertices(vertices1, vertices1+sizeof(vertices1)/sizeof(cv::Point));

    // Create vector of vectors, add the vertices defined above
    std::vector<std::vector<cv::Point>> verticesToFill;
    verticesToFill.push_back(vertices);

    // Fill in the vertices on the blank frame
    cv::fillPoly(mask, verticesToFill, cv::Scalar(255, 255, 255));

    // Create frame only where mask and edge detection frame are the same
    // Create masked frame, which takes 2 inputs and outputs only where the two
    // frames overlap
    cv::Mat maskedFrame = edgesFramePtr->clone();
    cv::bitwise_and(*edgesFramePtr, mask, *edgesFramePtr);
}

// Method for reisizing frames implementation
void frameProcessing::resize_frame(cv::Mat* framePtr, const size_t& x, const size_t& y)
{
    cv::resize(*framePtr, *framePtr, cv::Size(x,y));
}