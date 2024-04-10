#pragma once

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include "opencv2/opencv.hpp"

class LaneDetection
{
private:
    // Use low pass filter to remove noise, removes high frequency stuff like edges
    const int kernelSize = 9; // bigger kernel = more smoothing
    // Define values for edge detection
	const int minVal = 60;
	const int maxVal = 150;
    // Define values for hough lines calculations
    const float rho = 2;
	const float pi = 3.14159265358979323846;
	const float theta = pi/180;
	const float threshold = 45;
	const int minLineLength = 40;
	const int maxLineGap = 100;

public:
    // Method for finding median between lanes
    double median(std::vector<double> vector)
    {
        // Get size of vector 
        int vectorSize = vector.size();
        
        // If vector is empty throw error
        if (vectorSize == 0)
        {
            throw std::domain_error("Vector is empty!");
        }
        
        // Sort Vector
        std::sort(vector.begin(), vector.end());

        // Define middle and median
        int middle;
        double median;

        // If vector size is even, take average of two middle values
        if (vectorSize % 2 == 0)
        {
            // A value representing the middle of the vector
            middle = vectorSize/2;

            // Take average of middle values, assume vector [1,2,3,4], take average of 2 and 3
            median = (vector[middle -1] + vector[middle]) / 2;
        }

        // Odd numbered vector size
        else
        {
            // Result will be lowest int, so if vectorSize = 5, middle will be 2
            middle = vectorSize/2;
            median = vector[middle];
        }
        
        return median;
    }

    // Method for converting the current frame to grayscale
    cv::Mat grayFrame(cv::Mat frame)
    {   
        // Define grayscale frame
        cv::Mat grayFrame;

        // Convert frame to grayscale
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Return grayscale frame
        return grayFrame;
    }

    // Method to apply gaussian smoothing 
    cv::Mat smoothedFrame(cv::Mat grayFrame)
    {   
        // Define and apply gaussian filter to smoothed frame
        cv::Mat smoothedFrame;
        cv::GaussianBlur(grayFrame, smoothedFrame, cv::Size(kernelSize, kernelSize), 0, 0);

        // Return smoothed frame
        return smoothedFrame;
    }

    // Edge detection method
    // Finds gradient in x, y directions, gradient direction is perpendicular to edges
    cv::Mat edgesFrame(cv::Mat smoothedFrame)
    {   
        // Define edge detection frame
        cv::Mat edgesFrame;

        // Perform edge detection
        cv::Canny(smoothedFrame, edgesFrame, minVal, maxVal);

        // Return edge detection frame
        return edgesFrame;
    }

    // Method for creating and applying mask to frame
    cv::Mat maskedFrame(cv::Mat edgesFrame)
    {
        // Create mask to only keep area define by four corners
        // And black out every area outside

        // Define masked frame
        // Create all black frame with same dimensions as original
        // 3rd arg is CV_<bit-depth>{U|S|F}C(<number_of_channels>), so this is 8bit, unsigned int, channels: 1
        cv::Mat mask(edgesFrame.size().height, edgesFrame.size().width, CV_8UC1, cv::Scalar(0)); // CV_8UC3 to make it a 3 channel

        // Define the points for the mask
        cv::Point p1 = cv::Point(0, edgesFrame.size().height);
        cv::Point p2 = cv::Point(455, 320);
        cv::Point p3 = cv::Point(505, 320);
        cv::Point p4 = cv::Point(edgesFrame.size().width, edgesFrame.size().height);

        // Create vector from array with defined points
        cv::Point vertices1[] = {p1, p2, p3, p4};
        std::vector<cv::Point> vertices(vertices1, vertices1 + sizeof(vertices1) / sizeof(cv::Point));

        // Create vector of vectors, add the vertices defined above
        std::vector<std::vector<cv::Point>> verticesToFill;
        verticesToFill.push_back(vertices);

        // Fill in the vertices on the blank frame
        cv::fillPoly(mask, verticesToFill, cv::Scalar(255, 255, 255));

        // Create frame only where mask and edge detection frame are the same
        // Create masked frame, which takes 2 inputs and outputs only where the two frames overlap
        cv::Mat maskedFrame = edgesFrame.clone();
        cv::bitwise_and(edgesFrame, mask, maskedFrame);

        // Return mask applied frame
        return maskedFrame;
    }

    // Method for calculating hough lines
};