#include "frame_processing.hpp"

// Edge detection method
// Finds gradient in x, y directions, gradient direction is perpendicular to edges
void FrameProcessing::apply_edge_detection_to_frame(cv::Mat& frame) 
{
    cv::Canny(frame, frame, min_val, max_val);
}

// Method for converting the current frame to grayscale
void FrameProcessing::apply_gray_scale_to_frame(cv::Mat& frame) 
{
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
}

// Method for creating and applying mask to frame
// Create mask to only keep area defined by four corners
// Black out every area outside area
void FrameProcessing::apply_mask_to_frame(cv::Mat& frame) 
{
    // Define masked image
	// Create all black image with same dimensions as original image
	// 3rd arg is CV_<bit-depth>{U|S|F}C(<number_of_channels>), so this is 8bit, unsigned int, channels: 1
	cv::Mat mask(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));

    // Define the points for the mask - Bottom Mask
	// Use cv::Point type for x,y points 
    //cv::Point bottom_left = cv::Point(125, frame.rows);
	//cv::Point top_left = cv::Point(125, frame.rows/2);
	//cv::Point top_right = cv::Point(frame.cols - 25, frame.rows/2);
	//cv::Point bottom_right = cv::Point(frame.cols - 25, frame.rows);

    // Define the points for the mask - Top Mask
    // Use cv::Point type for x,y points 
	cv::Point bottom_left = cv::Point(50, frame.rows/2);
	cv::Point top_left = cv::Point(50, 0);
	cv::Point top_right = cv::Point(frame.cols - 50, 0);
	cv::Point bottom_right = cv::Point(frame.cols - 50, frame.rows/2);
    

    // create vector from array with points we just defined
	//cv::Point vertices1[] = {bottom,top_left_1,top_left_2,top_right_2,top_right_1};
    cv::Point vertices1[] = {bottom_left, top_left, top_right, bottom_right};
	std::vector<cv::Point> vertices (vertices1, vertices1 + sizeof(vertices1) / sizeof(cv::Point));

    // Create vector of vectors, add the vertices we defined above
	// (you could add multiple other similar contours to this vector)
	std::vector<std::vector<cv::Point> > vertices_to_fill;
	vertices_to_fill.push_back(vertices);

    // Fill in the vertices on the blank image, showing what the mask is
	cv::fillPoly(mask, vertices_to_fill, cv::Scalar(255,255,255));

    // Show the mask
    //cv::namedWindow("Mask", cv::WINDOW_NORMAL);
    //cv::imshow("Mask", mask);
    //cv::waitKey(0);

    // Create masked im, which takes input1, input2, and output. Only keeps where two images overlap
	cv::bitwise_and(frame, mask, frame);
}

// Method to apply gaussian smoothing
void FrameProcessing::apply_smoothing_to_frame(cv::Mat& frame) 
{
    cv::GaussianBlur(frame, frame, cv::Size(kernel_size, kernel_size), 0, 0);
}

// Implementation of apply pre processing method
void FrameProcessing::apply_pre_processing_to_frame(cv::Mat& frame)
{
    apply_gray_scale_to_frame(frame);
    apply_smoothing_to_frame(frame);
    apply_edge_detection_to_frame(frame);
    apply_mask_to_frame(frame);
}

// Implementation of blend frame method
cv::Mat FrameProcessing::blend_frame(const cv::Mat& frame, const LineParameters& positive_line_parameters, const LineParameters& negative_line_parameters)
{
    // Create image, lane lines on real image
    cv::Mat lane_filled_image = frame.clone();

    // Use cv::Point type for x,y points - Bottom Mask
    //cv::Point v1 = cv::Point(positive_line_parameters.x1, frame.rows - positive_line_parameters.y1);
    //cv::Point v2 = cv::Point(positive_line_parameters.x2, frame.rows - positive_line_parameters.y2);
    //cv::Point v3 = cv::Point(negative_line_parameters.x1, frame.rows - negative_line_parameters.y1);
    //cv::Point v4 = cv::Point(negative_line_parameters.x2, frame.rows - negative_line_parameters.y2);
    // Use cv::Point type for x,y points - Top Mask
    cv::Point v1 = cv::Point(positive_line_parameters.x1, frame.rows/2 - positive_line_parameters.y1);
    cv::Point v2 = cv::Point(positive_line_parameters.x2, frame.rows/2 - positive_line_parameters.y2);
    cv::Point v3 = cv::Point(negative_line_parameters.x1, frame.rows/2 - negative_line_parameters.y1);
    cv::Point v4 = cv::Point(negative_line_parameters.x2, frame.rows/2 - negative_line_parameters.y2);

    // create vector from array of corner points of lane
    cv::Point vertices_blend[] = {v1, v3, v4, v2}; 
    std::vector<cv::Point> vertices_vector_blend(vertices_blend, vertices_blend + sizeof(vertices_blend) / sizeof(cv::Point));

    // Create vector of vectors to be used in fillPoly, add the vertices we defined above
    std::vector<std::vector<cv::Point>> vertices_fill_poly;
    vertices_fill_poly.push_back(vertices_vector_blend);

    // Fill area created from vector points
    cv::fillPoly(lane_filled_image, vertices_fill_poly, cv::Scalar(0, 255, 255));

    // Blend image
    float opacity  {0.25};
    cv::Mat blended_image;
    cv::addWeighted(lane_filled_image, opacity, frame, 1 - opacity, 0, blended_image);

    int middle_line_x = (positive_line_parameters.x2 + negative_line_parameters.x2)/2;

    // Plot lane lines - Bottom Mask
    //cv::line(blended_image, cv::Point(positive_line_parameters.x1, frame.rows - positive_line_parameters.y1),
    //            cv::Point(positive_line_parameters.x2, frame.rows - positive_line_parameters.y2),
    //                cv::Scalar(0, 255, 0), 8, 8);
    //cv::line(blended_image, cv::Point(negative_line_parameters.x1, frame.rows - negative_line_parameters.y1),
    //                 cv::Point(negative_line_parameters.x2, frame.rows - negative_line_parameters.y2),
    //                 cv::Scalar(0, 255, 0), 8, 8);
    //cv::line(blended_image, cv::Point(middle_line_x, frame.rows),
    //                 cv::Point(middle_line_x, frame.rows - negative_line_parameters.y2),
    //                 cv::Scalar(0, 0, 255), 5, 8);
    //cv::line(blended_image, cv::Point(frame.cols/2, frame.rows),
    //                 cv::Point(frame.cols/2, frame.rows - negative_line_parameters.y2),
    //                 cv::Scalar(0, 0, 0), 5, 8);

    // Plot lane lines - Top Mask
    cv::line(blended_image, cv::Point(positive_line_parameters.x1, frame.rows/2 - positive_line_parameters.y1),
                cv::Point(positive_line_parameters.x2, frame.rows/2 - positive_line_parameters.y2),
                    cv::Scalar(0, 255, 0), 8, 8);
    cv::line(blended_image, cv::Point(negative_line_parameters.x1, frame.rows/2 - negative_line_parameters.y1),
                     cv::Point(negative_line_parameters.x2, frame.rows/2 - negative_line_parameters.y2),
                     cv::Scalar(0, 255, 0), 8, 8);
    cv::line(blended_image, cv::Point(middle_line_x, frame.rows/2),
                     cv::Point(middle_line_x, frame.rows/2 - negative_line_parameters.y2),
                     cv::Scalar(0, 0, 255), 5, 8);
    cv::line(blended_image, cv::Point(frame.cols/2, frame.rows/2),
                     cv::Point(frame.cols/2, frame.rows/2 - negative_line_parameters.y2),
                     cv::Scalar(0, 0, 0), 5, 8);

    return blended_image;
}

// Implementation of classify hough lines method
// This will return 2 2d matrices
// Mat3d[0] = Positive slope lines i.e., Right Lines
// Mat3d[1] = Negative slope lines i.e., Left Lines
MLPP::Mat3d<double> FrameProcessing::classify_hough_lines(const cv::Mat& frame, const std::vector<cv::Vec4i>& h_lines)
{
    // Create empty 2d matrices to store positive and negative slope lines
    MLPP::Mat2d<double> positive_slope_lines; // format will be [x1 y1 x2 y2 slope]
    MLPP::Mat2d<double> negative_slope_lines; // format will be [x1 y1 x2 y2 slope]
    std::vector<float> angles;
    float frame_center = frame.cols/2.0f;
    float frame_height = static_cast<float>(frame.rows);
    
    std::vector<cv::Vec4i> positive_lines;
    std::vector<cv::Vec4i> negative_lines;

    for (size_t i = 0; i < h_lines.size(); i++) {
        if (h_lines[i][0] <= frame_center) {
            positive_lines.push_back(h_lines[i]);
        } else {
            negative_lines.push_back(h_lines[i]);
        }
    }

    positive_slope_lines.resize(positive_lines.size());
    negative_slope_lines.resize(negative_lines.size());

    for (size_t i = 0; i < positive_slope_lines.size(); i++) {
        // Get points for current line
		float x1 = positive_lines[i][0];
		float y1 = positive_lines[i][1];
		float x2 = positive_lines[i][2];
		float y2 = positive_lines[i][3];
        float slope = (y2-y1)/(x2-x1);

        positive_slope_lines[i].resize(5);
        positive_slope_lines[i][0] = x1; 
        positive_slope_lines[i][1] = y1; 
        positive_slope_lines[i][2] = x2; 
        positive_slope_lines[i][3] = y2; 
        positive_slope_lines[i][4] = -slope; 
    }

    for (size_t i = 0; i < negative_lines.size(); i++) {
        // Get points for current line
		float x1 = negative_lines[i][0];
		float y1 = negative_lines[i][1];
		float x2 = negative_lines[i][2];
		float y2 = negative_lines[i][3];
        float slope = (y2-y1)/(x2-x1);

        negative_slope_lines[i].resize(5);
        negative_slope_lines[i][0] = x1; 
        negative_slope_lines[i][1] = y1; 
        negative_slope_lines[i][2] = x2; 
        negative_slope_lines[i][3] = y2; 
        negative_slope_lines[i][4] = -slope; 
    }

    MLPP::Mat3d<double> result;
    result.push_back(positive_slope_lines);
    result.push_back(negative_slope_lines);

    return result; 

    /* 
    std::vector<float> y_values; // Vector to store all y values

    int positive_counter = 0;
    int negative_counter = 0;

    // keep track of if we added one of each, want at least one of each to proceed
	bool added_positive = false;
	bool added_negative = false;

    // Loop through all lines
	for (size_t i = 0; i != h_lines.size(); ++i) {
		// Get points for current line
		float x1 = h_lines[i][0];
		float y1 = h_lines[i][1];
		float x2 = h_lines[i][2];
		float y2 = h_lines[i][3];

		// Get line length
		float line_length =  std::pow(std::pow(x2-x1,2) + std::pow(y2-y1,2), 0.5);

		// if line is long enough
		if (line_length > 30) {
			// Dont divide by zero check
			if (x2 != x1) {
				// Get slope
				float slope = (y2-y1)/(x2-x1);

				// Check if slope is positive
				if (slope > 0) {
					// Find angle of line wrt x axis.
					float tan_theta = std::tan((std::abs(y2-y1))/(std::abs(x2-x1))); // tan(theta) value
					float angle = atan (tan_theta) * 180/pi;
					// Only pass good line angles,  dont want verticalish/horizontalish lines
					if (std::abs(angle) < 85 && std::abs(angle) > 20) {
					    // Add a row to the matrix
						negative_slope_lines.resize(negative_counter+1);

					    // Reshape current row to 5 columns [x1, y1, x2, y2, slope]
					    negative_slope_lines[negative_counter].resize(5);

						// Add values to row
						negative_slope_lines[negative_counter][0] = x1;
						negative_slope_lines[negative_counter][1] = y1;
						negative_slope_lines[negative_counter][2] = x2;
						negative_slope_lines[negative_counter][3] = y2;
						negative_slope_lines[negative_counter][4] = -slope;

					    // add yValues
					    y_values.push_back(y1);
					    y_values.push_back(y2);

						// Note that we added a positive slope line
						added_positive = true;

						// iterate the counter
						negative_counter++;
					}
				}

				// Check if slope is Negative
				if (slope < 0) {
				    // Find angle of line wrt x axis.
					float tan_theta = std::tan ((std::abs(y2-y1))/(std::abs(x2-x1))); // tan(theta) value
					float angle = std::atan (tan_theta) * 180/pi;

					// Only pass good line angles,  dont want verticalish/horizontalish lines
					if (std::abs(angle) < 85 && std::abs(angle) > 20) {                        
						// Add a row to the matrix
						positive_slope_lines.resize(positive_counter+1);

						// Reshape current row to 5 columns [x1, y1, x2, y2, slope]
						positive_slope_lines[positive_counter].resize(5);

						// Add values to row
						positive_slope_lines[positive_counter][0] = x1;
						positive_slope_lines[positive_counter][1] = y1;
						positive_slope_lines[positive_counter][2] = x2;
						positive_slope_lines[positive_counter][3] = y2;
						positive_slope_lines[positive_counter][4] = -slope;

						// add yValues
						y_values.push_back(y1);
						y_values.push_back(y2);
							 
                        // Note that we added a positive slope line
					    added_negative = true;

						// iterate counter
						positive_counter++;

					}
				} // if slope < 0
                
			} 
		}
	} 

    // If we didn't get any positive lines, go though again and just add any
    // positive slope lines Be less strict
    if (added_positive == false) { // if we didnt add any positive lines

        // Loop through lines
        for (size_t i = 0; i != h_lines.size(); ++i) {

            // Get points for current line
            float x1 = h_lines[i][0];
            float y1 = h_lines[i][1];
            float x2 = h_lines[i][2];
            float y2 = h_lines[i][3];

            // Get slope
            float slope = (y2 - y1) / (x2 - x1);

            // Check if slope is positive
            if (slope > 0 && x2 != x1) {

                // Find angle of line wrt x axis.
                float tan_theta = std::tan((std::abs(y2 - y1))/(std::abs(x2 - x1))); // tan(theta) value
                float angle = std::atan(tan_theta) * 180/pi;

                // Add a row to the matrix
                positive_slope_lines.resize(negative_counter + 1);

                // Reshape current row to 5 columns [x1, y1, x2, y2, slope]
                positive_slope_lines[negative_counter].resize(5);

                // Add values to row
                negative_slope_lines[negative_counter][0] = x1;
                negative_slope_lines[negative_counter][1] = y1;
                negative_slope_lines[negative_counter][2] = x2;
                negative_slope_lines[negative_counter][3] = y2;
                negative_slope_lines[negative_counter][4] = -slope;

                // add yValues
                y_values.push_back(y1);
                y_values.push_back(y2);

                // Note that we added a positive slope line
                added_positive = true;

                // iterate the counter
                negative_counter++;
            }
        }
    }

    // If we didn't get any negative lines, go though again and just add any
    // positive slope lines Be less strict
    if (added_negative == false) { // if we didnt add any positive lines

        // Loop through lines
        for (size_t i = 0; i != h_lines.size(); ++i) {

            // Get points for current line
            float x1 = h_lines[i][0];
            float y1 = h_lines[i][1];
            float x2 = h_lines[i][2];
            float y2 = h_lines[i][3];

            // Get slope
            float slope = (y2 - y1) / (x2 - x1);

            // Check if slope is positive
            if (slope < 0 && x2 != x1) {

                // Find angle of line wrt x axis.
                float tan_theta = std::tan((std::abs(y2 - y1))/(std::abs(x2 - x1))); // tan(theta) value
                float angle = std::atan(tan_theta) * 180 / pi;

                // Add a row to the matrix
                positive_slope_lines.resize(positive_counter + 1);

                // Reshape current row to 5 columns [x1, y1, x2, y2, slope]
                positive_slope_lines[positive_counter].resize(5);

                // Add values to row
                positive_slope_lines[positive_counter][0] = x1;
                positive_slope_lines[positive_counter][1] = y1;
                positive_slope_lines[positive_counter][2] = x2;
                positive_slope_lines[positive_counter][3] = y2;
                positive_slope_lines[positive_counter][4] = -slope;

                // add yValues
                y_values.push_back(y1);
                y_values.push_back(y2);

                // Note that we added a positive slope line
                added_negative = true;

                // iterate the counter
                positive_counter++;
            }
        }
    } 

    // If we still dont have lines then display error
    if (added_positive == false || added_negative == false) {
        std::cout << "Not enough lines found" << std::endl;
        return MLPP::Mat3d<double>();
    }

    MLPP::Mat3d<double> result;
    result.push_back(positive_slope_lines);
    result.push_back(negative_slope_lines);

    return result; */

    /* for (const auto& line : h_lines)
    {
        float x1 = line[0];
        float y1 = line[1];
        float x2 = line[2];
        float y2 = line[3];

        // Normalize coordinates
        float n_x1 = x1 /frame.cols;
        float n_y1 = y1 / frame.rows;
        float n_x2 = x2 /frame.cols;
        float n_y2 = y2 / frame.rows;

        // Calculate angle 
        float angle = std::atan2(n_y2-n_y1, n_x2-n_x1) * (180.0f/CV_PI); 

        // Filter out near-horizontal and near-vertical lines
        if (std::abs(angle) > 20 && std::abs(angle) < 85) {
            float line_length = std::sqrt(std::pow(n_x2-n_x1, 2) + std::pow(n_y2-n_y1, 2));

            if (line_length > 0.1) {
                MLPP::Mat2d<double>& target_lines = (x1 + x2) / 2 < frame_center ? negative_slope_lines : positive_slope_lines;
                target_lines.push_back({x1, y1, x2, y2, angle});
                angles.push_back(angle);
            }
        }
    }

    auto ransac_filter = [](MLPP::Mat2d<double>& lines, float median_angle) {
        const int num_iterations = 100;
        const float inlier_threshold = 5.0f;
        MLPP::Mat2d<double> best_inliers;
        int max_inliers = 0;

        for (int i = 0; i < num_iterations; i++) {
            int idx = rand() % lines.size();
            float m = std::tan(lines[idx][4] * CV_PI/180.0f);
            float b = lines[idx][1] - (m * lines[idx][0]);

            MLPP::Mat2d<double> current_inliers;
            for (const auto& line : lines)
            {
                float dist = std::abs(line[1] - (m* line[0] + b)) + std::abs(line[3] - (m * line[2] + b));
                if (dist < inlier_threshold && std::abs(line[4] - median_angle) < 15) {
                    current_inliers.push_back(line);
                }
                
            }

            if (current_inliers.size() > max_inliers) {
                max_inliers = current_inliers.size();
                best_inliers = current_inliers;
            }
            
        }

        lines = best_inliers;
    };

    if (!angles.empty()) {
        std::sort(angles.begin(), angles.end());
        float median_angle = angles[angles.size()/2];

        ransac_filter(positive_slope_lines, median_angle);
        ransac_filter(negative_slope_lines, -median_angle);
    } */
    
}

// Implementation of extract hough lines method
std::vector<cv::Vec4i> FrameProcessing::extract_hough_lines(const cv::Mat& frame)
{
    std::vector<cv::Vec4i> lines; // A Vec4i is a type holding 4 integers
	cv::HoughLinesP(frame, lines, rho, theta, threshold, min_line_length, max_line_gap);
    return lines;
}

// Implementation of get average slopes method
// First value is positive slopes average
// Second value is negative slopes average
std::vector<float> FrameProcessing::get_average_positive_and_negative_slopes_lines(const MLPP::Mat3d<double>& slope_lines)
{
    // Add positive slopes from slope_lines[0] into a vector positive slopes
    std::vector<float> positive_slopes;
    float positive_sum {0.0}; // Initialize sum so we can get the mean
    for (size_t i = 0; i != slope_lines[0].size(); ++i) {
        positive_slopes.push_back(slope_lines[0][i][4]);
        positive_sum += positive_slopes[i];
    }

    // Get mean of valid positive slopes
    float positive_slope_mean = positive_sum / positive_slopes.size(); 

    // Add negative slopes from slope_lines[1] into a vector negative slopes
    std::vector<float> negative_slopes;
    float negative_sum {0.0}; // Initialize sum so we can get the mean
    for (size_t i = 0; i != slope_lines[1].size(); ++i) {
        negative_slopes.push_back(slope_lines[1][i][4]);
        negative_sum += negative_slopes[i];
    }

    float negative_slope_mean = negative_sum / negative_slopes.size(); 

    return std::vector<float> { positive_slope_mean, negative_slope_mean };  

    /* 
    // Add positive slopes from slope_lines[0] into a vector positive slopes
    std::vector<float> positive_slopes;
    for (size_t i = 0; i != slope_lines[0].size(); ++i) {
        positive_slopes.push_back(slope_lines[0][i][4]);
    }
    
    // Get median of positive slopes
    std::sort(positive_slopes.begin(), positive_slopes.end()); // sort slopes
    int middle_positive; // Define middle value
    double positive_slope_median; // Define positive slope median

    // If even number of elements in vector, take average of two middle values
    if (positive_slopes.size() % 2 == 0) {
        // Get middle index of vector
        middle_positive = positive_slopes.size() / 2;
        // Take average of middle values
        positive_slope_median = (positive_slopes[middle_positive-1] + positive_slopes[middle_positive+1]) / 2;
    }
    // Odd number of values in the vector
    else {
        // Get middle index of vector
        middle_positive = positive_slopes.size() / 2;
        // Get middle value
        positive_slope_median = positive_slopes[middle_positive];
    }

    // Define vector of valid positive slopes, outliers are discarded
    std::vector<float> valid_positive_slopes;
    float positive_sum {0.0}; // Initialize sum so we can get the mean

    // Loop through positive slopes and discard the outliers
    for (size_t i = 0; i != positive_slopes.size(); i++)
    {
        // Check difference between current slope and the median
        // If difference is too big we consider it an outlier and ignore
        if (std::abs(positive_slopes[i] - positive_slope_median) < positive_slope_median ) {
            valid_positive_slopes.push_back(positive_slopes[i]); // Add slope to valid slopes vector
            positive_sum += positive_slopes[i]; // Add to sum
        }
    }

    // Get mean of valid positive slopes
    float positive_slope_mean = positive_sum / valid_positive_slopes.size(); 

    // Now we do the same for negative slopes

    // Add negative slopes from slope_lines[1] into a vector negative slopes
    std::vector<float> negative_slopes;
    for (size_t i = 0; i != slope_lines[1].size(); ++i) {
        negative_slopes.push_back(slope_lines[1][i][4]);
    }

    // Get median of negative slopes
    std::sort(negative_slopes.begin(), negative_slopes.end()); // sort slopes
    int middle_negative; // Define middle value
    double negative_slope_median; // Define negative slope median

    // If even number of elements in vector, take average of two middle values
    if (negative_slopes.size() % 2 == 0) {
        // Get middle index of vector
        middle_negative = negative_slopes.size() / 2;
        // Take average of middle values
        negative_slope_median = (negative_slopes[middle_negative-1] + negative_slopes[middle_negative+1]) / 2;
    }
    // Odd number of values in the vector
    else {
        // Get middle index of vector
        middle_negative = negative_slopes.size() / 2;
        // Get middle value
        negative_slope_median = negative_slopes[middle_negative];
    }

     // Define vector of valid negative slopes, outliers are discarded
    std::vector<float> valid_negative_slopes;
    float negative_sum {0.0}; // Initialize sum so we can get the mean

    // Loop through negative slopes and discard the outliers
    for (size_t i = 0; i != negative_slopes.size(); i++)
    {
        // Check difference between current slope and the median
        // If difference is too big we consider it an outlier and ignore
        if (std::abs(negative_slopes[i] - negative_slope_median) < 0.9) {
            valid_negative_slopes.push_back(negative_slopes[i]); // Add slope to valid slopes vector
            negative_sum += negative_slopes[i]; // Add to sum
        }
    }

    // Get mean of valid positive slopes
    float negative_slope_mean = negative_sum / valid_negative_slopes.size();
    
    return std::vector<float> { positive_slope_mean, negative_slope_mean }; */
    
}

// Implementation of get X intercept coordinates method
// First value is average positive slopes X coordinate
// Second value is average negative slopes X coordinate
std::vector<double>  FrameProcessing::get_x_intercept_coordinates(const cv::Mat& frame, const MLPP::Mat3d<double>& slope_lines)
{
    // Define vector for X intercept coordinates for positive slope lines
    std::vector<double> positive_slope_lines_x_intercept_coordinates;

    // Loop through positive slope lines, find and store x intercept values
    for (size_t i = 0; i != slope_lines[0].size(); i++)
    {
        double x1 = slope_lines[0][i][0]; // Get current x value
        double y1 = frame.rows - slope_lines[0][i][1]; // Get current y value
        double slope = slope_lines[0][i][4];
        double y_intercept = y1 - slope * x1; // Y intercept of current line
        double x_intercept = -y_intercept/slope; // Find x intercept based off y = mx+b

        // Check for nan
        if (std::isnan(x_intercept) == 0) {
            positive_slope_lines_x_intercept_coordinates.push_back(x_intercept); // Add value to vector
        }
    }

    // Get median of X intercepts for positive slope lines
    double x_intercept_positive_median = median(positive_slope_lines_x_intercept_coordinates);
    
    // Define vector for storing valid X intercept values, outliers are discarded
    std::vector<double> valid_positive_x_intercept_values;
    double x_intercept_positive_sum; // Necessary for finding average

    // Now that we have the median, loop through lines and compare values against median
    for (size_t i = 0; i != slope_lines[0].size(); i++) {
        double x1 = slope_lines[0][i][0]; // Get current x value
        double y1 = frame.rows - slope_lines[0][i][1]; // Get current y value
        double slope = slope_lines[0][i][4];
        double y_intercept = y1 - slope * x1; // Y intercept of current line
        double x_intercept = -y_intercept/slope; // Find x intercept based off y = mx+b

        // Check for nan
        if (std::isnan(x_intercept) == 0) {
            valid_positive_x_intercept_values.push_back(x_intercept); // Add value to valid vector
            x_intercept_positive_sum += x_intercept;
        }
    }

    // Get mean x intercept value for positive slope lines
    double positive_x_intercept_mean = x_intercept_positive_sum / valid_positive_x_intercept_values.size();

    // Now we do the same for negative slope lines

    // Define vector for X intercept coordinates for positive slope lines
    std::vector<double> negative_slope_lines_x_intercept_coordinates;

    // Loop through negative slope lines, find and store x intercept values
    for (size_t i = 0; i != slope_lines[1].size(); i++)
    {
        double x1 = slope_lines[1][i][0]; // Get current x value
        double y1 = frame.rows - slope_lines[1][i][1]; // Get current y value
        double slope = slope_lines[1][i][4];
        double y_intercept = y1 - slope * x1; // Y intercept of current line
        double x_intercept = -y_intercept/slope; // Find x intercept based off y = mx+b

        // Check for nan
        if (std::isnan(x_intercept) == 0) {
            negative_slope_lines_x_intercept_coordinates.push_back(x_intercept); // Add value to vector
        }
    }

    if (negative_slope_lines_x_intercept_coordinates.empty()) {
        negative_slope_lines_x_intercept_coordinates.push_back(500);
    }

    // Get median of X intercepts for negative slope lines
    double negative_x_intercept_median = median(negative_slope_lines_x_intercept_coordinates);
    
    // Define vector for storing valid X intercept values, outliers are discarded
    std::vector<double> valid_negative_x_intercept_values;
    double x_intercept_negative_sum; // Necessary for finding average

    // Now that we have the median, loop through lines and compare values against median
    for (size_t i = 0; i != slope_lines[1].size(); i++) {
        double x1 = slope_lines[1][i][0]; // Get current x value
        double y1 = frame.rows - slope_lines[1][i][1]; // Get current y value
        double slope = slope_lines[1][i][4];
        double y_intercept = y1 - slope * x1; // Y intercept of current line
        double x_intercept = -y_intercept/slope; // Find x intercept based off y = mx+b
        // Check for nan
        if (std::isnan(x_intercept) == 0 && std::abs(x_intercept - negative_x_intercept_median) < negative_x_intercept_median * 0.35) {
            valid_negative_x_intercept_values.push_back(x_intercept); // Add value to valid vector
            x_intercept_negative_sum += x_intercept;
        }
    }

    // Get mean x intercept value for negative slope lines
    double negative_x_intercept_mean = x_intercept_negative_sum / valid_negative_x_intercept_values.size();

    //std::cout << positive_x_intercept_mean << std::endl;
    //std::cout << negative_x_intercept_mean << std::endl;

    return std::vector<double> { positive_x_intercept_mean, negative_x_intercept_mean };;
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

// Implementation of plot lane line method
void FrameProcessing::plot_lane_lines(const cv::Mat& frame, const std::vector<float>& avrg_slope_lines, const std::vector<double>& x_intercept_coord, 
                            LineParameters& positive_slope_parameters, LineParameters& negative_slope_parameters)
{
    if (!avrg_slope_lines.empty() && !x_intercept_coord.empty()) {
        // Positive Slope Line
        float positive_slope = avrg_slope_lines[0];
        double positive_x1  = x_intercept_coord[0];

        if (positive_x1 < 0 || std::isnan(positive_x1) || std::isinf(positive_x1)) {
            positive_x1 = 0;
        }

        int positive_y1 = 0;
        double positive_y2 = frame.rows - (frame.rows - frame.rows * 0.35);
        double positive_x2  = (positive_y2 - positive_y1) / positive_slope + positive_x1;

        if (positive_x2 < 0 || std::isnan(positive_x2) || std::isinf(positive_x2)) {
            positive_x2 = 10;
        }

        positive_x1 = int(positive_x1 + 0.5);
        positive_x2 = int(positive_x2 + 0.5);
        positive_y1 = int(positive_y1 + 0.5);
        positive_y2 = int(positive_y2 + 0.5);

        positive_slope_parameters.slope = positive_slope;
        positive_slope_parameters.x1 = positive_x1;
        positive_slope_parameters.x2 = positive_x2;
        positive_slope_parameters.y1 = positive_y1;
        positive_slope_parameters.y2 = positive_y2;

        //cv::line(frame, cv::Point(positive_x1, frame.size().height-positive_y1), cv::Point(positive_x2, frame.size().height - positive_y2), cv::Scalar(0,255,0), 3, 8 );

        // Negative Slope Line
        float negative_slope = avrg_slope_lines[1];
        double negative_x1  = x_intercept_coord[1];

        if (negative_x1 > frame.cols || std::isnan(negative_x1) || std::isinf(negative_x1)) {
            negative_x1 = frame.cols;
        }

        int negative_y1 = 0;
        double negative_y2 = frame.rows - (frame.rows - frame.rows * 0.35);
        double negative_x2  = (negative_y2 - negative_y1) / negative_slope + negative_x1;

        if (negative_x2 > frame.cols ||std::isnan(negative_x2) || std::isinf(negative_x2)) {
            negative_x2 = frame.cols -10;
        }

        negative_x1 = int(negative_x1 + 0.5);
        negative_x2 = int(negative_x2 + 0.5);
        negative_y1 = int(negative_y1 + 0.5);

        negative_slope_parameters.slope = negative_slope;
        negative_slope_parameters.x1 = negative_x1;
        negative_slope_parameters.x2 = negative_x2;
        negative_slope_parameters.y1 = negative_y1;
        negative_slope_parameters.y2 = negative_y2;

        //cv::line(frame, cv::Point(negative_x1, frame.size().height-negative_y1), cv::Point(negative_x2, frame.size().height - negative_y2), cv::Scalar(0,255,0), 3, 8 );
    }

    // Plot positive and negative lane lines
	//cv::imshow("Lane lines on image", frame);
	//cv::waitKey(0); // wait for a key press
}

// Method for reisizing frames implementation
void FrameProcessing::resize_frame(cv::Mat& frame, const size_t& x, const size_t& y)
{
    cv::resize(frame, frame, cv::Size(x,y));
}
