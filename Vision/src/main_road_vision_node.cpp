#include <iostream>
#include "frame_processing.cpp"
#include "guidance_system.cpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp/rclcpp.hpp"
#include "../ai_libraries/mlpp.hpp"
#include "std_msgs/std_msgs/msg/int8.hpp"

using namespace std;
using namespace std::chrono_literals;
using namespace MLPP;

// Timer callback function
void timer_callback(rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr publisher, rclcpp::Node::SharedPtr node, 
						std::shared_ptr<int8_t> servo_steering_angle)
{
    auto message = std_msgs::msg::Int8();
	message.data = *servo_steering_angle;
    RCLCPP_INFO(node->get_logger(), "Publishing: '%i'", message.data);
    publisher->publish(message);
}

int main(int argc, char * argv[])
{
	#pragma region ROS INITIALIZATION
	// Initialize ROS 2
	rclcpp::init(argc, argv);
	// Create a ROS 2 node
	auto node = std::make_shared<rclcpp::Node>("road_vision");
	// Create a ROS 2 publisher
	auto publisher = node->create_publisher<std_msgs::msg::Int8>("road_vision_topic", 10);
	// Create a shared variable for steering angle
    auto servo_steering_angle = std::make_shared<int8_t>(0);
	// Create a timer with a lambda function, so we publish steering angle every 100 ms 
    auto timer = node->create_wall_timer(
        100ms, 
        [publisher, node, servo_steering_angle]() {
            timer_callback(publisher, node, servo_steering_angle);
        }
    );

	// Start road vision node on a separate thread from video while loop
	thread spin_thread([&]() 
	{
		rclcpp::spin(node);
	});

	#pragma endregion

	// Set desired resolution to resize received footage
	int width = 240;
	int height = 240;
	
    // Open video file
    cv::VideoCapture cap(2);
	//cv::VideoCapture cap("/home/muzaodamassa/Downloads/VideosPista/WIN_20240819_17_34_14_Pro.mp4");

    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    // Create a window to display the video
    cv::namedWindow("Road Vision Stream", cv::WINDOW_NORMAL);

	// Create a window to display the video
    cv::namedWindow("Debug Stream", cv::WINDOW_NORMAL);
    
	// Set capture resolution
	cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	//cap.set(cv::CAP_PROP_FPS, 15);

	// Variables for fps calculation
    double fps;
    cv::TickMeter tm;

	// Frame Processing class constructor
	FrameProcessing frame_processing;
	// Guidance System class constructor
	GuidanceSystem guidance_system;
	// PID Controller class constructor
	PIDController pid_controller(0.35, 0.01, 0);

	// Constants for actual servo calculations
	const double servo_error = 0.7;
	const double servo_compensation = 3.5;
	const int servo_center_anlge = 90;

    while (cap.isOpened())
    {
		tm.start();

		// Capture frame-by-frame
		cv::Mat frame;
		cap >> frame;

		// If the frame is empty, break immediately
		if (frame.empty()) {
			break;
		}

		cv::Mat p_frame = frame.clone();

		frame_processing.resize_frame(frame, height, width);
		frame_processing.resize_frame(p_frame, height, width);
		frame_processing.apply_pre_processing_to_frame(p_frame);

		cv::imshow("Debug Stream", p_frame); // Show processed frame
		
		auto h_lines = frame_processing.extract_hough_lines(p_frame);

		if (!h_lines.empty() && h_lines.size() > 2) {
			// Separate Hough lines into posistive and negative sloped lines
			Mat3d<double> positive_and_negative_h_lines = frame_processing.classify_hough_lines(p_frame, h_lines);

			if (!positive_and_negative_h_lines.empty()) {
				// Get average of positive of and negative sloped hough lines
				std::vector<float> avrg_pos_and_neg_lines = frame_processing.get_average_positive_and_negative_slopes_lines(positive_and_negative_h_lines);
				// Get X intercet coordinates for average positive and negative sloped hough lines 
				std::vector<double> x_intercept_coordinates = frame_processing.get_x_intercept_coordinates(p_frame, positive_and_negative_h_lines);

				// Initialize structs to hold positive and negative sloped line parameters
				FrameProcessing::LineParameters positive_slope_line_parameters;
				FrameProcessing::LineParameters negative_slope_line_parameters;
				
				// Fill structs constructed above with x1,y1,x2,y2 and slope values for positive and negative slope lines
				frame_processing.plot_lane_lines(p_frame, avrg_pos_and_neg_lines, x_intercept_coordinates, 
													positive_slope_line_parameters, negative_slope_line_parameters);

				// Calculate steering angle based on plotted lane lines
				double steering_angle = guidance_system.calc_steering_angle(p_frame, positive_slope_line_parameters, negative_slope_line_parameters);
				
			
				// Get error correction with PID
				//double dt = 0.1; // 100 ms
				//double dt = pid_controller.get_dt();
				//double error = pid_controller.compute_error(steering_angle, dt);
				// Make necessary adjustments to compensate for servo
				//double new_servo_angle = (((error * servo_compensation) + servo_center_anlge) * 0.7);
				//*servo_steering_angle = static_cast<int8_t>(new_servo_angle);

				// Get error correction without PID
				double angle_correction = (servo_center_anlge - steering_angle) * servo_compensation;
				*servo_steering_angle = (servo_center_anlge + angle_correction) * servo_error;

				cv::Mat blended_frame = frame_processing.blend_frame(frame, positive_slope_line_parameters, negative_slope_line_parameters);

				// Calculate FPS
				tm.stop();
				fps = 1.0 / tm.getTimeSec();
				tm.reset();

				// Put FPS text on the frame
				cv::putText(blended_frame, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

				// Show final frame
				cv::imshow("Road Vision Stream", blended_frame);
			}
			else {
				// Calculate FPS
				tm.stop();
				fps = 1.0 / tm.getTimeSec();
				tm.reset();

				// Put FPS text on the frame
				cv::putText(frame, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

				// Dispay the resulting frame
				cv::imshow("Road Vision Stream", frame);
			}
        }
		else {
			// Calculate FPS
			tm.stop();
			fps = 1.0 / tm.getTimeSec();
			tm.reset();

			// Put FPS text on the frame
			cv::putText(frame, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
			
			// Dispay the resulting frame
			cv::imshow("Road Vision Stream", frame);
		}

		// Press q to exit the loop
		if (cv::waitKey(1) == 'q') {
			break;
		}	
    }

	// Whene everything is done, release the video capture and video writer objects
	cap.release();

    // Close all windows
	cv::destroyAllWindows();

 	// Shut down ROS 2
	rclcpp::shutdown();
	
	// Join the spin thread before exiting
    spin_thread.join();

	return 0;
}

// TESTING
int main2()
{
	// Set desired resolution to resize received footage
	int width = 500;
	int height = 500;
	
	cv::VideoCapture cap("/home/muzaodamassa/Downloads/VideosPista/WIN_20240819_17_34_14_Pro.mp4");

	// Check if camera opened successfully
    if (!cap.isOpened())
    {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    // Create a window to display the video
    cv::namedWindow("Video Stream", cv::WINDOW_NORMAL);

	// Set capture resolution
	//cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
	//cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	// Frame Processing class constructor
	FrameProcessing frame_processing;
	// Guidance System class constructor
	GuidanceSystem guidance_system;
	// PID Controller class constructor
	PIDController pid_controller(0.15, 0.001, 0.005);

	while (cap.isOpened())
	{
		cv::Mat frame;
		cap >> frame;

		// If the frame is empty, break immediately
		if (frame.empty()) {
			break;
		}

		cv::Mat p_frame = frame.clone(); // Store processed frame here

		frame_processing.resize_frame(frame, height, width);
		frame_processing.resize_frame(p_frame, height, width);
		frame_processing.apply_gray_scale_to_frame(p_frame);
		frame_processing.apply_smoothing_to_frame(p_frame);
		frame_processing.apply_edge_detection_to_frame(p_frame);
		frame_processing.apply_mask_to_frame(p_frame);
		auto h_lines = frame_processing.extract_hough_lines(p_frame);

		if (!h_lines.empty() && h_lines.size() > 2) {
			// Separate Hough lines into posistive and negative sloped lines
			Mat3d<double> positive_and_negative_h_lines = frame_processing.classify_hough_lines(p_frame, h_lines);

			if (!positive_and_negative_h_lines[0].empty() && !positive_and_negative_h_lines[1].empty()) {
				// Get average of positive of and negative sloped hough lines
				std::vector<float> avrg_pos_and_neg_lines = frame_processing.get_average_positive_and_negative_slopes_lines(positive_and_negative_h_lines);
				// Get X intercet coordinates for average positive and negative sloped hough lines 
				std::vector<double> x_intercept_coordinates = frame_processing.get_x_intercept_coordinates(p_frame, positive_and_negative_h_lines);

				// Initialize structs to hold positive and negative sloped line parameters
				FrameProcessing::LineParameters positive_slope_line_parameters;
				FrameProcessing::LineParameters negative_slope_line_parameters;
				
				// Fill structs constructed above with x1,y1,x2,y2 and slope values for positive and negative slope lines
				frame_processing.plot_lane_lines(p_frame, avrg_pos_and_neg_lines, x_intercept_coordinates, 
													positive_slope_line_parameters, negative_slope_line_parameters);

				//std::cout << positive_slope_line_parameters.x1 << std::endl;
				//std::cout << negative_slope_line_parameters.x1 << std::endl;

				// Calculate steering angle based on plotted lane lines
				double angle_correction = guidance_system.calc_steering_angle(p_frame, positive_slope_line_parameters, negative_slope_line_parameters);
				std::cout << "~~~~~~~~~~~~~" << std::endl;
				std::cout << angle_correction << std::endl;

				// Get delta time
				double dt = pid_controller.get_dt();
				double error = pid_controller.compute_error(angle_correction, dt);
				std::cout << error << std::endl;
				std::cout << "~~~~~~~~~~~~~" << std::endl;
				
				cv::Mat blended_frame = frame_processing.blend_frame(frame, positive_slope_line_parameters, negative_slope_line_parameters);

				// Show final frame
				cv::imshow("Video Stream", blended_frame);
			}
			else {
			// Dispay the resulting frame
			cv::imshow("Video Stream", frame);
			}
        }
		else {
			// Dispay the resulting frame
			cv::imshow("Video Stream", frame);
		}
		

		// Press q to exit the loop
		if (cv::waitKey(1) == 'q') {
			break;
		}

		//cv::waitKey(50);
	}

	return 0;
}

int main3()
{
	FrameProcessing frame_processing;
	GuidanceSystem guidance_system;

	cv::Mat image = cv::imread(
        "/home/muzaodamassa/Downloads/High_Angle_Straight.jpg"); // read the image into mat type variable, image, if
                        // funning from cmd line, path is ../src/1.jpg

    // Check for failure in reading the image
    if (image.empty()) // check if it's empty
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // name of window to show original image
    cv::String originalWindowName = "Original Image";

	frame_processing.resize_frame(image, 500, 500);

    // Show the original image
	cv::namedWindow(originalWindowName, cv::WINDOW_NORMAL);
    imshow(originalWindowName, image);
    cv::waitKey(0); // Wait for any keystroke in the window

	cv::Mat p_frame = image.clone(); // Store processed frame here

	frame_processing.apply_gray_scale_to_frame(p_frame);

	// window for grayscaled image
    cv::String grayscaleWindowName = "Grayscaled image";

    // Show grayscale image
	cv::namedWindow(grayscaleWindowName, cv::WINDOW_NORMAL);
    cv::imshow(grayscaleWindowName, p_frame);
    cv::waitKey(0); // wait for a key press

	frame_processing.apply_smoothing_to_frame(p_frame);

	// window for smoothed image
    cv::String smoothedWindowName = "Smoothed image";

    // Show smoothed image
	cv::namedWindow(smoothedWindowName, cv::WINDOW_NORMAL);
    cv::imshow(smoothedWindowName, p_frame);
    cv::waitKey(0); // wait for a key press

	frame_processing.apply_edge_detection_to_frame(p_frame);
	
	// window for edge detection image
    cv::String edgeWindowName = "edge detection image";

    // Show edge detection image
	cv::namedWindow(edgeWindowName, cv::WINDOW_NORMAL);
    cv::imshow(edgeWindowName, p_frame);
    cv::waitKey(0); // wait for a key press

	frame_processing.apply_mask_to_frame(p_frame);

	// Show masked image
	cv::namedWindow("Masked Image", cv::WINDOW_NORMAL);
    cv::imshow("Masked Image", p_frame);
    cv::waitKey(0);

	auto h_lines = frame_processing.extract_hough_lines(p_frame);

	// Check if we got more than one line
    if (!h_lines.empty() && h_lines.size() > 2) {

        // Initialize lines image
        cv::Mat allLinesIm(
            image.size().height, image.size().width, CV_8UC3,
            cv::Scalar(0, 0, 0)); // CV_8UC3 to make it a 3 channel)

        // Loop through lines
        // std::size_t can store the maximum size of a theoretically possible
        // object of any type
        for (size_t i = 0; i != h_lines.size(); ++i) {

            // Draw line onto image
            cv::line(allLinesIm, cv::Point(h_lines[i][0], h_lines[i][1]),
                     cv::Point(h_lines[i][2], h_lines[i][3]), cv::Scalar(0, 0, 255),
                     3, 8);
        }

        // Display images
		cv::namedWindow("Hough Lines", cv::WINDOW_NORMAL);
        cv::imshow("Hough Lines", allLinesIm);
        cv::waitKey(0);

		Mat3d<double> positive_and_negative_h_lines = frame_processing.classify_hough_lines(p_frame, h_lines);
		std::vector<float> avrg_pos_and_neg_lines = frame_processing.get_average_positive_and_negative_slopes_lines(positive_and_negative_h_lines);
		std::vector<double> x_intercept_coordinates = frame_processing.get_x_intercept_coordinates(image, positive_and_negative_h_lines);

		// Initialize structs to hold positive and negative sloped line parameters
		FrameProcessing::LineParameters positive_slope_line_parameters;
		FrameProcessing::LineParameters negative_slope_line_parameters;
			
		// Fill structs constructed above with x1,y1,x2,y2 and slope values for positive and negative slope lines
		frame_processing.plot_lane_lines(allLinesIm, avrg_pos_and_neg_lines, x_intercept_coordinates, 
												positive_slope_line_parameters, negative_slope_line_parameters);
 
		double angle_correction = guidance_system.calc_steering_angle(image, positive_slope_line_parameters, negative_slope_line_parameters);
		std::cout << angle_correction << std::endl;

		cv::Mat blended_image = frame_processing.blend_frame(image, positive_slope_line_parameters, negative_slope_line_parameters);

        // Plot positive and negative lane lines
		cv::namedWindow("Final Output", cv::WINDOW_NORMAL);
        cv::imshow("Final Output", blended_image);
        cv::waitKey(0); // wait for a key press  
	}

	return 0;
}



