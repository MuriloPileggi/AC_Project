#include <iostream>
#include "frameProcessing.cpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp/rclcpp.hpp"
#include "../ai_libraries/mlpp.hpp"
#include "std_msgs/std_msgs/msg/string.hpp"
#include "std_srvs/std_srvs/srv/trigger.hpp"

using namespace std;
using namespace MLPP;

// TESTING
void drawPolygon(cv::Mat& img, const vector<cv::Point>& points, cv::Scalar color, int thickness = 2) {
    // Draw the polygon using polylines
    const cv::Point* pts = &points[0];
    int npts = (int)points.size();
    polylines(img, &pts, &npts, 1, true, color, thickness);
}

int main(int argc, char * argv[])
{
	#pragma region ROS INTEGRATION
	// Initialize ROS 2
	rclcpp::init(argc, argv);
	// Create a ROS 2 node
	auto node = std::make_shared<rclcpp::Node>("vision_talker_1");
	// Create a ROS 2 publisher
	auto publisher = node->create_publisher<std_msgs::msg::String>("frame_classification", 10);
	// Create a ROS 2 service client
	auto client = node->create_client<std_srvs::srv::Trigger>("trigger_action");

	#pragma endregion

	// Set desired resolution to resize received footage
	int width = 500;
	int height = 500;
	
    // Open video file
    //cv::VideoCapture cap(0);
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
	cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	// Variables for fps calculation
    double fps;
    cv::TickMeter tm;

	// Lane detection class
	FrameProcessing frame_processing;
	const int car_center = width/2;
	double last_frame_anlge = 0.0;
	double diff_between_frames = 0.0;

	// Create neural network class to load model into
	//NeuralNetwork cnn;
	//cnn.load_model("/home/muzaodamassa/AC_Project/Vision/models/road_vision_high3_25.bin");
	// TESTING
	//int class_0_counter = 0;
	//int class_1_counter = 0;
	//int class_2_counter = 0;
	//int class_3_counter = 0;
	//int class_4_counter = 0;

    while (cap.isOpened())
    {
		tm.start();

		// Capture frame-by-frame
		cv::Mat* frame = new cv::Mat();
		cap >> *frame;

		// If the frame is empty, break immediately
		if (frame->empty()) {
			break;
		}

		// Create new pointer to store processed frame
		//cv::Mat* processed_frame_ptr = new cv::Mat();
		//*processed_frame_ptr = *frame;

		frame_processing.apply_gray_scale_to_frame(frame);
		frame_processing.apply_smoothing_to_frame(frame);
		frame_processing.apply_edge_detection_to_frame(frame);
		frame_processing.apply_mask_to_frame(frame);
		auto h_lines = frame_processing.find_hough_lines(*frame);
		double steering_angle = 0.0;
		
		if (!h_lines.empty() && h_lines.size() > 2) {
			steering_angle = frame_processing.calc_steering_angle(h_lines, car_center);
			diff_between_frames = steering_angle - last_frame_anlge;
		}

		/* Neural network code

		//frame_processing.apply_gray_scale_to_frame(processed_frame_ptr); // Convert the frame to grayscale
		//frame_processing.apply_smoothing_to_frame(processed_frame_ptr); // Apply gaussian smoothing to frame
		//frame_processing.apply_edge_detection_to_frame(processed_frame_ptr); // Apply edge detection to frame
        //frame_processing.resize_frame(processed_frame_ptr, 24, 24); // Resize frame to desired spacial dimensions (Rows and Columns)
		// Convert frame into MLPP matrix data structure
        //Mat3d<double>* converted_frame = OpenCvIntegration::convert_gray_image<double>(processed_frame_ptr);
        // Normalize pixel data
        //Mat3d<double> normalized_frame = ComputerVision::normalize_pixels<double>(converted_frame);
        // Feed converted frame data into neural network model so it can predict its class
        //size_t frame_predicted_class = cnn.predict_frame<double>(normalized_frame); 

		// Evaluating models
		switch (frame_predicted_class)
		{
		case 0:
			class_0_counter++;
			break;
		case 1:
			class_1_counter++;
			break;
		case 2:
			class_2_counter++;
			break;
		case 3:
			class_3_counter++;
			break;
		default:
			class_4_counter++;
			break;
		} */
		
		// Get processed frame and publish result
		auto message = std_msgs::msg::String();
		message.data = std::to_string(diff_between_frames);  
		RCLCPP_INFO(node->get_logger(), "Publishing: '%s'", message.data.c_str());
		publisher->publish(message);

		/* // Wait for the service to be available
		while (!client->wait_for_service(1s)) {
			if (!rclcpp::ok()) {
				RCLCPP_ERROR(node->get_logger(), "Error: Interrupted while waiting for the service. Exiting.");
				return -1;
			}
			RCLCPP_INFO(node->get_logger(), "Waiting for service to appear...");
		}
		
		// Call the service and wait for the response
		auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
		auto future = client->async_send_request(request);

		// Wait for service actions to complete
		if (rclcpp::spin_until_future_complete(node, future) == rclcpp::FutureReturnCode::SUCCESS) {
			auto response = future.get();
			RCLCPP_INFO(node->get_logger(), "Service responded with: %s", response->message.c_str());
		}
		else {
            RCLCPP_ERROR(node->get_logger(), "Service call failed");
        }
		 */

		// Calculate FPS
		tm.stop();
		fps = 1.0 / tm.getTimeSec();
		tm.reset();

		// Display purpose only
		//frame_processing.apply_gray_scale_to_frame(frame); // Convert the frame to grayscale
		//frame_processing.apply_smoothing_to_frame(frame); // Apply gaussian smoothing to frame
		//frame_processing.apply_edge_detection_to_frame(frame); // Apply edge detection to frame

		// Put detected frame class on Video Stream window
        //cv::putText(*frame, "Frame Class: " + to_string(frame_predicted_class), cv::Point(350, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
		// Put FPS text on the frame
		cv::putText(*frame, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
		
		// Dispay the resulting frame
		cv::imshow("Video Stream", *frame);

		// Press q to exit the loop
		if (cv::waitKey(1) == 'q') {
			break;
		}	

		delete frame;

		last_frame_anlge = steering_angle;
    }

	// Whene everything is done, release the video capture and video writer objects
	cap.release();

    // Close all windows
	cv::destroyAllWindows();

    // Shut down ROS 2
	rclcpp::shutdown();

	// Evaluating CNN models
	//std::cout << class_0_counter << std::endl;
	//std::cout << class_1_counter << std::endl;
	//std::cout << class_2_counter << std::endl;
	//std::cout << class_3_counter << std::endl;
	//std::cout << class_4_counter << std::endl;
	
	return 0;
}


// TESTING
int main2()
{	
	cv::Mat image = cv::imread("/home/muzaodamassa/Downloads/High_Angle_Right.jpg", 1);

	if (!image.data)
	{
		cout<<"No image data!"<<endl;
		return -1;
	}
	
	FrameProcessing frame_processing;

	frame_processing.resize_frame(&image, 500, 500);
	frame_processing.apply_gray_scale_to_frame(&image);
	frame_processing.apply_smoothing_to_frame(&image);
	frame_processing.apply_edge_detection_to_frame(&image);
	frame_processing.apply_mask_to_frame(&image);

	// Make mask shape visible for better visualization of how mask is being applied
	cv::Point topLeft(0,image.rows/3);  // Top-left of trapezoid
    cv::Point topRight(image.cols, image.rows/3);  // Top-right of trapezoid
    cv::Point bottomLeft(0, image.rows/3 + image.rows/3);  // Bottom-left of trapezoid
    cv::Point bottomRight(image.cols, image.rows/3 + image.rows/3);  // Bottom-right of trapezoid
    // Create the trapezoid points
    vector<cv::Point> trapezoid;
    trapezoid.push_back(topLeft);
    trapezoid.push_back(topRight);
    trapezoid.push_back(bottomRight);
    trapezoid.push_back(bottomLeft);
	drawPolygon(image, trapezoid, cv::Scalar(255, 255, 255), 3);  // Red trapezoid (Field of View)

	cv::namedWindow("Test Image", cv::WINDOW_NORMAL);
	cv::imshow("Test Image", image);
	cv::waitKey(0);

	auto h_lines = frame_processing.find_hough_lines(image);

	if (!h_lines.empty() && h_lines.size() > 2) {
		// Initialize lines image
		cv::Mat allLinesIm(image.size().height, image.size().width, CV_8UC1, cv::Scalar(0,0,0)); // CV_8UC3 to make it a 3 channel)

		// Loop through lines
		// std::size_t can store the maximum size of a theoretically possible object of any type
		for (size_t i = 0; i != h_lines.size(); ++i) {
			// Draw line onto image
			cv::line(allLinesIm, cv::Point(h_lines[i][0], h_lines[i][1]),
			cv::Point(h_lines[i][2], h_lines[i][3]), cv::Scalar(255,255,255), 3, 8 );
		}
		// Display images
		cv::namedWindow("Hough Lines", cv::WINDOW_NORMAL);
		cv::imshow("Hough Lines", allLinesIm);
		cv::waitKey(0);

		double steering_angle = frame_processing.calc_steering_angle(h_lines, image.cols/2);
		std::cout << steering_angle << std::endl;
	}

	

	return 0;
}
