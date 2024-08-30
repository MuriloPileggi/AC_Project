#include <iostream>
#include "frameProcessing.cpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp/rclcpp.hpp"
#include "std_msgs/std_msgs/msg/string.hpp"
#include "std_srvs/std_srvs/srv/trigger.hpp"
#include "../ai_libraries/mlpp.hpp"
#include "../../Serial/src/serialCommunication.cpp"

using namespace std;
using namespace MLPP;

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
    cv::VideoCapture cap(0);
	//cv::VideoCapture cap("/home/muzaodamassa/Downloads/VideosPista/Pista/Reta_4_2.mp4");

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
	frameProcessing pFrame;

	// Create neural network class to load model into
	NeuralNetwork cnn;
	cnn.load_model("/home/muzaodamassa/AC_Project/Vision/models/high_angle3_model_25.bin");

	// TESTING
	int class_0_counter = 0;
	int class_1_counter = 0;
	int class_2_counter = 0;
	int class_3_counter = 0;
	int class_4_counter = 0;

    while (cap.isOpened())
    {
		tm.start();

		// Capture frame-by-frame
		cv::Mat* framePtr = new cv::Mat();
		cap >> *framePtr;

		// If the frame is empty, break immediately
		if (framePtr->empty()) {
			break;
		}

		// Create new pointer to store processed frame
		cv::Mat* processed_frame_ptr = new cv::Mat();
		*processed_frame_ptr = *framePtr;
		pFrame.apply_gray_scale_to_frame(processed_frame_ptr); // Convert the frame to grayscale
		pFrame.apply_smoothing_to_frame(processed_frame_ptr); // Apply gaussian smoothing to frame
		pFrame.apply_edge_detection_to_frame(processed_frame_ptr); // Apply edge detection to frame
        pFrame.resize_frame(processed_frame_ptr, 32, 32); // Resize frame to desired spacial dimensions (Rows and Columns)
		// Convert frame into MLPP matrix data structure
        Mat3d<double>* converted_frame = OpenCvIntegration::convert_gray_image<double>(processed_frame_ptr);
        // Normalize pixel data
        Mat3d<double> normalized_frame = ComputerVision::normalize_pixels<double>(converted_frame);
        // Feed converted frame data into neural network model so it can predict its class
        size_t frame_predicted_class = cnn.predict_frame<double>(normalized_frame); 

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
		}
		
		// Get processed frame and publish result
		auto message = std_msgs::msg::String();
		message.data = std::to_string(frame_predicted_class);  
		RCLCPP_INFO(node->get_logger(), "Publishing: '%s'", message.data.c_str());
		publisher->publish(message);

		// Wait for the service to be available
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

		// Calculate FPS
		tm.stop();
		fps = 1.0 / tm.getTimeSec();
		tm.reset();

		// Display purpose only
		pFrame.apply_gray_scale_to_frame(framePtr); // Convert the frame to grayscale
		pFrame.apply_smoothing_to_frame(framePtr); // Apply gaussian smoothing to frame
		pFrame.apply_edge_detection_to_frame(framePtr); // Apply edge detection to frame

		// Put detected frame class on Video Stream window
        cv::putText(*framePtr, "Frame Class: " + to_string(frame_predicted_class), cv::Point(350, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
		// Put FPS text on the frame
		cv::putText(*framePtr, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
		
		// Dispay the resulting frame
		cv::imshow("Video Stream", *framePtr);

		// Press q to exit the loop
		if (cv::waitKey(1) == 'q') {
			break;
		}	

		delete framePtr;
    }

	// Whene everything is done, release the video capture and video writer objects
	cap.release();

    // Close all windows
	cv::destroyAllWindows();

    // Shut down ROS 2
	rclcpp::shutdown();

	// Evaluating CNN models
	std::cout << class_0_counter << std::endl;
	std::cout << class_1_counter << std::endl;
	std::cout << class_2_counter << std::endl;
	std::cout << class_3_counter << std::endl;
	std::cout << class_4_counter << std::endl;
	
	return 0;
}

int main2(int argc, char** argv)
{	
	if (argc != 2) 
	{ 
        cout<<"usage: DisplayImage.out <Image_Path>"<<endl; 
        return -1; 
    } 
	
	frameProcessing pFrame;
	cv::Mat* image = new cv::Mat();
	*image = cv::imread(argv[1], 1);

	if (!image->data)
	{
		cout<<"No image data!"<<endl;
		return -1;
	}

	cv::namedWindow("Test Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Test Image", *image);
	cv::waitKey(0);
	pFrame.apply_gray_scale_to_frame(image);
	cv::imshow("GrayScale Image", *image);
	cv::waitKey(0);
	pFrame.apply_smoothing_to_frame(image);
	cv::imshow("Smoothed Frame Image", *image);
	cv::waitKey(0);
	pFrame.apply_edge_detection_to_frame(image);
	cv::imshow("Egdes Frame Image", *image);
	cv::waitKey(0);
	pFrame.maskedFrame(image);
	cv::imshow("Masked Frame Image", *image);
	cv::waitKey(0);

	return 0;
}
