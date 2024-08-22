#include <iostream>
#include "frameProcessing.cpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp/rclcpp.hpp"
#include "std_msgs/std_msgs/msg/string.hpp"
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
	auto publisher = node->create_publisher<std_msgs::msg::String>("frame_class", 10);
	// Set loop rate
	rclcpp::Rate loop_rate(10);
	#pragma endregion

	// Set desired resolution to resize received footage
	int width = 512;
	int height = 512;
	
    // Open video file
    //cv::VideoCapture cap("../data/Video_Testing/drivingVideo.mp4");
	cv::VideoCapture cap(0);

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
	cnn.load_model("../models/model_r2_alpha.bin");

    while (cap.isOpened())
    {
		tm.start();

		// Capture frame-by-frame
		cv::Mat* framePtr = new cv::Mat();
		cap >> *framePtr;

		// If the frame is empty, break immediately
		if (framePtr->empty())
		{
			break;
		}

		cv::Mat* processed_frame_ptr = new cv::Mat();
		*processed_frame_ptr = *framePtr;
        pFrame.resize_frame(processed_frame_ptr, 16, 16); // Resize frame to desired spacial dimensions (Rows and Columns)
		pFrame.apply_gray_scale_to_frame(processed_frame_ptr); // Convert the frame to grayscale
		//pFrame.apply_smoothing_to_frame(framePtr); // Apply gaussian smoothing to frame
		//pFrame.apply_edge_detection_to_frame(framePtr); // Apply edge detection to frame
		//pFrame.maskedFrame(framePtr); // Apply mask to frame
		// Convert frame into MLPP matrix data structure
        Mat3d<double>* converted_frame = OpenCvIntegration::convert_gray_image<double>(processed_frame_ptr);
        // Normalize pixel data
        Mat3d<double> normalized_frame = ComputerVision::normalize_pixels<double>(converted_frame);
        // Feed converted frame data into neural network model so it can predict its class
        size_t frame_predicted_class = cnn.predict_frame<double>(normalized_frame); 
		
		// Convert frame class into ROS 2 message data structure
		auto message = std_msgs::msg::String();
		message.data = std::to_string(frame_predicted_class);  
		RCLCPP_INFO(node->get_logger(), "Publishing: '%s'", message.data.c_str());
		publisher->publish(message);
		rclcpp::spin_some(node);
		loop_rate.sleep();

		// Calculate FPS
		tm.stop();
		fps = 1.0 / tm.getTimeSec();
		tm.reset();

		// Display purpose only
		//pFrame.apply_gray_scale_to_frame(framePtr); // Convert the frame to grayscale
		//pFrame.apply_smoothing_to_frame(framePtr); // Apply gaussian smoothing to frame
		//pFrame.apply_edge_detection_to_frame(framePtr); // Apply edge detection to frame

		// Put detected frame class on Video Stream window
        cv::putText(*framePtr, "Frame Class: " + to_string(frame_predicted_class), cv::Point(350, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
		// Put FPS text on the frame
		cv::putText(*framePtr, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
		
		// Dispay the resulting frame
		cv::imshow("Video Stream", *framePtr);

		// Press q to exit the loop
		if (cv::waitKey(1) == 'q')
		{
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
