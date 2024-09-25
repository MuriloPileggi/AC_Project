#include <iostream>
#include "frame_processing.cpp"
#include "guidance_system.cpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp/rclcpp.hpp"
#include "../ai_libraries/mlpp.hpp"
#include "std_msgs/std_msgs/msg/bool.hpp"

using namespace std;
using namespace std::chrono_literals;
using namespace MLPP;

// Stop sign detection timer callback function
void stop_sign_timer_callback(rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher,
						rclcpp::Node::SharedPtr node, std::shared_ptr<bool> stop_sign_detected)
{
    auto message = std_msgs::msg::Bool();
    message.data = *stop_sign_detected;
    RCLCPP_INFO(node->get_logger(), "Publishing: '%s'", message.data ? "true" : "false");
    publisher->publish(message);
}

// Obstacle detection timer callback function
void obstacle_detection_timer_callback(rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher,
						rclcpp::Node::SharedPtr node, std::shared_ptr<bool> obstacle_detected)
{
    auto message = std_msgs::msg::Bool();
    message.data = *obstacle_detected;
    RCLCPP_INFO(node->get_logger(), "Publishing: '%s'", message.data ? "true" : "false");
    publisher->publish(message);
}

int main(int argc, char * argv[])
{
	#pragma region ROS INITIALIZATION

	// Initialize ROS 2
	rclcpp::init(argc, argv);
	// Create a ROS 2 node that will publish data relevant to stop sign detection
	auto stop_sign_detection_node = std::make_shared<rclcpp::Node>("stop_sign_detection");
	// Create stop sign detection ROS 2 publisher
	auto stop_sign_publisher = stop_sign_detection_node->create_publisher<std_msgs::msg::Bool>("stop_sign_detection_topic", 10);
	// Create a shared variable for stop sign detection
    auto stop_sign_detected = std::make_shared<bool>(false);
	// Create a timer with a lambda function, so we publish stop sign detection status every 100 ms 
    auto stop_sign_timer = stop_sign_detection_node->create_wall_timer(
        100ms, 
        [stop_sign_publisher, stop_sign_detection_node, stop_sign_detected]() {
            stop_sign_timer_callback(stop_sign_publisher, stop_sign_detection_node, stop_sign_detected);
        }
    );
	// Start stop sign detection node on a separate thread 
	thread stop_sign_thread([&]() 
	{
		rclcpp::spin(stop_sign_detection_node);
		
	});

	//OBSTACLE DETECTION NODE

	/* // Create a ROS 2 node that will publish data relevant to obstacle detection
	auto obstacle_detection_node = std::make_shared<rclcpp::Node>("obstacle_detection");
	// Create stop sign detection ROS 2 publisher
	auto obstacle_detection_publisher = obstacle_detection_node->create_publisher<std_msgs::msg::Bool>("obstacle_detection_topic", 10);
	// Create a shared variable for stop sign detection
    auto obstacle_detected = std::make_shared<bool>(false);
	// Create a timer with a lambda function, so we publish obstacle detection status every 100 ms 
    auto obstacle_detection_timer = obstacle_detection_node->create_wall_timer(
        100ms, 
        [obstacle_detection_publisher, obstacle_detection_node, obstacle_detected]() {
            obstacle_detection_timer_callback(obstacle_detection_publisher, obstacle_detection_node, obstacle_detected);
        }
    );

	// Start stop sign detection node on a separate thread 
	thread obstacles_thread([&]() 
	{
		rclcpp::spin(obstacle_detection_node);
		
	});  */

	#pragma endregion

	// Set desired resolution to resize received footage
	int width = 500;
	int height = 500;

	// Loading haar cascade xml files for object detection
	cv::CascadeClassifier stop_sign_detection_cascade;
	stop_sign_detection_cascade.load("/home/muzaodamassa/AC_Project/Vision/data/Pre_Trained_Models/Stopsign_HAAR_19Stages.xml");

    // Open default camera
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

	// Frame processing class constructor
	FrameProcessing frame_processing;
	// Neural network constructor from MLPP
	//NeuralNetwork cnn;
	// Load pre trained model into class
	//cnn.load_model("Vision/models/pedestrian_detection_25_2.bin");

    while (cap.isOpened())
    {
		tm.start();

		// Capture frame-by-frame
		cv::Mat frame;
		cap >> frame;

		// If the frame is empty, break immediately
		if (frame.empty())
		{
			break;
		}

		//cv::Mat* processed_frame = new cv::Mat(frame);

		// Visual/Debugging
		frame_processing.apply_gray_scale_to_frame(frame);
		frame_processing.resize_frame(frame, width, height);
		//frame_processing.apply_smoothing_to_frame(frame);

		// Process frame
		//frame_processing.apply_gray_scale_to_frame(*processed_frame);
		//frame_processing.resize_frame(*processed_frame, 128, 128);

		// Convert frame into MLPP matrix 
		//Mat3d<double>* mlpp_frame = OpenCvIntegration::convert_gray_image<double>(processed_frame);
		// Normalize frame data
		//Mat3d<double> normalized_frame = ComputerVision::normalize_pixels<double>(mlpp_frame);
		// Apply inference to normalized frame data using loaded model to store predicted class
		//size_t predicted_frame_class = cnn.predict_frame<double>(normalized_frame);
		// If predicted class is 1, i.e, there was some sort of obstacle in frame
		// We assign shared pointer data to true, else it remains false
		//*obstacle_detected = predicted_frame_class == 1 ? true : false;	

		// Detect stop signs in the frame
		vector<cv::Rect> stop_signs;
		stop_sign_detection_cascade.detectMultiScale(frame, stop_signs, 1.1, 5);
		*stop_sign_detected = !stop_signs.empty();

		// Draw bounding boxes around detected stop signs
		for (const auto& rect : stop_signs)
		{
			cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
			cv::putText(frame, "Stop Sign", cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);
		} 

        // Calculate FPS
		tm.stop();
		fps = 1.0 / tm.getTimeSec();
		tm.reset();

		// Put FPS text on the frame
		cv::putText(frame, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

		// Dispay the resulting frame
		cv::imshow("Video Stream", frame);

		// Press q to exit the loop
		if (cv::waitKey(1) == 'q')
		{			
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
    stop_sign_thread.join();
	//obstacles_thread.join();

	return 0;
}