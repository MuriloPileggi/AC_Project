#include <iostream>
#include "frame_processing.cpp"
#include "guidance_system.cpp"
#include "object_detector.cpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp/rclcpp.hpp"
#include "std_msgs/std_msgs/msg/bool.hpp"

using namespace std;
using namespace std::chrono_literals;

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
	int width = 320;
	int height = 240;

    // Open default camera
    cv::VideoCapture cap(0);

    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    // Create a window to display the video
    cv::namedWindow("Frontal Vision Stream", cv::WINDOW_NORMAL);
    
	// Set capture resolution
	cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	cap.set(cv::CAP_PROP_FPS, 15);

	// Variables for fps calculation
    double fps;
    cv::TickMeter tm;

	// Frame processing class constructor
	FrameProcessing frame_processing;

	try
	{
		ObjectDetector detector("/home/muzaodamassa/Downloads/yolov3.cfg", "/home/muzaodamassa/Downloads/yolov3.weights", "/home/muzaodamassa/Downloads/coco.names");

		cv::Mat frame;

		while (cap.isOpened())
		{
			tm.start();

			// Capture frame-by-frame
			cap >> frame;

			// If the frame is empty, break immediately
			if (frame.empty())
			{
				break;
			}

			frame_processing.resize_frame(frame, width, height);

			detector.detect(frame);
		
			// Calculate FPS
			tm.stop();
			fps = 1.0 / tm.getTimeSec();
			tm.reset();

			// Put FPS text on the frame
			cv::putText(frame, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

			// Dispay the resulting frame
			cv::imshow("Frontal Vision Stream", frame);

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
	catch(const exception& e)
	{
		cerr << "Error: " << e.what() << endl;
		return -1;
	}   
}