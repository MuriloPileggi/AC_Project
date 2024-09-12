#include <iostream>
#include "frame_processing.cpp"
#include "guidance_system.cpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp/rclcpp.hpp"
#include "../ai_libraries/mlpp.hpp"
#include "std_msgs/std_msgs/msg/bool.hpp"
#include "std_srvs/std_srvs/srv/trigger.hpp"

using namespace std;
using namespace std::chrono_literals;
using namespace MLPP;

// Timer callback function
void timer_callback(rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher,
						rclcpp::Node::SharedPtr node, std::shared_ptr<bool> stop_sign_detected)
{
    auto message = std_msgs::msg::Bool();
    message.data = *stop_sign_detected;
    RCLCPP_INFO(node->get_logger(), "Publishing: '%s'", message.data ? "true" : "false");
    publisher->publish(message);
}

int main(int argc, char * argv[])
{
	#pragma region ROS INITIALIZATION
	// Initialize ROS 2
	rclcpp::init(argc, argv);
	// Create a ROS 2 node
	auto node = std::make_shared<rclcpp::Node>("frontal_vision");
	// Create a ROS 2 publisher
	auto publisher = node->create_publisher<std_msgs::msg::Bool>("frontal_vision_topic", 10);
	// Create a ROS 2 service client
	//auto client = node->create_client<std_srvs::srv::Trigger>("trigger_action");
	// Create a shared variable for steering angle
    auto stop_sign_detected = std::make_shared<bool>(false);
	// Create a timer with a lambda function, so we publish steering angle every 20 ms (50Hz)
    auto timer = node->create_wall_timer(
        50ms, 
        [publisher, node, stop_sign_detected]() {
            timer_callback(publisher, node, stop_sign_detected);
        }
    );

	// Start road vision node on a separate thread from video while loop
	thread spin_thread([&]() 
	{
		rclcpp::spin(node);
	});

	#pragma endregion

	// Set desired resolution to resize received footage
	int width = 500;
	int height = 500;

	// Loading haar cascade xml files for object detection
	cv::CascadeClassifier stopSignCascade;
	//cv::CascadeClassifier trafficLightCascade;
	stopSignCascade.load("/home/muzaodamassa/AC_Project/Vision/data/Pre_Trained_Models/Stopsign_HAAR_19Stages.xml");
	//trafficLightCascade.load("../Data/Pre_Trained_Models/TrafficLight_HAAR_16Stages.xml");

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

		frame_processing.resize_frame(frame, width, height);
		frame_processing.apply_gray_scale_to_frame(frame);
		// Detect traffic lights
		/*
		vector<cv::Rect> trafficLights;
		trafficLightCascade.detectMultiScale(gray, trafficLights, 1.1, 5, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30), cv::Size(100, 100));

		// Draw bounding boxes around detected traffic lights
		for (const auto& rect : trafficLights)
		{
			cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
			cv::putText(frame, "Traffic Light", cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);
		}
		*/

		// Detect stop signs in the frame
		vector<cv::Rect> stopSigns;
		stopSignCascade.detectMultiScale(frame, stopSigns, 1.1, 5);

		// Draw bounding boxes around detected stop signs
		*stop_sign_detected = !stopSigns.empty();
		for (const auto& rect : stopSigns)
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
    spin_thread.join();

	return 0;
}