#include <iostream>
#include <chrono>
#include "frame_processing.cpp"
#include "guidance_system.cpp"
#include "object_detector.cpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp/rclcpp.hpp"
#include "std_msgs/std_msgs/msg/int8.hpp"
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

// Traffic light detection timer callback function
void traffic_light_detection_timer_callback(rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr publisher,
						rclcpp::Node::SharedPtr node, std::shared_ptr<int8_t> traffic_light_status)
{
    auto message = std_msgs::msg::Int8();
    message.data = *traffic_light_status;
    RCLCPP_INFO(node->get_logger(), "Publishing: '%i'", message.data);
    publisher->publish(message);
}

int main(int argc, char * argv[])
{
	#pragma region ROS INITIALIZATION
	// Initialize ROS 2
	rclcpp::init(argc, argv);
	
	//STOP SIGN DETECTION NODE
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

	// TRAFFIC LIGHT DETECTION NODE
	// Create a ROS 2 node that will publish data relevant to traffic light detection
	auto traffic_light_detection_node = std::make_shared<rclcpp::Node>("traffic_light_detection");
	// Create stop sign detection ROS 2 publisher
	auto traffic_light_detection_publisher = traffic_light_detection_node->create_publisher<std_msgs::msg::Int8>("traffic_light_detection_topic", 10);
	// Create a shared variable for person detection
    auto traffic_light_status = std::make_shared<int8_t>(0);
	// Create a timer with a lambda function, so we publish traffic light status every 100 ms 
    auto traffic_light_detection_timer = traffic_light_detection_node->create_wall_timer(
        100ms, 
        [traffic_light_detection_publisher, traffic_light_detection_node, traffic_light_status]() {
            traffic_light_detection_timer_callback(traffic_light_detection_publisher, traffic_light_detection_node, traffic_light_status);
        }
    );

	// Start traffic light detection node on a separate thread 
	thread traffic_light_detection_thread([&]() 
	{
		rclcpp::spin(traffic_light_detection_node);
		
	});  

	#pragma endregion

	// Set desired resolution to resize received footage
	int width = 320;
	int height = 320;

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
	//cap.set(cv::CAP_PROP_FPS, 15);

	// Variables for fps calculation
    double fps;
    cv::TickMeter tm;

	// Frame processing class constructor
	FrameProcessing frame_processing;

	try
	{
		ObjectDetector detector("/home/atom3/AC_Project/Vision/models/YOLO/yolov4-tiny.cfg",
								"/home/atom3/AC_Project/Vision/models/YOLO/yolov4-tiny.weights", 
								"/home/atom3/AC_Project/Vision/models/YOLO/coco.names", 0.3);
		
		cv::Mat frame;

		while (cap.isOpened())
		{
			tm.start();

            // Capture frame-by-frame
            cap >> frame;

            // If the frame is empty, break immediately
            if (frame.empty()) {
                break;
            }

            frame_processing.resize_frame(frame, width, height);

            // Apply detection to frame based on loaded model
            detector.detect(frame);

            // Check for any  in frame
			*stop_sign_detected = detector.stop_sign_detected();
			*traffic_light_status = detector.get_traffic_light_status();

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
		traffic_light_detection_thread.join();

		return 0;
	}
	catch(const exception& e)
	{
		cerr << "Error: " << e.what() << endl;
		return -1;
	}   
}