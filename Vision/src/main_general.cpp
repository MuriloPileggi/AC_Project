#include <iostream>
#include "../../Serial/src/serialCommunication.cpp"
#include "frameProcessing.cpp"
#include "opencv2/videoio.hpp"

using namespace std;

int main()
{
	// Create handle to serial communication class
	serialCommunication serial;
	std::string stop = "Stop";
	int loopIterationsSinceLastWrite = 0;
	int writeInterval = 10;

	// Set desired resolution to resize received footage
	int width = 360;
	int height = 240;

	// Loading haar cascade xml files for object detection
	cv::CascadeClassifier stopSignCascade;
	//cv::CascadeClassifier trafficLightCascade;
	stopSignCascade.load("../../Vision/data/Pre_Trained_Models/Stopsign_HAAR_19Stages.xml");
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

    while (true)
    {
		tm.start();

		// Capture frame-by-frame
		cv::Mat* frame = new cv::Mat();
		cap >> *frame;

		// If the frame is empty, break immediately
		if (frame->empty())
		{
			break;
		}

		// Convert the frame to grayscale
		cv::Mat* gray = new cv::Mat();
		cvtColor(*frame, *gray, cv::COLOR_BGR2GRAY);

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
		stopSignCascade.detectMultiScale(*gray, stopSigns, 1.1, 5);

		// Draw bounding boxes around detected stop signs
		bool stopSignDetected = !stopSigns.empty();
		for (const auto& rect : stopSigns)
		{
			cv::rectangle(*frame, rect, cv::Scalar(255, 0, 0), 2);
			cv::putText(*frame, "Stop Sign", cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);
		}

		// Send signal to arduino to stop motors
        if (loopIterationsSinceLastWrite >= writeInterval) {
            // Send signal to Arduino to stop motors only if writeToSerialPort
            // method has not been called yet
            serial.writeToSerialPort(stop, stopSignDetected);
            loopIterationsSinceLastWrite = 0; // Reset counter
        } 
		else {
            loopIterationsSinceLastWrite++; // Increment counter
        }

        // Calculate FPS
		tm.stop();
		fps = 1.0 / tm.getTimeSec();
		tm.reset();

		// Put FPS text on the frame
		cv::putText(*frame, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

		// Dispay the resulting frame
		cv::imshow("Video Stream", *frame);

		// Press q to exit the loop
		if (cv::waitKey(1) == 'q')
		{			
			break;
		}	

		delete frame;
		delete gray;
    }

	// Whene everything is done, release the video capture and video writer objects
	cap.release();

    // Close all windows
	cv::destroyAllWindows();

    return 0;	
}