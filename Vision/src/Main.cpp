#include <iostream>
#include "FrameProcessing.hpp"
#include "opencv2/videoio.hpp"

using namespace std;

int main()
{
	// Set desired resolution to resize received footage
	int width = 640;
	int height = 480;

    // Open video file
    cv::VideoCapture video("/home/muzaodamassa/AC_Project/Vision/data/Video_Testing/drivingVideo.mp4");
	//cv::VideoCapture video(0);

    // Check if camera opened successfully
    if (!video.isOpened())
    {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    // Create a window to display the video
    namedWindow("Video Stream", cv::WINDOW_NORMAL);
    
	// Set capture resolution
	video.set(cv::CAP_PROP_FRAME_WIDTH, width);
	video.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	// Variables for fps calculation
    double fps;
    cv::TickMeter tm;

	// Lane detection class
	FrameProcessing pFrame;

    while (video.isOpened())
    {
		tm.start();

		// Capture frame-by-frame
		cv::Mat* framePtr = new cv::Mat();
		video >> *framePtr;

		// If the frame is empty, break immediately
		if (framePtr->empty())
		{
			break;
		}

		// Convert the frame to grayscale
		pFrame.grayFrame(framePtr);

		// Apply gaussian smoothing to frame
		pFrame.smoothedFrame(framePtr);

		// Apply edge detection to frame
		pFrame.edgesFrame(framePtr);

		// Apply mask to frame
		pFrame.maskedFrame(framePtr);

		// Dispay the resulting frame
		imshow("Video Stream", *framePtr);

		// Press q to exit the loop
		if (cv::waitKey(1) == 'q')
		{
			break;
		}	

		delete framePtr;
    }

	// Whene everything is done, release the video capture and video writer objects
	video.release();

    // Close all windows
	cv::destroyAllWindows();

    return 0;	
}

int main2(int argc, char** argv)
{	
	if (argc != 2) 
	{ 
        cout<<"usage: DisplayImage.out <Image_Path>"<<endl; 
        return -1; 
    } 
	
	FrameProcessing pFrame;
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
	pFrame.grayFrame(image);
	cv::imshow("GrayScale Image", *image);
	cv::waitKey(0);
	pFrame.smoothedFrame(image);
	cv::imshow("Smoothed Frame Image", *image);
	cv::waitKey(0);
	pFrame.edgesFrame(image);
	cv::imshow("Egdes Frame Image", *image);
	cv::waitKey(0);
	pFrame.maskedFrame(image);
	cv::imshow("Masked Frame Image", *image);
	cv::waitKey(0);

	return 0;
}

int main3()
{
	// Set desired resolution to resize received footage
	int width = 640;
	int height = 480;

	// Loading haar cascade xml files for object detection
	cv::CascadeClassifier stopSignCascade;
	//cv::CascadeClassifier trafficLightCascade;
	stopSignCascade.load("../data/Pre_Trained_Models/Stopsign_HAAR_19Stages.xml");
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
    namedWindow("Video Stream", cv::WINDOW_NORMAL);
    
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
		cv::Mat frame;
		cap >> frame;

		// If the frame is empty, break immediately
		if (frame.empty())
		{
			break;
		}

		// Convert the frame to grayscale
		cv::Mat gray;
		cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

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
		stopSignCascade.detectMultiScale(gray, stopSigns, 1.1, 5);

		// Draw bounding boxes around detected stop signs
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
		putText(frame, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

		// Dispay the resulting frame
		imshow("Video Stream", frame);

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

    return 0;	
}