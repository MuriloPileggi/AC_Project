#include <iostream>
#include "frameProcessing.cpp"
#include "../../Serial/src/serialCommunication.cpp"
#include "opencv2/videoio.hpp"

using namespace std;

int main()
{
	// Set desired resolution to resize received footage
	int width = 320;
	int height = 240;

    // Open video file
    cv::VideoCapture video("../data/Video_Testing/drivingVideo.mp4");
	//cv::VideoCapture video(0);

    // Check if camera opened successfully
    if (!video.isOpened())
    {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    // Create a window to display the video
    cv::namedWindow("Video Stream", cv::WINDOW_NORMAL);
    
	// Set capture resolution
	video.set(cv::CAP_PROP_FRAME_WIDTH, width);
	video.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	// Variables for fps calculation
    double fps;
    cv::TickMeter tm;

	// Lane detection class
	frameProcessing pFrame;

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
		cv::imshow("Video Stream", *framePtr);

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
