#include "object_detector.hpp"

// Constructor implementation
ObjectDetector::ObjectDetector(const std::string& model_config, const std::string& model_weights, const std::string& classes_file, 
                                float confidence_threshold, float nms_threshold, float min_proximity_threshold) :
    m_confidence_threshold(confidence_threshold),
    m_nms_threshold(nms_threshold),
    m_min_proximity_threshold(min_proximity_threshold),
    m_stop_sign_size_ratio(0.0f),
    m_traffic_light_size_ratio(0.0f)
{
    try {
        // Load the network
        m_net = cv::dnn::readNetFromDarknet(model_config, model_weights);
        if (m_net.empty()) {
            throw std::runtime_error("Failed to load the network. Check model paths.");
        }
        m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // Load class names
        std::ifstream ifs(classes_file.c_str());
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open classes file: " + classes_file);
        }
        std::string line;
        while (std::getline(ifs, line)) {
            m_classes.push_back(line);
        }
        if (m_classes.empty()) {
            throw std::runtime_error("No classes loaded from file: " + classes_file);
        }

        std::cout << "Model and classes loaded successfully." << std::endl;
        std::cout << "Number of classes loaded: " << m_classes.size() << std::endl;
        } catch (const cv::Exception& e) {
            throw std::runtime_error("OpenCV exception: " + std::string(e.what()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Error in ObjectDetector constructor: " + std::string(e.what()));
    }
}

// Calculate size ratio method implementation
float ObjectDetector::calc_size_ratio(const cv::Rect& box, const cv::Mat& frame)
{
    // Calculate the area of the bounding box relative to the frame
    float box_area = static_cast<float>(box.width * box.height);
    float frame_area = static_cast<float>(frame.cols * frame.rows);
    return box_area / frame_area;
}

// Detection method implementation
void ObjectDetector::detect(cv::Mat& frame) 
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(320, 320), cv::Scalar(0,0,0), true, false);
    
    m_net.setInput(blob);
    
    std::vector<cv::Mat> outs;
    m_net.forward(outs, m_net.getUnconnectedOutLayersNames());

    // Reset member variables for new frame
    m_stop_sign_detected = false;
    m_traffic_light_status = NONE;

    post_process(frame, outs);
}

// Get traffic light status method implementation
int8_t ObjectDetector::get_traffic_light_status()
{
    if (m_traffic_light_size_ratio >= m_min_proximity_threshold) {
        return static_cast<int8_t>(m_traffic_light_status);
    }
    return NONE;
}

// Set traffic light status method implementation
ObjectDetector::TrafficLightStatus ObjectDetector::set_traffic_light_status(const cv::Mat& roi)
{
    // Convert the received region of interest to HSV color space
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    // Apply color masks for red, yellow and green
    cv::Mat red_mask, yellow_mask, green_mask;

    // Red (Using two ranges to cover the wrap-around of red in HSV)
    cv::Mat red_mask_1, red_mask_2;
    cv::inRange(hsv, cv::Scalar(0,100,100), cv::Scalar(10,255,255), red_mask_1);
    cv::inRange(hsv, cv::Scalar(160,100,100), cv::Scalar(180,255,255), red_mask_2);
    cv::bitwise_or(red_mask_1, red_mask_2, red_mask); // Combine both masks into one

    // Yellow
    cv::inRange(hsv, cv::Scalar(20,100,100), cv::Scalar(30,255,255), yellow_mask);

    // Green
    cv::inRange(hsv, cv::Scalar(35,100,100), cv::Scalar(85,255,255), green_mask);

    // Apply morphological operations for each mask
    cv::erode(red_mask, red_mask, cv::Mat());
    cv::dilate(red_mask, red_mask, cv::Mat());

    cv::erode(yellow_mask, yellow_mask, cv::Mat());
    cv::dilate(yellow_mask, yellow_mask, cv::Mat());

    cv::erode(green_mask, green_mask, cv::Mat());
    cv::dilate(green_mask, green_mask, cv::Mat());

    // Count number of colored pixels in each mask, if it's not in defined range
    // pixels were changed to black during maks application
    int total_pixels = roi.rows * roi.cols;
    int red_ratio = (cv::countNonZero(red_mask) * 100) / total_pixels;
    int yellow_ratio = (cv::countNonZero(yellow_mask) * 100) / total_pixels;
    int green_ratio = (cv::countNonZero(green_mask) * 100) / total_pixels;

    // Determine which color has the highest ratio and return current status
    if (red_ratio > yellow_ratio && red_ratio > green_ratio && red_ratio > 10) {
        return RED;
    }
    if (yellow_ratio > red_ratio && yellow_ratio > green_ratio && yellow_ratio > 10) {
        return YELLOW;
    }
    if (green_ratio > red_ratio && green_ratio > yellow_ratio && green_ratio > 10) {
        return GREEN;
    }

    return NONE;
}

// Draw predictions method implementation
void ObjectDetector::draw_predictions(int class_id, float conf, int left, int top, int right, int bottom, cv::Mat& frame, const std::string& label) 
{
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);
        
    //std::string label = cv::format("%.2f", conf);
    //if (!m_classes.empty()) {
    //    CV_Assert(class_id < (int)m_classes.size());
    //    label = m_classes[class_id] + ": " + label;
    //}
    
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    
}

// Post processing method implementation
void ObjectDetector::post_process(cv::Mat& frame, const std::vector<cv::Mat>& outs) 
{
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point class_id_point;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
            if (confidence > m_confidence_threshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                class_ids.push_back(class_id_point.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Reset size ratios at the start of processing
    m_stop_sign_size_ratio = 0.0f;
    m_traffic_light_size_ratio = 0.0f;
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_confidence_threshold, m_nms_threshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int id_x = indices[i];
        cv::Rect box = boxes[id_x];
        int class_id = class_ids[id_x];

        // Get current box size ratio
        float current_size_ratio = calc_size_ratio(box, frame);

        // Check for classes of interest in frame
        if(m_classes[class_id] == "stop sign") {
            m_stop_sign_detected = true;
            m_stop_sign_size_ratio = current_size_ratio;
        }
        if (m_classes[class_id] == "traffic light") {
            // Ensure the bounding box is within the frame dimensions
            cv::Rect box_copy;
            box_copy.x = std::max(0, box.x);
            box_copy.y = std::max(0, box.y);
            box_copy.width = std::min(box.width, frame.cols - box.x);
            box_copy.height = std::min(box.height, frame.rows - box.y);

            cv::Mat roi = frame(box_copy);
            m_traffic_light_status = set_traffic_light_status(roi);
            m_traffic_light_size_ratio = current_size_ratio;
        }
        
        // Add size ratio to the label if it's a stop sign or traffic light
        std::string label = cv::format("%.2f", confidences[id_x]);
        if (!m_classes.empty()) {
            CV_Assert(class_id < (int)m_classes.size());
            label = m_classes[class_id] + ": " + label;
            if (m_classes[class_id] == "stop sign" || m_classes[class_id] == "traffic light") {
                label += cv::format(" (Size: %.1f%%)", current_size_ratio * 100);
            }
        }

        draw_predictions(class_ids[id_x], confidences[id_x], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, label);
    }
}

// Stop sign detected method implementation
bool ObjectDetector::stop_sign_detected() 
{
    return m_stop_sign_detected && (m_stop_sign_size_ratio >= m_min_proximity_threshold);
}

