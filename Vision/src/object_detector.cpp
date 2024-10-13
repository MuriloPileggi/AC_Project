#include "object_detector.hpp"

// Constructor implementation
ObjectDetector::ObjectDetector(const std::string& model_config, const std::string& model_weights, 
               const std::string& classes_file, float confidence_threshold, float nms_threshold) :
    m_confidence_threshold(confidence_threshold),
    m_nms_threshold(nms_threshold) 
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

// Detection method implementation
void ObjectDetector::detect(cv::Mat& frame) 
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(320, 320), cv::Scalar(0,0,0), true, false);
    
    m_net.setInput(blob);
    
    std::vector<cv::Mat> outs;
    m_net.forward(outs, m_net.getUnconnectedOutLayersNames());
    
    m_person_detected = false; // Resets for new frame
    post_process(frame, outs);
}

// Draw predictions method implementation
void ObjectDetector::draw_predictions(int class_id, float conf, int left, int top, int right, int bottom, cv::Mat& frame) 
{
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);
        
    std::string label = cv::format("%.2f", conf);
    if (!m_classes.empty()) {
        CV_Assert(class_id < (int)m_classes.size());
        label = m_classes[class_id] + ": " + label;
    }
    
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
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_confidence_threshold, m_nms_threshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int id_x = indices[i];
        cv::Rect box = boxes[id_x];
        int class_id = class_ids[id_x];

        // Check for a person in the frame
        if (m_classes[class_id] == "person")
        {
            m_person_detected = true;
        }
        
        draw_predictions(class_ids[id_x], confidences[id_x], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Is person detected method implementation
bool ObjectDetector::person_detected() 
{
    return m_person_detected;
}

