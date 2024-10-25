# gesture_init/gesture_init/gesture_detector_node.py
import rclpy
from rclpy.node import Node
import cv2
import mediapipe as mp
import subprocess
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class GestureDetectorNode(Node):
    def __init__(self):
        super().__init__('gesture_detector_node')
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        
        # Initialize OpenCV Bridge
        self.bridge = CvBridge()
        
        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Adjust topic name as needed
            self.image_callback,
            10
        )
        
        # Launch configuration for your packages
        self.launch_config = [
            # Specify the full path to your launch files
            ('~/AC_Project/Vision/ros_cpp_workspace/road_vision_pkg', 'road_vision.launch.py'),
            ('~/AC_Project/Vision/ros_cpp_workspace/frontal_vision_pkg', 'frontal_vision.launch.py'),
            ('~/AC_Project/Vision/ros_python_workspace/bridge_node_pkg', 'bridge_node.launch.py')
        ]
        
        self.get_logger().info('Gesture Detector Node initialized')
        
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check for peace sign
                    if self.is_peace_sign(hand_landmarks):
                        self.get_logger().info('Peace sign detected! Starting packages...')
                        self.launch_packages()
                        rclpy.shutdown()
                        return
            
            # Display the image (optional)
            cv2.imshow('Gesture Detection', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def is_peace_sign(self, hand_landmarks):
        # Get relevant finger landmarks
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Check if index and middle fingers are extended while others are closed
        if (index_tip.y < hand_landmarks.landmark[6].y and  # Index extended
            middle_tip.y < hand_landmarks.landmark[10].y and  # Middle extended
            ring_tip.y > hand_landmarks.landmark[14].y and  # Ring closed
            pinky_tip.y > hand_landmarks.landmark[18].y):  # Pinky closed
            return True
        return False
    
    def launch_packages(self):
        for pkg_path, launch_file in self.launch_config:
            try:
                full_path = pkg_path.replace('~', os.path.expanduser('~'))
                subprocess.Popen(['ros2', 'launch', full_path, launch_file])
                self.get_logger().info(f'Successfully launched {launch_file}')
            except Exception as e:
                self.get_logger().error(f'Failed to launch {launch_file}: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    gesture_detector = GestureDetectorNode()
    rclpy.spin(gesture_detector)
    gesture_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()