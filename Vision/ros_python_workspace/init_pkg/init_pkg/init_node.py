# gesture_init/gesture_init/gesture_detector_node.py
import rclpy
from rclpy.node import Node
import cv2
import mediapipe as mp
import subprocess
import time
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

        self.pkg_directories = [
            "/home/atom3/AC_Project/Vision/ros_cpp_workspace/road_vision_pkg",
            "/home/atom3/AC_Project/Vision/ros_cpp_workspace/frontal_vision_pkg",
            "/home/atom3/AC_Project/Vision/ros_python_workspace/bridge_node_pkg"
        ]

        self.road_vision_node_terminal_commands = [
            "source /opt/ros/iron/setup.bash",
            "source install/setup.bash",
            "ros2 run road_vision_pkg road_vision_node"
        ]

        self.frontal_vision_node_terminal_commands = [
            "source /opt/ros/iron/setup.bash",
            "source install/setup.bash",
            "ros2 run frontal_vision_pkg frontal_vision_node"
        ]

        self.bridge_node_terminal_commands = [
            "source /opt/ros/iron/setup.bash",
            "source install/setup.bash",
            "ros2 run bridge_node_pkg bridge_node"
        ]
        
        self.get_logger().info('Gesture Detector Node initialized')

        self.video_processing()
    

    def video_processing(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check for peace sign
                    if self.is_peace_sign(hand_landmarks):
                        self.get_logger().info('Peace sign detected! Starting packages...')
                        cap.release()
                        cv2.destroyAllWindows()
                        self.launch_packages()
                        rclpy.shutdown()
                        return
            
            # Display the image (optional)
            cv2.imshow('Gesture Detection', frame)
            cv2.waitKey(1)
    
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
        command_string = f"cd {self.pkg_directories[0]}; " + "; ".join(self.road_vision_node_terminal_commands) + "; exec bash"
        subprocess.Popen(["gnome-terminal", "--", "bash", "-c", command_string])

        time.sleep(1)

        command_string = f"cd {self.pkg_directories[1]}; " + "; ".join(self.frontal_vision_node_terminal_commands) + "; exec bash"
        subprocess.Popen(["gnome-terminal", "--", "bash", "-c", command_string])

        time.sleep(1)

        command_string = f"cd {self.pkg_directories[2]}; " + "; ".join(self.bridge_node_terminal_commands) + "; exec bash"
        subprocess.Popen(["gnome-terminal", "--", "bash", "-c", command_string])
        


def main(args=None):
    rclpy.init(args=args)
    gesture_detector = GestureDetectorNode()
    rclpy.spin(gesture_detector)
    gesture_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()