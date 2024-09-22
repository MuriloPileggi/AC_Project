import serial.tools
import serial.tools.list_ports
import serial.tools.list_ports_linux
import rclpy
import serial
import time
import threading
from rclpy.node import Node
from std_msgs.msg import Int8, Bool

class BridgeNode(Node):
    def __init__(self):
        # Start ROS Node along with necessary variables
        super().__init__('bridge_node')
        self.angle_correction = 0
        self.stop_sign_count = 0
        self.stop_sign_detected = False
        self.obstacle_detected = False
        self.threads = []
        #self.stop_time = 0
        
        # Start node subscriptions
        self.subscription = self.create_subscription(Int8,'road_vision_topic',self.road_vision_callback,10)
        self.subscription = self.create_subscription(Bool,'stop_sign_detection_topic',self.stop_sign_detection_callback,10)
        #self.subscription = self.create_subscription(Bool,'obstacle_detection_topic',self.obstacle_detection_callback,10)

        # Create a timer to execute callback every X ms
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Initialize serial connection
        #self.serial_port =  serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        #if self.serial_port is None:
            #self.get_logger().error('Failed to initialize serial connection')
            #return
        #time.sleep(2)  # Wait for the connection to settle
        #self.serial_port =  serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        #if self.serial_port is None:
            #self.get_logger().error('Failed to initialize serial connection')
            #return
        #time.sleep(2)  # Wait for the connection to settle
  
    def road_vision_callback(self, msg):
        #self.get_logger().info('Received: "%s"' % msg.data)
        self.angle_correction = msg.data
    
    def stop_sign_detection_callback(self, msg):
        #self.get_logger().info('Received: "%s"' % msg.data)
        self.stop_sign_detected = msg.data
    
    def obstacle_detection_callback(self, msg):
        #self.get_logger().info('Received: "%s"' % msg.data)
        self.obstacle_detected = msg.data
    
    # We call this method to reset stop sign counter method on a different thread
    # This way, we can stop the car for 5 seconds for the stop sign, and only after
    # 30 seconds have passed, if we find another stop sign we repeat the process
    def start_reset_thread(self):
        reset_thread = threading.Thread(target=self.reset_stop_sign_counter)
        self.threads.append(reset_thread)
        reset_thread.start()

    def reset_stop_sign_counter(self):
        time.sleep(30)
        self.stop_sign_count = 0
    
    # This method will run at specified frequency
    def timer_callback(self):
        # Get current time so we can properly apply stop sign detection
        
        #current_time = self.get_clock().now().to_msg().sec
        
        # If an obstacle is detected in front of the car, we immediately stop sending data
        # to our arduino, and continue to do so as long as we have something obstructing our
        # way
        if self.obstacle_detected:
            return
        # Stop the car for 5 seconds if a stop sign has been detected
        elif self.stop_sign_detected and self.stop_sign_count < 1:
            self.stop_sign_count += 1
            self.get_logger().info("Stopping at stop sign")
            time.sleep(5)
            self.start_reset_thread()
            return
        else:
            # Resume normal operation
            self.stop_sign_detected = False
            data_to_send = f"{self.angle_correction},{1 if self.stop_sign_detected else 0}\n"
            #data_to_send = f"{self.angle_correction},{1 if self.stop_sign_detected else 0},{1 if self.obstacle_detected else 0}\n"
            self.get_logger().info('Timer triggered, sending data: %s' % data_to_send)
            #self.send_to_arduino(data_to_send) 
    
    def send_to_arduino(self, data):
        if self.serial_port is None:
            self.get_logger().error('Serial port not initialized')
            return
        try:
            self.serial_port.write((str(data)).encode('utf-8'))
            self.get_logger().info('Sent to Arduino: "%s"' % data)
        except serial.SerialException as e:
            self.get_logger().error('Failed to send data to Arduino: %s' % str(e))
     
    def receive_from_arduino(self):
        if self.serial_port is None:
           self.get_logger().error('Serial port not initialized')
           return None
        try:
           #Continuously wait for a response from the Arduino
            while True:
                if self.serial_port.in_waiting > 0:
                    ard = self.serial_port.readline().decode().strip()
                    if ard:
                        return ard
        except serial.SerialException as e:
           self.get_logger().error('Failed to receive data from Arduino: %s' % str(e))
           return None
    
    def join_threads(self):
        for thread in self.threads:
            thread.join()
        self.threads = []

    def cleanup(self):
        self.join_threads()
        
    def __del__(self):
        self.cleanup()
        if hasattr(self, 'serial_port'):
            self.serial_port.close()

def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
