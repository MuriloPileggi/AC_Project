import serial.tools
import serial.tools.list_ports
import serial.tools.list_ports_linux
import rclpy
import serial
import time
from rclpy.node import Node
from std_msgs.msg import Float32, Bool

class BridgeNode(Node):
    def __init__(self):
        # Start ROS Node along with necessary variables
        super().__init__('bridge_node')
        self.angle_correction = 0.0
        self.stop_sign_detected = False
        self.stop_time = 0
        
        # Start node subscriptions
        self.subscription = self.create_subscription(Float32,'road_vision_topic',self.road_vision_callback,10)
        self.subscription = self.create_subscription(Bool,'frontal_vision_topic',self.frontal_vision_callback,10)

        #self.service = self.createservice(Trigger, 'trigger_action', self.handle_trigger_action)
        self.received_data = None

        # Create a timer to execute callback every 20 ms (50 Hz)
        self.timer = self.create_timer(0.05, self.timer_callback)

        # Initialize serial connection
        #self.serial_port =  serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        #if self.serial_port is None:
            #self.get_logger().error('Failed to initialize serial connection')
            #return
        #time.sleep(2)  # Wait for the connection to settle
        self.serial_port =  serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        if self.serial_port is None:
            self.get_logger().error('Failed to initialize serial connection')
            return
        time.sleep(2)  # Wait for the connection to settle
  
    def road_vision_callback(self, msg):
        #self.get_logger().info('Received: "%s"' % msg.data)
        self.angle_correction = msg.data
    
    def frontal_vision_callback(self, msg):
        #self.get_logger().info('Received: "%s"' % msg.data)
        self.stop_sign_detected = msg.data
        if self.stop_sign_detected:
            self.stop_time = self.get_clock().now().to_msg().sec
    
    def timer_callback(self):
        # This method will run at 50 Hz
        # Get current time so we can properly apply stop sign detection
        current_time = self.get_clock().now().to_msg().sec
        
        # Stop the car for 5 seconds if a stop sign has been detected
        if self.stop_sign_detected and current_time - self.stop_time < 5:
            return
        else:
            # Resume normal operation
            self.stop_sign_detected = False
            data_to_send = f"{self.angle_correction},{1 if self.stop_sign_detected else 0}\n"
            self.get_logger().info('Timer triggered, sending data: %s' % data_to_send)
            self.send_to_arduino(data_to_send) 
    
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
       
    def __del__(self):
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
