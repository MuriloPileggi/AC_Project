import serial.tools
import serial.tools.list_ports
import serial.tools.list_ports_linux
import rclpy
import serial
import time
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

class FrameClassificationSubscriber(Node):
    def __init__(self):
        super().__init__('bridge_node')
        self.subscription = self.create_subscription(
            Float32,
            'road_vision_topic',
            self.road_vision_callback,
            10)
        #self.subscription = self.create_subscription(
            #Bool,
            #'frontal_vision_topic',
            #self.frontal_vision_callback,
            #10)
        #self.subscription  # prevent unused variable warning

        #self.service = self.create_service(Trigger, 'trigger_action', self.handle_trigger_action)
        self.received_data = None

        # Create a timer to execute callback every 20 ms (50 Hz)
        self.timer_period = 0.05  # 20 ms
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Initialize serial connection
        #self.serial_port =  serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        #if self.serial_port is None:
            #self.get_logger().error('Failed to initialize serial connection')
            #return
        #time.sleep(2)  # Wait for the connection to settle
        #self.serial_port =  serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        #if self.serial_port is None:
        #    self.get_logger().error('Failed to initialize serial connection')
        #    return
        #time.sleep(2)  # Wait for the connection to settle
  
    def road_vision_callback(self, msg):
        #self.get_logger().info('Received: "%s"' % msg.data)
        self.received_data = msg.data
    
    def frontal_vision_callback(self, msg):
        #self.get_logger().info('Received: "%s"' % msg.data)
        self.received_data = msg.data
    
    def timer_callback(self):
        # This method will run at 50 Hz
        if self.received_data is not None:
            self.get_logger().info('Timer triggered, sending data: %f' % self.received_data)
            #self.send_to_arduino(self.received_data)  # Uncomment to send data
        else:
            self.get_logger().info('Timer triggered, but no message received yet.')

    def handle_trigger_action(self, request, response):
        self.get_logger().info('Action Triggered!')
        
        # Simulate action being performed
        self.get_logger().info(f'Processing data: {self.received_data}')
        
        # Send the received data to the Arduino
        #self.send_to_arduino(self.received_data)
        
        # Wait for a response from the Arduino
        #arduino_response = self.receive_from_arduino()

        #if arduino_response is not None:
        #    self.get_logger().info(f'Received from Arduino: {arduino_response}')
        #    if arduino_response == "success":
        #        response.success = True
        #        response.message = 'Action Completed Successfully'
        #    else:
        #        response.success = False
        #        response.message = 'Action Failed'
        #else:
        #    response.success = False
        #    response.message = 'No response from Arduino'

        #return response
    
    def send_to_arduino(self, data):
        if self.serial_port is None:
            self.get_logger().error('Serial port not initialized')
            return
        try:
            self.serial_port.write(data.encode())
            self.get_logger().info('Sent to Arduino: "%s"' % data)
        except serial.SerialException as e:
            self.get_logger().error('Failed to send data to Arduino: %s' % str(e))
     
    #def receive_from_arduino(self):
       #if self.serial_port is None:
           #self.get_logger().error('Serial port not initialized')
           #return None
       #try:
           # Continuously wait for a response from the Arduino
            #while True:
                #if self.serial_port.in_waiting > 0:
                    #ard = self.serial_port.readline().decode().strip()
                    #if ard:
                        #return ard
       #except serial.SerialException as e:
           #self.get_logger().error('Failed to receive data from Arduino: %s' % str(e))
           #return None
       
    def __del__(self):
        if hasattr(self, 'serial_port'):
            self.serial_port.close()

def main(args=None):
    rclpy.init(args=args)
    subscriber = FrameClassificationSubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
