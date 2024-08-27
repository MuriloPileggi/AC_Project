import serial.tools
import serial.tools.list_ports
import serial.tools.list_ports_linux
import rclpy
import serial
import time
from rclpy.node import Node
from std_msgs.msg import String

class FrameClassificationSubscriber(Node):
    def __init__(self):
        super().__init__('frame_classification_subscriber')
        self.subscription = self.create_subscription(
            String,
            'frame_classification',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Initialize serial connection
        #self.serial_port =  serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        #if self.serial_port is None:
            #self.get_logger().error('Failed to initialize serial connection')
            #return
        #time.sleep(2)  # Wait for the connection to settle
  
    def listener_callback(self, msg):
        self.get_logger().info('Received: "%s"' % msg.data)
        #self.send_to_arduino(msg.data)

    def send_to_arduino(self, data):
        if self.serial_port is None:
            self.get_logger().error('Serial port not initialized')
            return
        try:
            self.serial_port.write(data.encode())
            self.get_logger().info('Sent to Arduino: "%s"' % data)
            ard = self.serial_port.readline().decode().strip()
            self.get_logger().info(ard)
        except serial.SerialException as e:
            self.get_logger().error('Failed to send data to Arduino: %s' % str(e))
    
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
