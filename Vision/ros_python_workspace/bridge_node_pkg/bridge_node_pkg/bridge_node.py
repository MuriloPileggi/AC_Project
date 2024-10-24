import serial.tools
import serial.tools.list_ports
import serial.tools.list_ports_linux
import rclpy
import serial
import time
import threading
import RPi.GPIO as GPIO
from rclpy.node import Node
from std_msgs.msg import Int8, Bool

class BridgeNode(Node):
    def __init__(self):
        # Start ROS Node along with necessary variables
        super().__init__('bridge_node')
        self.angle_correction = 0
        self.stop_sign_count = 0
        self.stop_sign_detected = False
        self.traffic_light_status = 0 # 0 = None, 1 = Red, 2 = Yellow, 3 = Green
        self.obstacle_detected = False
        self.ultrasonic_running = True # Flag to keep reading data from ultrasonic sensor
        self.threads = []
        #self.stop_time = 0

        # GPIO Setup for ultrasonic sensor
        self.TRIG_PIN = 23  # You can change these pin numbers
        self.ECHO_PIN = 24  # according to your setup
        self.setup_ultrasonic_sensor()

        # Start ultrasonic sensor thread
        self.start_ultrasonic_thread()
        
        # Start node subscriptions
        self.subscription = self.create_subscription(Int8,'road_vision_topic',self.road_vision_callback,10)
        self.subscription = self.create_subscription(Bool,'stop_sign_detection_topic',self.stop_sign_detection_callback,10)
        self.subscription = self.create_subscription(Int8,'traffic_light_detection_topic',self.traffic_light_detection_callback,10)

        # Create a timer to execute callback every X ms
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Initialize serial connection
        self.serial_port =  serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        if self.serial_port is None:
            self.get_logger().error('Failed to initialize serial connection')
            return
        time.sleep(2)  # Wait for the connection to settle
        
        #self.serial_port =  serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        #if self.serial_port is None:
        #   self.get_logger().error('Failed to initialize serial connection')
        #   return
        #time.sleep(2)  # Wait for the connection to settle
    

    def setup_ultrasonic_sensor(self):
        """Initialize GPIO pins for the ultrasonic sensor"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.TRIG_PIN, GPIO.OUT)
        GPIO.setup(self.ECHO_PIN, GPIO.IN)
        GPIO.output(self.TRIG_PIN, False)
        time.sleep(2)  # Let the sensor settle

    def measure_distance(self):
        """Measure distance using HC-SR04 sensor"""
        try:
            # Trigger pulse
            GPIO.output(self.TRIG_PIN, True)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.TRIG_PIN, False)

            # Wait for echo to start
            pulse_start = time.time()
            timeout_start = pulse_start
            while GPIO.input(self.ECHO_PIN) == 0:
                pulse_start = time.time()
                if pulse_start - timeout_start > 0.1:  # 100ms timeout
                    return float('inf')

            # Wait for echo to end
            pulse_end = time.time()
            timeout_start = pulse_end
            while GPIO.input(self.ECHO_PIN) == 1:
                pulse_end = time.time()
                if pulse_end - timeout_start > 0.1:  # 100ms timeout
                    return float('inf')

            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = (pulse_duration * 34300) / 2  # Speed of sound = 343 m/s
            return round(distance, 2)  # Return distance in cm

        except Exception as e:
            self.get_logger().error(f'Error measuring distance: {str(e)}')
            return float('inf')

    def ultrasonic_loop(self):
        """Continuous loop for reading ultrasonic sensor"""
        while self.ultrasonic_running:
            distance_cm = self.measure_distance()
            self.obstacle_detected = distance_cm < 60
            if self.obstacle_detected:
                self.get_logger().debug('Obstacle detected')
            

    def start_ultrasonic_thread(self):
        """Start the ultrasonic sensor reading thread"""
        ultrasonic_thread = threading.Thread(target=self.ultrasonic_loop)
        ultrasonic_thread.daemon = True  # Thread will stop when main program stops
        self.threads.append(ultrasonic_thread)
        ultrasonic_thread.start()

    def road_vision_callback(self, msg):
        #self.get_logger().info('Received: "%s"' % msg.data)
        self.angle_correction = msg.data
    
    def stop_sign_detection_callback(self, msg):
        #self.get_logger().info('Received: "%s"' % msg.data)
        self.stop_sign_detected = msg.data
    
    def traffic_light_detection_callback(self, msg):
        #self.get_logger().info('Received: "%s"' % msg.data)
        self.traffic_light_status = msg.data 
    
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
        # If an obstacle is detected in front of the car, we immediately stop sending data
        # to our arduino, and continue to do so as long as we have something obstructing our
        # way
        if self.traffic_light_status == 1 or self.obstacle_detected:
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
            #self.stop_sign_detected = False
            #data_to_send = f"{self.angle_correction}\n"
            data_to_send = f"{self.angle_correction},{self.traffic_light_status}\n"
            self.get_logger().info('Timer triggered, sending data: %s' % data_to_send)
            self.send_to_arduino(data_to_send) 
            #self.receive_from_arduino()
    
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
                    self.get_logger().info('Received from Arduino: "%s"' % ard)
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
        """Clean up GPIO and threads"""
        self.ultrasonic_running = False  # Stop the ultrasonic thread
        self.join_threads()
        GPIO.cleanup()  # Clean up GPIO
        
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
