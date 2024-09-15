#include <Servo.h>


Servo myservo;  // create servo object to control a servo

int center_pos = 90; // variable to store the servo central position
int corrected_angle; // Variable to store the actual angle to pass to servo motor
String angle_correction_str; // Variable to store the angle value
String stop_sign_str; // Variable to store if stop sign has been detected
bool stop_sign_detected = false; // Variable to store stop sign detection status
bool motors_on = false; // Variable to store dc motors status
unsigned long previousMillis = 0; // Variable to store the last time the serial was read
const long interval = 50; // 20 ms interval to match 50 Hz frequency

void setup() 
{
  Serial.begin(9600);
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
  myservo.write(center_pos * 0.7);
}

void loop() 
{
  unsigned long currentMillis = millis(); // Get the current time in milliseconds

  // Check if 20 ms have passed since the last read
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis; // Update the time of the last read

    // If we're receiving serial data this means that there are no obstacles detected 
    // from our frontal vision node
    if (Serial.available() > 0) { // Check if data is available on the serial port
      String input = Serial.readStringUntil('\n'); // Read serial data
      process_serial_input(input); // Process serial data
      
      // If stop sign detected flag is true and our motors are on, stop then and exit
      if (stop_sign_detected && motors_on) {
        stop_motors();
        return;
      }
      // If ours motors are off, start them
      if (!motors_on) {
        start_motors();
      }
      // After all checks we send relevant signals to system actuators
      update_servo(corrected_angle);
    }
    // If we're in this block of code this means an obstacle was detected
    // So we check if our motors are on, if they are, we turn them off
    else {
      if (motors_on) {
        stop_motors();
      }
    } 
  }
}

void process_serial_input(String input)
{
  // Split input string into two parts, using a comma as the separator character
  int comma_index = input.indexOf(',');
  if (comma_index != -1) {
    angle_correction_str = input.substring(0, comma_index);
    stop_sign_str = input.substring(comma_index+1);

    // Parse angle correction data
    int angle_correction_int = angle_correction_str.toInt();
    corrected_angle = (center_pos + angle_correction_int) * 0.7;

    // Parse stop sign status
    stop_sign_detected = (stop_sign_str == "1");
  }
}

void start_motors() 
{
  // Add code to initialize all 4 dc motors
  motors_on = true;
}

void stop_motors()
{
  // Add code to stop all 4 dc motors
  motors_on = false;
}

void update_servo(int angle)
{
  myservo.write(angle);
}



















