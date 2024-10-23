#include <Servo.h>

// Pins for Left Front Wheel Motor
int R_IS = 22;
int R_EN = 24;
int R_PWM = 9;
int L_IS = 26;
int L_EN = 28;
int L_PWM = 10;

// Pins for Left Back Wheel Motor
int R_IS_2 = 32;
int R_EN_2 = 34;
int R_PWM_2 = 11;
int L_IS_2 = 36;
int L_EN_2 = 38;
int L_PWM_2 = 12;

// Pins for Right Back Wheel Motor
int R_IS_3 = 42;
int R_EN_3 = 44;
int R_PWM_3 = 7;
int L_IS_3 = 46;
int L_EN_3 = 48;
int L_PWM_3 = 8;

// Pins for Right Front Wheel Motor
int R_IS_4 = 1;
int R_EN_4 = 2;
int R_PWM_4 = 5;
int L_IS_4 = 3;
int L_EN_4 = 4;
int L_PWM_4 = 6;


// Pins for ultrassonic sensor
const int trig_pin = 51;
const int echo_pin = 53;

Servo myservo;  // create servo object to control a servo

int center_pos = 90; // variable to store the servo central position
int corrected_angle; // Int variable to store the angle value to pass to servo motor
String corrected_angle_str; // String variable to store the angle value from serial input
String traffic_light_status_str; // String variable to store traffic light status from serial input
// Int variable to store and use traffic light status data
int traffic_light_status = 0; // 0 = NONE, 1 = RED, 2 = YELLOW, 3 = GREEN
//String stop_sign_str; // Variable to store if stop sign has been detected
//bool stop_sign_detected = false; // Variable to store stop sign detection status
bool motors_on = false; // Variable to store dc motors status
unsigned long previousMillis = 0; // Variable to store the last time the serial was read
const long interval = 100; // 20 ms interval to match 50 Hz frequency

void setup() 
{
  // Setting Pins for letf front wheel
  pinMode(R_IS, OUTPUT);
  pinMode(R_EN, OUTPUT);
  pinMode(R_PWM, OUTPUT);
  pinMode(L_IS, OUTPUT);
  pinMode(L_EN, OUTPUT);
  pinMode(L_PWM, OUTPUT);
  
  // Setting Pins for left back wheel  //start_motors();
  pinMode(R_IS_2, OUTPUT);
  pinMode(R_EN_2, OUTPUT);
  pinMode(R_PWM_2, OUTPUT);
  pinMode(L_IS_2, OUTPUT);
  pinMode(L_EN_2, OUTPUT);
  pinMode(L_PWM_2, OUTPUT);

  // Setting Pins for right back wheel
  pinMode(R_IS_3, OUTPUT);
  pinMode(R_EN_3, OUTPUT);
  pinMode(R_PWM_3, OUTPUT);
  pinMode(L_IS_3, OUTPUT);
  pinMode(L_EN_3, OUTPUT);
  pinMode(L_PWM_3, OUTPUT);

  // Setting Pins for right front wheel
  pinMode(R_IS_4, OUTPUT);
  pinMode(R_EN_4, OUTPUT);
  pinMode(R_PWM_4, OUTPUT);
  pinMode(L_IS_4, OUTPUT);
  pinMode(L_EN_4, OUTPUT);
  pinMode(L_PWM_4, OUTPUT);

  // Setting ultrasonic sensors pins
  pinMode(trig_pin, OUTPUT); // Trigger pin as output
  pinMode(echo_pin, INPUT);  // Echo pin as input
  
  digitalWrite(R_IS, LOW);
  digitalWrite(L_IS, LOW);
  digitalWrite(R_EN, HIGH);
  digitalWrite(L_EN, HIGH);
  
  digitalWrite(R_IS_2, LOW);
  digitalWrite(L_IS_2, LOW);
  digitalWrite(R_EN_2, HIGH);
  digitalWrite(L_EN_2, HIGH);
  
  digitalWrite(R_IS_3, LOW);
  digitalWrite(L_IS_3, LOW);
  digitalWrite(R_EN_3, HIGH);
  digitalWrite(L_EN_3, HIGH);
  
  digitalWrite(R_IS_4, LOW);
  digitalWrite(L_IS_4, LOW);
  digitalWrite(R_EN_4, HIGH);
  digitalWrite(L_EN_4, HIGH);

  myservo.attach(52);  // attaches the servo on pin 9 to the servo object
  myservo.write(center_pos * 0.7); // Set servo to starting position
  Serial.begin(115200); // Start Serial monitor to receive data
  //start_motors();
}

void loop() 
{ 
  //if (get_ultrasonic_sensor_distance() <= 60)
  //{
    //stop_motors();
  //}
  
  unsigned long currentMillis = millis(); // Get the current time in milliseconds

  // Check if 20 ms have passed since the last read
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis; // Update the time of the last read

    // If we're receiving serial data this means that there are no obstacles detected 
    // from our frontal vision node
    if (Serial.available() > 0) { // Check if data is available on the serial port
      String input = Serial.readStringUntil('\n'); // Read serial data
      process_serial_input(input); // Process serial data

      if (traffic_light_status == 2) {
        slow_motors();
      }
      else {
        start_motors();
      }
      
      // After processing serial input we send new angle data to update servo
      update_servo(corrected_angle);
    }
    // If we're in this block of code this means an that serial communication
    // Has been terminated, this happens if we detect a person in front of the 
    // Car or if we have detected a red traffic light either way, we stop our
    // motors
    else {
      stop_motors();
    } 
  }
}

int get_ultrasonic_sensor_distance()
{
  // Send a 10us pulse to trigger the sensor
  digitalWrite(trig_pin, LOW);
  delayMicroseconds(2);
  digitalWrite(trig_pin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig_pin, LOW);

  // Measure the time for the echo to return
  long duration = pulseIn(echo_pin, HIGH);

  // Calculate distance in cm (sound speed = 343 m/s, or 0.0343 cm/us)
  int distance = duration * 0.0343 / 2;  // Divide by 2 since sound has to travel to the object and back

  // Print the distance to the Serial Monitor
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  return distance;
}


void process_serial_input(String input)
{
  // Split input string into two parts, using a comma as the separator character
  int comma_index = input.indexOf(',');
  if (comma_index != -1) {
    // Parse serial input into correct variables
    corrected_angle_str = input.substring(0, comma_index);
    traffic_light_status_str = input.substring(comma_index+1);

    // Parse angle correction data
    corrected_angle = corrected_angle_str.toInt();
    //int angle_correction_int = angle_correction_str.toInt() * 4;
    //corrected_angle = (center_pos + angle_correction_int) * 0.7;

    // Parse traffic light status data
    traffic_light_status = traffic_light_status_str.toInt();
    
    // Parse stop sign status
    //stop_sign_str = input.substring(comma_index+1);
    //stop_sign_detected = (stop_sign_str == "1");
  }
}

void start_motors() 
{
  // Add code to initialize all 4 dc motors
  analogWrite(R_PWM, 65);
  analogWrite(R_PWM_2, 65);
  analogWrite(R_PWM_3, 65);
  analogWrite(R_PWM_4, 65);
  motors_on = true;
}

void slow_motors()
{
  // Slow down all motors
  analogWrite(R_PWM, 30);
  analogWrite(R_PWM_2, 30);
  analogWrite(R_PWM_3, 30);
  analogWrite(R_PWM_4, 30);
  motors_on = true;
}

void stop_motors()
{
  // Add code to stop all 4 dc motors
  analogWrite(R_PWM, 0);
  analogWrite(R_PWM_2, 0);
  analogWrite(R_PWM_3, 0);
  analogWrite(R_PWM_4, 0);
  motors_on = false;
}

void update_servo(int angle)
{
  myservo.write(angle);
}
