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

Servo myservo;  // create servo object to control a servo

int center_pos = 30; // variable to store the servo central position
int corrected_angle; // Variable to store the actual angle to pass to servo motor
String angle_correction_str; // Variable to store the angle value
String stop_sign_str; // Variable to store if stop sign has been detected
bool stop_sign_detected = false; // Variable to store stop sign detection status
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
  
  // Setting Pins for left back wheel
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
  
  myservo.attach(53);  // attaches the servo on pin 9 to the servo object
  myservo.write(center_pos * 0.7); // Set servo to starting position
  Serial.begin(115200); // Start Serial monitor to receive data
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
    corrected_angle = (center_pos+ angle_correction_int) * 0.7;
    Serial.println(String(corrected_angle));

    // Parse stop sign status
    stop_sign_detected = (stop_sign_str == "1");
  }
}

void start_motors() 
{
  // Add code to initialize all 4 dc motors
  analogWrite(R_PWM, 50);
  analogWrite(R_PWM_2, 50);
  analogWrite(R_PWM_3, 50);
  analogWrite(R_PWM_4, 50);
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