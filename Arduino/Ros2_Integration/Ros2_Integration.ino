#include <Servo.h>


Servo myservo;  // create servo object to control a servo

int center_pos = 90;    // variable to store the servo position
int corrected_angle;
String angle_correction_str; // Variable to store the angle value
unsigned long previousMillis = 0; // Variable to store the last time the serial was read
const long interval = 20; // 20 ms interval to match 50 Hz frequency

void setup() 
{
  Serial.begin(115200);
  myservo.attach(53);  // attaches the servo on pin 9 to the servo object
  //current_angle = center_pos * 0.7;
  myservo.write(center_pos * 0.7);
}

void loop() 
{
  unsigned long currentMillis = millis(); // Get the current time in milliseconds

  // Check if 20 ms have passed since the last read
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis; // Update the time of the last read

    if (Serial.available() > 0) { // Check if data is available on the serial port
      angle_correction_str = Serial.readStringUntil(','); // Read the integer input from the serial monitor
      int angle_correction_int = angle_correction_str.toInt();
      corrected_angle = (center_pos + angle_correction_int) * 0.7;
      myservo.write(corrected_angle);
    }
    
  }
}