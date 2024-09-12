#include <Servo.h>


Servo myservo;  // create servo object to control a servo

int center_pos = 90 * 0.7;    // variable to store the servo position
int angle; // Variable to store the angle value
unsigned long previousMillis = 0; // Variable to store the last time the serial was read
const long interval = 50; // 20 ms interval to match 50 Hz frequency

void setup() 
{
  Serial.begin(9600);
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
  myservo.write(center_pos);
}

void loop() 
{
  unsigned long currentMillis = millis(); // Get the current time in milliseconds

  // Check if 20 ms have passed since the last read
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis; // Update the time of the last read

    if (Serial.available() > 0) { // Check if data is available on the serial port
      angle = Serial.parseInt(); // Read the integer input from the serial monitor
      myservo.write(angle + center_pos);
      Serial.flush();
    }
    
  }
}
