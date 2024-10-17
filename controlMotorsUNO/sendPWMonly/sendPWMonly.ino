#include <Wire.h> //I2C Library 
#include <Adafruit_PWMServoDriver.h> //PWM Servo Driver Library

// Create an instance of the Adafruit PWM Servo Driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Limits pwm pulse length for sg90 servos
#define SERVOMIN 150  // Minimum pulse length count (out of 4096)
#define SERVOMAX 600  // Maximum pulse length count (out of 4096)

// Function to convert degrees to pulse length
uint16_t degreesToPulse(uint8_t degrees) {
  return map(degrees, 0, 180, SERVOMIN, SERVOMAX);
}

// Function to control the servo position
void servo_control(uint8_t servoNum, uint8_t degrees) {
  uint16_t pulseLength = degreesToPulse(degrees);
  pwm.setPWM(servoNum, 0, pulseLength);
}

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(60);  // Analog servos run at ~60 Hz
  delay(10);
}

void loop() {
  static String inputString = "";  // A string to hold incoming data
  static boolean stringComplete = false;  // Whether the string is complete

  // Check if data is available to read
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    if (inChar == '\n') {
      stringComplete = true;
    }
  }

  // Process the complete string
  if (stringComplete) {
    int servoAngles[4];
    int angleIndex = 0;
    int startIndex = 0;
    int endIndex = inputString.indexOf(',');

    // Parse the input string into servo angles
    while (endIndex != -1 && angleIndex < 4) {
      servoAngles[angleIndex++] = inputString.substring(startIndex, endIndex).toInt();
      startIndex = endIndex + 1;
      endIndex = inputString.indexOf(',', startIndex);
    }
    if (angleIndex < 4) {
      servoAngles[angleIndex] = inputString.substring(startIndex).toInt();
    }

    // Send angles to the PWM driver
    for (int i = 0; i < 4; i++) {
      if (servoAngles[i] >= 0 && servoAngles[i] <= 180) { 
        servo_control(i, servoAngles[i]);
        Serial.print("Servo ");
        Serial.print(i);
        Serial.print(" set to: ");
        Serial.println(servoAngles[i]);
      } else {
        Serial.print("Invalid angle for servo ");
        Serial.print(i);
        Serial.println(". Please enter a number between 0 and 180.");
      }
    }

    // Clear the inputString and reset the flag
    inputString = "";
    stringComplete = false;
  }
  delay(100);
}
