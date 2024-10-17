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

// Function to smoothly move the servos to the target positions
void smoothMove(int startAngles[], int endAngles[], unsigned long duration) {
  unsigned long startTime = millis();
  unsigned long endTime = startTime + duration;

  while (millis() < endTime) {
    unsigned long currentTime = millis();
    float progress = (float)(currentTime - startTime) / duration;

    for (int i = 0; i < 4; i++) {
      uint8_t newAngle = startAngles[i] + (endAngles[i] - startAngles[i]) * progress;
      servo_control(i, newAngle);
    }
    delay(20);  // Adjust this delay for smoother/faster transitions
  }
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
  static int currentAngles[4] = {90, 90, 90, 150};  // Default positions when switch is off
  static boolean switchState = false;  // Switch state

  // Check if data is available to read
  while (Serial.available()) {
    // Get the new angle
    char inChar = (char)Serial.read();
    // Add it to the inputString
    inputString += inChar;
    // If the incoming character is a newline, set a flag so the main loop can process the string
    if (inChar == '\n') {
      stringComplete = true;
    }
  }

  // Process the complete string
  if (stringComplete) {
    // Split the string by commas
    int servoAngles[5];
    int angleIndex = 0;
    int startIndex = 0;
    int endIndex = inputString.indexOf(',');

    // process input
    while (endIndex != -1 && angleIndex < 5) {
      servoAngles[angleIndex++] = inputString.substring(startIndex, endIndex).toInt();
      startIndex = endIndex + 1;
      endIndex = inputString.indexOf(',', startIndex);
    }
    if (angleIndex < 5) {
      servoAngles[angleIndex] = inputString.substring(startIndex).toInt();
    }

    // check for valid angles
    bool valid = true;
    for (int i = 0; i < 4; i++) {
      if (servoAngles[i] < 0 || servoAngles[i] > 180) { 
        Serial.print("Invalid angle for servo ");
        Serial.print(i);
        Serial.println(". Please enter a number between 0 and 180.");
        valid = false;
      }
    }

    // Validate switch state
    if (servoAngles[4] == 0 || servoAngles[4] == 1) { // Thumb position is converted to a Bool switch
      switchState = servoAngles[4];
    } else {
      Serial.println("Invalid switch state. Please enter 0 or 1.");
      valid = false;
    }

    if (valid) {
      // Move servos to specified positions if switch is on
      if (switchState) {
        // Move all servos in parallel to the specified positions over 1 second
        smoothMove(currentAngles, servoAngles, 1000);

        // Update the current angles
        for (int i = 0; i < 4; i++) {
          currentAngles[i] = servoAngles[i];
        }

        // Print the new positions
        for (int i = 0; i < 4; i++) {
          Serial.print("Servo ");
          Serial.print(i);
          Serial.print(" set to: ");
          Serial.println(currentAngles[i]);
        }
      } else {
        // Move all servos to default positions (90, 90, 90, 135)
        int targetAngles[] = {90, 90, 90, 150};
        smoothMove(currentAngles, targetAngles, 1000);

        // Update the current angles
        for (int i = 0; i < 4; i++) {
          currentAngles[i] = 90;
        }

        // Print default positions
        Serial.println("Servos set to default positions.");
      }
    }

    // Clear the inputString and reset the flag
    inputString = "";
    stringComplete = false;
  }
  delay(100);
}
