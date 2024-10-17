import sys
import os
import time

import cv2
import mediapipe as mp
import math
import serial


# Add the subfolder1 to the system path to import modules
sys.path.append(os.path.join(os.path.dirname("controlMotorsUNO/sendPWMonly"), 'endPointControl'))

import cv2
import mediapipe as mp

class CameraControl:
    def __init__(self):
        self.cap = None

    def turn_on(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            print("Camera is ON")
            return True
        else:
            print("Failed to open the camera")
            return False

    def capture_image(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def turn_off(self):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Camera is OFF")



import cv2
import mediapipe as mp

class HandProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

    def process_frame(self, frame):
        """
        Process the frame to detect hands and display the index finger tip and its coordinates.
        
        :param frame: The original frame captured by the camera.
        :return: Frame with the index finger tip and its coordinates drawn on it (if detected).
        """
        # Convert the image to RGB and flip for a natural display
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        # Set the image to non-writeable for performance optimization
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        # Convert back to BGR for rendering
        processed_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw only the index finger tip and its coordinates if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of the index finger tip
                tip_coords = self.get_index_finger_tip_position(hand_landmarks, processed_frame)
                if tip_coords:
                    # Draw a circle at the index finger tip
                    cv2.circle(processed_frame, tip_coords, 10, (0, 255, 0), -1)  # Green circle for the tip of the index finger

                    # Draw the coordinates as text near the fingertip
                    cv2.putText(processed_frame, f'{tip_coords}', (tip_coords[0] + 10, tip_coords[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)  # Blue text for coordinates

        return processed_frame

    def get_index_finger_tip_position(self, hand_landmarks, frame):
        """
        Get the x and y coordinates of the index finger tip.
        
        :param hand_landmarks: Landmark data for the detected hand.
        :param frame: The original frame to calculate coordinates in pixel values.
        :return: A tuple of (x, y) coordinates for the index finger tip, or None if not detected.
        """
        # Get the index finger tip landmark
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Convert the normalized coordinates to pixel values
        h, w, _ = frame.shape
        x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

        return (x, y)






class KineticsHandle:
    def initialize(self):
        print("Initializing Kinetics Handle...")

    def rotZ(self, angle):
        
        return angle
    
    def translate(self, dir, distance):
        return distance
    
    def translateX(self, distance):
        return self.translate("X", distance)
    
    def translateY(self, distance):
        return self.translate("Y", distance)
    
    def closeClaw(self):
        return True

    def forwardKinematics(self, coords):
        x,y,z = coords
        # Simulate forward kinematics
        return
        


class SerialHandle:
    def __init__(self, port='COM7', baud_rate=9600, timeout=1):
        """
        Initialize the serial connection.
        
        :param port: Serial port name (e.g., 'COM7' or '/dev/ttyUSB0')
        :param baud_rate: Communication speed
        :param timeout: Timeout for serial communication in seconds
        """
        self.arduino = None
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout

    def initialize_serial(self):
        """
        Opens the serial port connection.
        """
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            time.sleep(2)  # Wait for the serial connection to initialize
            print(f"Serial connection established on {self.port} at {self.baud_rate} baud rate.")
        except serial.SerialException as e:
            print(f"Error initializing serial port: {e}")

    def serial_encode(self, message):
        """
        Encodes the message for serial communication.
        
        :param message: The message to be encoded
        :return: Encoded message in the format ready to be sent via serial
        """
        try:
            encoded_message = message.encode('utf-8')
            return encoded_message
        except Exception as e:
            print(f"Error encoding message: {e}")
            return None

    def send_servo_positions(self, encoded_message):
        """
        Sends the encoded message to the connected device via serial communication.
        
        :param encoded_message: Message that has already been encoded for serial transmission
        """
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(encoded_message)
                print("Message sent successfully.")
            except Exception as e:
                print(f"Error sending message: {e}")
        else:
            print("Serial port is not open. Please initialize the serial connection first.")



class MainSystem:
    def __init__(self):
        self.coords = None
        self.camera_status = None
        self.image = None
        self.serial_com_status = None
        self.move_boolean = None

        # Initialize handles
        self.kinetics_handle = KineticsHandle()
        self.camera_handle = CameraControl()
        self.serial_handle = SerialHandle()
        
        

    def create_systems_and_environment(self):
        return None

    def move_to_position(self, coords):
        print(f"Robot moved to position with coordinates: {coords}")


# Example usage
if __name__ == "__main__":
    # Actors: Creator and Human interacting with the system
    human = "Human"
    creator = "Creator"

    main_system = MainSystem()
    main_system.create_systems_and_environment()
    # Instantiate the components
    camera = CameraControl()
    hand_processor = HandProcessor()

    # Turn on the camera and start processing the frames
    if camera.turn_on():
        while True:
            frame = camera.capture_image()
            if frame is None:
                break

            # Process the frame and detect hands
            processed_frame = hand_processor.process_frame(frame)

            # Display the processed frame
            cv2.imshow('Hand Tracking', processed_frame)

            # Exit on pressing 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Turn off the camera and clean up
        camera.turn_off()
