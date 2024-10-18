import sys
import os
import time

import cv2
import mediapipe as mp
import math
import serial

import numpy as np
import pandas as pd
import scipy.spatial.transform.rotation as rot


# Add communication to ---» serial message reciever
sys.path.append(os.path.join(os.path.dirname("controlMotorsUNO/sendPWMonly"), 'endPointControl'))




# getting Camera frame 





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

# mediapiepe Handmark detection and processing












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
# Geometric operations











class XYZTransformations:


    def rotation_matrix_x(self,angle_degrees):
        angle_radians = np.deg2rad(angle_degrees)
        return np.array([[1, 0, 0, 0],
                        [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
                        [0, np.sin(angle_radians), np.cos(angle_radians), 0],
                        [0, 0, 0, 1]])

    def rotation_matrix_y(self,angle_degrees):
        angle_radians = np.deg2rad(angle_degrees)
        return np.array([[np.cos(angle_radians), 0, np.sin(angle_radians), 0],
                        [0, 1, 0, 0],
                        [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
                        [0, 0, 0, 1]])

    def rotation_matrix_z(self,angle_degrees):
        angle_radians = np.deg2rad(angle_degrees)
        return np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
                        [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    def translation_matrix_x(self,tx):
        return np.array([[1, 0, 0, tx],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    def translation_matrix_y(self,ty):
        return np.array([[1, 0, 0, 0],
                        [0, 1, 0, ty],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    def translation_matrix_z(self,tz):
        return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, tz],
                        [0, 0, 0, 1]])
        
    def identity(self):
        return np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]]
                )

    def homogenous_matrix(self, operation, value):
        id = self.identity()  # Assuming this method exists in your class

        self.case = {
            0: id @ self.rotation_matrix_x(value),
            1: id @ self.rotation_matrix_y(value),  # Added self. to call the method
            2: id @ self.rotation_matrix_z(value),  # Added self. to call the method
            3: id @ self.translation_matrix_x(value),  # Added self. to call the method
            4: id @ self.translation_matrix_y(value),  # Added self. to call the method
            5: id @ self.translation_matrix_z(value)   # Added self. to call the method
        }

        return self.case.get(operation, "Invalid operation")  # Use self.case and operation parameter directly


    def generateHtotal(self,operations,values):
        htotal = self.identity()
        for i in range(len(operations)):
            htotal = htotal @ self.homogenous_matrix(operations[i],values[i])
        return htotal

    def apply_homogeneous_transform(self,matrix, vector):
        # Extend the 3D vector to homogeneous coordinates by adding 1
        extended_vector = np.array([vector[0], vector[1], vector[2], 1])
        
        # Apply the transformation matrix
        transformed_vector = matrix @ extended_vector
        
        # Return the transformed vector (in Cartesian coordinates)
        return transformed_vector[:3]  # We ignore the last element (homogeneous component)

    def clean_matrix(self,matrix, tolerance=1e-10):
        """
        Replaces very small values in a matrix with zero.

        Parameters:
        matrix (np.ndarray): The input matrix to clean.
        tolerance (float): The threshold below which values are set to zero (default: 1e-10).

        Returns:
        np.ndarray: The cleaned matrix with near-zero values set to zero.
        """
        cleaned_matrix = np.where(abs(matrix) < tolerance, 0, matrix)
        return np.array(cleaned_matrix)


    # # Example usage:
    # matrix = np.array([[6.123234e-17, 1.000000e+00, 0.000000e+00],
    #                    [1.000000e+00, 6.123234e-17, 0.000000e+00],
    #                    [0.000000e+00, 0.000000e+00, 6.123234e-17]])

    # cleaned_matrix = clean_matrix(matrix)
 
    """ tests

        
    # Test for Pure Translation
    m_translation = generateHtotal([3], [10])  # Translation along x-axis by 10
    vector = [1, 2, 3]
    transformed_vector = apply_homogeneous_transform(m_translation, vector)
    # print("Pure Translation Test - Expected: [11, 2, 3], Got:", transformed_vector)

    # Test for Pure Rotation around Z-axis
    m_rotation_z = clean_matrix(generateHtotal([2], [90]))  # 90 degree rotation around z-axis
    vector = [1, 0, 0]
    transformed_vector = apply_homogeneous_transform(m_rotation_z, vector)
    # print("Pure Rotation Test - Expected: [0, 1, 0], Got:", transformed_vector)

    # Test for Combination of Translation and Rotation
    m_combined = generateHtotal([3, 2], [5, 90])  # Translation by 5 and 90 degree rotation around z-axis
    vector = [1, 0, 0]
    transformed_vector = apply_homogeneous_transform(m_combined, vector)
    # print("Combined Test - Expected: [5, 1, 0], Got:", transformed_vector)
    """

class SerialHandle:
    def __init__(self, port='COM5', baud_rate=9600, timeout=1):
        """
        Initialize the serial connection.
        
        :param port: Serial port name (e.g., 'COM5' or '/dev/ttyUSB0')
        :param baud_rate: Communication speed
        :param timeout: Timeout for serial communication in seconds
        """
        self.arduino = None
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout

        # Create a dictionary to store motor angles
        self.motor_angles = {
            'm0': 0,
            'm1': 0,
            'm2': 0,
            'm3': 0
        }

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

    def update_motor_angles(self, motor, angle):
        """
        Update the angle for a specific motor in the motor_angles dictionary.
        
        :param motor: The motor identifier ('m0', 'm1', 'm2', 'm3')
        :param angle: The angle value to set for the specified motor (should be between 0 and 180)
        
        :return: bool: True if the update was successful, False if the angle is out of range
        """
        if motor in self.motor_angles and 0 <= angle <= 180:
            self.motor_angles[motor] = angle
            print(f"{motor} updated to {angle} degrees.")
            return True
        else:
            print(f"Invalid angle {angle} for motor {motor}. Must be between 0 and 180.")
            return False

    def encode_motor_angles(self):
        """
        Transform the motor angles dictionary into a comma-separated string format.
        
        :return: str: A string in the format 'angle1,angle2,angle3,angle4'
        """
        return ','.join(str(self.motor_angles[motor]) for motor in ['m0', 'm1', 'm2', 'm3'])
















# Main 
















class MainSystem:
    def __init__(self):
        self.coords = None
        self.camera_status = None
        self.image = None
        self.serial_com_status = None
        self.move_boolean = None

        # Initialize handles
        self.transformation = XYZTransformations()
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
    if camera.turn_off():
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


me = XYZTransformations()
angle_x = 45

m_combined = me.generateHtotal([3, 2], [5, 90])  # Translation by 5 and 90 degree rotation around z-axis
vector = [1, 1, 1]
me.apply_homogeneous_transform(m_combined, vector)

transformed_vector = me.apply_homogeneous_transform(m_combined, vector)
# print(transformed_vector)

# Example usage of the SerialHandle class

serial_handle = SerialHandle()
serial_handle.initialize_serial()

# Update motor angles
serial_handle.update_motor_angles('m0', 90)
serial_handle.update_motor_angles('m1', 45)
serial_handle.update_motor_angles('m2', 110)
serial_handle.update_motor_angles('m3', 80)

# Encode angles into the Arduino-compatible string format
encoded_string = serial_handle.encode_motor_angles()
print(f"Encoded Motor Angles: {encoded_string}")  # Output: '90,45,180,0'

# Send the encoded angles to the Arduino
encoded_message = serial_handle.serial_encode(encoded_string + '\n')  # Adding newline character
if encoded_message:
    serial_handle.send_servo_positions(encoded_message)