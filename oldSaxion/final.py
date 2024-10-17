import cv2
import mediapipe as mp
import math
import serial
import time

# Set to 1 to enable servos, 0 to disable them , usefull for testing the camera angle calculation without the servos ( debug )
servos_switch = 1  # 0: Camera only, 1: Camera and servos

# Initialize serial communication with Arduino
arduino = None

def initialize_serial(port='COM7', baud_rate=9600, timeout=1):
    global arduino
    arduino = serial.Serial(port, baud_rate, timeout=timeout)
    time.sleep(2)  # Wait for the serial connection to initialize

def calculate_angle(a, b, c):
    """Calculate the angle between three points.
    Angle is calculated using the cosine rule. cos(angle) = (AB o BC) / (|AB| * |BC|)
    """
    # Ensure the denominator is not zero
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    
    cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2))
    # Ensure the cosine value is within the valid range for acos
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    return angle

def send_servo_positions(angles):
    """
    data is sent over serial in the following format: "angle1,angle2,angle3,angle4,safety_switch\n" to the servo driver board
    safety_switch or last_args is used to enable or disable the servos for safety purposes 
    enable means the servos are allows to move freely
    disable means sending them to a default position
    """
    if len(angles) >= 5 and arduino:  # Ensure expected behavior
        # Reverse the order of angles to match the desired sequence
        angles = angles[::-1]
        # Check if angles[4] is greater than 150
        last_arg = 1 if angles[4] > 150 else 0 # 1: Enable servos, 0: Send servos to a default position
        data = f"{angles[0]},{angles[1]},{angles[2]},{angles[3]},{last_arg}\n"
        print("Sending over serial:", data)  # Print what is being sent over serial
        arduino.write(data.encode())

def map_angle(value, input_min, input_max, output_min, output_max):
    """Map a value from one range to another.
    input_min and input_max are the range of the input value (obtained from camera angle calculation) calculate_angle fn
    output_min and output_max are the range of the output value (servo angle range )
    value is the angle obained for one finger
    this function normalizes the input value, muliplies the pourcentage with the output value
    resulting in a value that is in the output range ( a mapping function )
    """
    relative_value = (value - input_min) / (input_max - input_min)
    calc_angle = output_min + (output_max - output_min) * relative_value
    if calc_angle < 1:
        calc_angle = 1 # fail safe
    elif calc_angle > 180:
        calc_angle = 180 # fail safe
    return calc_angle


def get_finger_angles(hand_landmarks, image):
    """
    Specifies which points of the hand to consider when calculating the angle for each finger.
    These values are obtained from the mediapipe hand landmarks, what landmarks are chosen is better seen in the Concise Report document
    """
    fingers_indices = {
        'Thumb': [mp.solutions.hands.HandLandmark.WRIST, mp.solutions.hands.HandLandmark.THUMB_MCP, mp.solutions.hands.HandLandmark.THUMB_TIP],
        'Index': [mp.solutions.hands.HandLandmark.WRIST, mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP],
        'Middle': [mp.solutions.hands.HandLandmark.WRIST, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP],
        'Ring': [mp.solutions.hands.HandLandmark.WRIST, mp.solutions.hands.HandLandmark.RING_FINGER_PIP, mp.solutions.hands.HandLandmark.RING_FINGER_TIP],
        'Pinky': [mp.solutions.hands.HandLandmark.WRIST, mp.solutions.hands.HandLandmark.PINKY_PIP, mp.solutions.hands.HandLandmark.PINKY_TIP]
    }

    angles = []

    # Get the points for each finger joint
    for finger_name, indices in fingers_indices.items():
        base = hand_landmarks.landmark[indices[0]]
        middle = hand_landmarks.landmark[indices[1]]
        tip = hand_landmarks.landmark[indices[2]]

        # Get the coordinates of the finger joint points
        base_coords = (int(base.x * image.shape[0]), int(base.y * image.shape[1]))
        middle_coords = (int(middle.x * image.shape[0]), int(middle.y * image.shape[1]))
        tip_coords = (int(tip.x * image.shape[0]), int(tip.y * image.shape[1]))

        # Calculate the angle between the finger joints, per finger
        angle = calculate_angle(base_coords, middle_coords, tip_coords)

        # Map finger to a bounded range. This is the displayed value on the screen ( not the finger angle value but the corresponding 
        # servo angle value)
        if finger_name == 'Pinky':
            angle = map_angle(angle, 50, 170, 1, 180)
            if angle<1:
                angle = 1
            if angle>180:
                angle = 180
        elif finger_name == 'Ring':
            angle = map_angle(angle, 60, 170, 50, 150)
            if angle<50:
                angle = 50
            if angle>150:
                angle = 150
        elif finger_name == 'Middle':
            angle = map_angle(angle, 40, 170, 70, 150)
            if angle<70:
                angle = 70
            if angle>150:
                angle = 150
        elif finger_name == 'Index':
            angle = map_angle(angle, 40, 170, 60, 80)
            if angle<60:
                angle = 60
            if angle>80:
                angle = 80
        
        angles.append(float(angle))
        # Display the angle on the image
        cv2.putText(image, f'{finger_name} Angle: {int(angle)}', (10, 30 + list(fingers_indices.keys()).index(finger_name) * 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return angles

def process_frame(frame, hands):
    # BGR 2 RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Flip on horizontal
    image = cv2.flip(image, 1)
    
    # Set flag
    image.flags.writeable = False
    
    # Detections
    results = hands.process(image)
    
    # Set flag to true
    image.flags.writeable = True
    
    # RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Rendering results
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get the label of the hand (left or right)
            hand_label = handedness.classification[0].label
            
            if hand_label == 'Left':
                # Draw landmarks for the left hand
                mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, 
                                          mp.solutions.drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp.solutions.drawing_utils.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
                
                # Get finger angles
                angles = get_finger_angles(hand_landmarks, image)
                return image, angles
    return image, []

def main():
    if servos_switch:
        initialize_serial()
    
    # Initialize mediapipe
    mp_hands = mp.solutions.hands
    
    # Open video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize timer
    last_send_time = time.time()

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            image, angles = process_frame(frame, hands)
            
            # Send angles to Arduino once every 0.25 seconds if servos are enabled
            if servos_switch:
                current_time = time.time()
                if current_time - last_send_time >= 0.25:
                    while len(angles) < 4: # fail safe
                        angles.append(90)  # default value for missing angles
                    send_servo_positions(angles)
                    last_send_time = current_time

            # Display the image
            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == ord('q'): #exit on q
                break

    cap.release()
    cv2.destroyAllWindows()
    
    if servos_switch:
        arduino.close()

if __name__ == "__main__":
    main()
