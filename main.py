# Import necessary libraries
import cv2  # OpenCV library for computer vision tasks
import mediapipe as mp  # MediaPipe library for hand tracking
import time  # Time library for calculating frames per second (FPS)

# Initialize the camera
camera_capture = cv2.VideoCapture(0)  # Capture video from the default camera (index 0)
camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the frame width to 640
camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the frame height to 480

# Initialize MediaPipe hands module
hand_tracking_module = mp.solutions.hands  # MediaPipe hands module
hand_tracker = hand_tracking_module.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)  # Create a Hands object with optimized settings
drawing_utilities = mp.solutions.drawing_utils  # MediaPipe drawing utilities

# Initialize variables for calculating FPS
previous_time = 0  # Previous time
current_time = 0  # Current time

while True:
    # Read a frame from the camera
    frame_read_success, frame = camera_capture.read()  # Read a frame from the camera

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB

    # Process the frame using MediaPipe hands
    hand_detection_results = hand_tracker.process(rgb_frame)  # Process the frame using MediaPipe hands

    # Check if hands are detected
    if hand_detection_results.multi_hand_landmarks:
        # Iterate over the detected hands
        for detected_hand in hand_detection_results.multi_hand_landmarks:
            # Get the shape of the frame
            frame_height, frame_width, frame_channels = frame.shape  # Get the shape of the frame

            # Get the coordinates of the first landmark (wrist)
            wrist_x, wrist_y = int(detected_hand.landmark[0].x * frame_width), int(detected_hand.landmark[0].y * frame_height)

            # Print the coordinates of the wrist
            print("Hand ID:", 0, "Wrist X:", wrist_x, "Wrist Y:", wrist_y)  # Print the coordinates of the wrist

            # Draw a circle at the wrist
            cv2.circle(frame, (wrist_x, wrist_y), 10, (0, 255, 0), cv2.FILLED)  # Draw a circle at the wrist

        # Draw the hand landmarks and connections
        drawing_utilities.draw_landmarks(frame, detected_hand, hand_tracking_module.HAND_CONNECTIONS, drawing_utilities.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))  # Draw the hand landmarks and connections

    # Calculate the FPS
    current_time = time.time()  # Get the current time
    frames_per_second = 1 / (current_time - previous_time)  # Calculate the FPS
    previous_time = current_time  # Update the previous time

    # Display the Author and FPS
    cv2.putText(frame, "Made by KieranMc", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)  # Display the author
    if frames_per_second < 50:
        cv2.putText(frame, "FPS: " + str(int(frames_per_second)), (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)  # Display the FPS in red
    else:
        cv2.putText(frame, "FPS: " + str(int(frames_per_second)), (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)  # Display the FPS in green
    
    cv2.putText(frame, "Press 'q' to quit", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 102), 2)
    
    cv2.imshow("Hand Tracking", frame)  # Display the frame

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop

# Release the camera and close all windows
camera_capture.release()  # Release the camera
cv2.destroyAllWindows()  # Close all windows