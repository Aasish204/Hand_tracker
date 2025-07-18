import cv2                               # OpenCV for real-time computer vision (video capture, image processing)
import mediapipe as mp                   # MediaPipe for building machine learning pipelines (e.g., face/hand tracking)
import time                              # Time module for handling time-related tasks (e.g., delays, measuring time)
import os

cap = cv2.VideoCapture(0)                # Start video capture from the default camera (camera index 0)

mpHands = mp.solutions.hands             # Initialize MediaPipe Hands
hands = mpHands.Hands()                  # Set up the hands object for processing
mpDraw = mp.solutions.drawing_utils      # Utility for drawing hand landmarks

# Customize the drawing style: Green points (landmarks)
drawSpec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

pTime = 0                                # Previous time (for FPS calculation)
cTime = 0                                # Current time (for FPS calculation)
 
while True:                              # Infinite loop that runs until manually stopped
    success, img = cap.read()            # Capture a frame from the camera
    if not success:      
        print("Failed to capture image") # Print error message if the frame is not captured
        break
    img = cv2.flip(img, 1)                         # Flip the image horizontally for a mirror-like view
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for MediaPipe
    result = hands.process(imgRGB)                 # Process the frame for hand landmarks

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            
            # Draw hand landmarks on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, drawSpec, drawSpec)

    # FPS Calculation
    cTime = time.time()
    fps = int(1 / (cTime - pTime))                 # Calculate FPS
    pTime = cTime

    # Display FPS on the image with blue color (BGR format)
    cv2.putText(img, f"FPS: {fps}", (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

    # Display the captured frame in a window
    cv2.imshow("Image", img)

    # Wait for 1 millisecond for a key pressed; if the 'a' key is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()                            # Release the video capture object
cv2.destroyAllWindows()                  # Close all OpenCV windows
