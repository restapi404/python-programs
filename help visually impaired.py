import cv2
import numpy as np
import pyttsx3
import threading
import time
from ultralytics import YOLO

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech speed

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height
cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for better performance

thres = 0.55  # Confidence threshold for detection
frame_skip = 5  # Skip frames to reduce lag
frame_count = 0
last_spoken_time = 0  # Time tracker for speech cooldown
speech_delay = 2  # Speak only once every 2 seconds

# Function to detect the color of the traffic light
def detect_traffic_light_color(traffic_light_img):
    hsv = cv2.cvtColor(traffic_light_img, cv2.COLOR_BGR2HSV)

    # Define color ranges for Red, Yellow, Green (in HSV)
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])

    green_lower = np.array([35, 50, 70])
    green_upper = np.array([85, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # Mask for each color
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Check if the mask has non-zero pixels
    if np.sum(red_mask) > 0:
        return "Red"
    elif np.sum(green_mask) > 0:
        return "Green"
    elif np.sum(yellow_mask) > 0:
        return "Yellow"
    else:
        return "Unknown"

# Function to speak detected objects and traffic light color (runs in a separate thread)
def speak_objects(detected_objects, traffic_light_color=None):
    global last_spoken_time
    if time.time() - last_spoken_time > speech_delay:  # Avoid repeating too often
        if traffic_light_color:
            objects_str = f"Traffic Light: {traffic_light_color}"  # Include traffic light color
        else:
            objects_str = ", ".join(detected_objects)  # Convert detected objects to a string
        last_spoken_time = time.time()
        print(f"Speaking: {objects_str}")  # Debug print
        engine.say(f"Detected {objects_str}")
        threading.Thread(target=engine.runAndWait, daemon=True).start()

# Main loop to process video frames
while True:
    success, img = cap.read() # Capture frame from webcam
    if not success:
        break # Exit if frame not captured

    frame_count += 1 # Increment frame counter

    if frame_count % frame_skip == 0:  # Skip frames for better performance
        results = model(img)  # Run YOLOv8 object detection

        detected_objects = set()  # Use a set to store detected object names (avoid duplicates)

        traffic_light_color = None  # Initialize traffic light color as None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0].item()  # Confidence score
                label = result.names[int(box.cls[0].item())]  # Get the class label of detected object

                if confidence > thres:  # Only detect objects above threshold
                    detected_objects.add(label)  # Store detected object name

                    # If a traffic light is detected, crop the image and detect its color
                    if label == 'traffic light':
                        traffic_light_img = img[y1:y2, x1:x2]  # Crop the image to the traffic light region
                        traffic_light_color = detect_traffic_light_color(traffic_light_img)  # Detect color

                    # Draw bounding box around detected object
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Display label and confidence score
                    cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Speak detected objects if any, including traffic light color if detected
        if detected_objects or traffic_light_color:
            speak_objects(detected_objects, traffic_light_color)

        # Show the video with detections
        cv2.imshow("YOLOv8 Detection", img)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam and close OpenCV windows   
cap.release()
cv2.destroyAllWindows()
