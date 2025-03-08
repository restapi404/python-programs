<<<<<<< HEAD
# OBJECT DETECTION AND OBSTACLE AVOIDANCE WITH TRAFFIC LIGHT DETECTION

# Project Overview
This project detects objects, avoids obstacles, identifies traffic light colors, and provides audible descriptions of detected objects along with their estimated distance. The program uses YOLOv8 for object detection and a webcam for real-time video processing.

# Features
- **Object Detection**: Identifies objects in real-time using YOLOv8.
- **Obstacle Avoidance**: Recognizes and avoids obstacles based on the detected objects.
- **Traffic Light Detection**: Detects the color of traffic lights (Red, Green, Yellow) using color detection techniques.
- **Voice Feedback**: Provides auditory descriptions of detected objects and traffic light colors using a text-to-speech engine (pyttsx3).

# Technology Stack
- **Python 3.x**
- **OpenCV**: For video capture and image processing.
- **YOLOv8 (Ultralytics)**: For real-time object detection.
- **pyttsx3**: For text-to-speech functionality.
- **NumPy**: For numerical operations.

# How It Works
1. The program captures frames from the webcam.
2. YOLOv8 is used to detect objects in each frame.
3. Traffic light detection is performed by extracting the region of the traffic light and identifying its color.
4. Detected objects and traffic light colors are spoken aloud using the pyttsx3 library.
5. The program continues to process video frames until the user stops it by pressing the 'q' key.

# Installation
1. Clone the repository or download the code.
2. Install dependencies:
   pip install opencv-python numpy pyttsx3 ultralytics
   
3. Download the YOLOv8 model (`yolov8n.pt`) from the [official YOLOv8 repository](https://github.com/ultralytics/yolov5/releases).
4. Run the program:
   python object_detection.py
   

## Usage
- Open the webcam, and the program will start detecting objects and traffic lights.
- Detected objects and traffic light colors will be spoken aloud.
- Press 'q' to exit the program.

## License
This project is licensed under the MIT License.
=======
"# project overview" 
detects objects, avoids obstacles, detects colors of traffic light, speaks out detected objects and estimated distance
>>>>>>> 6a41ca6de751d2d387f4742564ca9c3fe54a1767
