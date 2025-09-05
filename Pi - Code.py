import cv2
import numpy as np
from picamera2 import Picamera2
import time
import socket
import struct
import pickle

# Setup socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8485))  # Listen on all interfaces, port 8485
server_socket.listen(1)
print("Waiting for laptop connection...")
conn, addr = server_socket.accept()
print(f"Connected by: {addr}")

# Function for lane detection
def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def process_frame(frame):
    # Converting the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Performing Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Defining a region of interest (ROI)
    row, col = edges.shape
    roi_vertices = [(0, row), (col // 2, row // 2), (col, row)]
    roi_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    return roi_edges

def detect_lanes(frame):
    edges = process_frame(frame)

    # Applying Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Blue lane markings
    return frame

# Function for Obstacle Detection
def detect_humans(frame):
    # Converting the frame to grayscale before processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initializing HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detecting humans in the grayscale frame
    boxes, weights = hog.detectMultiScale(gray_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Drawing bounding boxes on the original color frame
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red box for human

    return frame, len(boxes) > 0  # Return the original frame with bounding boxes


# Function for turn decision
def detect_lane_curvature(frame):
    edges = process_frame(frame)
    height, width = frame.shape[:2]

    # Finding lane lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        # Fitting a second-degree polynomial to the detected lane points
        x_points = []
        y_points = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                x_points.extend([x1, x2])
                y_points.extend([y1, y2])

        # Fitting a second-degree polynomial to the lane points
        poly_coeffs = np.polyfit(x_points, y_points, 2)

        # Calculate the direction of curvature
        curvature_direction = "Straight"
        if poly_coeffs[0] > 0:
            curvature_direction = "Right Turn"
        elif poly_coeffs[0] < 0:
            curvature_direction = "Left Turn"

        # Drawing the fitted polynomial on the image if the curvature is detected
        poly_y = np.linspace(0, height - 1, height)
        poly_x = poly_coeffs[0] * poly_y ** 2 + poly_coeffs[1] * poly_y + poly_coeffs[2]
        for i in range(len(poly_y) - 1):
            cv2.line(frame, (int(poly_x[i]), int(poly_y[i])), (int(poly_x[i + 1]), int(poly_y[i + 1])), (0, 255, 255), 3)

        return frame, curvature_direction

    return frame, "Straight"

# Function to process camera feed
def process_camera_feed():
    # Initializing Picamera2
    picam2 = Picamera2()

    # Setting up camera stream
    stream_config = picam2.create_video_configuration(main={"size": (640, 480)}, raw={"size": (640, 480)})
    picam2.configure(stream_config)
    picam2.start()

    # Starting capturing frames from the camera
    while True:
        frame = picam2.capture_array()

        # Detect lanes
        frame = detect_lanes(frame)

        # Human detection and stop/move decision (only stop if human is detected)
        frame, stop_decision = detect_humans(frame)

        # Directional control (turn detection)
        frame, turn_decision = detect_lane_curvature(frame)

        # Display results on processed frame
        cv2.putText(frame, f"Stop/Move Decision: {'Stop' if stop_decision else 'Move'}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Turn Decision: {turn_decision}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = pickle.dumps(frame)
        message = struct.pack("Q", len(data)) + data
        conn.sendall(message)
        # Show the processed video in a separate window
        cv2.imshow("Processed Video", frame)

        # Real-time video playback: wait for the time corresponding to the FPS
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value for Escape
            break

    # Clean up
    picam2.stop()
    conn.close()
    cv2.destroyAllWindows()

process_camera_feed()
