import cv2
import numpy as np

# Helper functions for each task

# 1. Lane Detection
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def process_frame(frame):
    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest (ROI)
    height, width = edges.shape
    roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]
    roi_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    return roi_edges

def detect_lanes(frame):
    edges = process_frame(frame)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Blue lane markings
    return frame

# 2. Human Detection (Obstacle Detection)
def detect_humans(frame):
    # Initialize HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detect humans in the frame
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Draw bounding boxes around humans
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red box for human

    return frame, len(boxes) > 0  # Return frame and stop decision if any human is detected

# 3. Lane Curvature Detection (Turn Detection)
def detect_lane_curvature(frame):
    edges = process_frame(frame)
    height, width = frame.shape[:2]

    # Find lane lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        # Fit a second-degree polynomial to the detected lane points
        x_points = []
        y_points = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                x_points.extend([x1, x2])
                y_points.extend([y1, y2])

        # Fit a second-degree polynomial to the lane points
        poly_coeffs = np.polyfit(x_points, y_points, 2)

        # Calculate the direction of curvature
        curvature_direction = "Straight"
        if poly_coeffs[0] > 0:
            curvature_direction = "Right Turn"
        elif poly_coeffs[0] < 0:
            curvature_direction = "Left Turn"

        # Draw the fitted polynomial on the image if the curvature is detected
        poly_y = np.linspace(0, height - 1, height)
        poly_x = poly_coeffs[0] * poly_y ** 2 + poly_coeffs[1] * poly_y + poly_coeffs[2]
        for i in range(len(poly_y) - 1):
            cv2.line(frame, (int(poly_x[i]), int(poly_y[i])), (int(poly_x[i + 1]), int(poly_y[i + 1])), (0, 255, 255), 3)

        return frame, curvature_direction

    return frame, "Straight"

# 4. Main Function to Process Video
def process_video(video_path, output_path, frame_skip=10):
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Skip frames to speed up processing
        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue  # Skip processing for this frame

        # Keep original frame for side-by-side display
        original_frame = frame.copy()

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

        # Show the processed video in a separate window
        cv2.imshow("Processed Video", frame)

        # Write the processed frame to the output video file
        writer.write(frame)

        # Real-time video playback: wait for the time corresponding to the FPS
        if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:  # 27 is the ASCII value for Escape
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

# Call the function with the uploaded video
input_video_path = "C:/Users/dell/Downloads/Input Files/PXL_20250325_044603023.TS.mp4"
output_video_path = "C:/Users/dell/Downloads/Output Files/Processed.mp4"
process_video(input_video_path, output_video_path)
