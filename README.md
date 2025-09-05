# DIP Project — Lane & Obstacle Detection with Turn Decision (Raspberry Pi Streaming)

A Digital Image Processing project in **Python + OpenCV** that:
- detects **road lanes**,
- detects **humans/obstacles** with a HOG-based people detector,
- infers a **turn decision** (Left / Right / Straight) from lane geometry,
- runs either on a **local video** or in **real time on a Raspberry Pi** that streams frames to a laptop client.

---

## ✨ Features

- **Lane detection**  
  ROI selection → Gaussian blur → Canny edges → Hough Lines (probabilistic) → lane overlay.

- **Turn decision**  
  Collect lane points, fit a quadratic `y = ax² + bx + c`;  
  `a < 0 → Left`, `a ≈ 0 → Straight`, `a > 0 → Right`.

- **Human/obstacle detection**  
  OpenCV HOG descriptor + SVM people detector.  
  Displays **“Stop”** if a person is detected, otherwise **“Move”**.

- **Two run modes**
  1. **Local video processing**: read a video, annotate each frame, save an output video.
  2. **Raspberry Pi streaming**: Pi captures & processes frames (Picamera2) and streams them via TCP to a laptop client for display.

---

## 🧠 Pipeline Details

1. **Preprocessing & ROI**
   - Define a polygonal ROI covering the drivable area to suppress irrelevant edges.
   - Apply Gaussian blur to reduce noise.

2. **Edge & Line Extraction**
   - Canny edge detection with tuned thresholds.
   - `cv2.HoughLinesP` to obtain line segments; filter/merge and draw lane lines.

3. **Turn Estimation**
   - Accumulate lane points; polynomial fit (`numpy.polyfit`) of order 2.
   - Interpret the **leading coefficient** `a` for curvature-based direction:
     - `a < 0` → **Left**
     - `|a| ≈ 0` → **Straight**
     - `a > 0` → **Right**

4. **People Detection**
   - Initialize `cv2.HOGDescriptor_getDefaultPeopleDetector()`.
   - Run multi-scale detection and draw bounding boxes.
   - Overlay **Stop/Move** status based on detections.

5. **Visualization & Output**
   - Draw lanes, decision text, and detection boxes on each frame.
   - Show live preview windows; optionally write MP4 output (local mode).
   - In streaming mode, serialize frames (`pickle`) with length-prefix and send over TCP.

---

## 🚀 Run Modes

### Local Video
- Point the script to an input video and an output path.
- Optional: process every *n*-th frame to speed up (frame skipping).

### Raspberry Pi Streaming
- **Server (Pi):** capture frames with Picamera2, run the same vision pipeline, and send frames over TCP (default port `8485`).
- **Client (Laptop):** connect to the Pi’s IP, receive and display the processed stream in real time.

> Ensure Pi and laptop are on the same network and the chosen port is open.

---

## 🔧 Tuning Tips

- **ROI & Canny thresholds** strongly affect lane stability—tune to your camera height/FOV and lighting.
- For **HOG people detector** performance, consider resizing frames smaller, increasing `winStride`, or lowering `scale`.
- The polynomial-fit **turn heuristic** is simple; for robustness, add temporal smoothing, RANSAC, or lane-model fitting in meters.

