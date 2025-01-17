import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from flask import Flask, render_template, Response, jsonify
import matplotlib.pyplot as plt
import io
import threading
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Load TensorFlow model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded.")

app = Flask(__name__)

# Use external video source or a file instead of a local webcam
# Replace with a live IP camera URL or a video file path
cap = cv2.VideoCapture("your_video_file.mp4")  # Replace with your video file or IP camera URL

# Global variables
lock = threading.Lock()
bird_detected = False
frame_count = 0
confidence_scores = []
frame_numbers = []

# Detect bird in a single frame
def detect_bird_in_frame(frame, frame_number):
    global bird_detected
    resized_frame = cv2.resize(frame, (320, 320))
    resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor([resized_frame_rgb], dtype=tf.uint8)

    detections = detector(input_tensor)

    class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()
    boxes = detections['detection_boxes'][0].numpy()

    bird_detected = False
    for i, class_id in enumerate(class_ids):
        if class_id == 16 and scores[i] > 0.5:  # Class ID 16 is assumed to be "bird"
            bird_detected = True
            box = boxes[i]
            draw_bounding_box(frame, box)
            log_confidence(frame_number, scores[i])  # Log confidence
            print(f"Bird detected with confidence {scores[i]:.2f}")
            break
    
    if not bird_detected:
        log_confidence(frame_number, 0)  # Log zero confidence for no detection

    return bird_detected

# Draw bounding box for detected bird
def draw_bounding_box(frame, box):
    h, w, _ = frame.shape
    y_min, x_min, y_max, x_max = box
    x_min, x_max, y_min, y_max = int(x_min * w), int(x_max * w), int(y_min * h), int(y_max * h)
    
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, "Bird", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Log confidence scores for the graph
def log_confidence(frame_number, confidence):
    global confidence_scores, frame_numbers
    confidence_scores.append(confidence)
    frame_numbers.append(frame_number)
    if len(confidence_scores) > 100:  # Limit to last 100 frames
        confidence_scores.pop(0)
        frame_numbers.pop(0)

# Generate the video feed
def generate_video():
    global bird_detected, frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # Increment frame number
        with lock:
            bird_detected = detect_bird_in_frame(frame, frame_count)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Generate the confidence graph
@app.route('/graph')
def graph():
    global frame_numbers, confidence_scores
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(frame_numbers, confidence_scores, label="Confidence Score")
    axis.set_title("Bird Detection Confidence Over Time")
    axis.set_xlabel("Frame Number")
    axis.set_ylabel("Confidence Score")
    axis.set_ylim([0, 1])
    axis.legend()

    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bird_status')
def bird_status():
    global bird_detected
    return jsonify({'bird_detected': bird_detected})

# Release the camera on exit
import atexit
@atexit.register
def release_camera():
    cap.release()

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
