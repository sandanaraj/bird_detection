import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from flask import Flask, render_template, Response, jsonify
import threading
import winsound


detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded.")


app = Flask(__name__)


cap = cv2.VideoCapture(0)


lock = threading.Lock()


bird_detected = False

def detect_bird_in_frame(frame):
    """Detect bird in a single frame"""
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
        if class_id == 16 and scores[i] > 0.5:  
            bird_detected = True
            box = boxes[i]
            draw_bounding_box(frame, box)
            print(f"Bird detected with confidence {scores[i]:.2f}")
            break
    
    return bird_detected

def draw_bounding_box(frame, box):
    h, w, _ = frame.shape
    y_min, x_min, y_max, x_max = box
    x_min, x_max, y_min, y_max = int(x_min * w), int(x_max * w), int(y_min * h), int(y_max * h)
    

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, "Bird", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def play_alarm():
   
    winsound.PlaySound("alarm.wav", winsound.SND_FILENAME)

def generate_video():
    
    global bird_detected
    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        with lock:
            bird_detected = detect_bird_in_frame(frame)

        
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()

        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bird_status')
def bird_status():
    global bird_detected
    if bird_detected:
        play_alarm()  
    return jsonify({'bird_detected': bird_detected})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
