import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import winsound  


detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded.")


cap = cv2.VideoCapture(0) 



def detect_bird_in_frame(frame):
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
    winsound.PlaySound("alarm.wav",winsound.SND_FILENAME) 


try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bird_detected = detect_bird_in_frame(frame)

        
        if bird_detected:
            play_alarm()  

        cv2.imshow("Bird Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
