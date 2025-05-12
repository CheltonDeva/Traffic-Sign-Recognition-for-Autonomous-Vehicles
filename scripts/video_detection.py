import cv2
import torch
from keras.models import load_model
import numpy as np
from PIL import Image
import pyttsx3
import csv
import time

# Load YOLOv5 detection model and CNN classifier
yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')
classifier = load_model('models/traffic_classifier.h5')

# Define class names used by your classifier
class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 
    'No passing', 'No passing for vehicles over 3.5 metric tons', 
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 
    'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 
    'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 
    'Keep left', 'Roundabout mandatory', 'End of no passing', 
    'End of no passing by vehicles over 3.5 metric tons'
]

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
spoken_labels = set()

# Confidence threshold
confidence_threshold = 0.5

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = yolo(frame)

    # Extract predictions
    pred = results.pandas().xyxy[0].values  # [[x1, y1, x2, y2, conf, class_id, class_name], ...]

    if pred is not None and len(pred) > 0:
        for box in pred:
            if len(box) < 6:
                continue

            x1, y1, x2, y2, conf, cls = box[:6]

            # Apply confidence threshold
            if conf < confidence_threshold:
                continue  # Skip if confidence is too low

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # Crop detection
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            # Preprocess for CNN
            img = Image.fromarray(crop).resize((30, 30))
            img = np.array(img).astype('float32') / 255.0
            img = img.reshape(1, 30, 30, 3)

            # Predict traffic sign class
            pred_class = np.argmax(classifier.predict(img), axis=1)[0]
            label = class_names[pred_class] if pred_class < len(class_names) else f'Class {pred_class}'

            # Speak label (once)
            if label not in spoken_labels:
                engine.say(label)
                engine.runAndWait()
                spoken_labels.add(label)

            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)

    # Show video
    cv2.imshow('Traffic Sign Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
