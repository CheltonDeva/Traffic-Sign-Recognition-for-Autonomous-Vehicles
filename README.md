# Traffic-Sign-Recognition-for-Autonomous-Vehicles

This repository contains a complete pipeline for Traffic Sign Recognition, designed for autonomous vehicle applications. It includes scripts for data preprocessing, model training, evaluation, and real-time or image-based sign recognition using both command-line and GUI interfaces.

├── data_preprocessing.py      # Prepares and saves training/validation datasets
├── model_training.py          # Trains CNN classifier and saves model
├── test.py                    # Prepares and saves test dataset
├── evaluation.py              # Evaluates model accuracy on test set
├── video_detection.py         # Real-time webcam detection (YOLOv5 + CNN)
├── gui_tkinter.py             # Tkinter GUI for image/live detection
├── models/
│   ├── best.pt                # YOLOv5 custom weights
│   └── traffic_classifier.h5  # Trained CNN classifier
├── data/                      # Dataset folders (Train/Test)
│   ├── Train/
│   └── Test/
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation

Models
YOLOv5 Detector: Custom-trained for bounding box detection of traffic signs.
CNN Classifier: Classifies cropped sign images into 43 categories.

Requirements
Python 3.7+
numpy, pandas, scikit-learn, pillow
keras, tensorflow
torch, opencv-python
pyttsx3, tkinter
