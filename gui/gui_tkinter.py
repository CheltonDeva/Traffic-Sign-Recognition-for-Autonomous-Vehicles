import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
from keras.models import load_model
import threading
import pyttsx3  

# Load the model
model = load_model('models/traffic_classifier.h5')

# Class labels
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield', 14: 'Stop',
    15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve to the left', 20: 'Dangerous curve to the right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 29: 'Bicycles crossing',
    30: 'Beware of ice/snow', 31: 'Wild animals crossing', 32: 'End of all speed and passing limits',
    33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Initialize speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def classify(file_path):
    img = Image.open(file_path).resize((30, 30))
    img = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = np.argmax(model.predict(img), axis=1)[0]
    result_text = f"Prediction: {classes[pred]}"
    result_label.config(text=result_text)
    speak(result_text)

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_panel.config(image=img)
        image_panel.image = img
        classify(file_path)

def live_detection():
    def detect():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.resize(frame, (30, 30))
            img_input = np.expand_dims(img / 255.0, axis=0)
            pred = np.argmax(model.predict(img_input), axis=1)[0]
            result_text = classes[pred]
            cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Live Detection", frame)
            speak(result_text)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    thread = threading.Thread(target=detect)
    thread.start()

# GUI Setup
root = tk.Tk()
root.title("Traffic Sign Recognition System")
root.geometry("600x500")

# Load and set background image
bg_image = Image.open(r"C:\Users\Chelt\OneDrive\Desktop\Final Year Project\hq720.jpg")
bg_image = bg_image.resize((1500, 1000), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# GUI Widgets
title = tk.Label(root, text="Traffic Sign Recognition System", font=("Helvetica", 18, "bold"), bg="#ffffff")
title.pack(pady=10)

upload_frame = tk.Frame(root, bg="#ffffff")
upload_frame.pack(pady=10)

btn_upload = tk.Button(upload_frame, text="Upload Image", width=20, command=upload_image)
btn_upload.pack(side=tk.LEFT, padx=10)

btn_live = tk.Button(upload_frame, text="Live Camera Detection", width=20, command=live_detection)
btn_live.pack(side=tk.LEFT, padx=10)

image_panel = tk.Label(root, bg="#ffffff")
image_panel.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#ffffff")
result_label.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", command=root.quit, bg="red", fg="white", width=15)
exit_btn.pack(pady=10)

# Raise widgets above background
title.tkraise()
upload_frame.tkraise()
image_panel.tkraise()
result_label.tkraise()
exit_btn.tkraise()

root.mainloop()
