import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Supported image file extensions
image_extensions = ('.png', '.jpg', '.jpeg', '.ppm', '.bmp')

data = []
labels = []

for class_id in range(43):
    path = f"data/Train/{class_id}"
    for img in os.listdir(path):
        if img.lower().endswith(image_extensions):
            try:
                image = Image.open(os.path.join(path, img))
                image = image.resize((30, 30))
                data.append(np.array(image))
                labels.append(class_id)
            except Exception as e:
                print(f"Error loading image {img}: {e}")

data = np.array(data)
labels = np.array(labels)

# Normalize and one-hot encode
data = data / 255.0
labels = to_categorical(labels, 43)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Save datasets
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
