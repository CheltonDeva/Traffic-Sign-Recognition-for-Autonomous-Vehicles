import os
import numpy as np
from PIL import Image
from keras.utils import to_categorical

image_extensions = ('.png', '.jpg', '.jpeg', '.ppm', '.bmp')

X_test = []
y_test = []

for class_id in range(43):
    path = f"data/Test/{class_id}"
    if not os.path.exists(path):
        continue
    for img in os.listdir(path):
        if img.lower().endswith(image_extensions):
            try:
                image = Image.open(os.path.join(path, img))
                image = image.resize((30, 30))
                X_test.append(np.array(image))
                y_test.append(class_id)
            except Exception as e:
                print(f"Error loading image {img}: {e}")

X_test = np.array(X_test)
y_test = np.array(y_test)

print("Loaded test images:", X_test.shape)

# Normalize and one-hot encode
X_test = X_test / 255.0
y_test = to_categorical(y_test, 43)

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
