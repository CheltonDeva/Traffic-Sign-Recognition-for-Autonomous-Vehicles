from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score

model = load_model('models/traffic_classifier.h5')
X_test = np.load('X_test.npy')
y_test = np.argmax(np.load('y_test.npy'), axis=1)
pred_probs = model.predict(X_test)
pred = np.argmax(pred_probs, axis=1)
print(f"Test Accuracy: {accuracy_score(y_test, pred)}")
