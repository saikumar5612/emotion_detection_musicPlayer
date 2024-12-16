"""
import os
import cv2
import numpy as np
from pandas.io.common import file_path_to_url
from tensorflow.keras.models import load_model
from music_player import MusicPlayer
import threading
import tkinter as tk

# Load the trained model
model = load_model('C:\\Users\\atchi\\PycharmProjects\\emotion_detection_player\\emotion_cnn_model.h5')
file_path = "C:\\Users\\atchi\\PycharmProjects\\emotion_detection_player\\emotion_cnn_model.h5"
print("model saved successfully")
if not os.path.exists(file_path):
    print("File not found:", file_path)
else:
    print("File found:", file_path)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# Function to detect emotion from real-time video
def detect_emotion():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
            face = face / 255.0
            predictions = model.predict(face)
            emotion = emotion_labels[np.argmax(predictions)]
            print("Detected Emotion:", emotion)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to integrate music and detection
def start_app():
    root = tk.Tk()
    music_player = MusicPlayer(root)

    threading.Thread(target=detect_emotion, daemon=True).start()
    root.mainloop()


if __name__ == '__main__':
    start_app()
"""
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from music_player import MusicPlayer
import threading
import tkinter as tk

# Load the trained model
model = load_model('C:\\Users\\atchi\\PycharmProjects\\emotion_detection_player\\emotion_cnn_model.h5')
file_path = "C:\\Users\\atchi\\PycharmProjects\\emotion_detection_player\\emotion_cnn_model.h5"
print("Model loaded successfully")
if not os.path.exists(file_path):
    print("File not found:", file_path)
else:
    print("File found:", file_path)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Set the iteration limit for emotion detection
max_iterations = 25
iterations = 0

# Function to detect emotion from real-time video
def detect_emotion():
    global iterations
    cap = cv2.VideoCapture(0)

    while iterations < max_iterations:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=25)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
            face = face / 255.0
            predictions = model.predict(face)
            emotion = emotion_labels[np.argmax(predictions)]
            print("Detected Emotion:", emotion)

        cv2.imshow("Emotion Detection", frame)

        # Increment the iteration count after each frame
        iterations += 1

        # Wait for the 'q' key to stop early if desired
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to integrate music and detection with Tkinter GUI
def start_app():
    root = tk.Tk()
    music_player = MusicPlayer(root)

    # Start the emotion detection in a separate thread
    threading.Thread(target=detect_emotion, daemon=True).start()
    root.mainloop()

if __name__ == '__main__':
    start_app()

