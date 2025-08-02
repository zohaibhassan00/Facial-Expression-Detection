import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

# Load the emotion detection model
with open("client_model/emotion_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("client_model/emotion_model.h5")
emotion_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load Haar cascade face detector
face_classifier = cv2.CascadeClassifier("client_model/haarcascade_frontalface_default.xml")

# Emotion classes
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_expression_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print("Faces detected:", len(faces))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = emotion_model.predict(roi, verbose=0)[0]
        label = class_labels[np.argmax(prediction)]
        return label

    return "No face detected"

def predict_expression_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    label = "No face detected"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi, verbose=0)[0]
            label = class_labels[np.argmax(prediction)]
            break
        break

    cap.release()
    return label
