import cv2
import numpy as np
import os

def train_face_recognizer():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face_samples = []
    ids = []

    image_folder = 'D:/Study/DIP CEP/face security'
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    for image_path in image_paths:
        img = cv2.imread(image_path, 0)

        try:
            id = int(os.path.split(image_path)[-1].split(".")[1])
        except (IndexError, ValueError):
            print(f"Skipping file: {image_path} - Invalid format")
            continue

        faces = face_cascade.detectMultiScale(img)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y+h, x:x+w])
            ids.append(id)

    face_recognizer.train(face_samples, np.array(ids))
    face_recognizer.write('face_recognizer.yml')

if __name__ == "__main__":
    train_face_recognizer()
