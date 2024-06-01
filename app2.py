import cv2
import numpy as np
import os
import pygame

# Initialize the pygame mixer
pygame.mixer.init()

# Load the buzzer sound
buzzer_sound = pygame.mixer.Sound('buzzer.wav')

def capture_face_images():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"dataset/user.{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()

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

def security_system():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_recognizer.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            id, confidence = face_recognizer.predict(face_img)

            if confidence < 50:  # Adjust the confidence threshold as needed
                cv2.putText(frame, "Authorized", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                print("Access Granted")  # Simulate buzzer off
            else:
                cv2.putText(frame, "Unauthorized", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                print("Buzzer On! Alert!")  # Simulate buzzer on
                buzzer_sound.play()

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Security System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Uncomment the following lines to capture face images and train the recognizer
    # capture_face_images()
    # train_face_recognizer()
    security_system()
