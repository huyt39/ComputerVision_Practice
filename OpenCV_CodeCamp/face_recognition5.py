import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('OpenCV_CodeCamp/haar_face.xml')
people = ["Emma", "Messi", "Weeknd"]

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('faces_train5.yml')

img = cv.imread(r'/Users/macbook/Documents/Faces_val5/Weeknd/133891836d7f11fe4a7d63625682820a.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

#Detect the face in the img:
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 5)
for(x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x: x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    cv.imshow('Detected Face', img)
    
    cv.waitKey(0)