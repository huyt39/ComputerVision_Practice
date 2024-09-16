import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('OpenCV_CodeCamp/haar_face.yml')
people = ['Neymar', 'Jackie', 'Leonardo', 'Emma ']

# features = np.load('features2.npy', allow_pickle = True)
# labels = np.load('labels2.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('faces_trained2.yml')

img = cv.imread(r'/Users/macbook/Documents/Faces_val2/Jackie/Jackie-Chan-Birthday-040724-01-418f1dc2cc374649a9f56dd2867f32ed.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Person', gray)

#Detect the face in the img:
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
for(x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x: x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    cv.imshow('Detected Face', img)
    
    cv.waitKey(0)