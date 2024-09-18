import os
import numpy as np
import cv2 as cv

people = ['Weeknd', 'Emma', 'Messi']

# p = []

# for i in os.listdir(r'/Users/macbook/Documents/Faces_train5'):
#     p.append(i)

# print(p)

DIR = r'/Users/macbook/Documents/Faces_train5'
haar_cascade = cv.CascadeClassifier('OpenCV_Codecamp/haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors= 5)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done---------')
features = np.array(features, dtype = 'object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)
face_recognizer.save('faces_train5.yml')

np.save("features5.npy", features)
np.save("labels5.npy", labels)


