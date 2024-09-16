import os
import cv2 as cv
import numpy as np

people = ['Neymar', 'Jackie', 'Leonardo', 'Emma']

p = []

# for i in os.listdir(r'/Users/macbook/Documents/Faces_train2'):
#     p.append(i)

# print(p)
DIR = r'/Users/macbook/Documents/Faces_train2'

haar_cascade = cv.CascadeClassifier('OpenCV_CodeCamp/haar_face.xml')

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

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done---------')
features = np.array(features, dtype = 'object')
labels = np.array(labels)
# print(f'Length of the features = {len(features)}')
# print(f'Length of the labels = {len(labels)}')

faces_recognizer = cv.face.LBPHFaceRecognizer_create()

#Train the Recognizer on the features list and the labels list:
faces_recognizer.train(features, labels)

#save the model to use again:
faces_recognizer.save('faces_trained2.yml')

np.save('features2.npy', features)
np.save('labels2.npy', labels)
