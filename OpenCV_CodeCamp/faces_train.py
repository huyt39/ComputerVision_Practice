import os
import cv2 as cv
import numpy as np

people = ['Messi', 'Aguero', 'Di Maria']

p = []
# for i in os.listdir(r'/Users/macbook/Documents/Faces_train'):
#     p.append(i)

# print(p)

DIR = r'/Users/macbook/Documents/Faces_train'

haar_cascade = cv.CascadeClassifier('OpenCV_CodeCamp/haar_face.xml')

features = [] #images array of faces
labels = [] #faces


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            
            # Chỉ xử lý các tệp là hình ảnh (jpg, png, v.v.)
            if img.endswith('.jpg') or img.endswith('.png'):
                img_array = cv.imread(img_path)
                
                # Kiểm tra nếu img_array không rỗng
                if img_array is None:
                    print(f"Không thể đọc được ảnh: {img_path}")
                    continue
                
                gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

                faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

                for (x, y, w, h) in faces_rect:
                    faces_roi = gray[y:y+h, x:x+w]
                    features.append(faces_roi)
                    labels.append(label)

create_train()
print('Training done ----------------')

features = np.array(features, dtype = 'object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#Train the recognizer on the features list and labels list:
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

