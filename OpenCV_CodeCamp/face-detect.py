import cv2 as cv

img = cv.imread('OpenCV_CodeCamp/dimaria.png')
cv.imshow('Di Maria', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Di Maria', gray)

haar_cascade = cv.CascadeClassifier('OpenCV_CodeCamp/haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
#moi lan thuat toan giam kich thuoc hinh anh xuong 10%
#minNeighbors=3: Đây là số lượng khu vực lân cận cần được xác định là một khuôn mặt để thuật toán chấp nhận kết quả. 
# Giá trị này càng cao, kết quả càng chính xác nhưng đồng thời cũng có thể bỏ sót một số khuôn mặt.

print(f'Number of faces found = {len(faces_rect)}')

for(x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)

#Popular but not the most advanced