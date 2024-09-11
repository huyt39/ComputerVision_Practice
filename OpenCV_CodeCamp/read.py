import cv2 as cv

img = cv.imread('OpenCV_CodeCamp/Image1.png')
cv.imshow('dog', img)
cv.waitKey(0)