import cv2 as cv

img = cv.imread('OpenCV_CodeCamp/messi.png')
cv.imshow('Messi', img)

#Averaging:
average = cv.blur(img, (11, 11))
cv.imshow('Average blur', average)

#Gaussian blur: #less blurring, more natural than average
gauss = cv.GaussianBlur(img, (7, 7), 0)
cv.imshow('Gaussian', gauss)

#Median blur = average, better in reducing noise than average
median = cv.medianBlur(img, 7)
cv.imshow('Median blur', median)

#Bilateral blurring => most effective
bilateral = cv.bilateralFilter(img, 7, 15, 15)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)