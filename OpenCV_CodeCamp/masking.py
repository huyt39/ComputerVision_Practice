import cv2 as cv
import numpy as np

img = cv.imread('OpenCV_CodeCamp/Image1.png')
cv.imshow('Dog', img)

blank = np.zeros(img.shape[:2], dtype = 'uint8') #must be the same as ori 
cv.imshow('Blank', blank)

mask = cv.circle(blank, (img.shape[1]//2 + 45, img.shape[0]//2 - 45 ), 100, 255, -1) #can also use rectangle
cv.imshow('Mask', mask)

masked = cv.bitwise_and(img, img, mask = mask)
cv.imshow('Masked image', masked)

circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45, img.shape[0]//2 - 45 ), 100, 255, -1)
rectangle = cv.rectangle(blank.copy(), (img.shape[1]//2, img.shape[0]//2 ), (150, 350), 255, -1)
# weird_shape = cv.bitwise_and(circle, rectangle)
# cv.imshow('Weird shape', weird_shape)

cv.waitKey(0)