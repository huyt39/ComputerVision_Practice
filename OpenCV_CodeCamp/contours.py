import cv2 as cv
import numpy as np


img = cv.imread('OpenCV_CodeCamp/Image1.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blank = np.zeros(img.shape, dtype = 'uint8')
cv.imshow('Blank', blank)

# blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)

# canny = cv.Canny(img, 25, 75)
# canny = cv.Canny(blur, 25, 75)
# cv.imshow('Canny', canny)

ret, thresh = cv.threshold(gray, 125, 175, cv.THRESH_BINARY) #pixel under 125 -> set to zero or black, above 125 -> set to white or 175
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cv.imshow('Thresh', thresh)
#contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#vi du co 1 line => CHAIN_APPROX_NONE se tra ve tat ca cac diem thuoc line do
#CHAIN_APPROX_SIMPLE se tra ve diem dau va cuoi
print(f'{len(contours)} contour(s) found!')


#Draw contours on blank:
cv.drawContours(blank, contours, -1, (0, 0, 255), 1 ) #-1: all, 3rd: contours index
cv.imshow('Contours drawing', blank)
cv.waitKey(0)