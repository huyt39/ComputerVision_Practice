import cv2 as cv

img = cv.imread('OpenCV_CodeCamp/hanoi.png')
cv.imshow('Hanoi', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#Simple thresholding:
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
#threshold = 150
#thresh = threshold img of the binary img

cv.imshow('Simple threshold', thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Simple threshold inverse', thresh_inv)

#Adaptive thresholding: compute the optimal threshold value on the basis of the mean
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3) #max = 255
#kernel size = 11x11 => tinh nguong cho tung pixel
#c = 3 => gia tri duoc tru di tu nguong cuc bo (tinh trung vi roi tru 3) de dieu chinh do nhay cua nguong hoa
cv.imshow('Adaptive thresholding', adaptive_thresh)


cv.waitKey(0)