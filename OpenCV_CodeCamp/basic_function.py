import cv2 as cv

img = cv.imread('OpenCV_CodeCamp/Image1.png')
cv.imshow('dog', img)

#Converting to grayscale:
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#Blur = remove noise:
blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

#Edge Cascade:
canny = cv.Canny(blur, 50, 100)
cv.imshow('Canny', canny)

#Dilating the image:
dilated = cv.dilate(canny, (11, 11), iterations=5)
cv.imshow('Dilated', dilated)

#Eroding:
eroded = cv.erode(dilated, (11, 11), iterations=5) #back to canny?
cv.imshow('Eroded', eroded)

#Resize:
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC) #interpolation=>better quality
cv.imshow('Resized', resized)

#Cropping:
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)


cv.waitKey(0)