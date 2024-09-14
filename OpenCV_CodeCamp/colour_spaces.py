import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('OpenCV_CodeCamp/hanoi.png')
cv.imshow('Hanoi', img)

#BGR to Grayscale:
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#BGR to HSV:
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

#HSV to BGR:
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV2BGR', hsv_bgr)

#BGR to L*a*b:
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# plt.imshow(img[])
# plt.show()

#BGR to RGB:
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

plt.imshow(rgb) #original color
plt.show()



#cv.waitKey(0)
#Cannot convert gray 2 hsv directly => thong qua BGR