import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('OpenCV_CodeCamp/messi.png')
cv.imshow('Messi', img)

blank = np.zeros(img.shape[:2], dtype = 'uint8')

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
mask = cv.bitwise_and(img, img, mask = circle)
cv.imshow('Mask', mask)

#Gray scale histogram:
# gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
#0: color channel; gray => 0 (1 channel)
#None => mask = 0 => entire image
#[256] => bins (hist size)
#[0, 256] => pixel range

# plt.figure()
# plt.title('Grayscale histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0, 256]) #range of x
# plt.show()

plt.figure()
plt.title('Colour histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], circle, [256], [0, 256])
    plt.plot(hist, color = col)
    plt.xlim([0, 256])

plt.show()

cv.waitKey(0)