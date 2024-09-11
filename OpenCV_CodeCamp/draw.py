import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype= 'uint8')
cv.imshow('Blank', blank )

#Paint the image a certain colour:
blank[:] = 0, 255, 0
cv.imshow('Green', blank)
blank[200:300, 300:400] = 0, 0, 255
cv.imshow('Red', blank)

#Draw a rectangle:
#cv.rectangle(blank, (0, 0), (250, 250), (255, 0, 0), thickness=2)
#cv.rectangle(blank, (0, 0), (250, 250), (255, 0, 0), thickness=-1) #-1 = cv.FILLED
cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 0, 0), thickness=2)
cv.imshow('Rectangle', blank)

#Draw a circle:
cv.circle(blank, (250, 250), 40, (0, 0, 255), thickness=3)
cv.imshow('Circle', blank)

#Draw a line:
cv.line(blank, (0, 0), (250, 250), (255, 255, 255), thickness=3)
cv.imshow('Line', blank)

#Wrire text:
cv.putText(blank, 'Hello', (225, 225), cv.FONT_HERSHEY_TRIPLEX, 2.0, (0, 255, 255), 2) #2.0: scale
cv.imshow('Text', blank)
cv.waitKey(0)