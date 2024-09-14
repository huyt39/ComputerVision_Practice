import cv2 as cv
import numpy as np 

img = cv.imread('OpenCV_CodeCamp/hanoi.png')
cv.imshow('Hanoi', img)

b, g, r = cv.split(img)
cv.imshow('Blue', b)  #blue almost white
cv.imshow('Green', g) #green almost white
cv.imshow('Red', r) #red almost white
#All gray

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b, g, r])
cv.imshow('Merged image', merged)

blank = np.zeros(img.shape[:2], dtype = 'uint8')
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue', blue)  
cv.imshow('Green', green) 
cv.imshow('Red', red) 


cv.waitKey(0)