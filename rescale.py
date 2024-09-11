import cv2 as cv

img = cv.imread('OpenCV_CodeCamp/Image1.png')
cv.imshow('dog', img)

def rescaleFrame(frame, scale = 0.75):
    #image, video and live video:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    #live video:
    capture.set(3, width)
    capture.set(4, height)

resized_img = rescaleFrame(img)
cv.imshow('dog2', resized_img)



capture = cv.VideoCapture('OpenCV_CodeCamp/自己紹介 (1).mp4')

while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame, scale = .2)

    cv.imshow('Video', frame)
    cv.imshow('Video_resized', frame_resized)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break


capture.release()
cv.destroyAllWindows()


