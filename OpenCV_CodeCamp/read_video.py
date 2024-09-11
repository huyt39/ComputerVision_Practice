import cv2 as cv

capture = cv.VideoCapture('OpenCV_CodeCamp/自己紹介 (1).mp4')

while True:
    isTrue, frame = capture.read()

    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
