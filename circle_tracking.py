import cv2 as cv
import numpy as np

distanceBWPoints = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2


videoCapture = cv.VideoCapture(0)
previousCircle = None

while True:
    retval, frame = videoCapture.read()
    if not retval:
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17, 17), 0)
    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 0.5, 100, param1=100, param2=30, minRadius=75, maxRadius=400)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        currentCircle = None
        for circle in circles[0, :]:
            if currentCircle is None:
                currentCircle = circle
            if previousCircle is not None:
                if distanceBWPoints(currentCircle[0], currentCircle[1], previousCircle[0], previousCircle[1]) <= distanceBWPoints(circle[0], circle[1], previousCircle[0], previousCircle[1]):
                    currentCircle = circle

        cv.circle(frame, (currentCircle[0], currentCircle[1]), 1, (0, 100, 100), 3)
        cv.circle(frame, (currentCircle[0], currentCircle[1]), currentCircle[2], (255, 255, 0), 3)
        cv.putText(frame, str((currentCircle[0], currentCircle[1])), (currentCircle[0], currentCircle[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)
        previousCircle = currentCircle

    cv.imshow('Circle Tracking', frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

videoCapture.release()
cv.destroyAllWindows()
