import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (9, 9), 2)

    #Using the hough transform
    '''Params input image, method, 
    inverse ratio of accumulator array resolution to image resolution (1 means they have the same resolution, 2 means accumulator will have half the width and height),
    min distance between centers of detected circles, 
    upper threshold passed to Canny,
    threshold for circle detection positive (smaller should return more circles, but more false positives),
    min radius, max radius
    '''
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 40, param1 = 70, param2 = 40, minRadius = 5, maxRadius=80)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

    cv2.imshow('Live Circle Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()