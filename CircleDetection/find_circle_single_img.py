import cv2
import numpy as np

#Read image
img = cv2.imread('eyes2.jpg', cv2.IMREAD_COLOR)

greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#blur using 3x3 kernel
#blurred = cv2.blur(greyimg, (3,3))
blurred = cv2.GaussianBlur(greyimg, (9,9), 2)

#Using the hough transform
'''Params input image, method, 
inverse ratio of accumulator array resolution to image resolution (1 means they have the same resolution, 2 means accumulator will have half the width and height),
min distance between centers of detected circles, 
upper threshold passed to Canny,
threshold for circle detection positive (smaller should return more circles, but more false positives),
min radius, max radius
'''
detected_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 70, minRadius = 1)

print(f"detected_circles shape: {detected_circles.shape}")
for i, pt in enumerate(detected_circles[0, :]):
    a, b, r = pt
    print(f"Circle {i}: center=({a},{b}), radius={r}")

#Draw circles that are detected
if detected_circles is not None:
    #Convert circle params to ints
    detected_circles = np.uint16(np.around(detected_circles))

    for i, pt in enumerate(detected_circles[0, :]):
        a, b, r = pt
        print(f"Circle {i}: center=({a},{b}), radius={r}")
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        cv2.circle(img, (a,b), r, (0,255,0), 2)

        #Draw center

        cv2.circle(img, (a,b), 1, (0,0,255), 3)
    cv2.imshow("Detected cirle", img)
    cv2.waitKey(0)