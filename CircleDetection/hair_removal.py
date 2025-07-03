import numpy as np
import cv2
from PIL import Image

# Read the image and perfrom an OTSU threshold
img = cv2.imread('skinAngle.jpg')
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
height = img.shape[0]
width = img.shape[1]
print("Image size", width, " x ", height)

kernel = np.ones((10,10),np.uint8)

# Perform closing to remove hair and blur the image
closing = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel, iterations = 2)
blur = cv2.GaussianBlur(closing, (9,9), 2)

cv2.imshow("Closing img", closing)
cv2.imshow("Blur img", blur)

cv2.waitKey(0)