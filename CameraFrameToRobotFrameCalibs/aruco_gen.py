import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
marker_id = 9
image_size = 10

marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, image_size)

cv2.imwrite("aruco_id0.jpg", marker)