#The goal of this script is to find a matrix that represents the world coordinates in the camera frame.
#I want to define the world by choosing 3 points on the checkerboard.
#From those points, I can get an x, y, and z vector and define a coordinate frame.
#It should then be possible to use this frame to transform the detected aruco 
#movements from camera frame to world frame, which should align them with the commands
#given to the Stewart platform (if the platform is aligned with these axes).

import os
import pickle
import cv2
import numpy as np

def camera_robot_calibration(chessboard_img, camera_matrix_left, distortion_coefficients_left):
    COLS = 8
    ROWS = 6
    SQUARE_SIZE = 29 #mm

    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    axis = np.float32([[SQUARE_SIZE,0,0], [0,-SQUARE_SIZE,0], [0,0,-SQUARE_SIZE]])
    origin = np.float32([[0,0,0]])  # Chessboard center in object coordinates

    # Center offset so middle of board is origin
    #center_x = (COLS - 1) * SQUARE_SIZE / 2.0
    #center_y = (ROWS - 1) * SQUARE_SIZE / 2.0
    center_x = 4 * SQUARE_SIZE
    center_y = 2 * SQUARE_SIZE

    object_points = []
    for i in range(ROWS):
        for j in range(COLS):
            x = j * SQUARE_SIZE - center_x
            y = i * SQUARE_SIZE - center_y
            z = 0.0
            object_points.append([x, y, z])
            print(f"Point {i * COLS + j}: ({x:.3f}, {y:.3f}, {z:.3f})")

    object_points = np.array(object_points, dtype=np.float32)
    grey = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(grey, (COLS, ROWS), None)

    #If corners were found
    if ret:
        cornersRefined = cv2.cornerSubPix(grey, corners, (11,11), (-1,-1), term_criteria)
        pnp_ret, rvecs, tvecs = cv2.solvePnP(object_points, cornersRefined, camera_matrix_left, distortion_coefficients_left)

        imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix_left, distortion_coefficients_left)
        origin_imgpts, _ = cv2.projectPoints(origin, rvecs, tvecs, camera_matrix_left, distortion_coefficients_left)
        chessboard_img = drawAxes(chessboard_img, corners, origin_imgpts, imgpts)

        imgpts, _ = cv2.projectPoints(object_points, rvecs, tvecs, camera_matrix_left, distortion_coefficients_left)
        for pt in imgpts:
            cv2.circle(chessboard_img , tuple(pt.ravel().astype(int)), 2, (0,0,255),-1)
        cv2.imshow('Chessboard', chessboard_img)
        cv2.waitKey(0)




def drawAxes(img, corners, origin_imgpts, imgpts):
    def tupleOfInts(arr):
        return tuple(int(x) for x in arr)
    corner = tupleOfInts(corners[0].ravel())
    origin = tuple(int(x) for x in origin_imgpts[0].ravel())
    print("origin: ", origin)
    print("axesx end point: ", tupleOfInts(imgpts[0].ravel()))
    img = cv2.line(img, origin, tupleOfInts(imgpts[0].ravel()), (255,0,0),5)
    img = cv2.line(img, origin, tupleOfInts(imgpts[1].ravel()), (0,255,0),5)
    img = cv2.line(img, origin, tupleOfInts(imgpts[2].ravel()), (0,0,255),5)
    return img

def read_params(filePath):
    with open(filePath, 'rb') as f:
        data = pickle.load(f)
    cameraMatrix1 = data['cameraMatrix']
    distCoeffs1 = data['distortionCoeff']
    rotation_vecs = data['rotationVectors']
    translation_vecs = data['translationVectors']
    reprojection_err = data['reprojectionError']
    return cameraMatrix1, distCoeffs1, rotation_vecs, translation_vecs
    
def main():
    #Read in calibration parameters for the cameras
    #Need to go from camera frame -> robot base frame/world frame
    PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'catkin_ws/src/StewartIBVS/calibration_params_home_webcam/0/calibration_data.pkl'))
    cameraMat1, distCoeffs1, R, T = read_params(PARAM_DIR)
    img = cv2.imread("29_mm_chess.jpg")
    print("cameraMatrix1: ")
    print(cameraMat1)
    print("distortionCoeffs1:")
    print(distCoeffs1)
    #OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Triangulation'))
    T = camera_robot_calibration(img, cameraMat1, distCoeffs1)
    #test_with_aruco(4, cameraMat1, distCoeffs1, T, img=img)

if __name__ == "__main__":
    main()