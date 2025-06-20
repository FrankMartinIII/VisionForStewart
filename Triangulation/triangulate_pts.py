import cv2
import numpy as np
import os
import pickle

def read_params(filePath):
    with open(filePath, 'rb') as f:
        data = pickle.load(f)
    cameraMatrix1 = data['cameraMatrix1']
    distCoeffs1 = data['distortionCoeffs1']
    cameraMatrix2 = data['cameraMatrix2']
    distCoeffs2 = data['distortionCoeffs2']
    R_Mat = data['R_Mat']
    T_Vec = data['T_Vec']
    E = data['Essential_Mat']
    F = data['Fundamental_Mat']
    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_Mat, T_Vec

def get_undistort_rectification_maps(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_Matrix, T_Vec, img_size):
    '''
    Function to compute and return undistortion and rectification maps.
    Use with cv.remap to rectify images

    Params:
    cameraMatrix1 = Intrinsic matrix of left camera
    distCoeffs1 = Distortion coefficients of left camera
    cameraMatrix2 = Intrinsic matrix of right camera
    distCoeffs2 = Distortion coefficients of right camera
    R_Mat = 3x3 Rotation matrix between the 2 cameras
    T_Vecs = 1x3 translation vector between the cameras
    img_size = tuple of (width, height) of images

    Returns
    map1x
    map1y
    map2x
    map2y
    projection_matrix1
    projection_matrix2
    '''

    '''
    Documentation for stereoRectify
    R1	Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix brings points given in the unrectified first camera's coordinate system to points in the rectified first camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system.
    R2	Output 3x3 rectification transform (rotation matrix) for the second camera. 
    P1	Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified first camera's image.
    P2	Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified second camera's image.
    Q	Output 4×4 disparity-to-depth mapping matrix (see reprojectImageTo3D).
    validPixROI1	Optional output rectangles inside the rectified images where all the pixels are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
    validPixROI2	Optional output rectangles inside the rectified images where all the pixels are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller 
    '''
    #Compute rectification transforms, I didn't specify an alpha right now, may not be needed in our case
    R1, R2, P1, P2, Q, roil1, roil2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, img_size, R_Matrix, T_Vec, alpha=1)

    #Last argument is an int for type of the first output map, but idk what that means
    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, img_size, cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, P1, P2

def undistort_images(img_left, img_right, map1x, map1y, map2x, map2y):
    '''
    Apply undistortion and rectification to images
    '''
    rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)
    return rectified_left, rectified_right

def main():
    OUTPUT_DIRECTORY = OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stereo_calibration_params', 'stereo_calibration_data.pkl'))
    #IMAGE_SIZE = (640,480)
    print("Reading params from", OUTPUT_DIRECTORY)
    cameraMat1, distCoeffs1, cameraMat2, distCoeffs2, R, T = read_params(OUTPUT_DIRECTORY)
    print("cameraMatrix1: ")
    print(cameraMat1)
    print("distortionCoeffs1:")
    print(distCoeffs1)
    print("cameraMatrix2: ")
    print(cameraMat2)
    print("distortionCoeffs2:")
    print(distCoeffs2)
    print("Rotation Matrix:")
    print(R)
    print("Translation Vector")
    print(T)

    testL = cv2.imread("testL.jpg")
    testR = cv2.imread("testR.jpg")
    width = testL.shape[1]
    height = testL.shape[0]
    #combining images for display
    combinedImg = np.concatenate((testL, testR), axis=1)
    cv2.imshow('Combined images', combinedImg)
    cv2.waitKey(0)

    map1x, map1y, map2x, map2y, P1, P2 = get_undistort_rectification_maps(cameraMat1, distCoeffs1, cameraMat2, distCoeffs2, R,T, (width, height))
    testLUndist, testRUndist = undistort_images(testL, testR, map1x, map1y, map2x, map2y)
    combinedImgUndist = np.concatenate((testLUndist, testRUndist), axis=1)
    #cv2.imshow('Combined images2', combinedImgUndist)
    cv2.imshow("L undist",testLUndist)
    cv2.imshow("R undist",testRUndist)
    cv2.waitKey(0)

    # Example for drawing epipolar lines on rectified images
    # Assume rectified_img_left and rectified_img_right are your output images

    display_img_left = testLUndist.copy()
    display_img_right = testRUndist.copy()

    # Draw horizontal lines every 50 pixels
    for y in range(0, display_img_left.shape[0], 50):
        cv2.line(display_img_left, (0, y), (display_img_left.shape[1], y), (0, 255, 0), 1) # Green lines
        cv2.line(display_img_right, (0, y), (display_img_right.shape[1], y), (0, 255, 0), 1)

    cv2.imshow('Rectified Left (with lines)', display_img_left)
    cv2.imshow('Rectified Right (with lines)', display_img_right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()