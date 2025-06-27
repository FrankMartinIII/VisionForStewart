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
    R1, R2, P1, P2, Q, roil1, roil2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, img_size, R_Matrix, T_Vec, alpha=0)

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



def triangulate_points(points1_2d, points2_2d, P1, P2):
    if points1_2d.shape[1] != 2:
        print("Error: Points do not have 2 dimensions")
        return None
    if points1_2d.shape != points2_2d.shape:
        print("Error: Need the same amount of points in each image")
        return None
    #Points will be returned in first camera's rectified coordinate system
    pointsHomogenous4d = cv2.triangulatePoints(P1, P2, points1_2d.T, points2_2d.T)

    #Convert from homogeneous to euclidean (XYZW to XYZ)
    points_3d = (pointsHomogenous4d / pointsHomogenous4d[3]).T[:, :3]
    return points_3d

def test_Undistortion(cameraMat1, distCoeffs1, cameraMat2, distCoeffs2, R, T):
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

def live_circ_dist_detect(cameraMat1, distCoeffs1, cameraMat2, distCoeffs2, R, T):
    IMAGE_SIZE = (1280, 720)

    map1x, map1y, map2x, map2y, P1, P2 = get_undistort_rectification_maps(
        cameraMat1, distCoeffs1, cameraMat2, distCoeffs2, R, T, IMAGE_SIZE
    )

    capLeft = cv2.VideoCapture(0)
    capRight = cv2.VideoCapture(2)

    capLeft.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    capLeft.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[1])
    capRight.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    capRight.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[1])
    
    while True:
        retL, frameL = capLeft.read()
        retR, frameR = capRight.read()

        if not retL or not retR:
            print("ERROR: Failed to grab frames from cameras, exiting")
            break

        #Undistort and rectify
        rectifiedLeft, rectifiedRight = undistort_images(frameL, frameR, map1x, map1y, map2x, map2y)

        #Convert to greyscale
        greyL = cv2.cvtColor(rectifiedLeft, cv2.COLOR_BGR2GRAY)
        greyR = cv2.cvtColor(rectifiedRight, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blurL = cv2.GaussianBlur(greyL, (9, 9), 2)
        blurR = cv2.GaussianBlur(greyR, (9, 9), 2)

        circles_left = cv2.HoughCircles(blurL, cv2.HOUGH_GRADIENT, 1, 40, param1 = 70, param2 = 30, minRadius = 5, maxRadius=80)
        circles_right = cv2.HoughCircles(blurR, cv2.HOUGH_GRADIENT, 1, 40, param1 = 70, param2 = 30, minRadius = 5, maxRadius=80)

        #
        #
        #
        detected_circle_pair = None

        if circles_left is not None and circles_right is not None:
            xL, yL, rL = circles_left[0, 0] 
            xR, yR, rR = circles_right[0, 0]
            detected_circle_pair = ((xL, yL, rL), (xR, yR, rR))
            '''
            circles_left = np.uint16(np.around(circles_left[0, :]))
            circles_right = np.uint16(np.around(circles_right[0, :]))

            # 4. Find Corresponding Circles
            # Basic correspondence strategy:
            # - Look for similar Y-coordinates (due to rectification)
            # - Look for reasonable disparity (xL > xR)
            # - Look for similar radii
            y_tolerance = 5 # pixels
            x_disparity_min = 10 # min disparity (xL - xR) to avoid matching same point in flat image
            x_disparity_max = 300 # max disparity, adjust based on camera setup
            r_tolerance = 10 # pixels

            for (xL, yL, rL) in circles_left:
                best_match_idx = -1
                min_y_diff = float('inf')

                for i, (xR, yR, rR) in enumerate(circles_right):
                    y_diff = abs(yL - yR)
                    disparity = xL - xR
                    r_diff = abs(rL - rR)

                    if (y_diff < y_tolerance and
                        disparity > x_disparity_min and
                        disparity < x_disparity_max and
                        r_diff < r_tolerance):
                        
                        # Found a potential match, pick the one with closest Y
                        if y_diff < min_y_diff:
                            min_y_diff = y_diff
                            best_match_idx = i
                
                if best_match_idx != -1:
                    xR, yR, rR = circles_right[best_match_idx]
                    detected_circle_pair = ((xL, yL, rL), (xR, yR, rR))
                    # Break after finding the first good match (for simplicity)
                    break 
            '''
        # 5. Triangulate and Overlay Results
        if detected_circle_pair is not None:
            print("Circle pair found")
            (xL, yL, rL), (xR, yR, rR) = detected_circle_pair

            # Prepare points for triangulation (Nx2 array)
            point1_2d = np.array([[xL, yL]], dtype=np.float32)
            point2_2d = np.array([[xR, yR]], dtype=np.float32)

            # Triangulate
            points_3d = triangulate_points(point1_2d, point2_2d, P1, P2)

            if points_3d is not None and len(points_3d) > 0:
                X, Y, Z = points_3d[0] # Get the 3D coordinates

                center_left = (int(xL), int(yL))
                radius_left = int(rL)

                # Overlay on left image
                cv2.circle(rectifiedLeft, center_left, radius_left, (0, 255, 0), 2) # Green circle
                cv2.circle(rectifiedLeft, center_left, 2, (0, 0, 255), 3) # Red center
                text = f"X: {X:.2f} Y: {Y:.2f} Z: {Z:.2f}"
                cv2.putText(rectifiedLeft, text, (int(xL) + 10, int(yL) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Cyan text

                center_right = (int(xR), int(yR))
                radius_right = int(rR)

                # Overlay on right image
                cv2.circle(rectifiedRight, center_right, radius_right, (0, 255, 0), 2) # Green circle
                cv2.circle(rectifiedRight, center_right, 2, (0, 0, 255), 3) # Red center

        # Display Frames
        combined_rectified = np.concatenate((rectifiedLeft, rectifiedRight), axis=1)
        cv2.imshow('Live Rectified Stereo (Circles & 3D)', combined_rectified)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    capLeft.release()
    capRight.release()
    cv2.destroyAllWindows()

    

def main():
    OUTPUT_DIRECTORY = OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stereo_calibration_params', 'stereo_calibration_data.pkl'))
    #OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'stereo_calibration', 'stereo_calibration_data.pkl'))
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

    live_circ_dist_detect(cameraMat1, distCoeffs1, cameraMat2, distCoeffs2, R, T)



if __name__ == "__main__":
    main()