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

def camera_robot_calibration(chessboard_img, camera_matrix_left, distortion_coefficients_left, save_dir):
    og_img_copy = chessboard_img.copy()
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
        #I need three points in the camera frame to get the vectors to define out new frame
        #For that, I need to transform points from checkerboard frame to camera frame
        #I need the transformation matrix from solvePnP with the checkerboard
        R_checkerboard_to_camera, _ = cv2.Rodrigues(rvecs)
        T_checkerboard_to_camera = np.eye(4)
        T_checkerboard_to_camera[:3,:3] = R_checkerboard_to_camera
        T_checkerboard_to_camera[:3, 3] = tvecs[:,0]
        print("Transformation matrix from checkerboard to camera space")
        print(T_checkerboard_to_camera)

        #OK so now I have to take 3 points to define my axes. These are object points and must be transformed to camera frame
        #Origin
        p0 = np.array([0.0, 0.0, 0.0, 1.0])
        #One point along x-axis
        p1 = np.array([SQUARE_SIZE, 0.0, 0.0, 1.0])
        #One point along y-axis
        p2 = np.array([0.0, -SQUARE_SIZE, 0.0, 1.0])
        #Transform them to camera space
        p0_cam_space = T_checkerboard_to_camera @ p0
        p1_cam_space = T_checkerboard_to_camera @ p1
        p2_cam_space = T_checkerboard_to_camera @ p2
        print(f"Coordinates in camera space {p0_cam_space}   {p1_cam_space}   {p2_cam_space}")
        #Those coordinates are homogeneous so I want to make them just xyz
        p0_cam_space = p0_cam_space[:3]
        p1_cam_space = p1_cam_space[:3]
        p2_cam_space = p2_cam_space[:3]

        #Now to define the 3 vectors that will make up our basis in R3 space
        x_axis_cam = p1_cam_space - p0_cam_space
        y_axis_cam = p2_cam_space - p0_cam_space
        #The Z can be found by doing cross product of X and Y vectors to get a vector perpendicular to the plane
        #This ensures it is independent
        z_axis_cam = np.cross(x_axis_cam, y_axis_cam)
        print(f"Vectors in camera space {x_axis_cam}   {y_axis_cam}   {z_axis_cam}")
        #This may not be necessary, but there is something called Gram-Schmidt process
        #that can be used to make sure you have a true orthonormal basis
        y_axis_cam = np.cross(z_axis_cam, x_axis_cam)
        print(f"Vectors in camera space after Gram-Schmidt {x_axis_cam}   {y_axis_cam}   {z_axis_cam}")
        #These 3 vectors also have to be normalized
        x_axis_cam_normalized = x_axis_cam / np.linalg.norm(x_axis_cam)
        y_axis_cam_normalized  = y_axis_cam / np.linalg.norm(y_axis_cam)
        z_axis_cam_normalized = z_axis_cam / np.linalg.norm(z_axis_cam)
        print("")
        print(f"Normalized basis vectors {x_axis_cam_normalized}   {y_axis_cam_normalized}   {z_axis_cam_normalized}")
        print("")

        #Now build matrix to represent this. It is 3 basis vectors as the first 3 cols and last col is the origin.
        R_world_to_cam = np.column_stack((x_axis_cam_normalized, y_axis_cam_normalized, z_axis_cam_normalized))
        #print("R_world_to_cam", R_world_to_cam)
        world_origin_in_cam = p0_cam_space.reshape(3,1)
        T_world_to_camera = np.eye(4)
        T_world_to_camera[:3, :3] = R_world_to_cam
        T_world_to_camera[:3, 3] = world_origin_in_cam.ravel()
        print("T_world_to_cam ", T_world_to_camera)
        T_camera_to_world = np.linalg.inv(T_world_to_camera)

        #Save them to a file
        transformationMatrices = {
            'T_world_to_camera' : T_world_to_camera,
            'T_camera_to_world' : T_camera_to_world
        }
        with open(os.path.join(save_dir, "TransformationMatricesForStewCalib.pkl"), 'wb') as f:
            pickle.dump(transformationMatrices, f)
        np.savetxt(os.path.join(save_dir, 'T_camera_to_world_for_calibration.txt'), T_camera_to_world)
        np.savetxt(os.path.join(save_dir, 'T_world_to_camera_for_calibration.txt'), T_world_to_camera)

        #Some extra code for optional frame visualization
        axis_length = SQUARE_SIZE * 1  # for visualization
        world_axes = np.float32([
            [0, 0, 0],                 # origin
            [axis_length, 0, 0],       # X
            [0, axis_length, 0],       # Y
            [0, 0, axis_length]        # Z
        ])

        new_basis_viz = visualization(world_axes, og_img_copy, T_world_to_camera, camera_matrix_left, distortion_coefficients_left)
        cv2.imshow("World Axes Visualized", new_basis_viz)
        cv2.waitKey(0)



def visualization(axes, myimg, transformation_matrix, camera_matrix, distortion_coef):
    img = myimg.copy()
    R = transformation_matrix[:3,:3]
    tvec = transformation_matrix[:3,3]
    imgpts, _ = cv2.projectPoints(axes, R, tvec,
                              camera_matrix, distortion_coef)
    origin = tuple(imgpts[0].ravel().astype(int))
    cv2.line(img, origin, tuple(imgpts[1].ravel().astype(int)), (0,0,255), 4)   # X - Red
    cv2.line(img, origin, tuple(imgpts[2].ravel().astype(int)), (0,255,0), 4)   # Y - Green
    cv2.line(img, origin, tuple(imgpts[3].ravel().astype(int)), (255,0,0), 4)   # Z - Blue

    return img
    






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
    #PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'catkin_ws/src/StewartIBVS/calibration_params_home_webcam/0/calibration_data.pkl'))
    PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'catkin_ws/src/vs_stewart/calibration_params_home_webcam/0/calibration_data.pkl'))
    cameraMat1, distCoeffs1, R, T = read_params(PARAM_DIR)
    img = cv2.imread("29_mm_chess.jpg")
    print("cameraMatrix1: ")
    print(cameraMat1)
    print("distortionCoeffs1:")
    print(distCoeffs1)
    OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CameraFrameToRobotFrameCalibs'))
    T = camera_robot_calibration(img, cameraMat1, distCoeffs1, OUTPUT_DIRECTORY)
    #test_with_aruco(4, cameraMat1, distCoeffs1, T, img=img)

if __name__ == "__main__":
    main()