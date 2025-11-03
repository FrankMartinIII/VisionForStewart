import cv2
import cv2.aruco as aruco
import numpy as np
from collections import deque
import os
import pickle
from scipy.spatial.transform import Rotation


def read_camera_params(filePath):
    with open(filePath, 'rb') as f:
        data = pickle.load(f)
    cameraMatrix1 = data['cameraMatrix']
    distCoeffs1 = data['distortionCoeff']
    rotation_vecs = data['rotationVectors']
    translation_vecs = data['translationVectors']
    reprojection_err = data['reprojectionError']
    return cameraMatrix1, distCoeffs1

def read_robotframe_calibration(filePath):
    with open(filePath, 'rb') as f:
        data = pickle.load(f)
    T_camera_to_world = data['T_camera_to_world']
    T_world_to_camera = data['T_world_to_camera']
    return T_camera_to_world, T_world_to_camera

def draw_axes_on_image(img, T, camera_matrix, dist_coeffs, label="Frame", axis_length=50):
    """
    Draw a 3D axes for a given 4x4 transform T on an image.
    
    Args:
        img: OpenCV image
        T: 4x4 homogeneous transform (object -> camera frame)
        camera_matrix: Intrinsics
        dist_coeffs: Distortion coefficients
        label: Text label for the frame origin
        axis_length: Length of each axis in mm
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Make sure the origin is in front of the camera (positive Z)
    if t[2] <= 0:
        t[2] = 50.0  # minimum positive Z in mm

    # Define axes in local frame
    axes_3D = np.float32([
        [0, 0, 0],                  # origin
        [axis_length, 0, 0],        # X
        [0, axis_length, 0],        # Y
        [0, 0, axis_length]         # Z
    ])

    # Transform axes to camera frame
    axes_in_cam = (R @ axes_3D.T).T + t

    # Project points
    imgpts, _ = cv2.projectPoints(axes_in_cam, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    imgpts = np.clip(imgpts.squeeze(), 0, [img.shape[1]-1, img.shape[0]-1])  # keep in image bounds
    origin = tuple(imgpts[0].astype(int))

    # Draw axes lines
    cv2.line(img, origin, tuple(imgpts[1].astype(int)), (0,0,255), 3)  # X - Red
    cv2.line(img, origin, tuple(imgpts[2].astype(int)), (0,255,0), 3)  # Y - Green
    cv2.line(img, origin, tuple(imgpts[3].astype(int)), (255,0,0), 3)  # Z - Blue

    # Draw label
    cv2.putText(img, label, (origin[0]+5, origin[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img

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

def display(img, camera_matrix, distortion_coefficients, T_cam_to_world):
    drawn_axes = draw_axes_on_image(img, T_world_to_camera, camera_matrix, distortion_coefficients, label="World Frame")
    cv2.imshow("Axes on img", drawn_axes)
    cv2.waitKey(0)

    #Some extra code for optional frame visualization
    axis_length = 60 * 1  # for visualization
    world_axes = np.float32([
        [0, 0, 0],                 # origin
        [axis_length, 0, 0],       # X
        [0, axis_length, 0],       # Y
        [0, 0, axis_length]        # Z
    ])

    #newImg = visualization(world_axes, img, T_camera_to_world)


if __name__ == "__main__":
    #Read in calibration parameters for the cameras
    CAMERA_CALIB_PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'calibration_params_home_webcam/4', 'calibration_data.pkl'))
    cameraMat1, distCoeffs1 = read_camera_params(CAMERA_CALIB_PARAM_DIR)
    FRAME_PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'calibration_params_home_webcam/RobotBaseToCam/TransformationMatricesForStewCalib.pkl'))
    T_camera_to_world, T_world_to_camera = read_robotframe_calibration(FRAME_PARAM_DIR)

    img = cv2.imread("greenBase5-720p.jpg")
    print("image shape", img.shape)
    display(img, cameraMat1, distCoeffs1, T_camera_to_world)