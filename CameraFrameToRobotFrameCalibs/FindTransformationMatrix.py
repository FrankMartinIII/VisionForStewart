import os
import pickle
import cv2
import numpy as np

def camera_robot_calibration(chessboard_img, camera_matrix_left, distortion_coefficients_left, save_dir):
    #This is for a 12x8 grid technically to center everything in the middle of our workspace
    #COLS = 11  # inner corners per row
    #ROWS = 9  # inner corners per column
    COLS = 8
    ROWS = 6
    SQUARE_SIZE = 29  # mm
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    #For display purposes only
    axis = np.float32([[45,0,0], [0,45,0], [0,0,-45]])
    origin = np.float32([[0,0,0]])  # Chessboard center in object coordinates
    
    # Center offset so middle of board is origin
    center_x = (COLS - 1) * SQUARE_SIZE / 2.0
    center_y = (ROWS - 1) * SQUARE_SIZE / 2.0

    object_points = []
    for i in range(ROWS):
        for j in range(COLS):
            x = j * SQUARE_SIZE - center_x
            y = i * SQUARE_SIZE - center_y
            z = 0.0
            object_points.append([x, y, z])
            print(f"Point {i * COLS + j}: ({x:.3f}, {y:.3f}, {z:.3f})")

    object_points = np.array(object_points, dtype=np.float32)
    '''
    object_points = []
    for i in range(ROWS):
        for j in range(COLS):
            x = j * SQUARE_SIZE
            y = i * SQUARE_SIZE
            z = 0
            object_points.append([x, y, z])
            print(f"Point {i * COLS + j}: ({x:.3f}, {y:.3f}, {z:.3f})")

    object_points = np.array(object_points, dtype=np.float32)
    '''
    #object_points = np.zeros((ROWS*COLS,3), np.float32)
    #object_points[:,:2] = np.mgrid[0:ROWS, 0:COLS].T.reshape(-1,2)
    
    grey = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(grey, (COLS, ROWS), None)

    #If corners were found
    if ret:
        cornersRefined = cv2.cornerSubPix(grey, corners, (11,11), (-1,-1), term_criteria)
        pnp_ret, rvecs, tvecs = cv2.solvePnP(object_points, cornersRefined, camera_matrix_left, distortion_coefficients_left)
        print(pnp_ret)
        print("rvecs: ", rvecs.shape, rvecs)
        print("tvecs: ", tvecs)

        imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix_left, distortion_coefficients_left)
        origin_imgpts, _ = cv2.projectPoints(origin, rvecs, tvecs, camera_matrix_left, distortion_coefficients_left)
        chessboard_img = drawAxes(chessboard_img, corners, origin_imgpts, imgpts)

        imgpts, _ = cv2.projectPoints(object_points, rvecs, tvecs, camera_matrix_left, distortion_coefficients_left)
        for pt in imgpts:
            cv2.circle(chessboard_img , tuple(pt.ravel().astype(int)), 2, (0,0,255),-1)
        cv2.imshow('Chessboard', chessboard_img)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()
        R, _ = cv2.Rodrigues(rvecs) #Convert 3x1 rvecs to a 3x3 rotation matrix
        print("Rotation matrix R: ", R.shape)
        print(R)

        T_world_to_camera = np.eye(4) #Identity matrix
        T_world_to_camera[:3, :3] = R #Build the transformation matrix with the rotation matrix in the first 3 rows and cols
        T_world_to_camera[:3, 3] = tvecs[:, 0] #Make last column the translation part
        print("Transformation matrix from world space to camera space")
        print(T_world_to_camera)

        #Now invert it to get Transformation matrix of camera space to world space
        T_cam_to_world = np.linalg.inv(T_world_to_camera)
        print("Transformation matrix from camera space to world space")
        print(T_cam_to_world)

        transformationMatrices = {
            'T_camera_to_world' : T_cam_to_world,
            'T_world_to_camera': T_world_to_camera
        }
        with open(os.path.join(save_dir, 'T_camera_to_world.pkl'), 'wb') as f:
            pickle.dump(transformationMatrices, f)
        #Save as text files as well
        np.savetxt(os.path.join(save_dir, 'T_cam_to_world.txt'), T_cam_to_world)
        np.savetxt(os.path.join(save_dir, 'T_world_to_cam.txt'), T_world_to_camera)
        return T_cam_to_world
    else:
        print("No chessboard found")

#Draws 3D axes on image
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
    cameraMatrix1 = data['cameraMatrix1']
    distCoeffs1 = data['distortionCoeffs1']
    cameraMatrix2 = data['cameraMatrix2']
    distCoeffs2 = data['distortionCoeffs2']
    R_Mat = data['R_Mat']
    T_Vec = data['T_Vec']
    E = data['Essential_Mat']
    F = data['Fundamental_Mat']
    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_Mat, T_Vec

def test_with_aruco(camera_id, camera_matrix, distortion_coeffs, T_camera_to_world, aruco_size=10, img=None):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(camera_id)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera0 resolution: {width}x{height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(grey, aruco_dict, parameters=parameters)
        
        if ids is not None:
            frame_with_aruco = frame.copy()
            #estimatePose returns tvec that describes transformation from aruco center to camera optical center, it does not return coordinates of aruco in cam frame
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_size, camera_matrix, distortion_coeffs)
            cv2.aruco.drawDetectedMarkers(frame_with_aruco, corners)
            #cv2.aruco.drawAxis(frame, camera_matrix, distortion_coeffs, rvecs, tvecs, 0.03)
            
            '''
            coords_cam_frame = np.append(tvecs.flatten(), 1)
            coords_world_frame = T_camera_to_world @ coords_cam_frame
            position_world = coords_world_frame[:3] #0,1,2 are x, y, and z

            # Text position
            text_cam = f"Camera FrameX: {coords_cam_frame[0]:.3f} mm  Y: {coords_cam_frame[1]:.3f} mm  Z: {coords_cam_frame[2]:.3f} mm"
            text_world = f"World FrameX: {position_world[0]:.3f} mm  Y: {position_world[1]:.3f} mm  Z: {position_world[2]:.3f} mm"
            cv2.putText(frame_with_aruco, text_cam, (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_with_aruco, text_world, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Aruco detection", frame_with_aruco)
            '''
            for rvec, tvec in zip(rvecs, tvecs):
                #T marker to camera
                R_marker, _ = cv2.Rodrigues(rvec)
                T_marker_to_camera = np.eye(4)
                T_marker_to_camera[:3, :3] = R_marker
                T_marker_to_camera[:3, 3] = tvec.flatten()

                #T marker to world = T camera to world * T marker to camera
                T_marker_to_world = T_camera_to_world @ T_marker_to_camera
                marker_world_pos = T_marker_to_world[:3,3]

                #Marker top left in marker space
                half = aruco_size / 2
                top_left_aruco_marker_coordinates = np.array([-half, half, 0, 1], dtype=np.float32).reshape(4,1)
                top_left_aruco_camera_coordinates = T_marker_to_camera @ top_left_aruco_marker_coordinates #Should give [x,y,z,1] transpose in camera frame
                top_left_aruco_world_coordinates = T_camera_to_world @ top_left_aruco_camera_coordinates


                text_cam = f"CameraFrame X: {T_marker_to_camera[0,3]:.3f} mm  Y: {T_marker_to_camera[1,3]:.3f} mm  Z: {T_marker_to_camera[2,3]:.3f} mm"
                #text_world = f"World FrameX: {marker_world_pos[0]:.3f} mm  Y: {marker_world_pos[1]:.3f} mm  Z: {marker_world_pos[2]:.3f} mm"
                text_world = f"WorldFrame topleft X: {top_left_aruco_world_coordinates[0]:.3f} mm  Y: {top_left_aruco_world_coordinates[1]:.3f} mm  Z: {top_left_aruco_world_coordinates[2]:.3f} mm"
                cv2.putText(frame_with_aruco, text_cam, (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_with_aruco, text_world, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            cv2.imshow("Aruco detection", frame_with_aruco)
        
        cv2.imshow("frame", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    #Read in calibration parameters for the cameras
    #We really only need cameraMat1 and distCoef1 I think because triangulation of points is done in the left camera frame
    #Need to go from left camera frame -> robot base frame/world frame
    PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stereo_calibration_params', 'stereo_calibration_data.pkl'))
    cameraMat1, distCoeffs1, cameraMat2, distCoeffs2, R, T = read_params(PARAM_DIR)
    img = cv2.imread("chess3.jpg")
    print("cameraMatrix1: ")
    print(cameraMat1)
    print("distortionCoeffs1:")
    print(distCoeffs1)
    OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Triangulation'))
    T = camera_robot_calibration(img, cameraMat1, distCoeffs1, OUTPUT_DIRECTORY)
    #test_with_aruco(4, cameraMat1, distCoeffs1, T, img=img)

if __name__ == "__main__":
    main()