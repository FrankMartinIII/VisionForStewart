import numpy as np
import cv2
import glob
import os
import pickle

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #Based on this article: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
    '''
    Returns
    ret: RMS reprojection error
    camera_matrix1: camera matrix1
    distCoef1: distorition coefficient
    camera_matrix2: camera matrix2
    distCoef2: distorition coefficient
    R: 3x3 rotation matrix to fo from C1 coordinate system to C2 coordinate system
    T: 3x1 translation vector between coordinate system of C1 and C2
    E: 3x3 essential matrix
    F: 3x3 Fundamental matrix
    '''
    #CHESSBOARD_SIZE = (6,5)
    CHESSBOARD_SIZE = (11,7)
    #SQUARE_SIZE = 2.36 #centimeter for now, should use millimeters
    SQUARE_SIZE = 30
    OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stereo_calibration_params'))

    #Create output directory
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objPoints = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objPoints[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    #Scale object points by square size for real-world measurements
    objPoints = objPoints * SQUARE_SIZE
    '''
    #Read in synched frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]
    '''

    #frames_folder0 = os.path.abspath(os.path.join(frames_folder, 'stereo0/*.jpg'))
    frames_folder0 = os.path.abspath(os.path.join(frames_folder, 'kaggle0/*.png'))
    #frames_folder0 = frames_folder + "/stereo0/*.jpg"
    print(frames_folder0)
    #frames_folder1 = frames_folder + "/stereo1/*.jpg"
    #frames_folder1 = os.path.abspath(os.path.join(frames_folder, 'stereo1/*.jpg'))
    frames_folder1 = os.path.abspath(os.path.join(frames_folder, 'kaggle1/*.png'))
    images_names0 = glob.glob(frames_folder0)
    images_names1 = glob.glob(frames_folder1)
    c1_images_names = sorted(images_names0)
    c2_images_names = sorted(images_names1)


    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        cur_im = cv2.imread(im1, cv2.IMREAD_GRAYSCALE)
        c1_images.append(cur_im)

        cur_im = cv2.imread(im2, cv2.IMREAD_GRAYSCALE)
        c2_images.append(cur_im)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Create arrays to store object points and image points from all images
    objPointsStore = []
    imgPointsStore_left = []
    imgPointsStore_right = []

    imgCounter = 0
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = frame1 #cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = frame2 #cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        #Find chessboard corners
        frame1_ret, frame1_corners = cv2.findChessboardCorners(gray1, CHESSBOARD_SIZE, None)
        frame2_ret, frame2_corners = cv2.findChessboardCorners(gray2, CHESSBOARD_SIZE, None)

        if frame1_ret == True and frame2_ret == True:
            frame1_subCorners = cv2.cornerSubPix(gray1, frame1_corners, (11, 11), (-1, -1), criteria)
            frame2_subCorners = cv2.cornerSubPix(gray2, frame2_corners, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(frame1, CHESSBOARD_SIZE, frame1_subCorners, frame1_ret)
            cv2.imshow('img1', frame1)
            output_img_path = os.path.join(OUTPUT_DIRECTORY, f'corners1_{imgCounter}.jpg')
            cv2.imwrite(output_img_path, frame1)

            cv2.drawChessboardCorners(frame2, CHESSBOARD_SIZE, frame2_subCorners, frame2_ret)
            cv2.imshow('img2', frame2)
            output_img_path = os.path.join(OUTPUT_DIRECTORY, f'corners2_{imgCounter}.jpg')
            cv2.imwrite(output_img_path, frame2)

            k = cv2.waitKey(500)

            objPointsStore.append(objPoints)

            #If cornerSubPix fails, do I need to append frame_corners instead?
            imgPointsStore_left.append(frame1_subCorners)
            imgPointsStore_right.append(frame2_subCorners)
        imgCounter+=1


    stereoCalibrationFlags = cv2.CALIB_FIX_INTRINSIC
    #stereoCalibrationFlags |= cv2.CALIB_USE_INTRINSIC_GUESS
    #stereoCalibrationFlags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    #stereoCalibrationFlags |= cv2.CALIB_ZERO_TANGENT_DIST
    ret, camera_matrix1, distCoef1, camera_matrix2, distCoef2, R, T, E, F = cv2.stereoCalibrate(objPointsStore, imgPointsStore_left, imgPointsStore_right, mtx1, dist1, mtx2, dist2, (width, height), criteria=criteria, flags=stereoCalibrationFlags)
    print("Reprojection error: ", str(ret))

    #Save calibration data
    calibrationData = {
        'cameraMatrix1' : camera_matrix1,
        'distortionCoeffs1': distCoef1,
        'cameraMatrix2': camera_matrix2,
        'distortionCoeffs2': distCoef2,
        'reprojectionError': ret,
        'R_Mat': R,
        'T_Vec': T,
        'Essential_Mat': E,
        'Fundamental_Mat': F
    }
    with open(os.path.join(OUTPUT_DIRECTORY, 'stereo_calibration_data.pkl'), 'wb') as f:
        pickle.dump(calibrationData, f)

    #Save rotation matrix and translation vector as files
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'rotation_matrix.txt'), R)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'translation_vector.txt'), T)
    #I will also save these just for debugging but I believe they should be the same as the input matrices
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'camera_matrix1.txt'), camera_matrix1)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'camera_matrix2.txt'), camera_matrix2)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'distortion_coefficients1.txt'), distCoef1)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'distortion_coefficients2.txt'), distCoef2)
    return R, T


def main():
    OUTPUT_DIRECTORY_C0 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_params', str(7), 'calibration_data.pkl'))
    with open(OUTPUT_DIRECTORY_C0, 'rb') as f:
        data1 = pickle.load(f)

    for key in data1:
        print(key, type(data1[key]))
    mtx1 = data1['cameraMatrix']
    dist1 = data1['distortionCoeff']

    OUTPUT_DIRECTORY_C1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_params', str(8), 'calibration_data.pkl'))
    with open(OUTPUT_DIRECTORY_C1, 'rb') as f:
        data2 = pickle.load(f)

    for key in data2:
        print(key, type(data2[key]))
    mtx2 = data2['cameraMatrix']
    dist2 = data2['distortionCoeff']

    #CALIBRATION_IMAGES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FrankCalib/frames/synched/*.png'))
    CALIBRATION_IMAGES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_images'))
    R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, CALIBRATION_IMAGES_PATH)
if __name__ == "__main__":
    main()