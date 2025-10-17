#The goal of this script is to find a matrix that represents the world coordinates in the camera frame.
#Instead of using checkerboard, this script attempts to use the circles on the base itself.
#The center of the big circle center will be the center of the platform.
#Two sets of small circles will be used to find two vectors on the plane.
#Cross prod of the vectors can get the z, then cross of z and x to get y.
#It should then be possible to use this frame to transform the detected aruco 
#movements from camera frame to world frame, which should align them with the commands
#given to the Stewart platform (if the platform is aligned with these axes).

import os
import pickle
import cv2
import numpy as np

selected_points = []
labels = ["X1", "X2", "Y1", "Y2", "Center"]
current_label_idx = 0

def mouse_callback(event, x, y, flags, param):
    global selected_points, current_label_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_label_idx < len(labels):
            label = labels[current_label_idx]
            selected_points.append((x, y))
            print(f"Selected {label}: ({x}, {y})")
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(param, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Select Circles", param)
            current_label_idx += 1
        if current_label_idx == len(labels):
            print("✅ All circles selected.")
            #cv2.destroyAllWindows()

def detect_circles(img):
    greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grey",greyimg)
    blurred = cv2.GaussianBlur(greyimg, (3,3), 1)
    #canny = cv2.Canny(blurred, 1, 100)
    #cv2.imshow("Canny", canny)
    #cv2.imshow("blurred", blurred)
    #detected_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 10, param1 = 100, param2 = 50, minRadius = 0)
    detected_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT, 1.5, 10, param1 = 100, param2 = .8, minRadius = 0)
    return detected_circles
'''
def camera_robot_calibration(base_img, camera_matrix, distortion_coefficients, save_dir):
    detected_circles = detect_circles(base_img)
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

            cv2.circle(base_img, (a,b), r, (0,255,0), 2)

            #Draw center

            cv2.circle(base_img, (a,b), 1, (0,0,255), 3)
        cv2.imshow("Detected cirle", base_img)
        cv2.waitKey(0)

        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for i, pt in enumerate(detected_circles[0, :]):
                a, b, r = pt
                cv2.circle(base_img, (a, b), r, (0, 255, 0), 2)
                cv2.circle(base_img, (a, b), 2, (0, 0, 255), 3)
                cv2.putText(base_img, str(i), (a + 5, b - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            print(f"Detected {len(detected_circles[0])} circles. Click in this order:")
            print("  1️⃣ Two X-axis circles (left → right)")
            print("  2️⃣ Two Y-axis circles (front → back)")
            print("  3️⃣ Center circle last\n")
        else:
            print("❌ No circles detected.")
            return

        cv2.imshow("Select Circles", base_img)
        cv2.setMouseCallback("Select Circles", mouse_callback, base_img)

        if len(selected_points) == 5:
            print("\nFinal selected points:")
            for label, pt in zip(labels, selected_points):
                print(f"{label}: {pt}")
        else:
            print("⚠️ Not all circles selected.")
        cv2.waitKey(0)
        '''

def detect_circles(img):
    greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(greyimg, (3,3), 1)
    detected_circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT_ALT, 1.5, 10, param1=100, param2=0.8, minRadius=0
    )
    return detected_circles

def camera_robot_calibration(base_img, camera_matrix, distortion_coefficients, save_dir):
    detected_circles = detect_circles(base_img)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for i, pt in enumerate(detected_circles[0, :]):
            a, b, r = pt
            cv2.circle(base_img, (a, b), r, (0, 255, 0), 2)
            cv2.circle(base_img, (a, b), 2, (0, 0, 255), 3)
            cv2.putText(base_img, str(i), (a + 5, b - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        print(f"Detected {len(detected_circles[0])} circles.")
        print("👉 Click these 5 points in order:")
        print("  1️⃣ Two X-axis circles (left → right)")
        print("  2️⃣ Two Y-axis circles (front → back)")
        print("  3️⃣ Center circle last\n")
    else:
        print("❌ No circles detected.")
        return

    cv2.imshow("Select Circles", base_img)
    cv2.setMouseCallback("Select Circles", mouse_callback, base_img)

    # Keep window open until all points selected
    while current_label_idx < len(labels):
        if cv2.waitKey(20) & 0xFF == 27:  # ESC key to cancel
            break

    print("\nFinal selected points:")
    for label, pt in zip(labels, selected_points):
        print(f"{label}: {pt}")

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
    #PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'catkin_ws/src/vs_stewart/calibration_params_home_webcam/0/calibration_data.pkl'))
    PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'calibration_params_home_webcam/0', 'calibration_data.pkl'))
    print(PARAM_DIR)
    cameraMat1, distCoeffs1, R, T = read_params(PARAM_DIR)
    img = cv2.imread("greenBase2.jpg")
    print("cameraMatrix1: ")
    print(cameraMat1)
    print("distortionCoeffs1:")
    print(distCoeffs1)
    OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'RobotCameraCalibration'))
    T = camera_robot_calibration(img, cameraMat1, distCoeffs1, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()