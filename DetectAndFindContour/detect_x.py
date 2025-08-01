import cv2
import numpy as np
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
import os
import sys
TRIANGULATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Triangulation'))
sys.path.append(TRIANGULATION_DIR)
CIRCLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CircleDetection'))
sys.path.append(CIRCLE_DIR)
import triangulate_pts
import contour_det_lib
import AllSteps

def find_green_X_center(image_bgr):
    # Step 1: Original image
    cv2.imshow("Original", image_bgr)

    # Step 2: Convert to HSV

    # Step 3: Threshold for green
    low_hsv_val = np.array([40, 30, 0])
    high_hsv_val = np.array([99, 255, 255])

    mask = AllSteps.find_hsv_mask(image_bgr, low_hsv_val, high_hsv_val, blur=True)
    cv2.imshow("Mask", mask)
    # Step 4: Clean mask
    mask_clean = cv2.medianBlur(mask, 5)
    mask_clean = cv2.dilate(mask_clean, None, iterations=1)
    cv2.imshow("Green Mask (cleaned)", mask_clean)

    # Step 5: Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_bgr, contours, -1, (0, 255, 0), 2)
    cv2.imshow("CONTOURS", image_bgr)
    result_image = image_bgr.copy()
    centers = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 500:
            continue  # Filter out noise

        # Draw the raw contour
        cv2.drawContours(result_image, [cnt], -1, (255, 0, 0), 2)

        # Get centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

        # Draw center point
        cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)

        # Show individual contour mask (optional)
        temp_mask = np.zeros_like(mask)
        cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
        cv2.imshow(f"Contour Mask {i}", temp_mask)

    cv2.imshow("Contours + Centers", result_image)

    return result_image, centers

def main():
    #Read image
    OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stereo_calibration_params', 'stereo_calibration_data.pkl'))
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_Mat, T_Vec = triangulate_pts.read_params(OUTPUT_DIRECTORY)
    img1 = cv2.imread("stereo_images/70L.png")
    img2 = cv2.imread("stereo_images/70R.png")
    AllSteps.display_SideBySide(img1, img2, "Initial images")

    #Step 0: Rectify and undistort
    width = img1.shape[1]
    height = img1.shape[0]
    map1x, map1y, map2x, map2y, P1, P2 = triangulate_pts.get_undistort_rectification_maps(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_Mat, T_Vec, (width, height))
    LUndist, RUndist = triangulate_pts.undistort_images(img1, img2, map1x, map1y, map2x, map2y)
    AllSteps.display_SideBySide(LUndist, RUndist, "Undistorted images")
    img1 = LUndist
    img2 = RUndist

    find_green_X_center(img1)
    '''
    #Step 1: Perform histogram matching
    #img2 = histogram_match(img1, img2)
    img2 = AllSteps.color_transfer(img1, img2)
    AllSteps.display_SideBySide(img1, img2, "Post histogram matching")
    
 
    #Step 2: Develop the mask
    low_hsv_val = np.array([40, 30, 0])
    high_hsv_val = np.array([99, 255, 255])

    mask1 = AllSteps.find_hsv_mask(img1, low_hsv_val, high_hsv_val, blur=True)
    masked_img1 = AllSteps.apply_hsv_mask(img1, mask1)
    mask2 = AllSteps.find_hsv_mask(img2, low_hsv_val, high_hsv_val, blur=True)
    masked_img2 = AllSteps.apply_hsv_mask(img2, mask2)
    AllSteps.display_SideBySide(mask1, mask2, "Masks")
    AllSteps.display_SideBySide(masked_img1, masked_img2, "Masked images")

    gray = cv2.cvtColor(masked_img1
                        , cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 0, 40, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10)
    cv2.imshow("Edges", edges)
    '''
    


    cv2.waitKey(0)

if __name__ == "__main__":
    main()