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

def display_SideBySide(img1, img2, title):
    combined = np.hstack((img1, img2))
    cv2.imshow(title, combined)
    cv2.waitKey(0)

def histogram_match(img1, img2):
    # Match img2 colors to img1 and return the shifted img2
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    matched = match_histograms(img2_rgb, img1_rgb, channel_axis=-1)
    matched_bgr = cv2.cvtColor((matched).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return matched_bgr

def color_transfer(source, target):
    # Convert images to Lab color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # Split channels
    lSrc, aSrc, bSrc = cv2.split(source_lab)
    lTar, aTar, bTar = cv2.split(target_lab)

    # Compute mean and stddev of each channel
    lMeanSrc, lStdSrc = cv2.meanStdDev(lSrc)
    lMeanTar, lStdTar = cv2.meanStdDev(lTar)

    aMeanSrc, aStdSrc = cv2.meanStdDev(aSrc)
    aMeanTar, aStdTar = cv2.meanStdDev(aTar)

    bMeanSrc, bStdSrc = cv2.meanStdDev(bSrc)
    bMeanTar, bStdTar = cv2.meanStdDev(bTar)

    # Flatten scalar values
    lMeanSrc = lMeanSrc[0][0]; lStdSrc = lStdSrc[0][0]
    lMeanTar = lMeanTar[0][0]; lStdTar = lStdTar[0][0]

    aMeanSrc = aMeanSrc[0][0]; aStdSrc = aStdSrc[0][0]
    aMeanTar = aMeanTar[0][0]; aStdTar = aStdTar[0][0]

    bMeanSrc = bMeanSrc[0][0]; bStdSrc = bStdSrc[0][0]
    bMeanTar = bMeanTar[0][0]; bStdTar = bStdTar[0][0]

    # Apply color transfer
    l = ((lTar - lMeanTar) * (lStdSrc / lStdTar)) + lMeanSrc
    a = ((aTar - aMeanTar) * (aStdSrc / aStdTar)) + aMeanSrc
    b = ((bTar - bMeanTar) * (bStdSrc / bStdTar)) + bMeanSrc

    # Clip values to valid range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # Merge channels and convert back to BGR
    transferred = cv2.merge([l, a, b]).astype("uint8")
    transferred = cv2.cvtColor(transferred, cv2.COLOR_LAB2BGR)
    return transferred

def find_hsv_mask(img, low_colors, high_colors, blur=False):
    if blur:
        img = cv2.GaussianBlur(img, (9,9), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([48, 40, 0])
    upper_green = np.array([99, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

def apply_hsv_mask(img, mask):
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def image_processing(img):
    #Probably do not need to do this with the HSV mask
    greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(greyimg, (9,9), 0)
    edges = cv2.Canny(blurred,0,40, apertureSize=3, L2gradient=True)
    return edges, blurred

def find_closest_ellipse_by_y(reference_ellipse, candidate_ellipses, y_thresh=90):
    ref_y = reference_ellipse[0][1]  # y-coordinate of the center
    min_dist = float('inf')
    best_match = None
    for e in candidate_ellipses:
        y = e[0][1]
        dy = abs(ref_y - y)
        if dy < min_dist and dy < y_thresh:
            min_dist = dy
            best_match = e
    return best_match

def get_ellipse_mask(image_shape, ellipse, tighten=False):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if tighten:
        center, axes, angle = ellipse
        center = tuple(np.round(center).astype(int))
        axes = tuple(np.round(np.array(axes) / 2.5).astype(int))  # OpenCV takes half-length axes
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)  # Filled ellipse
    else:
        cv2.ellipse(mask, ellipse, 255, -1)
    return mask

def detect_and_compute_sift(img, mask=None):
    #SIFT should be used with greyscale images
    #greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, mask)
    return keypoints, descriptors

def draw_sift_keypoints(img, keypoints):
    img_with_kp = cv2.drawKeypoints(
    img,                     # original image
    keypoints,                      # list of keypoints
    None,                     # output image (None = new one will be created)
    color=(0, 255, 0),        # color of the keypoints
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # shows size/orientation
    )
    return img_with_kp

def clahe_preprocess(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grey_eq = clahe.apply(grey)
    return grey_eq

def match_features(descriptors1, descriptors2, ratio_threshold=0.75, distance_threshold=190):
    #Using brute force matcher https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html reference D. Lowe paper for this
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    #Apply Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)
    #Eliminate bad matches below distance threshold (max in SIFT is 300 I think)
    good_matches = [m for m in good_matches if m.distance < distance_threshold]
    #Sort by match quality
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    return good_matches

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3

def draw_matches_with_distances(img1, kp1, img2, kp2, matches, max_distance=None):
    # Use OpenCV's drawMatches to get initial visualization
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if not max_distance:
        max_distance = max(m.distance for m in matches)

    w1 = img1.shape[1]

    for m in matches:
        # Get keypoint locations
        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))
        pt2_offset = (pt2[0] + w1, pt2[1])  # Adjust for side-by-side layout

        # Normalize distance to get color
        norm_dist = min(m.distance / max_distance, 1.0)
        r = int(255 * norm_dist)
        g = int(255 * (1 - norm_dist))
        b = 0
        color = (b, g, r)

        # Put text (distance) near the second keypoint
        text_pos = (pt2_offset[0] + 5, pt2_offset[1] - 5)
        cv2.putText(match_img, f"{m.distance:.1f}", text_pos, 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=color, thickness=1, lineType=cv2.LINE_AA)

    return match_img

def main():
    OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stereo_calibration_params', 'stereo_calibration_data.pkl'))
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_Mat, T_Vec = triangulate_pts.read_params(OUTPUT_DIRECTORY)
    img1 = cv2.imread("stereo_images/70L.png")
    img2 = cv2.imread("stereo_images/70R.png")
    display_SideBySide(img1, img2, "Initial images")

    #Step 0: Rectify and undistort
    width = img1.shape[1]
    height = img1.shape[0]
    map1x, map1y, map2x, map2y, P1, P2 = triangulate_pts.get_undistort_rectification_maps(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_Mat, T_Vec, (width, height))
    LUndist, RUndist = triangulate_pts.undistort_images(img1, img2, map1x, map1y, map2x, map2y)
    display_SideBySide(LUndist, RUndist, "Undistorted images")
    img1 = LUndist
    img2 = RUndist
    img2copy = img2.copy()

    #Step 1: Perform histogram matching
    #img2 = histogram_match(img1, img2)
    img2 = color_transfer(img1, img2)
    display_SideBySide(img1, img2, "Post histogram matching")
    
 
    #Step 2: Develop the mask
    low_hsv_val = np.array([40, 30, 0])
    high_hsv_val = np.array([99, 255, 255])

    mask1 = find_hsv_mask(img1, low_hsv_val, high_hsv_val, blur=True)
    masked_img1 = apply_hsv_mask(img1, mask1)
    mask2 = find_hsv_mask(img2, low_hsv_val, high_hsv_val, blur=True)
    masked_img2 = apply_hsv_mask(img2, mask2)
    display_SideBySide(masked_img1, masked_img2, "Masked images")



    #Step 3A: Image processing (NOW I AM THINKING I DO NOT NEED THIS STEP AND EDGE DETECTION ANYMORE)
    '''
    edge1, blur1 = image_processing(img1)
    edge2, blur2 = image_processing(img2)
    combined = np.hstack((blur1, blur2))
    cv2.imshow("Blurred images", combined)
    combined = np.hstack((edge1, edge2))
    cv2.imshow("Edge images", combined)
    '''

    #Step 3B: Just skip straight to finding the contours and ellipses
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contImg1 = img1.copy()
    contImg2 = img2.copy()
    cv2.drawContours(contImg1, contours1, -1, (0, 255, 0), 2)
    cv2.drawContours(contImg2, contours2, -1, (0, 255, 0), 2)
    display_SideBySide(contImg1, contImg2, "Contours")

    ellipseContImg1 = img1.copy()
    ellipseContImg2 = img2.copy()
    ellipticalCon1, ellipses1 = contour_det_lib.find_best_circles2(contours1, min_perimeter=100, min_ellipse_perimeter=50, min_elliptical_aspect_ratio=1, max_elliptical_aspect_ratio=3)
    ellipticalCon2, ellipses2 = contour_det_lib.find_best_circles2(contours2, min_perimeter=100, min_ellipse_perimeter=50, min_elliptical_aspect_ratio=1, max_elliptical_aspect_ratio=3)
    for c in ellipticalCon1:
        cv2.drawContours(ellipseContImg1, [c], -1, (0, 255, 0), 2)
    for c in ellipticalCon2:
        cv2.drawContours(ellipseContImg2, [c], -1, (0, 255, 0), 2)
    display_SideBySide(ellipseContImg1, ellipseContImg2, "Circular contours")

    #Getting largest ellipse in each image
    ellipses1 = contour_det_lib.sort_ellipses_by_size(ellipses1)
    ellipses2 = contour_det_lib.sort_ellipses_by_size(ellipses2)
    #contour_det_lib.display_ellipses(ellipses1, ellipseContImg1)
    #contour_det_lib.display_ellipses(ellipses2, ellipseContImg2)
    cv2.ellipse(ellipseContImg1, ellipses1[0], (0, 255, 255), 2)
    cv2.ellipse(ellipseContImg2, ellipses2[0], (0, 255, 255), 2)
    display_SideBySide(ellipseContImg1, ellipseContImg2, "Ellipses")
    #Trying a new way of ellipse matching on epipolar lines
    ellipse1 = ellipses1[0]
    ellipse2 = find_closest_ellipse_by_y(ellipse1, ellipses2)
    if ellipse2 is None:
        print("No matching ellipse found in img2")
        return
    cv2.ellipse(ellipseContImg1, ellipse1, (0, 0, 255), 2)
    cv2.ellipse(ellipseContImg2, ellipse2, (0, 0, 255), 2)
    display_SideBySide(ellipseContImg1, ellipseContImg2, "Ellipses")

    #Step 4: Mask out just the area in the ellipse
    width = ellipseContImg1.shape[1]
    height = ellipseContImg1.shape[0]
    ellipse_mask1 = get_ellipse_mask((height, width), ellipse1, tighten=False)
    width = ellipseContImg2.shape[1]
    height = ellipseContImg2.shape[0]
    ellipse_mask2 = get_ellipse_mask((height, width), ellipse2, tighten=False)
    
    #Step 5: Image preprocessing for SIFT
    '''
    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey_eq1 = cv2.equalizeHist(grey1)
    display_SideBySide(grey1, grey_eq1, "Grey equalize hist img")
    '''
    grey1 = clahe_preprocess(img1)
    grey2 = clahe_preprocess(img2)
    #RIGHT HERE I AM DOING SIFT ON THE RECTIFIE IMG2 BEFORE HIST EQ
    grey2 = clahe_preprocess(img2copy)
    display_SideBySide(grey1, grey2, "CLAHE")
    #Step 6: Perform SIFT inside the ellipses
    kp1, desc1 = detect_and_compute_sift(grey1, ellipse_mask1)
    kp2, desc2 = detect_and_compute_sift(grey2, ellipse_mask2)
    sift1 = draw_sift_keypoints(img1, kp1)
    sift2 = draw_sift_keypoints(img2, kp2)
    display_SideBySide(sift1, sift2, "Keypoints in both images")

    #Step 7: Feature matching
    matches = match_features(desc1, desc2)
    print(matches)
    top_matches = matches[0:10]
    matched_img = draw_matches_with_distances(img1, kp1, img2, kp2, matches)
    cv2.imshow("Matches", matched_img)

    #Step 8: Triangulate the points
    # Get matched keypoint coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in top_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in top_matches])
    # Use your triangulation function
    pts3d = triangulate_pts.triangulate_points(pts1, pts2, P1, P2)
    print(pts3d)
    for pt3d, match in zip(pts3d, top_matches):
        print(f"3D Point: {pt3d}, Match distance: {match.distance:.2f}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()