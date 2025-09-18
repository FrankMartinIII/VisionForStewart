#Steps used
#Mask
#Morphological opening
#Skeletonization
#HoughLinesP
#Find the angles of the lines
#Group lines using Kmeans clustering according to their angles
#Use RANSAC on the endpoints of lines in the same group to get slope and intercept for each group
#Find the intersection point

import cv2
import numpy as np
from skimage.exposure import match_histograms
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import os
import sys
import math
TRIANGULATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Triangulation'))
sys.path.append(TRIANGULATION_DIR)
CIRCLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CircleDetection'))
sys.path.append(CIRCLE_DIR)
ALLSTEPS_LIB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DetectAndFindContour'))
sys.path.append(ALLSTEPS_LIB)
import triangulate_pts
import contour_det_lib
import AllSteps

def sketon_find_line_segments(skeletonImg, thresholdAccumVotesNeeded = 20, minLineLength = 10, maxLineGap = 5):
    #Creating an image so that I can see the line segments
    linesOnSkeleton = skeletonImg.copy()
    linesOnSkeleton = cv2.cvtColor(linesOnSkeleton, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(skeletonImg, 1, np.pi / 180, threshold=thresholdAccumVotesNeeded, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(linesOnSkeleton, (l[0], l[1]), (l[2], l[3]), (0,80*i,255-30*i), 3, cv2.LINE_AA)
    else:
        print("WARNING: NO LINES FOUND ON SKELETON")
    return lines, linesOnSkeleton


def line_to_angle_and_midpoint(line):
    #This function gets the angel and the midpoint of the given line segment
    x1, y1, x2, y2 = line
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    return angle % 180, midpoint  # Use angle mod 180 to treat parallel lines the same


def get_endpoints_of_lines(line_group):
    points = []
    for l in line_group:
        x1,y1,x2,y2 = l[0] #Get endpoints of that line
        points.append([x1, y1])
        points.append([x2, y2])
    return np.array(points)


def group_line_segments(lines, num_clusters=2, n_num_init=10):
    if(len(lines) < 2):
        print("Less than 2 lines detected in image. Cannot proceed")
        return
    angles = []
    for i in range(len(lines)):
        angle, midpoint = line_to_angle_and_midpoint(lines[i][0])
        print(f"Line {i}   Angle {angle},     Midpoint{midpoint}")
        angles.append([angle]) #Need brackets or reshape into column for kmeans

    #print(angles)
    kmeans = KMeans(n_clusters = num_clusters, n_init=n_num_init).fit(angles)
    labels = kmeans.labels_
    #print(labels)
    lineGroup1 = [lines[i] for i in range(len(lines)) if labels[i] == 0]
    lineGroup2 = [lines[i] for i in range(len(lines)) if labels[i] == 1]

    endPtsGrp1 = get_endpoints_of_lines(lineGroup1)
    endPtsGrp2 = get_endpoints_of_lines(lineGroup2)
    return endPtsGrp1, endPtsGrp2


#RANSAC to get a line from a list of points
def ransac_fit_line(points):
    X = points[:, 0].reshape(-1, 1)  # x values as 2D array
    y = points[:, 1]                # y values as 1D array
    ransac = RANSACRegressor(min_samples=2, residual_threshold=2.0)
    ransac.fit(X, y)
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    return slope, intercept


def draw_line_from_slope(img, m, b, color):
    h, w = img.shape[:2]
    pt1 = (0, int(b))
    pt2 = (w, int(m * w + b))
    cv2.line(img, pt1, pt2, color, 2)

def line_intersection(m1, b1, m2, b2):
    if m1 == m2:
        return None  # Parallel lines
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return int(x), int(y)

def detect_ShiTomasi_keypoints(imgrey, maxCorners=500, qualityLevel=0.01, minDistance=5, mask=None):
    corners = cv2.goodFeaturesToTrack(imgrey, maxCorners, qualityLevel, minDistance, mask=mask)
    keypoints = [cv2.KeyPoint(float(pt[0][0]), float(pt[0][1]), 3) for pt in corners]
    return keypoints




def main():
    OUTPUT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stereo_calibration_params', 'stereo_calibration_data.pkl'))
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_Mat, T_Vec = triangulate_pts.read_params(OUTPUT_DIRECTORY)
    #Assuming I had two images
    img1 = cv2.imread("capX1_L.png")
    img2 = cv2.imread("capX1_R.png")

    
    #Step 0: Rectify and undistort
    width = img1.shape[1]
    height = img1.shape[0]
    map1x, map1y, map2x, map2y, P1, P2 = triangulate_pts.get_undistort_rectification_maps(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_Mat, T_Vec, (width, height))
    LUndist, RUndist = triangulate_pts.undistort_images(img1, img2, map1x, map1y, map2x, map2y)
    AllSteps.display_SideBySide(LUndist, RUndist, "Undistorted images")
    img1 = LUndist
    img2 = RUndist
    

    #Step 1: Perform histogram matching
    img1OG = img1.copy()
    #img1 = AllSteps.histogram_match(img2, img1)
    img1 = AllSteps.color_transfer(img2, img1)
    AllSteps.display_SideBySide(img1, img2, "Post histogram matching")
    
    #Step 2: Develop the mask
    low_hsv_val = np.array([40, 40, 0])
    high_hsv_val = np.array([99, 255, 255])
    mask1 = AllSteps.find_hsv_mask(img1, low_hsv_val, high_hsv_val, blur=False, blur_alpha=2)
    masked_img1 = AllSteps.apply_hsv_mask(img1, mask1)
    mask2 = AllSteps.find_hsv_mask(img2, low_hsv_val, high_hsv_val, blur=False, blur_alpha=2)
    masked_img2 = AllSteps.apply_hsv_mask(img2, mask2)
    AllSteps.display_SideBySide(mask1, mask2, "Masks")
    print("Mask dtype:", mask1.dtype)
    print("Mask shape:", mask1.shape)
    print("Unique values in mask:", np.unique(mask1))

    #Step 3: Clean the mask
    #I originally was blurring above and using a lower value for the OPEN kernel
    #Issue was detecting too much noise in corner of img
    #Could be fixed by looking within an ellipse, my original plan
    clean1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    clean2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    AllSteps.display_SideBySide(mask1, mask2, "Clean masks")

    #Step 4: Skeletonize mask
    skeleton1 = cv2.ximgproc.thinning(clean1)
    skeleton2 = cv2.ximgproc.thinning(clean2)
    #cv2.imshow("Skeletonized", skeleton1)
    AllSteps.display_SideBySide(skeleton1, skeleton2, "Skeletonized")
    #print("Skeleton unique values:", np.unique(skeleton1))
    cv2.imwrite("debug_clean.png", clean1)
    cv2.imwrite("debug_skeleton.png", skeleton1)
    
    #Step 5: Find line segments in the skeleton
    lines1, lineSegsOnSkeletonImage1 = sketon_find_line_segments(skeleton1)
    lines2, lineSegsOnSkeletonImage2 = sketon_find_line_segments(skeleton2)
    #cv2.imshow("Line segments overlayed on skeleton", lineSegsOnSkeletonImage1)
    AllSteps.display_SideBySide(lineSegsOnSkeletonImage1, lineSegsOnSkeletonImage2, "Line segments overlayed on skeleton")

    #Step 6: Cluster line segments
    img1EndPtsGrp1, img1EndPtsGrp2 = group_line_segments(lines1)
    img2EndPtsGrp1, img2EndPtsGrp2 = group_line_segments(lines2)

    #Step 7: Use RANSAC to find two lines
    img1m1, img1b1 = ransac_fit_line(img1EndPtsGrp1)
    img1m2, img1b2 = ransac_fit_line(img1EndPtsGrp2)
    img1FittedLines = skeleton1.copy()
    img1FittedLines = cv2.cvtColor(img1FittedLines, cv2.COLOR_GRAY2BGR)
    draw_line_from_slope(img1FittedLines, img1m1, img1b1, (0, 255, 0))
    draw_line_from_slope(img1FittedLines, img1m2, img1b2, (255, 0, 0))
    img2m1, img2b1 = ransac_fit_line(img2EndPtsGrp1)
    img2m2, img2b2 = ransac_fit_line(img2EndPtsGrp2)
    img2FittedLines = skeleton2.copy()
    img2FittedLines = cv2.cvtColor(img2FittedLines, cv2.COLOR_GRAY2BGR)
    draw_line_from_slope(img2FittedLines, img2m1, img2b1, (0, 255, 0))
    draw_line_from_slope(img2FittedLines, img2m2, img2b2, (255, 0, 0))
    AllSteps.display_SideBySide(img1FittedLines, img2FittedLines, "RANSAC fitted lines on the skeletons")
    #cv2.imshow("RANSAC fitted lines on the skeleton", img1FittedLines)

    #Step 8: Find the intersection point
    fittedLinesOnOriginalImg1 = img1.copy()
    draw_line_from_slope(fittedLinesOnOriginalImg1, img1m1, img1b1, (0, 255, 0))
    draw_line_from_slope(fittedLinesOnOriginalImg1, img1m2, img1b2, (255, 0, 0))
    img1IntersectX, img1IntersectY = line_intersection(img1m1, img1b1, img1m2, img1b2)
    cv2.circle(fittedLinesOnOriginalImg1, (img1IntersectX, img1IntersectY), 5, (0, 255, 255), -1)
    fittedLinesOnOriginalImg2 = img2.copy()
    draw_line_from_slope(fittedLinesOnOriginalImg2, img2m1, img2b1, (0, 255, 0))
    draw_line_from_slope(fittedLinesOnOriginalImg2, img2m2, img2b2, (255, 0, 0))
    img2IntersectX, img2IntersectY = line_intersection(img2m1, img2b1, img2m2, img2b2)
    cv2.circle(fittedLinesOnOriginalImg2, (img2IntersectX, img2IntersectY), 5, (0, 255, 255), -1)
    AllSteps.display_SideBySide(fittedLinesOnOriginalImg1, fittedLinesOnOriginalImg2, "Intersection points on original rect and undist image")
    #cv2.imshow("Intersection points on original rect and undist image", fittedLinesOnOriginalImg1)
    print("INTERSECTION POINT IN IMAGE1 LOCATED AT: X=", img1IntersectX, " Y=", img1IntersectY)
    print("INTERSECTION POINT IN IMAGE2 LOCATED AT: X=", img2IntersectX, " Y=", img2IntersectY)
    #cv2.waitKey(0)



    #------AFTER THIS LINE IS EXPERIMENTAL-------
    #Step 9: Make a circle mask around the X intersection point
    radius = 50
    circle_mask1 = np.zeros_like(mask1, dtype=np.uint8)
    cv2.circle(circle_mask1, (img1IntersectX,img1IntersectY), radius, 255, -1)
    circle_mask2 = np.zeros_like(mask2, dtype=np.uint8)
    cv2.circle(circle_mask2, (img2IntersectX,img2IntersectY), radius, 255, -1)
    AllSteps.display_SideBySide(circle_mask1, circle_mask2, "Generated circle mask")
    #cv2.imshow("Generated circle mask", circle_mask1)

    #Step 10: Preprocessing for SIFT
    grey1 = AllSteps.clahe_preprocess(img1)
    grey2 = AllSteps.clahe_preprocess(img2)
    AllSteps.display_SideBySide(grey1, grey2, "CLAHE")
    #grey1 = cv2.cvtColor(img1OG, cv2.COLOR_BGR2GRAY)
    #grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Step 11: Perform SIFT inside circles
    kp1, desc1 = AllSteps.detect_and_compute_sift(grey1, circle_mask1)
    kp2, desc2 = AllSteps.detect_and_compute_sift(grey2, circle_mask2)
    sift1 = AllSteps.draw_sift_keypoints(img1, kp1)
    sift2 = AllSteps.draw_sift_keypoints(img2, kp2)
    #cv2.imshow("Keypoints in image1", sift1)
    AllSteps.display_SideBySide(sift1, sift2, "Keypoints")

    '''
    #Step 7: Feature matching
    matches = AllSteps.match_features(desc1, desc2, ratio_threshold=0.75, distance_threshold=190)
    print(matches)
    top_matches = matches[0:10]
    matched_img = AllSteps.draw_matches_with_distances(img1OG, kp1, img2, kp2, matches)
    cv2.imshow("Matches", matched_img)
    cv2.waitKey(0)
    '''

    #Step 12: Shi-Tomasi Corner Detection
    #Shi-Tomasi corner detection
    keypoints1 = detect_ShiTomasi_keypoints(grey1, mask=circle_mask1)
    keypoints2 = detect_ShiTomasi_keypoints(grey2, mask=circle_mask2)
    
    sift1 = AllSteps.draw_sift_keypoints(img1, keypoints1)
    sift2 = AllSteps.draw_sift_keypoints(img2, keypoints2)
    #cv2.imshow("Keypoints in image1", sift1)
    AllSteps.display_SideBySide(sift1, sift2, "Shi-Tomasi Keypoints")
    
    #Step 13: Perform SIFT
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.compute(grey1, keypoints1)
    kp2, desc2 = sift.compute(grey2, keypoints2)

    #Step 14: Feature matching
    matches = AllSteps.match_features(desc1, desc2, ratio_threshold=0.75, distance_threshold=190)
    print(matches)
    matched_img = AllSteps.draw_matches_with_distances(img1, kp1, img2, kp2, matches)
    cv2.imshow("Matches", matched_img)
    cv2.waitKey(0)

    #Step 15: Get top matches and triangulate
    top_matches = matches[0:10]
    # Get matched keypoint coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in top_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in top_matches])

    pts3d = triangulate_pts.triangulate_points(pts1, pts2, P1, P2)
    print(pts3d)
    for pt3d, match in zip(pts3d, top_matches):
        print(f"3D Point: {pt3d}, Match distance: {match.distance:.2f}")

    cv2.waitKey(0)

    #Step 17: RANSAC plane fitting

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

