import os
import pickle
import cv2
import numpy as np

# ---------- User settings ----------
IMAGE_PATH = "greenBase2.jpg"
# How many clicks we expect (X1,X2,Y1,Y2,Center)
LABELS = ["TopLeft", "MidLeft", "BotLeft", "TopRight", "MidRight", "BotRight", "h1", "h2", "j1", "j2"]
# -----------------------------------

selected_points = []          # raw pixel coords where user clicked
clicked_overlay = None        # image to draw live click markers on
selection_done = False        # set True when finished
current_label_idx = 0

def detect_circles(img):
    """Return Nx3 array of circles (x,y,r) or None if none found."""
    greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(greyimg, (3, 3), 1)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=10,
        param1=100, param2=0.8, minRadius=0, maxRadius=0
    )
    if circles is None:
        return None
    # convert to Nx3 float32
    circles = np.squeeze(circles).astype(np.float32)
    if circles.ndim == 1:
        circles = circles[np.newaxis, :]
    return circles  # shape (N,3): x,y,r

def mouse_callback(event, x, y, flags, param):
    """Mouse callback: record clicks and draw markers on overlay image."""
    global selected_points, current_label_idx, clicked_overlay, selection_done

    if event == cv2.EVENT_LBUTTONDOWN and current_label_idx < len(LABELS):
        selected_points.append((int(x), int(y)))
        label = LABELS[current_label_idx]
        # Draw on overlay (so we can keep original base image intact)
        cv2.circle(clicked_overlay, (x, y), 6, (0, 0, 255), -1)          # filled red dot
        cv2.putText(clicked_overlay, label, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        current_label_idx += 1
        print(f"Selected {label}: ({x}, {y})")
        # If done, set flag but DON'T destroy windows here
        if current_label_idx == len(LABELS):
            selection_done = True
            print("✅ All circles selected. Finalizing...")
            
def user_circle_selection(base_img):
    """Show detected circles, let user click 5 points in order, then display final annotated result."""
    global clicked_overlay, selected_points, selection_done, current_label_idx
    
    # reset
    selected_points = []
    current_label_idx = 0
    selection_done = False

    img = base_img.copy()
    detected = detect_circles(img)

    # Draw detected circles (green) and indexes (white)
    if detected is not None and len(detected) > 0:
        # convert to ints for drawing
        det_int = np.round(detected).astype(np.int32)
        for i, (cx, cy, r) in enumerate(det_int):
            cv2.circle(img, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            cv2.circle(img, (int(cx), int(cy)), 2, (0, 0, 255), 3)
            cv2.putText(img, str(i), (int(cx) + 6, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        print(f"Detected {len(detected)} circles. Click these 8 points in order:")
    else:
        print("No circles detected by Hough. You can still click manually.")
        detected = None

    print("  1) top left corner")
    print("  2) middle left")
    print("  3) bottom left corner")
    print("  4) top right corner")
    print("  5) middle right")
    print("  6) bottom right corner")
    print("  7) left hand side base attachment pt")
    print("  6) closer attachment point to middle hole")
    print("Click with left mouse button. Press ESC anytime to cancel.")

    # Make an overlay copy for drawing click markers so we can still re-show unmodified image
    clicked_overlay = img.copy()
    window_name = "Select Circles (click 8)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, clicked_overlay)
    cv2.setMouseCallback(window_name, mouse_callback, clicked_overlay)

    # Main GUI loop — update display until user has clicked or pressed ESC
    while True:
        cv2.imshow(window_name, clicked_overlay)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC cancels
            print("Selection cancelled by user (ESC).")
            break
        if selection_done:
            break

    # Finalize: show final annotated image with both clicks and nearest detected centers (if available)
    final = img.copy()

    # Draw clicked points (red) and labels (green)
    for idx, pt in enumerate(selected_points):
        cv2.circle(final, pt, 6, (0, 0, 255), -1)
        cv2.putText(final, LABELS[idx], (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # If detected circles exist, match each clicked point to the nearest detected circle center
    matched_centers = []
    if detected is not None:
        centers = detected[:, :2]  # Nx2
        for pt in selected_points:
            pt_arr = np.array(pt, dtype=np.float32)
            dists = np.linalg.norm(centers - pt_arr, axis=1)
            idx_near = int(np.argmin(dists))
            matched_centers.append(tuple(map(int, centers[idx_near])))
            # draw matched center as cyan circle and label its index
            cx, cy = matched_centers[-1]
            cv2.circle(final, (cx, cy), 6, (255, 255, 0), 2)
            cv2.putText(final, f"C{idx_near}", (cx - 20, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        print("\nNearest detected circle centers to your clicks (if any):")
        for lab, click_pt, center_pt in zip(LABELS, selected_points, matched_centers):
            print(f" {lab}: clicked {click_pt} -> matched center {center_pt}")
    else:
        print("\nNo detected circle centers to match; using clicked pixel coords only.")

    # Show final annotated image until keypress
    final_win = "Final Selection (press any key to close)"
    cv2.namedWindow(final_win, cv2.WINDOW_NORMAL)
    cv2.imshow(final_win, final)
    cv2.waitKey(0)
    return matched_centers, detected

def simple_fit_line(points):
    pts = np.array(points)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    return vx,vy,x0,y0

def camera_robot_calibration_show_clicks(base_img, camera_matrix, distortion_coefficients):
    display = base_img.copy()
    #This gives the coordinates of the circles in the camera frame, but only xy coordinates
    matched_centers, detected_circles = user_circle_selection(base_img)
    print(matched_centers)

    #However, I know where they should be in world space by measuring the actual base.
    #ACTUALLY I could have chosen any known points. It may be better to choose some other ones actually.
    #In mm
    object_points = np.array([
        [-109.982, 59.944, 0.0], #topL
        [-109.982, 0.0, 0.0], #midl
        [-109.982, -59.944 , 0.0], #botL
        [109.982, 59.944, 0.0], #topR
        [109.982, 0, 0.0], #midR
        [109.982, -59.944, 0.0], #botR
        [-59.055,0,0], #h1
        [-43.815,0,0], #h2
        [29.5148, -51.1431302256, 0.0],
        [21.9075, -37.9449030788, 0.0]
    ], dtype=np.float32)

    
    #Now I can use solvePnP to find the transformation for these points from world to camera
    image_points = np.array(matched_centers, dtype=np.float32)

    #object_points = object_points[0:6]
    #image_points = image_points[0:6]
    object_points = object_points[[1,4,6,6,8,9]]
    image_points = image_points[[1,4,6,6,8,9]]
    ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients, flags=cv2.SOLVEPNP_IPPE)
    
    print("Object points ", object_points)
    print("Image points ", image_points)

    print("Ret: ", ret)
    print("rvec: ", rvec)
    print("tvec: ", tvec)


    #I want to visualize the points I circled to make sure they are correct
    temp_disp_img = base_img.copy()
    axis = np.float32([[40,0,0], [0,-40,0], [0,0,-40]])
    origin = np.float32([[0,0,0]])  
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, distortion_coefficients)
    origin_imgpt = cv2.projectPoints(origin, rvec, tvec, camera_matrix, distortion_coefficients)
    temp_disp_img = drawAxes(temp_disp_img, origin_imgpt, imgpts)
    imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coefficients)
    # Flatten and cast to Python int tuples outside the loop
    imgpts_int = [tuple(int(float(c)) for c in pt.ravel()) for pt in imgpts]

    # Debug: print all points before drawing
    print("Projected points (as Python int tuples):")
    for p in imgpts_int:
        print(p)
    # Draw all points
    for pt in imgpts_int:
        cv2.circle(temp_disp_img, pt, 2, (0, 0, 255), -1)
    cv2.imshow('Test', temp_disp_img)
    cv2.waitKey(0)




def drawAxes(img, origin_imgpts, imgpts):
    def tupleOfInts(arr):
        return tuple(int(x) for x in arr)
    origin = tuple(int(x) for x in origin_imgpts[0].ravel())
    print("origin: ", origin)
    print("axesx end point: ", tupleOfInts(imgpts[0].ravel()))
    img = cv2.line(img, origin, tupleOfInts(imgpts[0].ravel()), (255,0,0),5)
    img = cv2.line(img, origin, tupleOfInts(imgpts[1].ravel()), (0,255,0),5)
    img = cv2.line(img, origin, tupleOfInts(imgpts[2].ravel()), (0,0,255),5)
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
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Couldn't open '{IMAGE_PATH}'")
    PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'calibration_params_home_webcam/0', 'calibration_data.pkl'))
    PARAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'catkin_ws/src/vs_stewart/calibration_params_home_webcam/0/calibration_data.pkl'))
    print(PARAM_DIR)
    cameraMat1, distCoeffs1, R, T = read_params(PARAM_DIR)
    print("cameraMatrix1: ")
    print(cameraMat1)
    print("distortionCoeffs1:")
    print(distCoeffs1)
    result = camera_robot_calibration_show_clicks(img, cameraMat1, distCoeffs1)


if __name__ == "__main__":
    main()