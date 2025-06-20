import cv2
import numpy as np
import os
import sys

def capture_stereo_calib_images(camNum0, camNum1):
    OUTPUT_DIRECTORY0 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_images', 'stereo0'))
    OUTPUT_DIRECTORY1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_images', 'stereo1'))
    CHESSBOARD_SIZE = (6,5)
    if not os.path.exists(OUTPUT_DIRECTORY0):
        os.makedirs(OUTPUT_DIRECTORY0)
    if not os.path.exists(OUTPUT_DIRECTORY1):
        os.makedirs(OUTPUT_DIRECTORY1)

    #Open camera CHANGE HERE AND BELOW
    cap0 = cv2.VideoCapture(camNum0)
    cap1 = cv2.VideoCapture(camNum1)

    if not cap0.isOpened():
        print("Error: could not open camera ", camNum0)
        return
    if not cap1.isOpened():
        print("Error: could not open camera ", camNum1)
        return
    

    # Get camera resolution
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera0 resolution: {width}x{height}")
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera1 resolution: {width}x{height}")

    imgCounter = 0
    while True:
        ret0, frame0 = cap0.read()
        rawFrame0 = frame0
        if not ret0:
            print("Cam0 Failed to capture image")
            break
        ret1, frame1 = cap1.read()
        rawFrame1 = frame1
        if not ret1:
            print("Cam1 Failed to capture image")
            break

        #Convert image to greyscale
        greyImg0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        greyImg1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        #cv2.imshow("TEST", greyImg)
        #cv2.waitKey(0)

        #Find chessboard corners
        ret_chess0, corners0 = cv2.findChessboardCorners(greyImg0, CHESSBOARD_SIZE, None)
        ret_chess1, corners1 = cv2.findChessboardCorners(greyImg1, CHESSBOARD_SIZE, None)
        #Draw corners if found
        if ret_chess0:
            rawFrame0 = frame0.copy()
            cv2.drawChessboardCorners(frame0, CHESSBOARD_SIZE, corners0, ret_chess0)
            cv2.putText(frame0, "Chessboard detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if ret_chess1:
            rawFrame1 = frame1.copy()
            cv2.drawChessboardCorners(frame1, CHESSBOARD_SIZE, corners1, ret_chess1)
            cv2.putText(frame1, "Chessboard detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        print("Image ", imgCounter, " captured")
        cv2.imshow("Camera calibration0", frame0)
        cv2.imshow("Camera calibration1", frame1)

        key = cv2.waitKey(1) & 0xFF

        # 'q' or Escape to quit
        if key == ord('q') or key == 27:  # 27 is the ASCII code for Escape
            print("Exiting...")
            break
        
        # 'c' to capture
        elif key == ord('c'):
            # Save the images
            img_name0 = os.path.join(OUTPUT_DIRECTORY0, f"calibration_{imgCounter:02d}.jpg")
            img_name1 = os.path.join(OUTPUT_DIRECTORY1, f"calibration_{imgCounter:02d}.jpg")
            cv2.imwrite(img_name0, rawFrame0)
            cv2.imwrite(img_name1, rawFrame1)
            print(f"Captured {img_name0}")
            print(f"Captured {img_name1}")
            
            imgCounter += 1

    #end while
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    print("Captured ", imgCounter, " images")

if __name__ == "__main__":
    capture_stereo_calib_images(0,2)