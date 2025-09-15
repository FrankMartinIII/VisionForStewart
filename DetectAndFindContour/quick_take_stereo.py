import cv2
import os
from datetime import datetime

def capture_stereo_images(cam_index_L=0, cam_index_R=1, save_dir='stereo_images'):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Open both cameras
    capL = cv2.VideoCapture(cam_index_L)
    capR = cv2.VideoCapture(cam_index_R)

    if not capL.isOpened() or not capR.isOpened():
        print("Error: Could not open both cameras.")
        return

    print("Press 'c' to capture, 'q' to quit.")
    img_counter = 0

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if not retL or not retR:
            print("Warning: Couldn't read from one of the cameras.")
            continue

        # Show live preview
        cv2.imshow('Left Camera', frameL)
        cv2.imshow('Right Camera', frameR)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"capture_{timestamp}"

            fileL = os.path.join(save_dir, base_name + '_L.png')
            fileR = os.path.join(save_dir, base_name + '_R.png')

            cv2.imwrite(fileL, frameL)
            cv2.imwrite(fileR, frameR)

            print(f"Captured: {fileL}, {fileR}")
            img_counter += 1

        elif key == ord('q'):
            print("Exiting.")
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_stereo_images(cam_index_L=4, cam_index_R=6)
