import cv2
import os
from datetime import datetime

def capture_stereo_images(cam_index_L=0, save_dir='temp_imgs'):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Open both cameras
    capL = cv2.VideoCapture(cam_index_L, cv2.CAP_V4L2)

    

    if not capL.isOpened():
        print("Error: Could not open camera.")
        return

    capL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = capL.read()
    print("Frame shape:", frame.shape)
    print("Actual width:", capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Actual height:", capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Press 'c' to capture, 'q' to quit.")
    img_counter = 0

    while True:
        retL, frameL = capL.read()


        # Show live preview
        cv2.imshow('Camera', frameL)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"capture_{timestamp}"

            fileL = os.path.join(save_dir, base_name + '.png')

            cv2.imwrite(fileL, frameL)

            print(f"Captured: {fileL}")
            img_counter += 1

        elif key == ord('q'):
            print("Exiting.")
            break

    capL.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_stereo_images(cam_index_L=4)
