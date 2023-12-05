import cv2
import numpy as np


def plate_detection(image):
    """
    In this file, you need to define plate_detection function.
    To do:
        1. Localize the plates and crop the plates
        2. Adjust the cropped plate images
    Inputs:(One)
        1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
        type: Numpy array (imread by OpenCV package)
    Outputs:(One)
        1. plate_imgs: cropped and adjusted plate images
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Hints:
        1. You may need to define other functions, such as crop and adjust function
        2. You may need to define two ways for localizing plates(yellow or other colors)
    """

    # Color segmentation
    # Create mask
    hsi_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsi_image, np.array([0, 0, 0]), np.array([40, 200, 255]))

    # Improve mask using morphology
    n8 = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)

    # Return coordinates
    hsi_image = hsi_image[:, :, 0]
    image_with_mask = np.bitwise_and(hsi_image, mask)
    # TODO: make it work for rotated licence plates
    image_with_mask = np.nonzero(image_with_mask)
    y = image_with_mask[1]
    x = image_with_mask[0]
    y_min = np.argmin(y)
    y_max = np.argmax(y)
    x_min = np.argmin(x)
    x_max = np.argmax(x)
    coordinates = np.array([x_min, x_max, y_min, y_max])
    return coordinates
