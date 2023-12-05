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
    hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsi_image, np.array([7, 122, 122]), np.array([30, 255, 255]))

    # Improve mask using morphology    
    n8 = np.array([     [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], np.uint8)
    
    # Improve the mask using morphological dilation and erosion
    mask = cv2.erode(mask, n8)
    mask = cv2.erode(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)   
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8) 
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)

    # Return coordinates
    hsi_image = hsi_image[:, :, 0]
    image_with_mask = np.bitwise_and(hsi_image, mask)
    nonzero_indices = np.argwhere(image_with_mask)
    

    if(not nonzero_indices.any()):
        return [0,0,0,0]

    top_left = np.min(nonzero_indices, axis=0)
    bottom_right = np.max(nonzero_indices, axis=0)

    x_min = top_left[0]
    y_min = top_left[1]

    x_max = bottom_right[0]
    y_max = bottom_right[1]


    coordinates = np.array([x_min, y_min, x_max, y_max])
    return coordinates
