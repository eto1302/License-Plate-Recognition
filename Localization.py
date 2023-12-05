import cv2
import numpy as np

def improveMask(mask):
    n8 = np.array([     [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], np.uint8)
    n4 = np.array([     [0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], np.uint8)
    
    # Improve the mask using morphological dilation and erosion    
    mask = cv2.erode(mask, n4)  
    mask = cv2.dilate(mask, n8) 
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)   
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)
    mask = cv2.dilate(mask, n8)     
    # Return the improved mask
    return mask

    # Define a function to create a gaussian kernel
def gaussianFilter (img, kernelSize, sigma):
    result = np.zeros((kernelSize, kernelSize), dtype=float)
    for row in range(kernelSize):
        for col in range(kernelSize):
            coeff = 1 / (2 * np.pi * sigma**2)
            exp = -(row**2 + col**2) / (sigma**2)
            result[row, col] = coeff * np.exp(exp)
    result /= np.sum(result)
    return cv2.filter2D(img, ddepth=-1, kernel=np.array(result))

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

    hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsi_image = gaussianFilter(hsi_image, 3, 2)
    mask = cv2.inRange(hsi_image, np.array([17, 140, 122]), np.array([30, 255, 200]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = improveMask(mask)

    hsi_image = hsi_image[:, :, 0]
    image_with_mask = np.bitwise_and(hsi_image, mask)
    nonzero_indices = np.argwhere(image_with_mask)

    if(not nonzero_indices.any()):
        return[0,0,0,0]

    top_left = np.min(nonzero_indices, axis=0)
    bottom_right = np.max(nonzero_indices, axis=0)

    x_min = top_left[0]
    y_min = top_left[1]

    x_max = bottom_right[0]
    y_max = bottom_right[1]

    coordinates = np.array([x_min, y_min, x_max, y_max])
    return coordinates
