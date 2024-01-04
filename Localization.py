import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 100, 122])
    upper_yellow = np.array([40, 255, 200])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_HSV2BGR)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (largest cluster of nonzero values)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    largest_contour_mask = np.zeros_like(mask)
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the filtered image to get the largest cluster
    largest_cluster_image = cv2.bitwise_and(filtered_image, filtered_image, mask=largest_contour_mask)

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding box
    cropped_image = filtered_image[y:y + h, x:x + w]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')

    axs[0, 1].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('Yellow Filtered Image')

    axs[0, 2].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title('Largest Cluster')

    axs[1, 0].hist(image.flatten(), bins=256, color='red', alpha=0.7, rwidth=0.8)
    axs[1, 0].set_title('Histogram - Original Image')

    axs[1, 1].hist(filtered_image.flatten(), bins=256, color='yellow', alpha=0.7, rwidth=0.8)
    axs[1, 1].set_title('Histogram - Yellow Filtered Image')

    axs[1, 2].hist(cropped_image.flatten(), bins=256, color='green', alpha=0.7, rwidth=0.8)
    axs[1, 2].set_title('Histogram - Largest Cluster')

    fig.canvas.manager.full_screen_toggle()
    plt.show(block=False)

    # Pause for 5 seconds
    plt.pause(5)

    # Close the plot
    plt.close()
    # hsi_image = gaussianFilter(hsi_image, 3, 2)
    # mask = cv2.inRange(hsi_image, np.array([17, 140, 122]), np.array([30, 255, 200]))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # mask = improveMask(mask)

    # hsi_image = hsi_image[:, :, 0]
    # image_with_mask = np.bitwise_and(hsi_image, mask)
    return [x, y, x+w, y+h]