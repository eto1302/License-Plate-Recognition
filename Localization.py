import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

def get_next_filename(folder):
    i = 1
    while True:
        filename = os.path.join(folder, f"{i}.jpg")
        if not os.path.exists(filename):
            return filename
        i += 1

def plate_detection(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 60, 100])
    upper_yellow = np.array([40, 255, 200])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = improveMask(mask)

    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_HSV2RGB)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:            
        largest_contour = max(contours, key=cv2.contourArea)

        largest_contour_mask = np.zeros_like(mask)
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        x1, y1, w1, h1 = cv2.boundingRect(largest_contour)
        x2, y2, w2, h2 = 0,0,1,1

        mask_without_largest = cv2.bitwise_xor(mask, largest_contour_mask)
        contours, _ = cv2.findContours(mask_without_largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            second_largest_contour = max(contours, key=cv2.contourArea)

            second_largest_contour_mask = np.zeros_like(mask)
            cv2.drawContours(second_largest_contour_mask, [second_largest_contour], -1, 255, thickness=cv2.FILLED)

            x2, y2, w2, h2 = cv2.boundingRect(second_largest_contour)
    else:
        x1, y1, w1, h1 = 0,0,1,1
        x2, y2, w2, h2 = 0,0,1,1

    firstArea = w1 * h1
    secondArea = w2 * h2

    if firstArea < 100:        
        x1, y1, w1, h1 = 0,0,1,1
    if secondArea < firstArea * 0.75:
        x2, y2, w2, h2 = 0,0,1,1

    cropped_image_largest = image[y1:y1 + h1, x1:x1 + w1]
    cropped_image_second_largest = image[y2:y2 + h2, x2:x2 + w2]

    fig, axs = plt.subplots(2, 2, figsize=(20, 8))

    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')

    axs[0, 1].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('Yellow Filtered Image')

    axs[1, 0].imshow(cv2.cvtColor(cropped_image_largest, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Largest Cluster')

    axs[1, 1].imshow(cv2.cvtColor(cropped_image_second_largest, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title('Second Largest Cluster')

    # axs[1, 0].hist(image.flatten(), bins=256, color='red', alpha=0.7, rwidth=0.8)
    # axs[1, 0].set_title('Histogram - Original Image')

    # axs[1, 1].hist(filtered_image.flatten(), bins=256, color='yellow', alpha=0.7, rwidth=0.8)
    # axs[1, 1].set_title('Histogram - Yellow Filtered Image')

    # axs[1, 2].hist(cropped_image_largest.flatten(), bins=256, color='green', alpha=0.7, rwidth=0.8)
    # axs[1, 2].set_title('Histogram - Largest Cluster')

    # axs[1, 3].hist(cropped_image_second_largest.flatten(), bins=256, color='blue', alpha=0.7, rwidth=0.8)
    # axs[1, 3].set_title('Histogram - Second Largest Cluster')

    save_path = "LocalizationLogs"

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(get_next_filename(save_path))
    else:
        plt.show()

    # plt.show(block=False)

    # # Pause for 5 seconds
    # plt.pause(1)

    plt.close()

    return [x1, y1, x1+w1, y1+h1], [x2, y2, x2+w2, y2+h2]