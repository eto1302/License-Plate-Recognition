import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def mostCommonColor(image):
    if not image.any() or (image.any() and image.shape[0] * image.shape[1] == 1):
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    pixels = hsv.reshape((-1, 3))

    mask = pixels[:, 2] >= 100

    filtered_pixels = pixels[mask]

    return np.mean(filtered_pixels, axis=0).astype(np.uint8)


def rotate_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edge_thresholds = np.arange(50, 255, 20)
    line_thresholds = np.arange(100, 50, -20)

    for edge_threshold in edge_thresholds:
        for line_threshold in line_thresholds:
            edges = cv2.Canny(gray, 50, edge_threshold)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, line_threshold)

            if lines is not None:
                dominant_line = lines[0][0]
                angle = np.degrees(dominant_line[1]) - 90
                center = tuple(np.array(image.shape[1::-1]) / 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

                # plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')

                # rho, theta = dominant_line
                # a = np.cos(theta)
                # b = np.sin(theta)
                # x0 = a * rho
                # y0 = b * rho
                # x1 = int(x0 + 10 * (-b))
                # y1 = int(y0 + 10 * (a))
                # x2 = int(x0 - 10 * (-b))
                # y2 = int(y0 - 10 * (a))
                # plt.subplot(122), plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)), plt.title('Rotated Image')
                # plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)

                # plt.show()
                # plt.pause(2)
                # plt.close()

                return rotated_image

    # print("No lines detected")
    return None


def improveMask(mask):
    n8 = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], np.uint8)
    n4 = np.array([[0, 1, 0],
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
def gaussianFilter(img, kernelSize, sigma):
    result = np.zeros((kernelSize, kernelSize), dtype=float)
    for row in range(kernelSize):
        for col in range(kernelSize):
            coeff = 1 / (2 * np.pi * sigma ** 2)
            exp = -(row ** 2 + col ** 2) / (sigma ** 2)
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


def twoBiggestPlates(image):
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
        x2, y2, w2, h2 = 0, 0, 1, 1

        mask_without_largest = cv2.bitwise_xor(mask, largest_contour_mask)
        contours, _ = cv2.findContours(mask_without_largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            second_largest_contour = max(contours, key=cv2.contourArea)

            second_largest_contour_mask = np.zeros_like(mask)
            cv2.drawContours(second_largest_contour_mask, [second_largest_contour], -1, 255, thickness=cv2.FILLED)

            x2, y2, w2, h2 = cv2.boundingRect(second_largest_contour)
    else:
        x1, y1, w1, h1 = 0, 0, 1, 1
        x2, y2, w2, h2 = 0, 0, 1, 1

    firstArea = w1 * h1
    secondArea = w2 * h2

    if firstArea < 100:
        x1, y1, w1, h1 = 0, 0, 1, 1
    if secondArea < firstArea * 0.75:
        x2, y2, w2, h2 = 0, 0, 1, 1

    cropped_image_largest = image[y1:y1 + h1, x1:x1 + w1]
    cropped_image_second_largest = image[y2:y2 + h2, x2:x2 + w2]

    rotated_cropped_largest = rotate_image(cropped_image_largest)
    rotated_cropped_second_largest = rotate_image(cropped_image_second_largest)
    return rotated_cropped_largest, rotated_cropped_second_largest


def cropPlate(plate):
    height, width = plate.shape[:2]

    top_crop = int(height * 0.2)
    bottom_crop = int(height * 0.2)
    left_crop = int(width * 0.1)
    right_crop = int(width * 0.1)

    cropped_image = plate[top_crop:(height - bottom_crop), left_crop:(width - right_crop)]

    return cropped_image


def plate_detection(image):
    first, second = twoBiggestPlates(image)

    if (first is not None):
        first = cropPlate(first)
    if (second is not None):
        second = cropPlate(second)

    # fig, axs = plt.subplots(2, 2, figsize=(20, 8))

    # axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axs[0, 0].set_title('Original Image')

    # axs[1, 0].imshow(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
    # axs[1, 0].set_title('Largest Cluster')

    # axs[1, 1].imshow(cv2.cvtColor(second, cv2.COLOR_BGR2RGB))
    # axs[1, 1].set_title('Second Largest Cluster')

    # save_path = "LocalizationLogs"

    # if save_path:
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     plt.savefig(get_next_filename(save_path))
    # else:
    #     plt.show()

    # # plt.show(block=False)

    # # # Pause for 5 seconds
    # # plt.pause(1)

    # plt.close('all')

    return first, second
