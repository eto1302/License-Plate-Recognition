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
	mask = cv2.dilate (mask, n4)  
	mask = cv2.erode(mask, n8)
	return mask

def get_next_filename(folder):
	i = 1
	while True:
		filename = os.path.join(folder, f"{i}.jpg")
		if not os.path.exists(filename):
			print(i)
			return filename
		i += 1


def cropImage(image):
    topRows = np.any(image > 200, axis=1)
    bottomRows = np.any(image[::-1] > 200, axis=1)

    firstRow = np.argmax(topRows)
    lastRow = len(image) - 1 - np.argmax(bottomRows)

    return image[firstRow:lastRow + 1, :]

def segment_and_recognize(image):
	"""
	In this file, you will define your own segment_and_recognize function.
	To do:
		1. Segment the plates character by character
		2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
		3. Recognize the character by comparing the distances
	Inputs:(One)
		1. plate_imgs: cropped plate images by Localization.plate_detection function
		type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
	Outputs:(One)
		1. recognized_plates: recognized plate characters
		type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
	Hints:
		You may need to define other functions.
	"""
	if not image.any() or (image.any() and image.shape[0] * image.shape[1] == 1):
		return ""
	frameNumber = ""
	greyscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# greyscaleImage = cropImage(greyscaleImage)
	# greyscaleImage = cv2.equalizeHist(greyscaleImage)
	greyscaleImage = (255 / (np.max(greyscaleImage) - np.min(greyscaleImage))) * (greyscaleImage - np.min(greyscaleImage)) 

	# TODO change the coefficient, when the plates are rotated properly
	# print(np.mean(greyscaleImage))
	ret,greyscaleImage = cv2.threshold(greyscaleImage,0.77 * np.mean(greyscaleImage),255,cv2.THRESH_BINARY_INV)

	threshold = np.mean(greyscaleImage) * 0.8

	ret,foreground = cv2.threshold(greyscaleImage,threshold,255,cv2.THRESH_BINARY_INV)

	foreground = improveMask(foreground)

	fig, axs = plt.subplots(1, 2, figsize=(20, 8))

	axs[0].imshow(greyscaleImage)
	axs[0].set_title('Original Image')

	axs[1].imshow(foreground)
	axs[1].set_title('Foreground Image')

	save_path = "SegmentationLogs"

	if save_path:
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		plt.savefig(get_next_filename(save_path))
	else:
		plt.show()

	plt.close()

	return frameNumber
