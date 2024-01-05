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
	mask = cv2.erode (mask, n4)  
	mask = cv2.dilate(mask, n8) 
	return mask

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
	# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# low_range = np.array([10, 30, 20])
	# high_range = np.array([80, 255, 120])

	# mask = cv2.inRange(hsv, low_range, high_range)
	# mask = improveMask(mask)

	# filtered_image = cv2.bitwise_and(image, image, mask=mask)
	# filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_HSV2BGR)
	greyscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# TODO change the coefficient, when the plates are rotated properly
	threshold = np.mean(greyscaleImage) * 0.75

	ret,foreground = cv2.threshold(greyscaleImage,threshold,255,cv2.THRESH_BINARY_INV)

	foreground = improveMask(foreground)

	fig, axs = plt.subplots(2, 2, figsize=(20, 8))

	axs[0, 0].imshow(cv2.cvtColor(greyscaleImage, cv2.COLOR_BGR2RGB))
	axs[0, 0].set_title('Original Image')

	axs[0, 1].imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
	axs[0, 1].set_title('Foreground Image')

	save_path = "SegmentationLogs"

	# plt.show(block=False)

	# plt.pause(3)

	# plt.close()

	return frameNumber
