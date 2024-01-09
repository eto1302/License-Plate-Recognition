import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def improveMask(mask):
	n8 = np.array([     [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], np.uint8)
	n4 = np.array([     [0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], np.uint8)

	mask = cv2.dilate(mask, n4)
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

def crop_unnecessary_horizontal_borders(image):
	while(True):
		height, width = image.shape[:2]
		toRemove = None
		for ind, currRow in enumerate(image):
			if(np.sum(currRow) == 0 or np.sum(currRow) == 255 * width):
				toRemove = ind
				break

		if(toRemove is None):
			break

		if(toRemove < height / 2):
			image = image[toRemove + 1:]			
		elif(toRemove >= height / 2):
			image = image[:toRemove]
	return image

def indices_to_crop(image):
	height, width = image.shape[:2]
	indices = []
	for i in range(width):
		non_zero_elem = np.count_nonzero(image[:, i])
		if non_zero_elem == height:
			indices.append(i)
	indices = np.array(indices)
	return indices

def split_image(image, indices):
	characters = []
	number_characters_detected = 0
	# Split the image based on zero columns
	prev_index = 0
	for index in  indices:
		if index - prev_index > 5:
			char = image[:, prev_index: index]
			characters.append(char)
			prev_index = index
			number_characters_detected += 1

	# characters = np.array(characters)
	return characters, int(number_characters_detected)

# def cropImage(image):
#     topRows = np.any(image > 200, axis=1)
#     bottomRows = np.any(image[::-1] > 200, axis=1)
#
#     firstRow = np.argmax(topRows)
#     lastRow = len(image) - 1 - np.argmax(bottomRows)
#
#     return image[firstRow:lastRow + 1, :]

def load_sample_images():
	sample_characters = np.array(
		['B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '0', '1', '2', '3', '4',
		 '5', '6', '7', '8', '9'])

	reference_characters = {}

	# Load the letters
	file = 1
	for i in range(17):
		char = sample_characters[i]
		path = f"dataset/SameSizeLetters/{file}.bmp"
		reference_characters[char] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
		file += 1

	# Load the numbers
	file = 0
	for i in range(17, 27):
		char = sample_characters[i]
		path = f"dataset/SameSizeNumbers/{file}.bmp"
		reference_characters[char] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
		file += 1

	return sample_characters, reference_characters

def recognize_character(character, sample_characters, reference_characters):
	character = np.array(reshape_found_characters(character))
	lowest_score = 99999999
	character_match = None
	for char in sample_characters:
		xor = cv2.bitwise_xor(character, reference_characters[char])
		score = np.count_nonzero(xor)
		if score < lowest_score:
			lowest_score = score
			character_match = char
	return character_match

def reshape_found_characters(found_character):
	template = cv2.imread(f"dataset/SameSizeLetters/1.bmp")
	template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	template_height, template_width = template.shape[:2]

	found_character = cv2.resize(found_character, (template_width, template_height))

	return found_character
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

	cropped = crop_unnecessary_horizontal_borders(foreground)
	# improved_cropped = improveMask(cropped)

	# indices = indices_to_crop(improved_cropped)
	# plate_characters, number_of_characters = split_image(improved_cropped, indices)
	# print(f"Number of characters found = {number_of_characters}")

	sample_characters, reference_characters = load_sample_images()

	fig, axs = plt.subplots(1, 3, figsize=(20, 8))
	axs[0].imshow(greyscaleImage)
	axs[0].set_title('Original Image')

	axs[1].imshow(foreground)
	axs[1].set_title('Foreground Image')

	axs[2].imshow(cropped)
	axs[2].set_title('Improved Foreground Image')


	# if number_of_characters > 1:
	# 	axs[1].imshow(plate_characters[0])
	# 	axs[1].set_title('A')

	# 	axs[2].imshow(plate_characters[1])
	# 	axs[2].set_title('B')
	# else:
	# 	axs[1].imshow(cropped)
	# 	axs[1].set_title('Foreground Image')

	# 	axs[2].imshow(improved_cropped)
	# 	axs[2].set_title('Improved Foreground Image')

	# plate_number = ""
	# for char in plate_characters:
	# 	plate_number += recognize_character(char, sample_characters, reference_characters)
	#
	# print(plate_number)

	# save_path = "SegmentationLogs"

	# if save_path:
	# 	if not os.path.exists(save_path):
	# 		os.makedirs(save_path)
	# 	plt.savefig(get_next_filename(save_path))
	# else:
	# 	plt.show()

	plt.show(block=False)

	plt.pause(3)

	plt.close()

	return frameNumber
