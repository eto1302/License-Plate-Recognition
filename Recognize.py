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
			return filename
		i += 1

def crop_unnecessary_borders(image):
	while(True):
		height, width = image.shape[:2]
		toRemoveH = None
		for ind, currRow in enumerate(image):
			if(np.sum(currRow) == 0 or np.sum(currRow) == 255 * width):
				toRemoveH = ind
				break
		if(toRemoveH is not None):
			if(toRemoveH < height / 2):
				image = image[toRemoveH + 1:]			
			elif(toRemoveH >= height / 2):
				image = image[:toRemoveH]	
		if(toRemoveH is None):
			break
	
	nonzero_indices = np.nonzero(np.sum(image, axis=0))[0]
	if len(nonzero_indices) == 0:
		return image
	left = nonzero_indices[0]
	right = nonzero_indices[-1]
	image = image[:, left:right]
	return image

def indices_to_crop(image):
	height, width = image.shape[:2]
	indices = []
	for i in range(width):
		all_zero = np.all(image[:, i] == 0)
		if all_zero:
			indices.append(i)
	indices.append(width)
	indices = np.array(indices)
	return indices

def split_image(image, indices):
	height, width = image.shape[:2]
	characters = []
	number_characters_detected = 0
	# Split the image based on zero columns
	prev_index = 0
	for index in  indices:
		if index - prev_index > 0.07 * width:
			char = image[:, prev_index : index]
			characters.append(crop_unnecessary_borders(char))
			prev_index = index
			number_characters_detected += 1

	# characters = np.array(characters)
	return characters, int(number_characters_detected)

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
		reference_characters[char] = crop_unnecessary_borders(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
		file += 1

	# Load the numbers
	file = 0
	for i in range(17, 27):
		char = sample_characters[i]
		path = f"dataset/SameSizeNumbers/{file}.bmp"
		reference_characters[char] = crop_unnecessary_borders(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
		file += 1

	return sample_characters, reference_characters

def recognize_character(character, sample_characters, reference_characters):
	if character.shape[0] == 0 or character.shape[1] == 0:
		return "-"

	lowest_score = 99999999
	character_match = None
	second_lowest_score = 99999999
	second_character_match = None
	for char in sample_characters:

		reference_characters[char] = reshape_to_image(reference_characters[char].astype(character.dtype), character)
		xor = cv2.bitwise_xor(character, reference_characters[char])
		score = np.count_nonzero(xor)
		# print(char, score)
		if score < lowest_score:
			second_lowest_score = lowest_score
			lowest_score = score
			second_character_match = character_match
			character_match = char
		elif score < second_lowest_score:
			second_lowest_score = score
			second_character_match = char
		
		# plt.figure(figsize=(15, 5))

		# plt.subplot(1, 3, 1)
		# plt.imshow(character, cmap='gray')
		# plt.title('Character')

		# plt.subplot(1, 3, 2)
		# plt.imshow(reference_characters[char], cmap='gray')
		# plt.title('Current Character: ' + str(score))

		# plt.subplot(1, 3, 3)
		# plt.imshow(reference_characters[character_match], cmap='gray')
		# plt.title('Character Match: ' + str(lowest_score))

		# plt.show(block=False)
		# plt.pause(2)
		# plt.close()

	print(character_match, lowest_score)
	return character_match, lowest_score, second_character_match, second_lowest_score

def reshape_to_image(image1, image2):
	target_height, target_width = image2.shape[:2]
	return cv2.resize(image1, (target_width, target_height))

def reshape_found_characters(found_character):
	template = cv2.imread(f"dataset/SameSizeLetters/1.bmp")

	template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	if found_character.shape[0] != 0 and found_character.shape[1] != 0:
		template_height, template_width = template.shape[:2]
		found_height, found_width = found_character.shape[:2]

		width_difference = found_width - template_width

		found_character = cv2.resize(found_character, (found_width, template_height))

		if width_difference < 0:
			width_difference = abs(width_difference)
			while width_difference > 0:
				found_character = np.c_[found_character, np.zeros(template_height)]
				width_difference -= 1
		else:
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

	greyscaleImage = crop_unnecessary_borders(greyscaleImage)
	greyscaleImage = cv2.equalizeHist(greyscaleImage)
	# greyscaleImage = (255 / (np.max(greyscaleImage) - np.min(greyscaleImage))) * (greyscaleImage - np.min(greyscaleImage)) 

	# TODO change the coefficient, when the plates are rotated properly
	# print(np.mean(greyscaleImage))
	ret,greyscaleImage = cv2.threshold(greyscaleImage,0.65 * np.mean(greyscaleImage),255,cv2.THRESH_BINARY)
	greyscaleImage = cv2.bitwise_not(greyscaleImage)

	ret, foreground = cv2.threshold(greyscaleImage, np.mean(greyscaleImage) * 0.5, 255, cv2.THRESH_BINARY)

	cropped = crop_unnecessary_borders(foreground)
	improved_cropped = improveMask(cropped)

	indices = indices_to_crop(improved_cropped)
	plate_characters, number_of_characters = split_image(improved_cropped, indices)

	sample_characters, reference_characters = load_sample_images()
	
	num_cols_first_row = max(number_of_characters, 3)

	fig, axs = plt.subplots(2, num_cols_first_row, figsize=(20, 16))
    
	axs[0, 0].imshow(greyscaleImage, cmap='gray')
	axs[0, 0].set_title('Original Image')

	axs[0, 1].imshow(foreground, cmap='gray')
	axs[0, 1].set_title('Foreground Image')

	axs[0, 2].imshow(improved_cropped, cmap='gray')
	axs[0, 2].set_title('Improved Foreground Image')

	# Second row
	plate_number = ""
	for i in range(number_of_characters):
		if plate_characters[i].shape[0] == 0 or plate_characters[i].shape[1] == 0:
			continue

		# plate_characters[i] = reshape_found_characters(plate_characters[i])
		#print(recognize_character(plate_characters[i], sample_characters, reference_characters))
		match, score, second_match, second_score = recognize_character(plate_characters[i], sample_characters, reference_characters)
		plate_number += match

		axs[1, i].imshow(plate_characters[i])
		axs[1, i].set_title(f'{match} : {score}\n {second_match} : {second_score}')

	# plt.show(block=False)
	# plt.pause(3)

	print(plate_number)

	save_path = "SegmentationLogs"

	if save_path:
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		plt.savefig(get_next_filename(save_path))
	else:
		plt.show()

	plt.close()

	return frameNumber
