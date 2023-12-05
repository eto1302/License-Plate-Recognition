import pandas as pd
import argparse
import numpy as np
import os
import CaptureFrame_Process

treshold = 0.85

def get_coordinates():
	input_folder = "dataset\TrainingSet\Categorie I"
	output_file = 'dataset/OutputCoordinates.csv'
	video_files = [f for f in os.listdir(input_folder) if f.endswith('.avi')]

	if os.path.exists(output_file):
		os.remove(output_file)

	for video_file in video_files:
		file_path = input_folder + "\\" + video_file
		CaptureFrame_Process.CaptureFrame_Process(file_path, 36, output_file)

def get_args():
	# ground truth header: '#', 'Category', 'Video name', 'x0', 'y0', 'x1', 'y1'
	parser = argparse.ArgumentParser()
	parser.add_argument('--student_file_path', type=str, default='dataset/OutputCoordinates.csv')
	parser.add_argument('--ground_truth_path', type=str, default='dataset/groundTruthCoordinates.csv')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_args()

	get_coordinates()

	student_results = pd.read_csv(args.student_file_path)
	ground_truth = pd.read_csv(args.ground_truth_path)	
	totalInput = len(student_results['Video name'])
	correctCoordinates = 0

	# For each line in the input list
	for i in range(totalInput):
		student_videoName = student_results['Video name'][i]
		student_x0 = student_results['x0'][i]
		student_y0 = student_results['y0'][i]
		student_x1 = student_results['x1'][i]
		student_y1 = student_results['y1'][i]
		gt_videoName = ground_truth['Video name'][i] 
		gt_x0 = ground_truth['x0'][i]
		gt_y0 = ground_truth['y0'][i]
		gt_x1 = ground_truth['x1'][i]
		gt_y1 = ground_truth['y1'][i]
		if(student_videoName != gt_videoName):
			continue
		
		x0 = min(student_x0, gt_x0)
		x1 = max(student_x1, gt_x1)
		y0 = min(student_y0, gt_y0)
		y1 = max(student_y1, gt_y1)

		intersection = max(0, (x1 - x0)) * max(0,(y1 - y0))

		student_box = abs((student_x1 - student_x0) * (student_y1 - student_y0))
		gt_box = abs((gt_x1 - gt_x0) * (gt_y1 - gt_y0))

		iou = intersection / (student_box + gt_box - intersection + 1e-6)

		print(f"Video: {student_videoName}, IOU: {iou}, Intersection: {intersection}, Student Box: {student_box}, GT Box: {gt_box}")

		if(iou >= treshold):
			correctCoordinates+=1

	accuracy = correctCoordinates / totalInput
	print(f"Accuracy: {accuracy}")