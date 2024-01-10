import argparse
import os
import CaptureFrame_Process
import numpy
import sys
import shutil


# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_path', type=str, default='dataset/trainingvideo.avi')
	parser.add_argument('--output_path', type=str, default='dataset/Output.csv')
	parser.add_argument('--sample_frequency', type=int, default=36)
	args = parser.parse_args()
	return args


# In this file, you need to pass three arguments into CaptureFrame_Process function.
if __name__ == '__main__':
	numpy.set_printoptions(threshold=sys.maxsize)
	args = get_args()
	if args.output_path is None:
		output_path = os.getcwd()
	else:
		output_path = args.output_path
	file_path = args.file_path
	sample_frequency = args.sample_frequency
	if os.path.exists("LocalizationLogs"):
		shutil.rmtree("LocalizationLogs")
	if os.path.exists("SegmentationLogs"):
		shutil.rmtree("SegmentationLogs")
	CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path)
