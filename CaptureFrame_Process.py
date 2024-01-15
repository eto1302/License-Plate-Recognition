import cv2
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Localization
import Recognize
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim

def getConfidence(predicted_probabilities):    
    scores = [pred[1] for pred in predicted_probabilities]
    correct_class_prob = scores[0] + 1e-6
    incorrect_class_prob = scores[1] + 1e-6

    total_prob = correct_class_prob + incorrect_class_prob
    normalized_correct_prob = correct_class_prob / total_prob
    normalized_incorrect_prob = incorrect_class_prob / total_prob

    margin = normalized_correct_prob - normalized_incorrect_prob

    return margin

def combine(init, toAdd):
    confidence = getConfidence(toAdd)
    for curr in toAdd:
        key = curr[0]
        value = curr[1]
        if key in init:
            init[key].append(value / (1 + confidence))
        else:
            init[key] = [value / (1 + confidence)]
    return init

def sameScene(frame1, frame2, threshold=100):
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    avg_hsv1 = np.mean(hsv1, axis=(0, 1))
    avg_hsv2 = np.mean(hsv2, axis=(0, 1))

    diff_hsv = np.linalg.norm(avg_hsv1 - avg_hsv2)

    result = diff_hsv < threshold
    print(diff_hsv)
    return result

def CaptureFrame_Process(file_path, sample_frequency, save_path):
    """
    In this file, you will define your own CaptureFrame_Process funtion. In this function,
    you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
    To do:
        1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
        2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
        3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
    Inputs:(three)
        1. file_path: video path
        2. sample_frequency: second
        3. save_path: final .csv file path
    Output: None
    """

    # We use sampling_frequency, based on frames
    # TODO: Read frames from the video (saved at `file_path`) by making use of `sample_frequency`
    video = cv2.VideoCapture(file_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    prev = None
    first = [{},{},{},{},{},{},{},{}]
    second = [{},{},{},{},{},{},{},{}]
    
    with open(save_path, "w") as output:        
        output.write("License plate,Frame no.,Timestamp(seconds)\n")
        while True:
            # Capture frame
            ret, frame = video.read()

            # Break the loop if the video has ended
            if not ret:
                break

            # Record frame number and timestamp
            frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = frame_number / fps
            if((frame_number - 1) % sample_frequency != 0): 
                continue
            
            # TODO: Implement actual algorithms for Localizing Plates
            # The plate_detection function should return the coordinates of detected plates
            firstPlate, secondPlate = Localization.plate_detection(frame)
            
            plate, firstOut = Recognize.segment_and_recognize(firstPlate)   
            if(firstOut is not None):
                for i, curr in enumerate(first):
                    combine(curr, firstOut[i])
                # plate = [min(d, key=d.get) for d in first]
                # print(plate)
                avgConf = np.average([getConfidence(scores) for scores in firstOut])
                # print(avgConf)
                output.write(f"{plate}, {frame_number}, {timestamp}\n")  
                
            plate, secondOut = Recognize.segment_and_recognize(secondPlate)   
            if(secondOut is not None):
                for i, curr in enumerate(second):
                    combine(curr, secondOut[i])
                # plate = [min(d, key=d.get) for d in secondOut]
                # print(plate)
                avgConf = np.average([getConfidence(scores) for scores in secondOut])
                # print(avgConf)
                output.write(f"{plate}, {frame_number}, {timestamp}\n")  
                
            if(prev is not None and not sameScene(prev, frame)):
                first = [{},{},{},{},{},{},{},{}]
                second = [{},{},{},{},{},{},{},{}]
                fig, axs = plt.subplots(2, figsize=(20, 16))
    
                axs[0].imshow(prev, cmap='gray')
                axs[0].set_title('Previous')

                axs[1].imshow(frame, cmap='gray')
                axs[1].set_title('Frame')
                plt.show()
                plt.pause(1)
                plt.close()
                print("Scene Change!")
            # prev = frame


    # Release the video capture object
    video.release()
