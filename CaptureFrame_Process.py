import cv2
import os
import sys
import numpy as np
import pandas as pd
import Localization
import Recognize
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim

def combine(init, toAdd):
    for curr in toAdd:
        key = curr[0]
        value = curr[1]
        if key in init:
            first = init[key]
            init[key] = (first + value) / 2
        else:
            init[key] = value

    return init

def sameScene(frame1, frame2, threshold=0.8):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(gray1, gray2, full=True)
    print(score)
    return score >= threshold

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
            
            
            plate = Recognize.segment_and_recognize(firstPlate)   
            print(plate)
            if(plate is not None):
                output.write(f"{plate}, {frame_number}, {timestamp}\n")  
                
            plate = Recognize.segment_and_recognize(secondPlate)   
            print(plate)
            if(plate is not None):
                output.write(f"{plate}, {frame_number}, {timestamp}\n") 
                
            if(prev is not None):
                print(sameScene(prev, frame))
            prev = frame


    # Release the video capture object
    video.release()
