import cv2
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Localization
import Recognize

def calculate_difference(str1, str2):
    diff = 0
    for ind, currChar in enumerate(str1):
        if(currChar != str2[ind]):
            diff += 1
    return diff
    
def appendIfSimilar(toAdd, plate):
    different = len(toAdd)
    if(different == 0):
        toAdd.append(plate)
        return True       
    
    for existing_plate in toAdd:
        if calculate_difference(existing_plate, plate) <= 1:
            different-=1
    
    # print(toAdd, plate, different)

    if (different == 0):
        toAdd.append(plate)
        return True

    toAdd.sort(key=lambda x: toAdd.count(x), reverse=True)
    
    return False

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
    video = cv2.VideoCapture(file_path)
    fps = video.get(cv2.CAP_PROP_FPS)    
    first = []
    second = []
    firstFrame = 0
    firstTimeStamp = 0
    secondFrame = 0
    secondTimeStamp = 0
    
    startFrame = 0
    
    video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    
    with open(save_path, "w") as output:        
        output.write("License plate,Frame no.,Timestamp(seconds)\n")
        while True:
            ret, frame = video.read()

            if not ret:
                break

            frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = frame_number / fps
            if((frame_number - 1) % sample_frequency != 0): 
                continue
            
            firstPlate, secondPlate = Localization.plate_detection(frame)
            
            plate, firstOut = Recognize.segment_and_recognize(firstPlate)

            if(firstOut is not None):
                if(not appendIfSimilar(first, plate)):
                    output.write(f"{first[0]}, {firstFrame}, {firstTimeStamp}\n")
                    first = []
                    first.append(plate)
                    firstFrame = frame_number
                    firstTimeStamp = timestamp

            plate, secondOut = Recognize.segment_and_recognize(secondPlate)

            if(secondOut is not None):
                if(not appendIfSimilar(first, plate)):
                    output.write(f"{second[0]}, {secondFrame}, {secondTimeStamp}\n")
                    second = []
                    second.append(plate)
                    secondFrame = frame_number
                    secondTimeStamp = timestamp
                
    video.release()
