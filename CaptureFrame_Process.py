import cv2
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Localization
import Recognize

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
                output.write(f"{plate}, {frame_number}, {timestamp}\n")  
                
            plate, secondOut = Recognize.segment_and_recognize(secondPlate)   
            
            if(secondOut is not None):
                output.write(f"{plate}, {frame_number}, {timestamp}\n")  
                
    video.release()
