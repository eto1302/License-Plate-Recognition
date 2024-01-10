import cv2
import os
import sys
import numpy as np
import pandas as pd
import Localization
import Recognize
from collections import defaultdict

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

    
    with open(save_path, "w") as output:        
        output.write("License plate,Frame no.,Timestamp(seconds)\n")
        previousFrame = None
        hasSecond = False
        cnt = 0
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
            
            cnt += 1
            # TODO: Implement actual algorithms for Localizing Plates
            # The plate_detection function should return the coordinates of detected plates
            firstPlate, secondPlate = Localization.plate_detection(frame)
            
            first_scores = [{},{},{},{},{},{},{},{}]
            second_scores = [{},{},{},{},{},{},{},{}]
            
            scores = Recognize.segment_and_recognize(firstPlate)   
            if(scores is not None):
                for ind, curr in enumerate(first_scores):
                    curr = combine(curr, scores[ind])             
            
            scores = Recognize.segment_and_recognize(secondPlate)
            if(scores is not None):
                hasSecond = True
                for ind, curr in enumerate(second_scores):
                    curr = combine(curr, scores[ind])
            print(cnt)
            if(cnt == 3):
                cnt = 0
                print("CHANGED")
                firstPlate_text = ""
                for scores_dict in first_scores:
                    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1])

                    min_char = sorted_scores[0][0]

                    firstPlate_text += min_char

                output.write(f"{firstPlate_text}, {frame_number}, {timestamp}\n")

                if(hasSecond):
                    secondPlate_text = ""
                    for scores_dict in second_scores:
                        sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1])

                        min_char = sorted_scores[0][0]

                        secondPlate_text += min_char

                    output.write(f"{secondPlate_text}, {frame_number}, {timestamp}\n")
                
                first_scores = [{},{},{},{},{},{},{},{}]
                second_scores = [{},{},{},{},{},{},{},{}]
                hasSecond = False


    # Release the video capture object
    video.release()
