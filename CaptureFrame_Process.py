import cv2
import os
import sys
import numpy as np
import pandas as pd
import Localization
import Recognize
''
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

    
    with open(save_path, "a") as output:        
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
            
            firstPlate_text = Recognize.segment_and_recognize(firstPlate)
            output.write(F"{firstPlate_text}, {frame_number}, {timestamp}\n")

            secondPlate_text = Recognize.segment_and_recognize(secondPlate)
            if(secondPlate_text != ""):
                output.write(F"{secondPlate_text}, {frame_number}, {timestamp}\n")


    # Release the video capture object
    video.release()
