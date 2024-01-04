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
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Initialize variables for saving results
    plate_data = []
    coordinates_data = []

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
        coordinatesFirst, coordinatesSecond = Localization.plate_detection(frame)

        # Save coordinates to csv
        file = os.path.basename(file_path)
        video_name = os.path.splitext(file)[0]
        coordinates_data.append([1, 1, video_name, coordinatesFirst[0], coordinatesFirst[1], coordinatesFirst[2], coordinatesFirst[3],  
                                 coordinatesSecond[0], coordinatesSecond[1], coordinatesSecond[2], coordinatesSecond[3]])

        # TODO: Implement actual algorithms for Recognizing Characters
        # The segment_and_recognize function should return the recognized license plate text
        plate = None
        plate_text = Recognize.segment_and_recognize(plate)
        plate_data.append([plate_text, frame_number, timestamp])

    # Save the results to a CSV file using pandas
    columns = ["License plate", "Frame no.", "Timestamp(seconds)"]
    coordinateColumns=["#","Category","Video name","x0","y0","x1","y1"]
    coordinates_df = pd.DataFrame(coordinates_data, columns=coordinateColumns)
    
    try:
        existing_df = pd.read_csv(save_path)
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    coordinates_df['#'] = range(existing_df.shape[0] + 1, existing_df.shape[0] + coordinates_df.shape[0] + 1)

    # Concatenate the existing DataFrame and the new DataFrame
    updated_df = pd.concat([existing_df, coordinates_df])

    # Save the updated DataFrame to the CSV file
    updated_df.to_csv(save_path, index=False)


    # Release the video capture object
    video.release()
