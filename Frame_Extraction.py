import cv2
import os
import pandas as pd

def extract_first_frame(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    ret, frame = cap.read()

    cap.release()

    video_name = os.path.basename(video_path)
    frame_filename = os.path.splitext(video_name)[0] + '_first_frame.jpg'
    output_path = output_folder + "\\" + frame_filename
    cv2.imwrite(output_path, frame)

def process_videos(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(input_folder) if f.endswith('.avi')]


    for video_file in video_files:
        extract_first_frame(input_folder + "\\" + video_file, output_folder)


if __name__ == "__main__":
    input_folder = "dataset\TrainingSet\Categorie I"
    output_folder = "dataset\TrainingSet\Categorie I\\frames"

    process_videos(input_folder, output_folder)
