"""
This script prepares the urbansas data for evaluation
"""

import pandas as pd 
import numpy as np
import cv2 as cv
import os 
import sys
import argparse
from tqdm import tqdm
import librosa 
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")



def main():

    vid_annot = pd.read_csv(vid_annot_path)
    aud_annot = pd.read_csv(aud_annot_path)

    # filter offscreen and non-identifiable vehicle sounds
    aud_annot = aud_annot[aud_annot["label"] != "offscreen"]
    aud_annot = aud_annot[aud_annot["non_identifiable_vehicle_sound"] == 0]
    
    # filter negative values for bbox coordinates
    vid_annot = vid_annot[(vid_annot["x"]>=0)&(vid_annot["w"]>=0)]
    vid_annot = vid_annot[(vid_annot["y"]>=0)&(vid_annot["h"]>=0)]

    #filter video frames where the vehicle isn't visible
    vid_annot = vid_annot[vid_annot["visibility"] == 1]

    # filter bounding boxes that don't have corresponding audio annotations
    vid_annot["has_audio"] = [has_aud_annot(aud_annot,
                                            fname,
                                            tstamp) for fname, tstamp in zip(vid_annot.filename, vid_annot.time)]
    
    vid_annot = vid_annot[vid_annot["has_audio"] == True]

    print(f"Number of annotations:\n\
          Video - {len(vid_annot)}\
          Audio - {len(aud_annot)}")

    # make a list of all video files 
    filenames = np.array(vid_annot["filename"].unique())
    print(f"Processing {len(filenames)} videos...")

    # process all videos 
    for vid in tqdm(filenames): 

        # load video
        vid_path = os.path.join(video_dir, f"{vid}.mp4")
        cap=cv.VideoCapture(vid_path)                     
        nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # load audio
        aud_path = os.path.join(audio_dir, f"{vid}.wav")
        aud, sr = librosa.load(aud_path)

        # get annotations for the video
        sub_annot = vid_annot[vid_annot["filename"] == vid]
        sub_annot["frame_num"] = [round(i) for i in sub_annot["time"]*fps]

        # TODO: Interpolate annotations 

        for frame_num in sorted(sub_annot["frame_num"]):
            # get annotations for frame 
            frame_annot = sub_annot[sub_annot["frame_num"] == frame_num]
            frame_id = frame_num//4
            # get starting point of the audio (1s snippet around the image)
            start = frame_num/fps - 0.5
            # get image 
            cap.set(1,frame_id) 
            ret, img = cap.read()

            if ret:
                # save audio
                aud_seg = aud[int(start*sr):int(start*sr)+sr]
                
                aud_name = f"{vid}_{frame_num}.wav"
                sf.write(f"data/urbansas/Data/{aud_name}", aud_seg, sr)

                # save image 
                img_name = f"{vid}_{frame_num}.jpg"
                cv.imwrite(f"data/urbansas/Data/{img_name}", img)

                # write label file
                annot = create_annot_txt(frame_annot)
                annot_name = f"{vid}_{frame_num}.txt"
                with open (f"data/urbansas/Annotations/{annot_name}", "w") as filehandler:
                    filehandler.writelines(annot)



# helper functions

def create_annot_txt(annotations):
    annot = []
    for i in range(annotations.shape[0]):
        row = annotations.iloc[i]
        annot.append(" ".join(["0",
                                str(row["x"]),
                                str(row["y"]),
                                str(row["w"]),
                                str(row["h"])]))
    annot = [i+"\n" for i in annot]
    return annot

def has_aud_annot(aud_annot, filename, timestamp):
    has_audio = False
    file_annot = aud_annot[aud_annot.filename == filename]

    # timestamp should lie between start and end
    file_annot = file_annot[file_annot["end"] > timestamp]
    file_annot = file_annot[file_annot["start"] < timestamp]

    if len(file_annot) > 0:
        has_audio = True
    return has_audio


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare the urbansas dataset')
    parser.add_argument("-fps", default=8, type=int)
    parser.add_argument("-d", help='path to urbansas dataset', required=True)
    fps = parser.parse_args().fps           
    dataset = parser.parse_args().d



    video_dir = f"{dataset}/video/video_2fps/"
    audio_dir = f"{dataset}/audio/"
    vid_annot_path = f"{dataset}/annotations/video_annotations.csv"
    aud_annot_path = f"{dataset}/annotations/audio_annotations.csv"

    # setup directories 
    dirs = [f"data/urbansas/Data",
            f"data/urbansas/Annotations"]

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

    main()
