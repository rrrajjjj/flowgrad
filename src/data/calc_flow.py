import numpy as np
import cv2 as cv
import os 
import argparse
from tqdm import tqdm
from glob import glob
import warnings
warnings.filterwarnings("ignore")


def main():
    
    # make a list of all video files 
    img_filenames = [os.path.basename(i)[:-4] for i in glob("data/urbansas/Data/*jpg")]
    filenames = np.array(list(set(["_".join(i.split("_")[:-1]) for i in img_filenames])))

    # create a dictionary mapping filenames to the list of frames for which to calculate flow
    file_frame_dict = {}
    for f in img_filenames:
        fname = "_".join(f.split("_")[:-1])
        frame = f.split("_")[-1]
        if fname not in file_frame_dict:
            file_frame_dict[fname] = []
        file_frame_dict[fname].append(int(frame))
    
    print(f"Processing {len(filenames)} videos...")

    # process all videos 
    for vid in tqdm(filenames): 

        # load video
        vid_path = os.path.join(video_dir, f"{vid}.mp4")
        cap=cv.VideoCapture(vid_path)                     

        for frame_num in sorted(file_frame_dict[vid]):
            # get image 
            cap.set(1,frame_num) 
            _, img = cap.read()
            ret, img_next = cap.read()

            if ret:
                # convert images to grayscale
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img_next = cv.cvtColor(img_next, cv.COLOR_BGR2GRAY)

                # calculate optical flow
                flow = cv.calcOpticalFlowFarneback(img, img_next,
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
                # Compute the magnitude and angle of the flow vectors
                magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

                # save optical flow 
                fname = f"{vid}_{frame_num}.jpg"
                cv.imwrite(f"data/urbansas/Flow/{fname}", magnitude)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Calculate dense optical flow for urbansas')
    parser.add_argument("-d", help='path to urbansas dataset', required=True)


    dataset = parser.parse_args().d


    video_dir = f"{dataset}/video/video_8fps/"
       
    # setup directories 
    flow_dir = f"data/urbansas/Flow/"   
    if not os.path.isdir(flow_dir):
        os.makedirs(flow_dir)

    main()
