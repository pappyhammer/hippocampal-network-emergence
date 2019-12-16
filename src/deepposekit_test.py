import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
# from deepposekit.io import VideoReader, DataGenerator, initialize_dataset
# from deepposekit.annotate import KMeansSampler
import tqdm
import glob
import pandas as pd
from deepposekit import Annotator
from os.path import expanduser
import glob
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, destroyAllWindows, VideoCapture
import os

from os.path import expanduser


HOME = ""


def change_frame_resolution(frame, new_width, new_height):
    if frame.shape[1] > new_width and frame.shape[0] > new_height:
        return frame[:new_height, :new_width]
    return frame


def change_video_resolution(new_width=1792, new_height=1024):
    """
    Change a video resolution from 1920*1200 to 1792*1024
    Or add black pixels and extend to : 2048*1280
    Returns:

    """
    fpath = "/media/julien/Not_today/hne_not_today/data/deepposekit_test/"
    file_name = "behavior_p8_19_09_29_1_cam_22983298_cam1_a000_fps_20.avi"
    new_avi_file_name = "new_res_" + file_name
    file_name = os.path.join(fpath, file_name)
    new_avi_file_name = os.path.join(fpath, new_avi_file_name)

    # TO READ
    cap = VideoCapture(file_name)
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        return
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_avi = cap.get(cv2.CAP_PROP_FPS)

    # for writing
    is_color = True
    # put fourcc to 0 for no compression
    # fourcc = 0
    fourcc = VideoWriter_fourcc(*"XVID")

    size_avi = (new_width, new_height)
    vid_avi = VideoWriter(new_avi_file_name, fourcc, fps_avi, size_avi, is_color)

    n_frames = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if n_frames == 0:
            print(f"frame.shape {frame.shape}")

        if ret == True:
            frame = change_frame_resolution(frame=frame, new_width=new_width, new_height=new_height)
            if n_frames == 0:
                print(f"new frame.shape {frame.shape}")
            n_frames += 1
            vid_avi.write(frame)
        # Break the loop
        else:
            break
        if n_frames % 5000 == 0:
            print(f"{n_frames} converted over {length}")
    print(f"{n_frames} frames converted")
    # Closes all the frames
    cv2.destroyAllWindows()
    # When everything done, release the video capture object
    vid_avi.release()
    cap.release()

def prepare_annotation():
    pass

def annotate():
    # ANNOTATE
    """

    Annotation Hotkeys
    +- = rescale image by ±10%
    left mouse button = move active keypoint to cursor location
    WASD = move active keypoint 1px or 10px
    space = change WASD mode (swaps between 1px or 10px movements)
    JL = next or previous image
    <> = jump 10 images forward or backward
    I,K or tab, shift+tab = switch active keypoint
    R = mark image as unannotated ("reset")
    F = mark image as annotated ("finished")
    esc or Q = quit
    """

    app = Annotator(datapath=HOME + '/deepposekit-data/datasets/fly/example_annotation_set.h5',
                    dataset='images',
                    skeleton=HOME + '/deepposekit-data/datasets/fly/skeleton.csv',
                    shuffle_colors=False,
                    text_scale=0.2)
    app.run()


if __name__ == "__main__":
    change_video_resolution()
    # prepare_annotation()
    # annotate()
