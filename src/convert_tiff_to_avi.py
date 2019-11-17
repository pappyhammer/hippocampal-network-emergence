import os
import numpy as np
import PIL
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, destroyAllWindows, VideoCapture
from time import time
from sortedcontainers import SortedDict
import cv2
import pims

import os


def sorted_tiff_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime

    files_in_dir = [item for item in os.listdir(path)
                    if os.path.isfile(os.path.join(path, item)) and
                    (item.endswith("tiff") or item.endswith("tif")) and (not item.startswith("."))]

    return list(sorted(files_in_dir, key=mtime))

def test_avi():
    # loading the root_path
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")
    path_data = os.path.join(root_path, "data/test_behavior_movie/results/behavior_test_cam_test_fps_50.avi")

    vs = pims.Video(path_data)
    print(f"vs.frame_shape {vs.frame_shape}")

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = VideoCapture(path_data)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        return

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"length {length}, width {width}, height {height}, fps {fps}")
    return

    n_frames = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            n_frames += 1

            # # Display the resulting frame
            # cv2.imshow('Frame', frame)
            #
            # # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        # Break the loop
        else:
            break
    print(f"n_frames {n_frames}")
    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def main():
    open_avi_for_test = True
    if open_avi_for_test:
        test_avi()
        return

    tiffs_path_dir = '/media/julien/Not_today/hne_not_today/data/test_behavior_movie/a000'
    subject_id = "test"
    cam_id = "test"

    # subject_id = "p8_19_09_29_1_a001"
    # cam_id = "22983298"
    # cam_id = "23109588"
    # tiffs_path_dir = f'/media/julien/Not_today/hne_not_today/data/p8/{subject_id}/cams/{cam_id}'

    results_path = '/media/julien/Not_today/hne_not_today/data/test_behavior_movie/results'

    files_in_dir = [item for item in os.listdir(tiffs_path_dir)
                    if os.path.isfile(os.path.join(tiffs_path_dir, item)) and
                    (item.endswith("tiff") or item.endswith("tif")) and (not item.startswith("."))]

    # files_in_dir = sorted_tiff_ls(tiffs_path_dir)
    # print(f"len(files_in_dir) {len(files_in_dir)}")
    # for file_name in files_in_dir[-1000:]:
    #     print(f"{file_name}")

    files_in_dir_dict = SortedDict()
    for file_name in files_in_dir:
        index_ = file_name[::-1].find("_")
        frame_number = int(file_name[-index_:-5])
        files_in_dir_dict[frame_number] = file_name
        # print(f"{file_name[-index_:-5]}")
        # break


    # # leave only regular files, insert creation date
    # entries = ((stat[ST_CTIME], path)
    #            for stat, path in entries if S_ISREG(stat[ST_MODE]))
    # # NOTE: on Windows `ST_CTIME` is a creation date
    # #  but on Unix it could be something else
    # # NOTE: use `ST_MTIME` to sort by a modification date
    #
    # for cdate, path in sorted(entries):
    #     print(time.ctime(cdate), os.path.basename(path))

    # sort by alaphabatical order

    size_avi = None
    vid_avi = None
    fps_avi = 1
    avi_file_name = os.path.join(results_path, f"behavior_{subject_id}_cam_{cam_id}_fps_{fps_avi}.avi")
    print(f"creating behavior_{subject_id}_cam_{cam_id}_fps_{fps_avi}.avi from {len(files_in_dir_dict)} tiff files")
    is_color = True
    # put fourcc to 0 for no compression
    # fourcc = 0
    fourcc = VideoWriter_fourcc(*"XVID")
    # fourcc = VideoWriter_fourcc(*"MPEG")

    # https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
    start_time = time()
    for tiff_frame, tiff_file in files_in_dir_dict.items():
        # temporary for testing
        if tiff_frame > 300:
            break

        if (tiff_frame > 0) and (tiff_frame % 5000 == 0):
            print(f"{tiff_frame} frames done")
        # img = PIL.Image.open(os.path.join(tiffs_path_dir, tiff_file))
        # img = np.array(img)
        if vid_avi is None:
            if size_avi is None:
                img = PIL.Image.open(os.path.join(tiffs_path_dir, tiff_file))
                img = np.array(img)
                size_avi = img.shape[1], img.shape[0]
            # vid_avi = VideoWriter(avi_file_name, fourcc, float(fps_avi), size_avi, is_color)
            vid_avi = VideoWriter(avi_file_name, fourcc, fps_avi, size_avi, is_color)
        # vid_avi.write(img)
        vid_avi.write(imread(os.path.join(tiffs_path_dir, tiff_file)))
    destroyAllWindows()
    vid_avi.release()

    time_to_convert = time() - start_time

    print(f"time_to_convert: {time_to_convert} sec")


if __name__ == "__main__":
    main()
