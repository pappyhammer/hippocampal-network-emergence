import os
import numpy as np
import PIL
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, destroyAllWindows, VideoCapture
from time import time
from sortedcontainers import SortedDict
import cv2
import yaml
# import pims

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
    file_name = "behavior_P6 19_12_11_0"
    path_data = f"/media/julien/My Book/robin_tmp/cameras/p9_19_09_30/{file_name}"
    path_data = "/media/julien/Not_today/hne_not_today/data/p8/p8_19_09_29_1_a001/cams/behavior_p8_19_09_29_1_cam_22983298_cam1_a001_fps_20.avi"
    path_data = "/media/julien/Not_today/hne_not_today/data/p8/p8_19_09_29_1_a001/cams/behavior_p8_19_09_29_1_cam_23109588_cam2_a001_fps_20.avi"
    path_data = "/media/julien/Not_today/hne_not_today/data/behavior_movies/converted_so_far/p6_20_01_09/behavior_p6_20_01_09_cam_23109588_cam2_a002_fps_20.avi"
    path_data = "/media/julien/Not_today/hne_not_today/data/behavior_movies/dlc_predictions/p7_200103_200110_200110_a000_2020_02/data/behavior_p7_20_01_10_cam_22983298_cam1_a000_fps_20.avi"

    file_name = "behavior_p7_20_03_13_rem_cam_23109588_cam2_a001_fps_20.avi"
    path_data = f"/media/julien/Not_today/hne_not_today/data/behavior_movies/converted_so_far/{file_name}"

    # 37254
    # vs = pims.Video(path_data)
    # print(f"vs.frame_shape {vs.frame_shape}")

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

    print(f"file: {file_name}")
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
    # test_file = "/run/user/1000/gvfs/smb-share:server=cossartlab.local,share=picardoteam/Behavior Camera/p5_20_02_17/cam 1"
    # print(f"is dir {os.path.isdir(test_file)}")
    # return

    open_avi_for_test = False
    if open_avi_for_test:
        test_avi()
        return

    subject_id = "p8_20_02_27" # P12_20_01_20 p8_20_01_16
    cam_folder_id_1 = "cam2" # "cam2"
    cam_folder_id_2 = "a001" # a000  a001
    if cam_folder_id_2 is None:
        cam_folder_id = "20190430_a002"  # ex cam1_a002, movie1, etc...
    else:
        cam_folder_id = f"{cam_folder_id_1}_{cam_folder_id_2}"
    tiffs_path_dir = '/media/julien/My Book/robin_tmp/cameras/'
    tiffs_path_dir = '/media/julien/My Book/robin_tmp/cameras/to_convert/'
    # tiffs_path_dir = '/media/julien/My Book/robin_tmp/cameras/basler_recordings/'
    # tiffs_path_dir = '/media/julien/dream team/camera/'
    tiffs_path_dir = '/media/julien/Not_today/hne_not_today/data/behavior_movies/to_convert/'
    # On NAS
    # tiffs_path_dir = '/run/user/1000/gvfs/smb-share:server=cossartlab.local,share=picardoteam/Behavior Camera/'
    if cam_folder_id_2 is not None:
        tiffs_path_dir = os.path.join(tiffs_path_dir, subject_id, cam_folder_id_1, cam_folder_id_2)
        # tiffs_path_dir = os.path.join(tiffs_path_dir, subject_id, cam_folder_id_2, cam_folder_id_1)
    else:
        tiffs_path_dir = os.path.join(tiffs_path_dir, subject_id, cam_folder_id)
    # print(f"is dir {os.path.isdir(tiffs_path_dir)}")
    if cam_folder_id_1 is None:
        cam_id = "22983298"
    elif cam_folder_id_1 == "cam1":
        cam_id = "22983298"
    else:
        cam_id = "23109588"  #  cam1: 22983298  cam2: 23109588

    # results_path = '/media/julien/My Book/robin_tmp/cameras/'
    # results_path = os.path.join(results_path, subject_id)
    results_path = "/media/julien/Not_today/hne_not_today/data/behavior_movies/converted_so_far/"

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

    # looking for a gap between frames
    last_tiff_frame = 0
    error_detected = False
    for tiff_frame, tiff_file in files_in_dir_dict.items():
        if tiff_frame - 1 != last_tiff_frame:
            print(f"Gap between frame n° {last_tiff_frame} and {tiff_frame}. File {tiff_file}")
            error_detected = True
        last_tiff_frame = tiff_frame

    if error_detected:
        raise Exception("ERROR: gap between 2 frames")

    # keep the name of the tiffs files
    yaml_file_name = os.path.join(results_path, f"behavior_{subject_id}_cam_{cam_id}_{cam_folder_id}.yaml")
    with open(yaml_file_name, 'w') as outfile:
        yaml.dump(list(files_in_dir_dict.values()), outfile, default_flow_style=False)

    # raise Exception("TEST YAML")
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
    fps_avi = 20
    avi_file_name = os.path.join(results_path, f"behavior_{subject_id}_cam_{cam_id}_{cam_folder_id}_fps_{fps_avi}.avi")
    print(f"creating behavior_{subject_id}_cam_{cam_id}_{cam_folder_id}_fps_{fps_avi}.avi from {len(files_in_dir_dict)} tiff files")
    is_color = True
    # put fourcc to 0 for no compression
    # fourcc = 0
    fourcc = VideoWriter_fourcc(*"XVID")
    # fourcc = VideoWriter_fourcc(*"MPEG")

    # https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
    start_time = time()
    for tiff_frame, tiff_file in files_in_dir_dict.items():
        if (tiff_frame > 0) and (tiff_frame % 5000 == 0):
            print(f"{tiff_frame} frames done")
        # img = PIL.Image.open(os.path.join(tiffs_path_dir, tiff_file))
        # img = np.array(img)
        if vid_avi is None:
            if size_avi is None:
                img = PIL.Image.open(os.path.join(tiffs_path_dir, tiff_file))
                img = np.array(img)
                print(f"img.shape {img.shape}")
                size_avi = img.shape[1], img.shape[0]
            # vid_avi = VideoWriter(avi_file_name, fourcc, float(fps_avi), size_avi, is_color)
            vid_avi = VideoWriter(avi_file_name, fourcc, fps_avi, size_avi, is_color)
        # vid_avi.write(img)
        vid_avi.write(imread(os.path.join(tiffs_path_dir, tiff_file)))
    cv2.destroyAllWindows()
    vid_avi.release()

    time_to_convert = time() - start_time

    print(f"time_to_convert: {time_to_convert} sec")


if __name__ == "__main__":
    main()
