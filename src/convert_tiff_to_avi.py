import os
import numpy as np
import PIL
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, destroyAllWindows
from time import time
from sortedcontainers import SortedDict

import os


def sorted_tiff_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime

    files_in_dir = [item for item in os.listdir(path)
                    if os.path.isfile(os.path.join(path, item)) and
                    (item.endswith("tiff") or item.endswith("tif")) and (not item.startswith("."))]

    return list(sorted(files_in_dir, key=mtime))


def main():
    tiffs_path_dir = '/media/julien/Not_today/hne_not_today/data/test_behavior_movie/a000'
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
    avi_file_name = os.path.join(results_path, "test.avi")
    fps_avi = 20
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
