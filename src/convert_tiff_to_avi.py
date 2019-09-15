import os
import numpy as np
import PIL
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, destroyAllWindows
from time import time

def main():

    tiffs_path_dir = '/media/julien/Not_today/hne_not_today/data/test_behavior_movie/a000'
    results_path = '/media/julien/Not_today/hne_not_today/data/test_behavior_movie/results'

    files_in_dir = [item for item in os.listdir(tiffs_path_dir)
                        if os.path.isfile(os.path.join(tiffs_path_dir, item))]
    # sort by alaphabatical order
    files_in_dir.sort()

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
    for tiff_index, tiff_file in enumerate(files_in_dir):
        if (tiff_index > 0) and (tiff_index % 5000 == 0):
            print(f"{tiff_index} frames done")
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