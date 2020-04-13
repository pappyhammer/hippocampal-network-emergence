import PIL
from ScanImageTiffReader import ScanImageTiffReader
from PIL import ImageSequence
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tifffile
from bisect import bisect_right

def load_tiff_movie(tiff_file_name):
    """
    Load a tiff movie from tiff file name.
    Args:
        tiff_file_name:

    Returns: a 3d array: n_frames * width_FOV * height_FOV

    """
    try:
        # start_time = time.time()
        tiff_movie = ScanImageTiffReader(tiff_file_name).data()
        # stop_time = time.time()
        # print(f"Time for loading movie with ScanImageTiffReader: "
        #       f"{np.round(stop_time - start_time, 3)} s")
    except Exception as e:
        im = PIL.Image.open(tiff_file_name)
        n_frames = len(list(ImageSequence.Iterator(im)))
        dim_y, dim_x = np.array(im).shape
        tiff_movie = np.zeros((n_frames, dim_y, dim_x), dtype="uint16")
        for frame, page in enumerate(ImageSequence.Iterator(im)):
            tiff_movie[frame] = np.array(page)
    return tiff_movie

def binarized_frame(movie_frame, filled_value=1, percentile_threshold=90,
                    threshold_value=None, with_uint=True, with_4_blocks=False):
    """
    Take a 2d-array and return a binarized version, thresholding using a percentile value.
    It could be filled with 1 or another value
    Args:
        movie_frame:
        filled_value:
        percentile_threshold:
        with_4_blocks: make 4 square of the images

    Returns:

    """
    img = np.copy(movie_frame)
    if threshold_value is None:
        threshold = np.percentile(img, percentile_threshold)
    else:
        threshold = threshold_value

    img[img < threshold] = 0
    img[img >= threshold] = filled_value

    if with_uint:
        img = img.astype("uint8")
    else:
        img = img.astype("int8")
    return img


if __name__ == '__main__':
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/tbi_microglia_github/"
    root_path = "/media/julien/Not_today/"
    data_path = os.path.join(root_path, "data/")

    results_path = os.path.join(root_path, "results")
    # time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    #
    # results_path = os.path.join(results_path, time_str)
    # os.mkdir(results_path)

    tiff_file_name = "/media/julien/Not_today/hne_not_today/data/axon_flex/p12_19_02_08_a003_part1.tif"
    movie = load_tiff_movie(tiff_file_name=tiff_file_name)
    print(movie.shape)

    mean_img = np.mean(movie, axis=0)

    percentile_threshold = 70
    threshold_value = np.percentile(mean_img, percentile_threshold)

    binary_frame = binarized_frame(movie_frame=mean_img, filled_value=1, threshold_value=None,
                                   percentile_threshold=percentile_threshold, with_uint=False,
                                   with_4_blocks=True)
    test_display = True

    if test_display:
        print(f"threshold_value {threshold_value}")
        plt.imshow(binary_frame, cmap=cm.Greys)
        plt.show()
