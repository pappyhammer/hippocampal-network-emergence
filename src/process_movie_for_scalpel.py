import numpy as np
import hdf5storage
import os
import time
import PIL
from PIL import ImageSequence
import scipy.io as sio

def main():
    data_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/scalpel/video_to_process/"
    data_path = "/home/julien/these_inmed/scalpel/video_to_process/"

    movie_file_name = None

    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(data_path):
        for file_name in local_filenames:
            if (file_name.endswith(".tif") or file_name.endswith(".tiff")) and (not file_name.startswith(".")):
                movie_file_name = file_name
        break
    if movie_file_name is None:
        print("no movie found")
        return

    start_time = time.time()
    im = PIL.Image.open(data_path + movie_file_name)
    n_frames = len(list(ImageSequence.Iterator(im)))
    dim_x, dim_y = np.array(im).shape
    print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")
    n_pixels = dim_x * dim_y
    tiff_movie = np.zeros((n_frames, dim_x, dim_y), dtype="uint16")
    for frame, page in enumerate(ImageSequence.Iterator(im)):
        tiff_movie[frame] = np.array(page)
    stop_time = time.time()
    print(f"Time for loading movie: "
          f"{np.round(stop_time - start_time, 3)} s")

    chunck_it = True

    if chunck_it:
        size_chunck = 500
        frame_indices = np.arange(0, n_frames, size_chunck)

        for i, frame_index in enumerate(frame_indices):
            y = np.zeros((n_pixels, size_chunck))
            for y_frame, frame in enumerate(np.arange(frame_index, frame_index+size_chunck)):
                y[:, y_frame] = tiff_movie[frame, :, :].flatten('F')
            sio.savemat(f"{data_path}/" + f"Y_{i+1}.mat",
                        {'video': y})
    else:
        y = np.zeros((n_pixels, n_frames))
        for frame in np.arange(n_frames):
            y[:, frame] = tiff_movie[frame, :, :].flatten('F')
        sio.savemat(f"{data_path}/" + f"Y_1.mat",
                    {'video': y})
main()
