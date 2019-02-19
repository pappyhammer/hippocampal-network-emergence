import numpy as np
from datetime import datetime
from mouse_session_loader import load_mouse_sessions
import pattern_discovery.tools.param as p_disc_tools_param
import scipy.io as sio
import os
import hdf5storage
from pattern_discovery.display.cells_map_module import CoordClass
from mouse_session import MouseSession
from PIL import Image
import matplotlib.pyplot as plt
import tifffile

from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize


class DataForMs(p_disc_tools_param.Parameters):
    def __init__(self, path_data, path_results, time_str=None):
        if time_str is None:
            self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        else:
            self.time_str = time_str
        super().__init__(path_results=path_results, time_str=self.time_str, bin_size=1)
        self.path_data = path_data
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None


def produce_cell_coord_from_cnn_validated_cells(param):
    path_cnn_classifier = "cell_classifier_results_txt/"

    ms_to_use = ["p7_171012_a000_ms", "p8_18_10_24_a005_ms", "p9_18_09_27_a003_ms", "p11_17_11_24_a000_ms",
                 "p12_171110_a000_ms", "p13_18_10_29_a001_ms"]

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False,
                                            for_transient_classifier=True)
    coords_to_keep = []

    for ms in ms_str_to_ms_dict.values():
        path_data = param.path_data

        cnn_file_name = None
        # finding the cnn_file coresponding to the ms
        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(path_data, path_cnn_classifier)):
            for file_name in local_filenames:
                if file_name.endswith(".txt"):
                    if ms.description.lower() in file_name.lower():
                        cnn_file_name = file_name
                        break
            # looking only in the top directory
            break

        if cnn_file_name is None:
            print(f"{ms.description} no cnn file_name")
            continue

        cell_cnn_predictions = []
        with open(os.path.join(path_data, path_cnn_classifier, cnn_file_name), "r", encoding='UTF-8') as file:
            for nb_line, line in enumerate(file):
                line_list = line.split()
                cells_list = [float(i) for i in line_list]
                cell_cnn_predictions.extend(cells_list)
        cell_cnn_predictions = np.array(cell_cnn_predictions)
        cells_predicted_as_true = np.where(cell_cnn_predictions >= 0.5)[0]

        print(f"ms.coord_obj.coord[0].shape {ms.coord_obj.coord[0].shape}")
        print(f"{ms.description}: n_cells: {ms.coord_obj.n_cells}")
        print(f"{ms.description}: cells_predicted_as_true: {len(cells_predicted_as_true)}")

        for cell in np.arange(ms.coord_obj.n_cells):
            if cell in cells_predicted_as_true:
                coords_to_keep.append(ms.coord_obj.coord[cell])

    print(f"len(coords_to_keep): {len(coords_to_keep)}")
    coords_matlab_style = np.empty((len(coords_to_keep),), dtype=np.object)
    for i in range(len(coords_to_keep)):
        coords_matlab_style[i] = coords_to_keep[i]
    sio.savemat(os.path.join(param.path_results, "test_coords_cnn.mat"), {"coord_python": coords_matlab_style})


def generate_artificial_map(coords_to_use, dimensions=(200, 200), n_cells=50,
                            n_overlap_by_cell_range=(1, 4), overlap_ratio_range=(0.1, 0.5)):
    coords_left = None
    map_coords, cells_with_overlap = (None, None)

    return coords_left, map_coords, cells_with_overlap


def make_video(images, outvid=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video file_name
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        # if not os.path.exists(image):
        #     raise FileNotFoundError(image)
        # img = imread(image)
        img = image
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())  # "RGBA"


def test_generate_movie_with_cells(coords, param):
    n_frames = len(coords) // 4
    lw = 400
    lh = 400
    images = np.zeros((n_frames, lw, lh))
    first_im = None
    im_to_append = []

    for frame_index, n_cells in enumerate(np.arange(1, len(coords) // 2, 2)):
        new_coords = []
        for n_cell in np.arange(0, min(n_cells+1, len(coords) // 2)):
            new_coords.append(coords[n_cell])
        # raise Exception()
        coord_obj = CoordClass(coord=new_coords, nb_col=200,
                               nb_lines=200)
        ms_fusion = MouseSession(age=10, session_id="fusion", nb_ms_by_frame=100, param=param)
        ms_fusion.coord_obj = coord_obj
        fig = ms_fusion.plot_all_cells_on_map(save_plot=False, return_fig=True)
        im = fig2img(fig)
        if first_im is None:
            first_im = im
        else:
            im_to_append.append(im)
        plt.close()
        im = im.convert('L')
        # print(f"coord[0].shape {coord[0].shape}")

        im.thumbnail((lw, lh), Image.ANTIALIAS)
        # im.show()
        im_array = np.asarray(im)
        # print(f"im_array.shape {im_array.shape}")
        produce_cell_coords = False

        if produce_cell_coords:
            produce_cell_coord_from_cnn_validated_cells(param)

        images[frame_index] = im_array

    outvid_avi = os.path.join(param.path_data, param.path_results, "test_vid.avi")
    outvid_tiff = os.path.join(param.path_data, param.path_results, "test_vid.tiff")
    outvid_tiff_bis = os.path.join(param.path_data, param.path_results, "test_vid_bis.tiff")
    # to avoid this error: error: (-215) src.depth() == CV_8U
    images = np.uint8(255 * images)
    make_video(images, outvid=outvid_avi, fps=5, size=None,
                is_color=False, format="XVID")
    # first_im.save(outvid_tiff, format="tiff", append_images=im_to_append, save_all=True,
    #               compression="tiff_jpeg")

    with tifffile.TiffWriter(outvid_tiff_bis) as tiff:
        for img in im_to_append:
            tiff.save(np.asarray(img), compress=6)



def main():
    """
    Objective is to produce fake movies of let's say 1000 frames with like 50 cells with targeted cells that would have
    between 1 and 3 overlaping cells.
    The function should produce a tiff movie and a file .mat with a coord variable containing the coords of the cells
    :return:
    """
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")
    path_data = root_path + "data/"
    path_results = root_path + "results_hne/"
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results + "/" + time_str
    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    param = DataForMs(path_data=path_data, path_results=path_results, time_str=time_str)

    data = hdf5storage.loadmat(os.path.join(path_data, "artificial_movie_generator", "test_coords_cnn.mat"))

    coords = data["coord_python"][0]

    test_generate_movie_with_cells(coords=coords, param=param)

    # coord_obj = CoordClass(coord=coords, nb_col=200,
    #                        nb_lines=200)
    #
    # ms_fusion = MouseSession(age=10, session_id="fusion", nb_ms_by_frame=100, param=param)
    # ms_fusion.coord_obj = coord_obj
    # fig = ms_fusion.plot_all_cells_on_map(save_plot=False, return_fig=True)
    # im = fig2img(fig)
    # im = im.convert('L')
    # # print(f"coord[0].shape {coord[0].shape}")
    # lw = 400
    # lh = 400
    # im.thumbnail((lw, lh), Image.ANTIALIAS)
    # # im.show()
    # im_array = np.asarray(im)
    # print(f"im_array.shape {im_array.shape}")
    # produce_cell_coords = False
    #
    # if produce_cell_coords:
    #     produce_cell_coord_from_cnn_validated_cells(param)
    #
    # outvid = os.path.join(path_data, path_results, "test_vid.avi")
    # n_frames = 200
    # images = np.zeros((n_frames, lw, lh))
    # for i in np.arange(0, n_frames, 2):
    #     images[i] = im_array
    # # to avoid this error: error: (-215) src.depth() == CV_8U
    # images = np.uint8(255 * images)
    # make_video(images, outvid=outvid, fps=1, size=None,
    #            is_color=False, format="XVID")

    coords_left, map_coords, cells_with_overlap = \
        generate_artificial_map(coords_to_use=coords, dimensions=(200, 200), n_cells=50,
                                n_overlap_by_cell_range=(1, 4), overlap_ratio_range=(0.1, 0.5))


main()
