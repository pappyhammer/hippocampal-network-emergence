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
from shapely import geometry
import shapely
import random

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


def shift_cell_coord_to_centroid(centroid, cell_coord):
    # it is necessary to remove one, as data comes from matlab, starting from 1 and not 0
    cell_coord = cell_coord - 1
    cell_coord = cell_coord.astype(int)
    coord_list_tuple = []
    for n in np.arange(cell_coord.shape[1]):
        coord_list_tuple.append((cell_coord[0, n], cell_coord[1, n]))

    poly_cell = geometry.Polygon(coord_list_tuple)
    centroid_point = poly_cell.centroid
    # print(f"centroid {centroid} centroid[0] {centroid[0]}")
    # print(f"centroid_point.x {centroid_point.x}, centroid_point.y {centroid_point.y}")
    x_shift = centroid[0] - centroid_point.x
    y_shift = centroid[1] - centroid_point.y
    # print(f"x_shift {x_shift}, y_shift {y_shift}")
    for n in np.arange(cell_coord.shape[1]):
        cell_coord[0, n] = cell_coord[0, n] + x_shift
        cell_coord[1, n] = cell_coord[1, n] + y_shift

    coord_list_tuple = []
    for n in np.arange(cell_coord.shape[1]):
        coord_list_tuple.append((cell_coord[0, n], cell_coord[1, n]))
    poly_cell = geometry.Polygon(coord_list_tuple)

    cell_coord = cell_coord + 1

    return cell_coord, poly_cell


def generate_artificial_map(coords_to_use,
                            n_overlap_by_cell_range=(1, 4), overlap_ratio_range=(0.1, 0.5)):
    dimensions = (120, 120)
    # model cells, then we'll put cells around with some overlaping
    n_cells = 16
    sub_window_size = (30, 30)
    x_padding = 1  # sub_window_size[1] // 6
    y_padding = 1  # sub_window_size[0] // 6
    centroids = []
    line = 0
    col = 0
    max_lines = dimensions[0] // sub_window_size[0]
    max_cols = dimensions[1] // sub_window_size[1]
    x_borders = []
    y_borders = []
    for c in np.arange(n_cells):
        x_borders.append((col*sub_window_size[1], (col+1)*sub_window_size[1]))
        y_borders.append((line*sub_window_size[0], (line+1)*sub_window_size[0]))
        centroids.append((int((col+0.5)*sub_window_size[1]), int((line+0.5)*sub_window_size[0])))
        line += 1
        if (line % max_lines) == 0:
            line = 0
            col += 1
    # print(f"centroids {centroids}")

    coords_to_use = coords_to_use
    cells_with_overlap = []
    # key is an int (one of the cells_with_overlap), and value an int correspdongin
    overlapping_cells = dict()
    map_coords = []
    cell_index = 0
    for c in np.arange(n_cells):
        cell_coord = coords_to_use[0]
        # print(f"cell_coord {cell_coord}")
        # print(f"cell_coord.shape {cell_coord.shape}")
        # we center the cell and change its coordinates
        centroid = centroids[c]
        # poly_cell = shapely.affinity.translate(poly_cell, xoff=x_shift, yoff=y_shift)
        # list(poly_cell.exterior.coords)
        cell_coord, poly_main_cell = shift_cell_coord_to_centroid(centroid=centroid, cell_coord=cell_coord)
        cells_with_overlap.append(cell_index)
        main_cell_index = cell_index
        overlapping_cells[main_cell_index] = []
        map_coords.append(cell_coord)
        cell_index += 1
        coords_to_use = coords_to_use[1:]

        # we decide how many cells will be overlaping it (more like intersect)
        n_overlaps = random.randint(n_overlap_by_cell_range[0], n_overlap_by_cell_range[1])
        n_non_overlaps = random.randint(2, 10)
        n_over_added = 0
        n_non_over_added = 0
        centroids_added = []
        max_n_overall_trial = 200
        n_overall_trial = 0

        while (n_over_added < n_overlaps) or (n_non_over_added < n_non_overlaps):
            if n_overall_trial >= max_n_overall_trial:
                print("n_overall_trial >= max_n_overall_trial")
                break
            n_overall_trial += 1
            over_cell_coord = coords_to_use[0]
            coords_to_use = coords_to_use[1:]
            new_centroid_x_values = np.concatenate((np.arange(x_borders[c][0] + x_padding, centroid[0]-2),
                                                   np.arange(centroid[0]+2, x_borders[c][1]+1 - x_padding)))
            new_centroid_y_values = np.concatenate((np.arange(y_borders[c][0] + y_padding, centroid[1] - 2),
                                                    np.arange(centroid[1] + 2, y_borders[c][1] + 1 - y_padding)))
            not_added = True
            # one cell might be too big to fit in, then we give up and go to the next window
            max_n_trial = 1000
            n_trial = 0
            while not_added:
                if n_trial >= max_n_trial:
                    print("n_trial >= max_n_trial")
                    break
                n_trial += 1
                # random x and y for centroid
                np.random.shuffle(new_centroid_x_values)
                np.random.shuffle(new_centroid_y_values)
                x = new_centroid_x_values[0]
                y = new_centroid_y_values[1]
                # first we want this centroid to be at least 2 pixels away of any added centroid
                if (x, y) in centroids_added:
                    continue
                to_close = False
                for centr in centroids_added:
                    if (abs(x - centr[0]) <= 2) and (abs(y - centr[1]) <= 2):
                        to_close = True
                        break
                if to_close:
                    continue
                cell_coord, poly_new_cell = shift_cell_coord_to_centroid(centroid=(x, y), cell_coord=over_cell_coord)
                # first we need to make sure the cell don't go out of the frame
                minx, miny, maxx, maxy = np.array(list(poly_new_cell.bounds))
                if (minx <= 0) or (miny <= 0) or (maxx >= dimensions[1]-1) or (maxy >= dimensions[0]-1):
                    continue
                # if intersects and not just touches (means commun border)
                if poly_main_cell.intersects(poly_new_cell) and (not poly_main_cell.touches(poly_new_cell)) \
                        and n_over_added < n_overlaps:
                    n_over_added += 1
                    not_added = False
                elif n_non_over_added < n_non_overlaps:
                    n_non_over_added += 1
                    not_added = False
                if not not_added:
                    map_coords.append(cell_coord)
                    overlapping_cells[main_cell_index].append(cell_index)
                    centroids_added.append((x, y))
                    cell_index += 1
                if n_trial >= max_n_trial:
                    print("n_trial >= max_n_trial")
                    break

    print(f"cells_with_overlap {cells_with_overlap}")
    return coords_to_use, map_coords, cells_with_overlap, overlapping_cells


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
    n_frames = len(coords) // 8
    lw = 400
    lh = 400
    images = np.zeros((n_frames, lw, lh))
    first_im = None
    im_to_append = []
    #
    # for frame_index, n_cells in enumerate(np.arange(1, len(coords) // 4, 2)):
    #     new_coords = []
    #     for n_cell in np.arange(0, min(n_cells+1, len(coords) // 4)):
    #         new_coords.append(coords[n_cell])
    #     # raise Exception()
    #     coord_obj = CoordClass(coord=new_coords, nb_col=200,
    #                            nb_lines=200)
    #     ms_fusion = MouseSession(age=10, session_id="fusion", nb_ms_by_frame=100, param=param)
    #     ms_fusion.coord_obj = coord_obj
    #     fig = ms_fusion.plot_all_cells_on_map(save_plot=False, return_fig=True)
    #     im = fig2img(fig)
    #     if first_im is None:
    #         first_im = im
    #     else:
    #         im_to_append.append(im)
    #     plt.close()
    #     im = im.convert('L')
    #     # print(f"coord[0].shape {coord[0].shape}")
    #
    #     im.thumbnail((lw, lh), Image.ANTIALIAS)
    #     # im.show()
    #     im_array = np.asarray(im)
    #     # print(f"im_array.shape {im_array.shape}")
    #     produce_cell_coords = False
    #
    #     if produce_cell_coords:
    #         produce_cell_coord_from_cnn_validated_cells(param)
    #
    #     images[frame_index] = im_array

    # outvid_avi = os.path.join(param.path_data, param.path_results, "test_vid.avi")
    outvid_tiff = os.path.join(param.path_data, param.path_results, "test_vid.tiff")
    outvid_tiff_bis = os.path.join(param.path_data, param.path_results, "test_vid_bis.tiff")
    outvid_tiff_bw = os.path.join(param.path_data, param.path_results, "test_vid_b_w_gauss.tiff")

    images = np.ones((n_frames, lw, lh))
    images *= 0.1

    for noise_str in ["s&p", "poisson", "gauss", "speckle"]:
        outvid_tiff_noisy = os.path.join(param.path_data, param.path_results,
                                      f"test_noisy_vid_{noise_str}.tiff")

        with tifffile.TiffWriter(outvid_tiff_noisy) as tiff:
            for img_array in images:
                img_array = noisy(noise_str, img_array)
                img_array = normalize_array_0_255(img_array)
                tiff.save(img_array, compress=6)

    raise Exception()


    # to avoid this error: error: (-215) src.depth() == CV_8U
    # images = np.uint8(255 * images)
    # make_video(images, outvid=outvid_avi, fps=5, size=None,
    #             is_color=False, format="XVID")

    # doesn't work
    # first_im.save(outvid_tiff, format="tiff", append_images=im_to_append, save_all=True,
    #               compression="tiff_jpeg")

    with tifffile.TiffWriter(outvid_tiff_bis) as tiff:
        for img in im_to_append:
            # to convert in gray
            # img = img.convert('L')
            # to reduce the size
            # img.thumbnail((lw, lh), Image.ANTIALIAS)
            tiff.save(np.asarray(img), compress=6)

    with tifffile.TiffWriter(outvid_tiff_bw) as tiff:
        for img in im_to_append:
            # to convert in gray
            img = img.convert('L')
            # to reduce the size
            # img.thumbnail((lw, lh), Image.ANTIALIAS)
            img_array = np.asarray(img)
            img_array = noisy("gauss", img_array)
            img_array = normalize_array_0_255(img_array)
            tiff.save(img_array, compress=6)

    outvid_tiff_sp = os.path.join(param.path_data, param.path_results, "test_vid_b_w_sp.tiff")

    with tifffile.TiffWriter(outvid_tiff_sp) as tiff:
        for img in im_to_append:
            # to convert in gray
            img = img.convert('L')
            # to reduce the size
            # img.thumbnail((lw, lh), Image.ANTIALIAS)
            img_array = np.asarray(img)
            img_array = noisy("s&p", img_array)
            img_array = normalize_array_0_255(img_array)
            tiff.save(img_array, compress=6)

    outvid_tiff_bw = os.path.join(param.path_data, param.path_results, "test_vid_b_w_poisson.tiff")

    with tifffile.TiffWriter(outvid_tiff_bw) as tiff:
        for img in im_to_append:
            # to convert in gray
            img = img.convert('L')
            # to reduce the size
            # img.thumbnail((lw, lh), Image.ANTIALIAS)
            img_array = np.asarray(img)
            img_array = noisy("poisson", img_array)
            img_array = normalize_array_0_255(img_array)
            tiff.save(img_array, compress=6)

    outvid_tiff_bw = os.path.join(param.path_data, param.path_results, "test_vid_b_w_speckle.tiff")

    with tifffile.TiffWriter(outvid_tiff_bw) as tiff:
        for img in im_to_append:
            # to convert in gray
            img = img.convert('L')
            # to reduce the size
            # img.thumbnail((lw, lh), Image.ANTIALIAS)
            img_array = np.asarray(img)
            img_array = noisy("speckle", img_array)
            img_array = normalize_array_0_255(img_array)
            tiff.save(img_array, compress=6)


def normalize_array_0_255(img_array):
    minv = np.amin(img_array)
    maxv = np.amax(img_array)
    img_array = (255 * (img_array - minv) / (maxv - minv)).astype(np.uint8)
    return img_array


# from https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv
def noisy(noise_typ, image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
    :param img_array:
    :return:
    """
    if noise_typ == "gauss":
        mean = 0
        var = 0.1
        sigma = var ** 0.5

        if len(image.shape) == 3:
            row,col,ch= image.shape
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
        else:
            row, col = image.shape
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        if len(image.shape) == 3:
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
        else:
            row, col = image.shape
            gauss = np.random.randn(row, col)
            gauss = gauss.reshape(row, col)
        noisy = image + image * gauss
        return noisy


def produce_movie(map_coords, raster_dur, param, dimensions):
    n_frames = raster_dur.shape[1]
    images = np.ones((n_frames, dimensions[0], dimensions[1]))
    images *= 0.1

    noise_str = "speckle"
    # in ["s&p", "poisson", "gauss", "speckle"]:
    outvid_tiff_noisy = os.path.join(param.path_data, param.path_results,
                                     f"artificial_movie.tiff")

    with tifffile.TiffWriter(outvid_tiff_noisy) as tiff:
        for img_array in images:
            img_array = noisy(noise_str, img_array)
            img_array = normalize_array_0_255(img_array)
            tiff.save(img_array, compress=6)


def build_raster_dur(map_coords, cells_with_overlap, overlapping_cells, n_frames):
    n_cells = len(map_coords)
    raster_dur = np.zeros((n_cells, n_frames), dtype="int8")

    return raster_dur


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

    do_test_generate_movie_with_cells = False
    if do_test_generate_movie_with_cells:
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

    coords_left, map_coords, cells_with_overlap, overlapping_cells = \
        generate_artificial_map(coords_to_use=coords,
                                n_overlap_by_cell_range=(1, 4), overlap_ratio_range=(0.1, 0.5))

    coord_obj = CoordClass(coord=map_coords, nb_col=120,
                           nb_lines=120)

    ms_fusion = MouseSession(age=10, session_id="fusion", nb_ms_by_frame=100, param=param)
    ms_fusion.coord_obj = coord_obj
    ms_fusion.plot_all_cells_on_map()

    n_frames = 2500

    # we need to generate a raster_dur, with some synchronicity between overlapping cells
    raster_dur = build_raster_dur(map_coords=map_coords, cells_with_overlap=cells_with_overlap,
                                    overlapping_cells=overlapping_cells, n_frames=2500)

    # then we build the movie based on cells_coords and the raster_dur
    produce_movie(map_coords=map_coords, raster_dur=raster_dur, param=param,
                  dimensions=(120, 120))

    coords_matlab_style = np.empty((len(map_coords),), dtype=np.object)
    for i in range(len(map_coords)):
        coords_matlab_style[i] = map_coords[i]
    sio.savemat(os.path.join(param.path_results, "map_coords.mat"), {"coord_python": coords_matlab_style})

main()
