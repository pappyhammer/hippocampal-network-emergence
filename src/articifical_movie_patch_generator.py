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
import math
from pattern_discovery.tools.misc import get_continous_time_periods
from pattern_discovery.display.raster import plot_spikes_raster
import PIL
from PIL import ImageDraw

from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
from pattern_discovery.tools.signal import smooth_convolve


class DataForMs(p_disc_tools_param.Parameters):
    def __init__(self, path_data, path_results, with_mvt, use_fake_cells, time_str=None):
        if time_str is None:
            self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        else:
            self.time_str = time_str
        super().__init__(path_results=path_results, time_str=self.time_str, bin_size=1)
        self.path_data = path_data
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None
        self.with_mvt = with_mvt
        self.use_fake_cells = use_fake_cells


def produce_cell_coord_from_cnn_validated_cells(param):
    path_cnn_classifier = "cell_classifier_results_txt/"

    ms_to_use = ["p7_171012_a000_ms", "p8_18_10_24_a005_ms", "p9_18_09_27_a003_ms", "p11_17_11_24_a000_ms",
                 "p12_171110_a000_ms", "p13_18_10_29_a001_ms"]

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False,
                                            for_transient_classifier=True)
    coords_to_keep = []
    true_cells = []
    fake_cells = []
    global_cell_index = 0

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
            coords_to_keep.append(ms.coord_obj.coord[cell])
            if cell in cells_predicted_as_true:
                true_cells.append(global_cell_index)
            else:
                fake_cells.append(global_cell_index)
            global_cell_index += 1

    print(f"len(coords_to_keep): {len(coords_to_keep)}")
    coords_matlab_style = np.empty((len(coords_to_keep),), dtype=np.object)
    for i in range(len(coords_to_keep)):
        coords_matlab_style[i] = coords_to_keep[i]

    true_cells = np.array(true_cells)
    fake_cells = np.array(fake_cells)

    sio.savemat(os.path.join(param.path_results, "coords_artificial_movie.mat"),
                {"coord": coords_matlab_style, "true_cells": true_cells, "fake_cells": fake_cells})


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


def change_polygon_centroid(new_centroid, poly_cell):
    centroid_point = poly_cell.centroid
    x_shift = new_centroid[0] - centroid_point.x
    y_shift = new_centroid[1] - centroid_point.y
    coords = poly_cell.exterior.coords
    new_coords = []
    for coord in coords:
        new_coords.append((coord[0] + x_shift, coord[1] + y_shift))
    poly_cell = geometry.Polygon(new_coords)

    return poly_cell


def generate_artificial_map(coords_to_use, padding, true_cells, fake_cells, param,
                            n_overlap_by_cell_range=(1, 4), overlap_ratio_range=(0.1, 0.5)):
    # padding is used to to add mvts
    dimensions_without_padding = (90, 90)
    dimensions = (dimensions_without_padding[0] + (padding * 2), dimensions_without_padding[1] + (padding * 2))
    # model cells, then we'll put cells around with some overlaping
    n_cells = 9
    sub_window_size = (30, 30)
    # cell padding
    x_padding = 1  # sub_window_size[1] // 6
    y_padding = 1  # sub_window_size[0] // 6
    centroids = []
    line = 0
    col = 0
    max_lines = dimensions_without_padding[0] // sub_window_size[0]
    max_cols = dimensions_without_padding[1] // sub_window_size[1]
    x_borders = []
    y_borders = []
    for c in np.arange(n_cells):
        x_borders.append((col * sub_window_size[1] + padding, (col + 1) * sub_window_size[1] + padding))
        y_borders.append((line * sub_window_size[0] + padding, (line + 1) * sub_window_size[0] + padding))
        centroids.append((int((col + 0.5) * sub_window_size[1] + padding),
                          int((line + 0.5) * sub_window_size[0] + padding)))
        line += 1
        if (line % max_lines) == 0:
            line = 0
            col += 1
    # print(f"centroids {centroids}")

    # coords_to_use = coords_to_use
    coords_true_cells = []
    for true_cell in true_cells:
        coords_true_cells.append(coords_to_use[true_cell])

    coords_fake_cells = []
    for fake_cell in fake_cells:
        coords_fake_cells.append(coords_to_use[fake_cell])

    cells_with_overlap = []
    # key is an int (one of the cells_with_overlap), and value an int correspdongin
    overlapping_cells = dict()
    map_coords = []
    # in order to randomly choose a true cell coord
    true_cells_index = np.arange(len(coords_true_cells))
    random.shuffle(true_cells_index)
    # in order to randomly choose a fake cell coord
    fake_cells_index = np.arange(len(coords_fake_cells))
    random.shuffle(fake_cells_index)
    cell_index = 0
    n_non_target_cells_added = 0
    for c in np.arange(n_cells):
        cell_coord = coords_true_cells[true_cells_index[0]]
        true_cells_index = true_cells_index[1:]
        # we center the cell and change its coordinates
        centroid = centroids[c]
        cell_coord, poly_main_cell = shift_cell_coord_to_centroid(centroid=centroid, cell_coord=cell_coord)
        cells_with_overlap.append(cell_index)
        main_cell_index = cell_index
        overlapping_cells[main_cell_index] = []
        map_coords.append(cell_coord)
        cell_index += 1

        # we decide how many cells will be overlaping it (more like intersect)
        n_overlaps = random.randint(n_overlap_by_cell_range[0], n_overlap_by_cell_range[1])
        n_non_overlaps = random.randint(4, 15)
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
            if param.use_fake_cells and (n_non_target_cells_added % 4 == 0):
                over_cell_coord = coords_fake_cells[fake_cells_index[0]]
                fake_cells_index = fake_cells_index[1:]
            else:
                over_cell_coord = coords_true_cells[true_cells_index[0]]
                true_cells_index = true_cells_index[1:]
            new_centroid_x_values = np.concatenate((np.arange(x_borders[c][0] + x_padding, centroid[0] - 2),
                                                    np.arange(centroid[0] + 2, x_borders[c][1] + 1 - x_padding)))
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
                if (minx <= padding) or (miny <= padding) or (maxx >= (dimensions[1] - 1 + padding)) or \
                        (maxy >= (dimensions[0] - 1 + padding)):
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
                    n_non_target_cells_added += 1
                if n_trial >= max_n_trial:
                    print("n_trial >= max_n_trial")
                    break

    print(f"cells_with_overlap {cells_with_overlap}")
    return coords_to_use, map_coords, cells_with_overlap, overlapping_cells, dimensions


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
    # minv = 0
    maxv = np.amax(img_array)
    if maxv - minv == 0:
        img_array = img_array.astype(np.uint8)
    else:
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
            row, col, ch = image.shape
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


def build_traces(raster_dur, param, n_pixels_by_cell, dimensions, baseline):
    n_cells = raster_dur.shape[0]
    n_frames = raster_dur.shape[1]

    decay_factor = 10
    traces = np.ones((n_cells, n_frames))
    original_baseline = baseline
    # cell_baseline_ratio: by how much increasing the baseline of the cell comparing to the baseline of map
    # The ratio change for each cell, goes from 1.4 to 1.6
    cell_baseline_ratio_values = np.arange(1.7, 1.9, 0.1)
    for cell in np.arange(n_cells):
        random.shuffle(cell_baseline_ratio_values)
        cell_baseline_ratio = cell_baseline_ratio_values[0]
        baseline = original_baseline * n_pixels_by_cell[cell] * cell_baseline_ratio
        traces[cell] *= baseline
        active_periods = get_continous_time_periods(raster_dur[cell])
        for period in active_periods:
            last_frame = period[1] + 1
            len_period = last_frame - period[0]
            x_coords = [period[0], last_frame]
            low_amplitude = traces[cell, period[0]]
            if len_period <= 2:
                amplitude_max = random.randint(2, 5)
            elif len_period <= 5:
                amplitude_max = random.randint(3, 8)
            else:
                amplitude_max = random.randint(5, 10)
            amplitude_max *= n_pixels_by_cell[cell]
            amplitude_max += low_amplitude
            y_coords = [low_amplitude, amplitude_max]
            traces_values = give_values_on_linear_line_between_2_points(x_coords, y_coords)
            traces[cell, period[0]:last_frame + 1] = traces_values
            if (last_frame + 1) == n_frames:
                continue
            len_decay = max(len_period * decay_factor, 12)
            growth_rate = finding_growth_rate(t=len_decay, a=amplitude_max, end_value=baseline)
            # print(f"len_decay {len_decay}")
            # decay_frames = np.arange(last_frame, last_frame + len_decay)
            traces_decay_values = exponential_decay_formula(t=np.arange(len_decay), a=amplitude_max,
                                                            k=growth_rate, c=0)

            if last_frame + len_decay <= n_frames:
                traces[cell, last_frame:last_frame + len_decay] = traces_decay_values
            else:
                offset = (last_frame + len_decay) - n_frames
                traces[cell, last_frame:] = traces_decay_values[:-offset]

    z_score_traces = np.copy(traces)
    for cell in np.arange(n_cells):
        z_score_traces[cell] = (z_score_traces[cell] - np.mean(z_score_traces[cell])) / np.std(z_score_traces[cell])
    plot_spikes_raster(spike_nums=raster_dur,
                       display_spike_nums=True,
                       display_traces=True,
                       traces=z_score_traces,
                       raster_face_color="white",
                       param=param,
                       spike_train_format=False,
                       title=f"traces",
                       file_name=f"traces_artificial",
                       y_ticks_labels_size=4,
                       save_raster=True,
                       without_activity_sum=True,
                       show_raster=False,
                       plot_with_amplitude=False,
                       spike_shape="o",
                       spike_shape_size=0.4,
                       save_formats="pdf")

    return traces


class CellPiece:

    def __init__(self, id, poly_gon, dimensions, activity_mask=None):
        self.id = id
        self.poly_gon = poly_gon
        self.dimensions = dimensions
        self.mask = self.get_mask()
        # TODO: keep a smaller mask to save memory
        self.activity_mask = activity_mask

    def fill_movie_images(self, images):
        images[:, self.mask] = self.activity_mask[:, self.mask]

    def set_activity_mask_from_other(self, other_activity_mask):
        self.activity_mask = np.zeros(other_activity_mask.shape)
        self.activity_mask[:, self.mask] = other_activity_mask[:, self.mask]

    def set_activity_mask_from_two_other(self, other_1, other_2):
        self.activity_mask = np.zeros(other_1.shape)
        for frame in np.arange(other_1.shape[0]):
            other_1_sum = np.sum(other_1[frame, self.mask])
            other_2_sum = np.sum(other_2[frame, self.mask])
            if other_1_sum >= other_2_sum:
                self.activity_mask[frame, self.mask] = other_1[frame, self.mask]
            else:
                self.activity_mask[frame, self.mask] = other_2[frame, self.mask]

    def get_mask(self):
        img = PIL.Image.new('1', (self.dimensions[0], self.dimensions[1]), 0)
        try:
            ImageDraw.Draw(img).polygon(list(self.poly_gon.exterior.coords), outline=1,
                                        fill=1)
        except AttributeError:
            ImageDraw.Draw(img).polygon(list(self.poly_gon.coords), outline=1,
                                        fill=1)
        return np.array(img)

    def split(self, other):
        new_pieces = []
        intersection_polygon = self.poly_gon.intersection(other.poly_gon)
        # we need to re-build the activity mask depending on which cell in the most active at each frame
        new_id = self.id + "-" + other.id
        geoms = []
        try:
            geoms.extend(intersection_polygon.geoms)
        except AttributeError:
            geoms.append(intersection_polygon)
        id_ext = ["", "*", ".", "%", "a", "b"]
        for geom_index, geom in enumerate(geoms):
            inter_cell_piece = CellPiece(id=new_id + id_ext[geom_index], poly_gon=geom, dimensions=self.dimensions)
            inter_cell_piece.set_activity_mask_from_two_other(self.activity_mask, other.activity_mask)
            new_pieces.append(inter_cell_piece)

        diff_poly_gon = self.poly_gon.difference(other.poly_gon)
        geoms = []
        try:
            geoms.extend(diff_poly_gon.geoms)
        except AttributeError:
            geoms.append(diff_poly_gon)
        for geom_index, geom in enumerate(geoms):
            diff_cell_piece = CellPiece(id=self.id + id_ext[geom_index], poly_gon=geom, dimensions=self.dimensions)
            diff_cell_piece.set_activity_mask_from_other(self.activity_mask)
            new_pieces.append(diff_cell_piece)

        diff_other_poly_gon = other.poly_gon.difference(self.poly_gon)
        geoms = []
        try:
            geoms.extend(diff_other_poly_gon.geoms)
        except AttributeError:
            geoms.append(diff_other_poly_gon)
        for geom_index, geom in enumerate(geoms):
            diff_other_cell_piece = CellPiece(id=other.id + id_ext[geom_index], poly_gon=geom,
                                              dimensions=other.dimensions)
            diff_other_cell_piece.set_activity_mask_from_other(other.activity_mask)
            new_pieces.append(diff_other_cell_piece)

        return new_pieces

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


def get_mask(dimensions, poly_gon):
    img = PIL.Image.new('1', (dimensions[0], dimensions[1]), 0)
    try:
        ImageDraw.Draw(img).polygon(list(poly_gon.exterior.coords), outline=1,
                                    fill=1)
    except AttributeError:
        ImageDraw.Draw(img).polygon(list(poly_gon.coords), outline=1,
                                    fill=1)
    return np.array(img)


def get_weighted_activity_mask_for_a_cell(mask, soma_mask, n_pixels, n_pixels_soma):
    mu, sigma = 100, 15  # mean and standard deviation
    mu_soma = random.randint(50, 70)
    sigma_soma = mu_soma // 6

    weighted_mask = np.zeros(mask.shape)

    weighted_mask = weighted_mask.reshape(mask.shape[0] * mask.shape[1])  # flattening
    n_pixels = np.sum(mask)
    weighted_mask[(mask.reshape(mask.shape[0] * mask.shape[1])) > 0] = \
        np.random.normal(loc=mu, scale=sigma, size=n_pixels)
    if len(np.where(weighted_mask < 0)[0]) > 0:
        print(f"weighted_mask < 0 {len(np.where(weighted_mask < 0)[0])}")
    weighted_mask[weighted_mask < 0] = 0
    weighted_mask = weighted_mask.reshape((mask.shape[0], mask.shape[1]))  # back to original shape
    # print(f"weighted_mask {np.sum(weighted_mask)}")

    if soma_mask is not None:
        weighted_soma_mask = np.zeros(soma_mask.shape)

        weighted_soma_mask = weighted_soma_mask.reshape(soma_mask.shape[0] * soma_mask.shape[1])  # flattening
        weighted_soma_mask[(soma_mask.reshape(soma_mask.shape[0] * soma_mask.shape[1])) > 0] = \
            np.random.normal(loc=mu_soma, scale=sigma_soma, size=n_pixels_soma)
        if len(np.where(weighted_soma_mask < 0)[0]) > 0:
            print(f"weighted_soma_mask < 0 {len(np.where(weighted_soma_mask < 0)[0])}")
        weighted_soma_mask[weighted_soma_mask < 0] = 0
        weighted_soma_mask = weighted_soma_mask.reshape(
            (soma_mask.shape[0], soma_mask.shape[1]))  # back to original shape
        # print(f"weighted_soma_mask {np.sum(weighted_soma_mask)}")

        weighted_mask[soma_mask] = weighted_soma_mask[soma_mask]
        # print(f"weighted_mask with soma {np.sum(weighted_mask)}")

    return weighted_mask


def construct_movie_images(coord_obj, traces, dimensions, baseline, soma_geoms, n_pixels_by_cell=None):
    print("construct_movie_images begins")
    cell_pieces = set()
    n_frames = traces.shape[1]
    n_cells = coord_obj.n_cells
    soma_indices = np.arange(len(soma_geoms))
    baseline_traces = np.min(traces)

    # change_polygon_centroid(new_centroid, poly_cell)
    for cell in np.arange(n_cells):
        # print(f"construct_movie_images begins, cell {cell}")
        mask = coord_obj.get_cell_mask(cell, dimensions)
        n_pixels = np.sum(mask)

        add_soma = not (cell % 5 == 0)

        same_weight_for_all_frame = True

        if add_soma:
            # first we pick a soma for this cell
            random.shuffle(soma_indices)
            soma_geom = soma_geoms[soma_indices[0]]
            # then we want to move it in the cell
            # centroid_x = random.randint(-1, 1)
            # centroid_y = random.randint(-1, 1)
            centroid_x = 0
            centroid_y = 0
            cell_poly = coord_obj.cells_polygon[cell]
            centroid_x += cell_poly.centroid.x
            centroid_y += cell_poly.centroid.y
            soma_geom = change_polygon_centroid(new_centroid=(centroid_x, centroid_y), poly_cell=soma_geom)
            # print(f"cell_poly x {cell_poly.centroid.x} y {cell_poly.centroid.y}")
            # print(f"soma_geom x {soma_geom.centroid.x} y {soma_geom.centroid.y}")
            soma_mask = get_mask(dimensions=dimensions, poly_gon=soma_geom)
            n_pixels_soma = np.sum(soma_mask)
        else:
            soma_mask = None
            n_pixels_soma = None

        if same_weight_for_all_frame:
            weighted_mask = get_weighted_activity_mask_for_a_cell(mask=mask, soma_mask=soma_mask,
                                                                  n_pixels=n_pixels, n_pixels_soma=n_pixels_soma)
            # del soma_mask
            del mask

        # TODO: decrease size of the mask so it just fit the cell
        # TODO: and add coord to where the top right of the mask starts
        activity_mask = np.zeros((n_frames, dimensions[0], dimensions[1]))

        for frame in np.arange(n_frames):
            if not same_weight_for_all_frame:
                weighted_mask = get_weighted_activity_mask_for_a_cell(mask=mask, soma_mask=soma_mask,
                                                                      n_pixels=n_pixels, n_pixels_soma=n_pixels_soma)

            amplitude = traces[cell, frame]
            if amplitude == baseline_traces:
                weighted_mask_tmp = np.copy(weighted_mask)
                weighted_mask_tmp[soma_mask] = 0
                activity_mask[frame] = weighted_mask_tmp * (amplitude / np.sum(weighted_mask))
                del weighted_mask_tmp
            else:
                activity_mask[frame] = weighted_mask * (amplitude / np.sum(weighted_mask))
        del weighted_mask
        cell_pieces.add(CellPiece(id=f"{cell}", poly_gon=coord_obj.cells_polygon[cell],
                                  activity_mask=activity_mask, dimensions=dimensions))

    images = np.ones((n_frames, dimensions[0], dimensions[1]))
    images *= baseline

    # then we collect all pieces of cell, by piece we mean part of the cell with no overlap and part
    # with one or more intersect, and get it as a polygon object

    #
    while len(cell_pieces) > 0:
        cell_piece = cell_pieces.pop()
        no_intersections = True
        for other_index, other_cell_piece in enumerate(cell_pieces):
            # check if they interesect and not just touches
            if cell_piece.poly_gon.intersects(other_cell_piece.poly_gon) and \
                    (not cell_piece.poly_gon.touches(other_cell_piece.poly_gon)):
                no_intersections = False
                cell_pieces.remove(other_cell_piece)
                # then we split those 2 cell_pieces in 3, and loop again
                new_cell_pieces = cell_piece.split(other_cell_piece)
                cell_pieces.update(new_cell_pieces)
                break
        if no_intersections:
            cell_piece.fill_movie_images(images)

    print("construct_movie_images is over")

    return images


def build_somas(coord_obj, dimensions):
    soma_geoms = []

    for cell, poly_gon in coord_obj.cells_polygon.items():
        img = PIL.Image.new('1', (dimensions[0], dimensions[1]), 0)
        ImageDraw.Draw(img).polygon(list(poly_gon.exterior.coords), outline=1,
                                    fill=1)
        img = np.array(img)
        n_pixel = np.sum(img)
        # print(f"soma {cell}: {n_pixel}")
        n_trial = 0
        max_trial = 200
        while True:
            if n_trial > max_trial:
                # print("point break")
                break
            n_trial += 1
            distances = np.arange(-3, -0.3, 0.2)
            random.shuffle(distances)
            soma = poly_gon.buffer(distances[0])
            if hasattr(soma, 'geoms') or (soma.exterior is None):
                # means its a MultiPolygon object
                continue
            img = PIL.Image.new('1', (dimensions[0], dimensions[1]), 0)
            ImageDraw.Draw(img).polygon(list(soma.exterior.coords), outline=1,
                                        fill=1)
            img = np.array(img)
            n_pixel_soma = np.sum(img)
            ratio = n_pixel / n_pixel_soma
            if (ratio > 3) and (ratio < 5):
                print(f"soma {cell}: {n_pixel} {n_pixel_soma} {str(np.round(ratio, 2))}")
                soma_geoms.append(soma)
                break
    return soma_geoms


def produce_movie(map_coords, raster_dur, param, dimensions, cells_with_overlap, overlapping_cells,
                  padding):
    n_frames = raster_dur.shape[1]
    n_cells = raster_dur.shape[0]

    coord_obj = CoordClass(coord=map_coords, nb_col=dimensions[0],
                           nb_lines=dimensions[1])

    # build polygons representing somas
    soma_geoms = build_somas(coord_obj, dimensions=dimensions)

    ms_fusion = MouseSession(age=10, session_id="fusion", nb_ms_by_frame=100, param=param)
    ms_fusion.coord_obj = coord_obj
    ms_fusion.plot_all_cells_on_map()

    n_pixels_by_cell = dict()
    for cell in np.arange(n_cells):
        mask = coord_obj.get_cell_mask(cell, dimensions)
        n_pixels = np.sum(mask)
        n_pixels_by_cell[cell] = n_pixels

    # default value at rest for a pixel
    baseline = 1
    traces = build_traces(raster_dur, param, n_pixels_by_cell, dimensions, baseline)

    images = construct_movie_images(coord_obj=coord_obj, traces=traces, dimensions=dimensions,
                                    n_pixels_by_cell=n_pixels_by_cell, baseline=baseline, soma_geoms=soma_geoms)
    # cells_activity_mask
    noise_str = "gauss"
    # in ["s&p", "poisson", "gauss", "speckle"]:
    outvid_tiff = os.path.join(param.path_data, param.path_results,
                               f"p10_artificial.tiff")

    # used to add mvt
    # images_with_padding = np.ones((images.shape[0], dimensions[0], dimensions[1]))
    # images_with_padding *= baseline

    images_mask = np.zeros((images.shape[1], images.shape[2]), dtype="uint8")
    # print(f"images_with_padding.shape {images_with_padding.shape}")
    # print(f"images_mask.shape {images_mask.shape}")

    x_shift = 0
    y_shift = 0
    shaking_frames = []
    if param.with_mvt:
        # adding movement
        shaking_rate = 1 / 60
        x_shift_range = (-2, 2)
        y_shift_range = (-2, 2)
        n_continuous_shaking_frames_range = (1, 5)
        shake_it_when_it_fired = True
        if shake_it_when_it_fired:
            shaking_frames = []
            # we put a frame contained in each active period of target cells or overlaping cells
            cells = list(cells_with_overlap) + list(overlapping_cells)
            for cell in cells:
                periods = get_continous_time_periods(raster_dur[cell])
                for period in periods:
                    shaking_frames.append(random.randint(max(period[0] - 1, 0), period[1]))
            shaking_frames = np.unique(shaking_frames)
        else:
            shaking_frames = np.arange(n_frames)
        random.shuffle(shaking_frames)
        shaking_frames = shaking_frames[:int(n_frames * shaking_rate)]
        shaking_frames_to_concat = np.zeros(0, dtype='int16')
        for frame in shaking_frames:
            n_to_add = random.randint(n_continuous_shaking_frames_range[0], n_continuous_shaking_frames_range[1])
            shaking_frames_to_concat = np.concatenate(
                (shaking_frames_to_concat, np.arange(frame + 1, frame + n_to_add)))
        shaking_frames = np.concatenate((shaking_frames, shaking_frames_to_concat))
        # doing it for the saving on file
        shaking_frames = np.unique(shaking_frames)
        np.ndarray.sort(shaking_frames)
    with tifffile.TiffWriter(outvid_tiff) as tiff:
        for frame, img_array in enumerate(images):
            if param.with_mvt:
                shaked = False
                if frame in shaking_frames:
                    x_shift = random.randint(x_shift_range[0], x_shift_range[1])
                    y_shift = random.randint(y_shift_range[0], y_shift_range[1])
                    shaked = True
            last_y = (-padding + y_shift) if (-padding + y_shift) < 0 else images.shape[1]
            last_x = (-padding + x_shift) if (-padding + x_shift) < 0 else images.shape[1]
            images[frame, padding + y_shift:last_y, padding + x_shift:last_x] = \
                img_array[padding:-padding, padding:-padding]
            img_array = images[frame]
            img_array = noisy(noise_str, img_array)
            img_array = normalize_array_0_255(img_array)
            tiff.save(img_array, compress=6)
            images[frame] = img_array
            if param.with_mvt:
                if shaked:
                    x_shift = 0
                    y_shift = 0

    save_traces(coord_obj=coord_obj, movie=images, param=param)
    print(f"n shaking frames : {len(shaking_frames)}")
    return shaking_frames


def do_traces_smoothing(traces):
    # smoothing the trace
    windows = ['hanning', 'hamming', 'bartlett', 'blackman']
    i_w = 1
    window_length = 11
    for i in np.arange(traces.shape[0]):
        smooth_signal = smooth_convolve(x=traces[i], window_len=window_length,
                                        window=windows[i_w])
        beg = (window_length - 1) // 2
        traces[i] = smooth_signal[beg:-beg]


def save_traces(coord_obj, movie, param):
    raw_traces = np.zeros((coord_obj.n_cells, movie.shape[0]))
    for cell in np.arange(coord_obj.n_cells):
        mask = coord_obj.get_cell_mask(cell=cell,
                                       dimensions=(movie.shape[1], movie.shape[2]))
        raw_traces[cell, :] = np.mean(movie[:, mask], axis=1)

    smooth_traces = np.copy(raw_traces)
    do_traces_smoothing(smooth_traces)

    sio.savemat(os.path.join(param.path_results, "artificial_traces.mat"), {'C_df': smooth_traces,
                                                                            'raw_traces': raw_traces})


def build_raster_dur(map_coords, cells_with_overlap, overlapping_cells, n_frames, param):
    n_cells = len(map_coords)
    raster_dur = np.zeros((n_cells, n_frames), dtype="int8")
    for cell in np.arange(n_cells):
        if cell in cells_with_overlap:
            n_transient = random.randint(5, 20)
        elif cell in overlapping_cells:
            n_transient = random.randint(20, 40)
        else:
            n_transient = random.randint(5, 40)
        onsets = np.zeros(0, dtype="int16")
        for transient in np.arange(n_transient):
            while True:
                onset = random.randint(2, n_frames - 20)
                sub_array = np.abs(onsets - onset)
                if (len(onsets) == 0) or (np.min(sub_array) > 3):
                    onsets = np.append(onsets, [onset])
                    break
        # useless
        onsets.sort()
        for transient in np.arange(n_transient):
            onset = onsets[transient]
            duration_transient = random.randint(1, 8)
            raster_dur[cell, onset:onset + duration_transient] = 1

    plot_spikes_raster(spike_nums=raster_dur, param=param,
                       spike_train_format=False,
                       title=f"raster plot test",
                       file_name=f"spike_nums__dur_artificial",
                       y_ticks_labels_size=4,
                       save_raster=True,
                       without_activity_sum=False,
                       sliding_window_duration=1,
                       show_sum_spikes_as_percentage=True,
                       show_raster=False,
                       plot_with_amplitude=False,
                       spike_shape="o",
                       spike_shape_size=4,
                       save_formats="pdf")

    return raster_dur


def give_values_on_linear_line_between_2_points(x_coords, y_coords):
    # x_coords = [100, 400]
    # y_coords = [240, 265]
    # print(f"x_coords {x_coords} y_coords {y_coords}")
    # Calculate the coefficients. This line answers the initial question.
    coefficients = np.polyfit(x_coords, y_coords, 1)

    # 'a =', coefficients[0]
    # 'b =', coefficients[1]

    # Let's compute the values of the line...
    polynomial = np.poly1d(coefficients)
    x_axis = np.arange(x_coords[0], x_coords[1] + 1)
    y_axis = polynomial(x_axis)

    return y_axis


def exponential_decay_formula(t, a, k, c):
    """
    :param t: time that has passed
    :param a: initial value (amount before measuring growth or decay)
    :param k: continuous growth rate (also called constant of proportionality)
    (k > 0, the amount is increasing (growing); k < 0, the amount is decreasing (decaying))
    :param c: lowest value
    :return:
    """
    return a * np.exp(k * t) + c


def finding_growth_rate(t, a, end_value):
    k = np.log(end_value / a) / t
    return k


def save_raster_dur_for_gui(raster_dur, param):
    spike_nums = np.zeros(raster_dur.shape, dtype="int8")
    peak_nums = np.zeros(raster_dur.shape, dtype="int8")
    for cell in np.arange(raster_dur.shape[0]):
        active_periods = get_continous_time_periods(raster_dur[cell])
        for period in active_periods:
            spike_nums[cell, period[0]] = 1
            peak_nums[cell, period[1] + 1] = 1
    sio.savemat(os.path.join(param.path_results, "gui_data.mat"), {'Bin100ms_spikedigital_Python': spike_nums,
                                                                   'LocPeakMatrix_Python': peak_nums})

def vessels_test():
    dimensions = (120, 120)
    img_array = None
    plt.imshow(img_array)

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

    # TODO: put in param all options for generating the movie
    param = DataForMs(path_data=path_data, path_results=path_results, time_str=time_str,
                      with_mvt=False, use_fake_cells=False)

    produce_cell_coords = False
    if produce_cell_coords:
        produce_cell_coord_from_cnn_validated_cells(param)
        raise Exception("produce_cell_coord_from_cnn_validated_cells")

    data = hdf5storage.loadmat(os.path.join(path_data, "artificial_movie_generator", "coords_artificial_movie.mat"))

    coords = data["coord"][0]
    true_cells = data["true_cells"][0]
    fake_cells = data["fake_cells"][0]

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
    padding = 5

    coords_left, map_coords, cells_with_overlap, overlapping_cells, dimensions = \
        generate_artificial_map(coords_to_use=coords,
                                true_cells=true_cells, fake_cells=fake_cells,
                                n_overlap_by_cell_range=(1, 4), overlap_ratio_range=(0.1, 0.5),
                                padding=padding, param=param)

    n_frames = 2500

    # we need to generate a raster_dur, with some synchronicity between overlapping cells
    raster_dur = build_raster_dur(map_coords=map_coords, cells_with_overlap=cells_with_overlap,
                                  overlapping_cells=overlapping_cells, n_frames=2500, param=param)

    save_raster_dur_for_gui(raster_dur, param)

    # then we build the movie based on cells_coords and the raster_dur
    shaking_frames = produce_movie(map_coords=map_coords, raster_dur=raster_dur, param=param,
                                   dimensions=dimensions, cells_with_overlap=cells_with_overlap,
                                   overlapping_cells=overlapping_cells, padding=padding)

    # saving cells' number of interest

    file_name_txt = 'artificial_cells_listing.txt'

    with open(os.path.join(param.path_results, file_name_txt), "w", encoding='UTF-8') as file:
        file.write(f"Targets cells: {', '.join(list(map(str, cells_with_overlap)))}" + '\n')
        file.write(f"Shaking frames: {', '.join(list(map(str, shaking_frames)))}" + '\n')

    coords_matlab_style = np.empty((len(map_coords),), dtype=np.object)
    for i in range(len(map_coords)):
        coords_matlab_style[i] = map_coords[i]
    sio.savemat(os.path.join(param.path_results, "map_coords.mat"), {"coord_python": coords_matlab_style})


main()
