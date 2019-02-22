import keras
from keras import models
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Activation
from keras import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import get_custom_objects
from keras.optimizers import RMSprop, adam
from keras.utils import to_categorical
from mouse_session_loader import load_mouse_sessions
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend as K
import pattern_discovery.tools.param as p_disc_tools_param
from datetime import datetime
import time
from PIL import ImageSequence, ImageDraw
import PIL
from shapely import geometry
from matplotlib import pyplot as plt
import scipy.io as sio

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.callbacks import ReduceLROnPlateau

from keras.models import model_from_json


class TTA_ModelWrapper():
    """A simple TTA wrapper for keras computer vision models.
    From: https://github.com/tsterbak/keras_tta
    the wrapper flips the images horizontally and vertically and averages the predictions of all flipped images.

    The intuition behind this is that even if the test image is not too easy to make a prediction,
    the transformations change it such that the model has higher chances of capturing
    the target shape and predicting accordingly.
    Args:
        model (keras model): A fitted keras model with a predict method.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, **args):
        """Wraps the predict method of the provided model.
        Augments the testdata with horizontal and vertical flips and
        averages the results.
        Args:
            data (numpy array of dim 4): The data to get predictions for.
        """
        data = None
        data_masked = None
        if 'data' in args:
            data = args['data']
        if 'data_masked' in args:
            data_masked = args['data_masked']
        if data is None:
            raise Exception("data is None")

        pred = []
        for index, img in enumerate(data):
            if data_masked is None:
                p0 = self.model.predict(self._expand(img[:, :, 0]))
                p1 = self.model.predict(self._expand(np.fliplr(img[:, :, 0])))
                p2 = self.model.predict(self._expand(np.flipud(img[:, :, 0])))
                p3 = self.model.predict(self._expand(np.fliplr(np.flipud(img[:, :, 0]))))
            else:
                img_masked = data_masked[index]
                p0 = self.model.predict({'first_input': self._expand(img[:, :, 0]),
                                         'second_input': self._expand(img_masked[:, :, 0])})
                p1 = self.model.predict({'first_input': self._expand(np.fliplr(img[:, :, 0])),
                                         'second_input': self._expand(np.fliplr(img_masked[:, :, 0]))})
                p2 = self.model.predict({'first_input': self._expand(np.flipud(img[:, :, 0])),
                                         'second_input': self._expand(np.flipud(img_masked[:, :, 0]))})
                p3 = self.model.predict({'first_input': self._expand(np.fliplr(np.flipud(img[:, :, 0]))),
                                         'second_input': self._expand(np.fliplr(np.flipud(img_masked[:, :, 0])))})

            p = (p0[0][0] + p1[0][0] + p2[0][0] + p3[0][0]) / 4
            show_differences = False
            if show_differences:
                if abs(p - p0[0][0]) > 0.1:
                    print(f"p0 {str(np.round(p0, 3))}, p1 {str(np.round(p1, 3))}, "
                          f"p2 {str(np.round(p2, 3))}, p3 {str(np.round(p3, 3))}")
                    print(f"Cell {index}: p {str(np.round(p, 3))}")
                    print("")

            pred.append(p)

        return np.array(pred)

    def _expand(self, x):
        return np.expand_dims(np.expand_dims(x, axis=0), axis=3)


class DataForMs(p_disc_tools_param.Parameters):
    def __init__(self, path_data, result_path):
        self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        super().__init__(path_results=result_path, time_str=self.time_str, bin_size=1)
        self.path_data = path_data
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None


def load_movie(ms):
    if ms.tif_movie_file_name is not None:
        if ms.tiff_movie is None:
            start_time = time.time()
            im = PIL.Image.open(ms.tif_movie_file_name)
            n_frames = len(list(ImageSequence.Iterator(im)))
            dim_x, dim_y = np.array(im).shape
            print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")
            ms.tiff_movie = np.zeros((n_frames, dim_x, dim_y))
            for frame, page in enumerate(ImageSequence.Iterator(im)):
                ms.tiff_movie[frame] = np.array(page)
            stop_time = time.time()
            print(f"Time for loading movie: "
                  f"{np.round(stop_time-start_time, 3)} s")
            ms.normalize_movie()
        return True
    return False


# ------------------------------------------------------------
# needs to be defined as activation class otherwise error
# AttributeError: 'Activation' object has no attribute '__name__'
# From: https://github.com/keras-team/keras/issues/8716
class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x):
    """
    Implementing a the swish activation function.
    From: https://www.kaggle.com/shahariar/keras-swish-activation-acc-0-996-top-7
    Paper describing swish: https://arxiv.org/abs/1710.05941

    :param x:
    :return:
    """
    return K.sigmoid(x) * x

def scale_polygon_to_source(poly_gon, minx, miny):
    coords = list(poly_gon.exterior.coords)
    scaled_coords = []
    for coord in coords:
        scaled_coords.append((coord[0] - minx, coord[1] - miny))
    return geometry.Polygon(scaled_coords)


def get_source_profile(cell, ms, binary_version=True, pixels_around=0, bounds=None, buffer=None):
    len_frame_x = ms.tiff_movie_normalized[0].shape[1]
    len_frame_y = ms.tiff_movie_normalized[0].shape[0]

    # determining the size of the square surrounding the cell
    poly_gon = ms.coord_obj.cells_polygon[cell]
    if bounds is None:
        minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
    else:
        minx, miny, maxx, maxy = bounds

    minx = max(0, minx - pixels_around)
    miny = max(0, miny - pixels_around)
    maxx = min(len_frame_x - 1, maxx + pixels_around)
    maxy = min(len_frame_y - 1, maxy + pixels_around)

    len_x = maxx - minx + 1
    len_y = maxy - miny + 1

    # mask used in order to keep only the cells pixel
    # the mask put all pixels in the polygon, including the pixels on the exterior line to zero
    scaled_poly_gon = scale_polygon_to_source(poly_gon=poly_gon, minx=minx, miny=miny)
    img = PIL.Image.new('1', (len_x, len_y), 1)
    if buffer is not None:
        scaled_poly_gon = scaled_poly_gon.buffer(buffer)
    ImageDraw.Draw(img).polygon(list(scaled_poly_gon.exterior.coords), outline=0, fill=0)
    mask = np.array(img)

    if binary_version:
        source_profile = np.ones((len_y, len_x))
    else:
        source_profile = np.zeros((len_y, len_x))

    if not binary_version:
        # selectionning the best peak to produce the source_profile
        peaks = np.where(ms.spike_struct.peak_nums[cell, :] > 0)[0]
        threshold = np.percentile(ms.traces[cell, peaks], 95)
        selected_peaks = peaks[np.where(ms.traces[cell, peaks] > threshold)[0]]
        # max 10 peaks, min 5 peaks
        if len(selected_peaks) > 10:
            p = 10 / len(peaks)
            threshold = np.percentile(ms.traces[cell, peaks], (1 - p) * 100)
            selected_peaks = peaks[np.where(ms.traces[cell, peaks] > threshold)[0]]
        elif (len(selected_peaks) < 5) and (len(peaks) > 5):
            p = 5 / len(peaks)
            threshold = np.percentile(ms.traces[cell, peaks], (1 - p) * 100)
            selected_peaks = peaks[np.where(ms.traces[cell, peaks] > threshold)[0]]

        onsets_frames = np.where(ms.spike_struct.spike_nums[cell, :] > 0)[0]
        raw_traces = np.copy(ms.raw_traces)
        # so the lowest value is zero
        raw_traces += abs(np.min(raw_traces))
        for peak in selected_peaks:
            tmp_source_profile = np.zeros((len_y, len_x))
            onsets_before_peak = np.where(onsets_frames <= peak)[0]
            if len(onsets_before_peak) == 0:
                # shouldn't arrive
                continue
            onset = onsets_frames[onsets_before_peak[-1]]
            # print(f"onset {onset}, peak {peak}")
            frames_tiff = ms.tiff_movie_normalized[onset:peak + 1]
            for frame_index, frame_tiff in enumerate(frames_tiff):
                tmp_source_profile += (frame_tiff[miny:maxy + 1, minx:maxx + 1] * raw_traces[cell, onset + frame_index])
            # averaging
            tmp_source_profile = tmp_source_profile / (np.sum(raw_traces[cell, onset:peak + 1]))
            source_profile += tmp_source_profile

        source_profile = source_profile / len(selected_peaks)
        # normalized so that value are between 0 and 1
        source_profile = source_profile / np.max(source_profile)

    return source_profile, minx, miny, mask


# used to get prediction, in the gui for exemple.
def get_source_profile_to_classify(ms, buffer=None, max_width=20, max_height=20, binary_version=False):
    n_cells = ms.spike_struct.n_cells
    if n_cells is None:
        n_cells = len(ms.raw_traces)

    if binary_version:
        full_data = np.zeros((n_cells, max_height, max_width), dtype="uint8")
        full_data_masked = np.zeros((n_cells, max_height, max_width), dtype="uint8")
    else:
        full_data = np.zeros((n_cells, max_height, max_width))
        full_data_masked = np.zeros((n_cells, max_height, max_width))

    # for each cell, we will extract a 2D array representing the cell shape
    # all 2D array should have the same shape
    for cell in np.arange(n_cells):
        pixels_around = 3
        # if use_mask:
        #     buffer = buffer
        # else:
        #     buffer = None

        if binary_version:
            buffer = None
        source_profile, minx, miny, mask_source_profile = get_source_profile(ms=ms, binary_version=binary_version,
                                                                             cell=cell,
                                                                             pixels_around=pixels_around,
                                                                             buffer=buffer,
                                                                             bounds=None)
        visualize_cells = False
        if visualize_cells:
            # print(f"max value {np.max(source_profile)}")
            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    gridspec_kw={'height_ratios': [1],
                                                 'width_ratios': [1]},
                                    figsize=(5, 5))
            c_map = plt.get_cmap('gray')
            img_src_profile = ax1.imshow(source_profile, cmap=c_map)
            plt.show()
        source_profile_masked = np.copy(source_profile)
        source_profile_masked[mask_source_profile] = 0
        if binary_version:
            profile_fit = np.zeros((max_height, max_width), dtype="uint8")
            profile_fit_masked = np.zeros((max_height, max_width), dtype="uint8")
        else:
            profile_fit = np.zeros((max_height, max_width))
            profile_fit_masked = np.zeros((max_height, max_width))
        # we center the source profile
        y_coord = (profile_fit.shape[0] - source_profile.shape[0]) // 2
        x_coord = (profile_fit.shape[1] - source_profile.shape[1]) // 2
        profile_fit[y_coord:source_profile.shape[0] + y_coord, x_coord:source_profile.shape[1] + x_coord] = \
            source_profile
        profile_fit_masked[y_coord:source_profile.shape[0] + y_coord, x_coord:source_profile.shape[1] + x_coord] = \
            source_profile_masked

        visualize_cells = False
        if visualize_cells:
            # print(f"max value {np.max(source_profile)}")
            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    gridspec_kw={'height_ratios': [1],
                                                 'width_ratios': [1]},
                                    figsize=(5, 5))
            c_map = plt.get_cmap('gray')
            img_src_profile = ax1.imshow(profile_fit, cmap=c_map)
            plt.show()
        full_data[cell] = profile_fit
        full_data_masked[cell] = profile_fit_masked
    return full_data, full_data_masked


def predict_cell_from_saved_model(ms, weights_file, json_file):
    # Model reconstruction from JSON file
    with open(json_file, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(weights_file)

    multi_inputs = (model.layers[0].output_shape == model.layers[1].output_shape)
    max_height = model.layers[0].output_shape[1]
    max_width = model.layers[0].output_shape[2]

    cells_profiles_to_predict, cells_profiles_to_predict_masked = get_source_profile_to_classify(ms=ms,
                                                                                                 binary_version=False,
                                                                                                 buffer=1,
                                                                                                 max_width=max_width,
                                                                                                 max_height=max_height)
    cells_profiles_to_predict = cells_profiles_to_predict.reshape(
        (cells_profiles_to_predict.shape[0], cells_profiles_to_predict.shape[1], cells_profiles_to_predict.shape[2], 1))
    cells_profiles_to_predict_masked = cells_profiles_to_predict_masked.reshape(
        (cells_profiles_to_predict_masked.shape[0], cells_profiles_to_predict_masked.shape[1],
         cells_profiles_to_predict_masked.shape[2], 1))

    tta_model = TTA_ModelWrapper(model)
    if multi_inputs:
        predictions = tta_model.predict(data=cells_profiles_to_predict, data_masked=cells_profiles_to_predict_masked)

        # predictions = np.ndarray.flatten(model.predict({'first_input': cells_profiles_to_predict,
        #                                                 'second_input': cells_profiles_to_predict_masked}))
    else:

        predictions = tta_model.predict(data=cells_profiles_to_predict)
        # predictions = np.ndarray.flatten(model.predict(cells_profiles_to_predict))

    return predictions


def load_data(ms_to_use, param, split_values=(0.5, 0.3), buffer=2,
              with_shuffling=True, with_data_augmentation=True, cells_shuffling=None):
    # ms_to_use: list of string representing the mouse_session
    # return normalized data
    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=True, load_abf=False,
                                            for_cell_classifier=True)

    # first we determine the max dimension of the array that will contains the cell contours
    max_width = 0
    max_height = 0
    padding_value = 10
    total_n_cells = 0
    img_descr = []
    for ms_str in ms_to_use:
        ms = ms_str_to_ms_dict[ms_str]
        movie_loaded = load_movie(ms)
        if not movie_loaded:
            raise Exception(f"could not load movie of ms {ms.description}")
        n_cells = ms.spike_struct.n_cells
        if n_cells is None:
            n_cells = len(ms.traces)
        print(f"{ms.description} n_cells {n_cells}")
        total_n_cells += n_cells
        for cell in np.arange(n_cells):
            poly_gon = ms.coord_obj.cells_polygon[cell]
            minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
            max_width = max(max_width, maxx - minx)
            max_height = max(max_height, maxy - miny)
    # then we add a few pixels
    max_width += padding_value
    max_height += padding_value

    if (max_width > 25) or (max_height > 25):
        print(f"cell_classifier: max_width {max_width}, max_height {max_height}")

    # we fix the max_width and max_height so it won't change for predicting other ms cells
    max_width = 20
    max_height = 20

    # data will be 0 or 1
    full_data = np.zeros((total_n_cells, max_height, max_width))
    full_data_masked = np.zeros((total_n_cells, max_height, max_width))
    full_labels = np.zeros(total_n_cells, dtype="uint8")

    cells_count = 0
    for ms in ms_str_to_ms_dict.values():
        n_cells = ms.spike_struct.n_cells
        if n_cells is None:
            n_cells = len(ms.traces)

        labels = np.ones(n_cells, dtype="uint8")
        labels[ms.cells_to_remove] = 0
        full_labels[cells_count:cells_count + n_cells] = labels

        for cell in np.arange(n_cells):
            img_descr.append(f"{ms.description}_cell_{cell}")

        # for each cell, we will extract a 2D array representing the cell shape
        # all 2D array should have the same shape
        profiles_fit, profiles_fit_masked = get_source_profile_to_classify(ms=ms, buffer=buffer,
                                                                           max_width=20, max_height=20)

        full_data[cells_count: cells_count + n_cells] = profiles_fit
        full_data_masked[cells_count: cells_count + n_cells] = profiles_fit_masked

        cells_count += n_cells

    print(f"total_n_cells {total_n_cells}")
    print(f"full_labels {len(np.where(full_labels)[0])}")

    # 0.3 of the data will be for the test, the rest for training
    # train_data, test_data, train_labels, test_labels = train_test_split(full_data, full_labels, train_size=.7)

    # cells shuffling
    if cells_shuffling is None:
        cells_shuffling = np.arange(total_n_cells)
        if with_shuffling:
            np.random.shuffle(cells_shuffling)
    n_cells_for_training = int(total_n_cells * split_values[0])
    n_cells_for_validation = int(total_n_cells * split_values[1])

    train_data = full_data[cells_shuffling[:n_cells_for_training]]
    train_data_masked = full_data_masked[cells_shuffling[:n_cells_for_training]]
    train_labels = full_labels[cells_shuffling[:n_cells_for_training]]
    if with_data_augmentation:
        train_data_augmented = np.zeros((train_data.shape[0] * 4, max_height, max_width))
        train_data_masked_augmented = np.zeros((train_data.shape[0] * 4, max_height, max_width))
        train_labels_augmented = np.zeros((train_labels.shape[0] * 4), dtype="uint8")
        n_transform = 4
        for index_img in np.arange(train_data.shape[0]):
            train_data_augmented[index_img * n_transform] = train_data[index_img]
            train_data_masked_augmented[index_img * n_transform] = train_data_masked[index_img]
            train_labels_augmented[index_img * n_transform] = train_labels[index_img]
            # horizontal flip
            train_data_augmented[(index_img * n_transform) + 1] = np.fliplr(train_data[index_img])
            train_data_masked_augmented[(index_img * n_transform) + 1] = np.fliplr(train_data_masked[index_img])
            # vertical flip
            train_data_augmented[(index_img * n_transform) + 2] = np.flipud(train_data[index_img])
            train_data_masked_augmented[(index_img * n_transform) + 2] = np.flipud(train_data_masked[index_img])
            # horizontal and vertical flip
            train_data_augmented[(index_img * n_transform) + 3] = np.fliplr(np.flipud(train_data[index_img]))
            train_data_masked_augmented[(index_img * n_transform) + 3] = np.fliplr(
                np.flipud(train_data_masked[index_img]))
            train_labels_augmented[(index_img * n_transform) + 1:(index_img * n_transform) + 4] = train_labels[
                index_img]

            # for i_angle, angle in enumerate([90, 180, 270]):
            #     first_index = (index_img * 4) + i_angle + 1
            #     to_rotate = train_data[index_img]
            #     train_data_augmented[first_index] = ndimage.rotate(input=to_rotate, angle=angle,
            #                                                        reshape=False)
            #     to_rotate = train_data_masked[index_img]
            #     train_data_masked_augmented[first_index] = ndimage.rotate(input=to_rotate, angle=angle,
            #                                                               reshape=False)

            #   train_labels_augmented[first_index] = train_labels[index_img]
            visualize_cells = False
            if visualize_cells and (index_img == 0):
                root_path = "/Users/pappyhammer/Documents/academique/these_inmed/hne_project/"
                path_data = root_path + "data/"
                result_path = root_path + "results_classifier/"

                images = [train_data_augmented[0], train_data_augmented[1], train_data_augmented[2],
                          train_data_augmented[3]]
                images_masked = [train_data_masked_augmented[0], train_data_masked_augmented[1],
                                 train_data_masked_augmented[2], train_data_masked_augmented[3]]
                for i_img in np.arange(len(images)):
                    # print(f"max value {np.max(source_profile)}")
                    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                            gridspec_kw={'height_ratios': [1],
                                                         'width_ratios': [1]},
                                            figsize=(5, 5))
                    c_map = plt.get_cmap('gray')
                    img_profile = ax1.imshow(images[i_img], cmap=c_map)
                    fig.savefig(f'{result_path}/cell_{i_img}.pdf',
                                format=f"pdf")
                    # plt.show()
                    plt.close()
                    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                            gridspec_kw={'height_ratios': [1],
                                                         'width_ratios': [1]},
                                            figsize=(5, 5))
                    c_map = plt.get_cmap('gray')
                    img_profile = ax1.imshow(images_masked[i_img], cmap=c_map)
                    fig.savefig(f'{result_path}/cell_masked_{i_img}.pdf',
                                format=f"pdf")
                    # plt.show()
                    plt.close()
        train_data = train_data_augmented
        train_data_masked = train_data_masked_augmented
        train_labels = train_labels_augmented
    valid_data = full_data[cells_shuffling[n_cells_for_training:n_cells_for_training + n_cells_for_validation]]
    valid_data_masked = full_data_masked[
        cells_shuffling[n_cells_for_training:n_cells_for_training + n_cells_for_validation]]
    valid_labels = full_labels[cells_shuffling[n_cells_for_training:n_cells_for_training + n_cells_for_validation]]
    test_data = full_data[cells_shuffling[n_cells_for_training + n_cells_for_validation:]]
    test_data_masked = full_data_masked[cells_shuffling[n_cells_for_training + n_cells_for_validation:]]
    test_labels = full_labels[cells_shuffling[n_cells_for_training + n_cells_for_validation:]]
    # print(f"test cells: {cells_shuffling[n_cells_for_training + n_cells_for_validation:]}")
    test_img_descr = []
    for cell in cells_shuffling[n_cells_for_training + n_cells_for_validation:]:
        test_img_descr.append(img_descr[cell])

    data_set = (train_data, valid_data, test_data)
    data_masked_set = (train_data_masked, valid_data_masked, test_data_masked)
    labels_set = (train_labels, valid_labels, test_labels)

    return data_set, data_masked_set, labels_set, test_img_descr, cells_shuffling


def hyperas_data(train_data, train_labels, valid_data, valid_labels, input_shape):
    return train_data, train_labels, valid_data, valid_labels, input_shape


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def build_hyperas_model(train_data, train_labels, valid_data, valid_labels, input_shape):
    # input_tensor = Input(shape=input_shape)
    #
    # x = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    # x = layers.MaxPooling2D((2, 2))(x)
    # # If we choose 'three', add an additional third layer
    # if {{choice(['two', 'three'])}} == 'three':
    #     x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    #     x = layers.MaxPooling2D((2, 2))(x)
    #
    # x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    # x = layers.Flatten()(x)
    # x = layers.Dropout({{uniform(0, 1)}})(x)
    #
    # x = layers.Dense({{choice([32, 64, 128])}})(x)  # used to be 64
    #
    # x = layers.Activation({{choice(['relu', 'tanh'])}})(x)
    #
    # output_tensor = layers.Dense(1, activation='sigmoid')(x)
    # model = Model(input_tensor, output_tensor)
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    if ({{choice(['two', 'three'])}} == 'three'):
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout({{uniform(0, 1)}}))
    model.add(layers.Dense({{choice([32, 64, 128])}}))
    model.add(layers.Activation({{choice(['relu', 'tanh'])}}))
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    result = model.fit(train_data, train_labels,
                       batch_size={{choice([32, 64])}},
                       epochs=5,
                       validation_data=(valid_data, valid_labels))

    # get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def build_model(input_shape, use_mulimodal_inputs, with_dropout=0.5):
    use_sequential_mode = False
    get_custom_objects().update({'swish': Swish(swish)})
    # to choose between siwsh and relu
    activation_fct = "swish"
    if use_sequential_mode and (not use_mulimodal_inputs):
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation=activation_fct, input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation=activation_fct))
        model.add(layers.MaxPooling2D((2, 2)))
        # print(model.summary())
        model.add(layers.Conv2D(64, (3, 3), activation=activation_fct))
        model.add(layers.Flatten())
        if with_dropout > 0:
            model.add(layers.Dropout(with_dropout))
        model.add(layers.Dense(64, activation=activation_fct))
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        input_tensor = Input(shape=input_shape, name='first_input')
        if use_mulimodal_inputs:
            input_tensor_bis = Input(shape=input_shape, name='second_input')
        use_depthwise_separable = False
        if use_depthwise_separable:
            x = layers.SeparableConv2D(32, 3, activation=activation_fct)(input_tensor)
            x = layers.SeparableConv2D(64, 3, activation=activation_fct)(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.SeparableConv2D(64, 3, activation=activation_fct)(x)
            x = layers.SeparableConv2D(128, 3, activation=activation_fct)(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.SeparableConv2D(64, 3, activation=activation_fct)(x)
            x = layers.SeparableConv2D(128, 3, activation=activation_fct)(x)
            x = layers.GlobalAveragePooling2D()(x)
            if with_dropout > 0:
                x = layers.Dropout(with_dropout)(x)
            first_x = layers.Dense(32, activation=activation_fct)(x)
        else:
            x = layers.Conv2D(32, (3, 3), activation=activation_fct)(input_tensor)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(64, (3, 3), activation=activation_fct)(x)
            x = layers.MaxPooling2D((2, 2))(x)
            if with_dropout > 0:
                x = layers.Dropout(with_dropout)(x)
            x = layers.Conv2D(64, (3, 3), activation=activation_fct)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Flatten()(x)
            if with_dropout > 0:
                x = layers.Dropout(with_dropout)(x)

            first_x = layers.Dense(64, activation=activation_fct)(x)  # used to be 64

            if use_mulimodal_inputs:
                x = layers.Conv2D(32, (3, 3), activation=activation_fct)(input_tensor_bis)
                x = layers.BatchNormalization()(x)
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Conv2D(64, (3, 3), activation=activation_fct)(x)
                x = layers.MaxPooling2D((2, 2))(x)
                if with_dropout > 0:
                    x = layers.Dropout(with_dropout)(x)
                x = layers.Conv2D(64, (3, 3), activation=activation_fct)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Flatten()(x)
                if with_dropout > 0:
                    x = layers.Dropout(with_dropout)(x)

                second_x = layers.Dense(64, activation=activation_fct)(x)

        if use_mulimodal_inputs:
            concatenated = layers.concatenate([first_x, second_x], axis=-1)
            output_tensor = layers.Dense(1, activation='sigmoid')(concatenated)
        else:
            output_tensor = layers.Dense(1, activation='sigmoid')(first_x)

        print("Creating Model instance")
        if use_mulimodal_inputs:
            model = Model([input_tensor, input_tensor_bis], output_tensor)
        else:
            model = Model(input_tensor, output_tensor)
        print("After Creating Model instance")

    return model


def plot_training_and_validation_values(history, key_name, n_epochs, result_path, param):
    history_dict = history.history
    acc_values = history_dict[key_name]
    val_acc_values = history_dict['val_' + key_name]
    epochs = range(1, n_epochs + 1)
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1],
                                         'width_ratios': [1]},
                            figsize=(5, 5))
    ax1.plot(epochs, smooth_curve(acc_values), 'bo', label=f'Training {key_name}')
    ax1.plot(epochs, smooth_curve(val_acc_values), 'b', label=f'Validation {key_name}')
    plt.title(f'Training and validation {key_name}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{key_name}')
    plt.legend()
    save_formats = "pdf"
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{result_path}/training_and_validation_{key_name}_{param.time_str}.{save_format}',
                    format=f"{save_format}")

    plt.close()



def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def event_lambda(f, *args, **kwds):
    return lambda f=f, args=args, kwds=kwds: f(*args, **kwds)


def load_data_from_file():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    result_path = root_path + "results_classifier/"
    dict_res = np.load(result_path + "data_hyperas.npz")

    train_data = dict_res["train_images"]
    train_labels = dict_res["train_labels"]
    valid_data = dict_res["valid_images"]
    valid_labels = dict_res["valid_labels"]
    test_images = dict_res["test_images"]
    test_labels = dict_res["test_labels"]
    input_shape = dict_res["input_shape"]
    train_data_masked = dict_res["train_images_masked"]
    valid_data_masked = dict_res["valid_images_masked"]
    test_images_masked = dict_res["test_images_masked"]

    return train_data, train_labels, valid_data, valid_labels, test_images, test_labels, input_shape, \
           train_data_masked, valid_data_masked, test_images_masked


# so far the best with 0.83 on test data (an 0.86 on another session),
# Is 0.5 dropout x 2, home datagen with just flip horizontal and vertical,  two  inputs (masked with buffer 1, and
# image with 3 pixels around)
def train_model():
    print("train_model()")
    np.set_printoptions(threshold=np.inf)
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")

    path_data = root_path + "data/"
    result_path = root_path + "results_classifier/"
    use_mulimodal_inputs = True

    param = DataForMs(path_data=path_data, result_path=result_path)

    # "p12_171110_a000_ms"
    descr_model = "p7_171012_a000-p12-p12_171110_a000_ms_buffer_1"
    # "p7_171012_a000-p9_18_09_27_a003-p12-p12_171110_a000_ms_buffer_1"
    # "p9_18_09_27_a003_ms",
    data_set, data_masked_set, labels_set, test_img_descr, cells_shuffling \
        = load_data(["p7_171012_a000_ms", "p12_171110_a000_ms"], param=param, buffer=1,
                    split_values=(0.7, 0.2), with_shuffling=False, with_data_augmentation=True)
    (train_images, valid_images, test_images) = data_set
    (train_images_masked, valid_images_masked, test_images_masked) = data_masked_set
    (train_labels, valid_labels, test_labels) = labels_set

    print(f"train_images {train_images.shape}, train_labels {train_labels.shape}, "
          f"valid_images {valid_images.shape}, valid_labels {valid_labels.shape}, "
          f"test_images {test_images.shape}, test_labels {test_labels.shape}")
    train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
    valid_images = valid_images.reshape((valid_images.shape[0], valid_images.shape[1], valid_images.shape[2], 1))
    test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
    input_shape = train_images.shape[1:]

    train_images_masked = train_images_masked.reshape((train_images_masked.shape[0], train_images_masked.shape[1],
                                                       train_images_masked.shape[2], 1))
    valid_images_masked = valid_images_masked.reshape((valid_images_masked.shape[0], valid_images_masked.shape[1],
                                                       valid_images_masked.shape[2], 1))
    test_images_masked = test_images_masked.reshape((test_images_masked.shape[0], test_images_masked.shape[1],
                                                     test_images_masked.shape[2], 1))

    save_in_npz_file = False
    if save_in_npz_file:
        print("save_in_npz_file data_hyperas.npz")
        for i in np.arange(len(test_img_descr)):
            print(f"{i}: {test_img_descr[i]}")
        if use_mulimodal_inputs:
            np.savez(result_path + "data_hyperas.npz",
                     train_images=train_images,
                     train_labels=train_labels,
                     valid_images=valid_images,
                     valid_labels=valid_labels,
                     test_images=test_images,
                     test_labels=test_labels,
                     train_images_masked=train_images_masked,
                     valid_images_masked=valid_images_masked,
                     test_images_masked=test_images_masked,
                     input_shape=train_images.shape[1:])
        else:
            np.savez(result_path + "data_hyperas.npz",
                     train_images=train_images_masked,
                     train_labels=train_labels,
                     valid_images=valid_images_masked,
                     valid_labels=valid_labels,
                     test_images=test_images_masked,
                     test_labels=test_labels,
                     input_shape=train_images.shape[1:])
        return
    load_data_npz_file = False
    if load_data_npz_file:
        train_images, train_labels, valid_images, valid_labels, test_images, test_labels, input_shape, \
        train_data_masked, valid_data_masked, test_images_masked = \
            load_data_from_file()

    print(f"train_images {train_images.shape}, train_labels {train_labels.shape}, "
          f"valid_images {valid_images.shape}, valid_labels {valid_labels.shape}, "
          f"test_images {test_images.shape}, test_labels {test_labels.shape}")

    hyperas_version = False

    if hyperas_version:
        best_run, best_model = optim.minimize(model=build_hyperas_model,
                                              data=event_lambda(hyperas_data, train_data=train_images,
                                                                train_labels=train_labels,
                                                                valid_data=valid_images,
                                                                valid_labels=valid_labels,
                                                                input_shape=input_shape),
                                              algo=tpe.suggest,
                                              max_evals=2,
                                              trials=Trials())
        print("Evalutation of best performing model:")
        print(best_model.evaluate(test_images, test_labels))
        prediction = np.ndarray.flatten(best_model.predict(test_images))
        for i, predict_value in enumerate(prediction):
            predict_value = str(round(predict_value, 2))
            print(f"test predict / real: {predict_value} / {test_labels[i]}")
        print("Best performing model chosen hyper-parameters:")
        print(best_run)
    else:
        print("before building model")
        model = build_model(input_shape=train_images.shape[1:], with_dropout=0.5,
                            use_mulimodal_inputs=use_mulimodal_inputs)
        print("before printing summary")
        print(model.summary())

        raise Exception("TOTOOO")
        # Define the optimizer
        # from https://www.kaggle.com/shahariar/keras-swish-activation-acc-0-996-top-7
        optimizer = adam(lr=0.001, epsilon=1e-08, decay=0.0)
        # optimizer = 'rmsprop'
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy', sensitivity, specificity])

        # Set a learning rate annealer
        # from: https://www.kaggle.com/shahariar/keras-swish-activation-acc-0-996-top-7
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_sensitivity',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)

        n_epochs = 30
        batch_size = 16

        with_datagen = False

        if with_datagen:
            train_datagen = ImageDataGenerator(
                fill_mode='constant',
                cval=0,
                # rescale=1,
                # rotation_range=60,
                # width_shift_range=0.2,
                # height_shift_range=0.2,
                # zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True
            )
            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            # train_datagen.fit(train_images)

            test_datagen = ImageDataGenerator(rescale=1)

            # fits the model on batches with real-time data augmentation:
            if use_mulimodal_inputs:
                # in that case, needs to do a manual data augmentation
                history = None
            else:
                history = model.fit_generator(train_datagen.flow(train_images,
                                                                 train_labels, batch_size=batch_size,
                                                                 shuffle=False),
                                              steps_per_epoch=len(train_images) / batch_size,
                                              epochs=n_epochs,
                                              shuffle=False,
                                              validation_steps=len(valid_images) / batch_size,
                                              validation_data=test_datagen.flow(valid_images, valid_labels,
                                                                                batch_size=batch_size,
                                                                                shuffle=False),
                                              callbacks=[learning_rate_reduction])
            # validation_data=(valid_images, valid_labels))
        else:
            # model.fit(train_images, train_labels, epochs=n_epochs, batch_size=batch_size)
            if use_mulimodal_inputs:
                history = model.fit({'first_input': train_images, 'second_input': train_images_masked}, train_labels,
                                    epochs=n_epochs,
                                    batch_size=batch_size,
                                    validation_data=({'first_input': valid_images, 'second_input': valid_images_masked},
                                                     valid_labels),
                                    shuffle=True,
                                    callbacks=[learning_rate_reduction])
            else:
                history = model.fit(train_images,
                                    train_labels,
                                    epochs=n_epochs,
                                    batch_size=batch_size,
                                    validation_data=(valid_images, valid_labels),
                                    shuffle=True,
                                    callbacks=[learning_rate_reduction])

        show_plots = True

        if show_plots:
            key_names = ["loss", "acc", "sensitivity", "specificity"]
            for key_name in key_names:
                plot_training_and_validation_values(history=history, key_name=key_name, n_epochs=n_epochs,
                                                    result_path=result_path, param=param)

        # if use_mulimodal_inputs:
        #     prediction = np.ndarray.flatten(model.predict({'first_input': valid_images,
        #                                                    'second_input': valid_images_masked}))
        # else:
        #     prediction = np.ndarray.flatten(model.predict(valid_images))
        # for i, predict_value in enumerate(prediction):
        #     print(f"valid: {str(round(predict_value, 2))} / {valid_labels[i]}")

        if use_mulimodal_inputs:
            test_loss, test_acc, test_sensitivity, test_specificity = model.evaluate({'first_input': test_images,
                                                  'second_input': test_images_masked},
                                                 test_labels)
        else:
            test_loss, test_acc, test_sensitivity, test_specificity = model.evaluate(test_images, test_labels)

        print(f"test_acc {test_acc}, test_sensitivity {test_sensitivity}, test_specificity {test_specificity}")

        model.save(f'{param.path_results}cell_classifier_model_{descr_model}_test_acc_{test_acc}_{param.time_str}.h5')
        model.save_weights(f'{param.path_results}cell_classifier_weights_{descr_model}_test_acc_{test_acc}_{param.time_str}.h5')
        # Save the model architecture
        with open(
                f'{param.path_results}cell_classifier_model_architecture_{descr_model}_test_acc_{test_acc}_{param.time_str}.json',
                'w') as f:
            f.write(model.to_json())

        # if use_mulimodal_inputs:
        #     prediction = np.ndarray.flatten(model.predict({'first_input': test_images,
        #                                                    'second_input': test_images_masked}))
        # else:
        #     prediction = np.ndarray.flatten(model.predict(test_images))
        # for i, predict_value in enumerate(prediction):
        #     predict_value = str(round(predict_value, 2))
        #     print(f"{i}: {test_img_descr[i]}: {predict_value} / {test_labels[i]}")
        # print(f"test_acc {test_acc}")
