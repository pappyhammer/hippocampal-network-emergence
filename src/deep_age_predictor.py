from tensorflow.python.client import device_lib

print(f"device_lib.list_local_devices(): {device_lib.list_local_devices()}")

import tensorflow as tf
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Bidirectional, BatchNormalization
from keras.layers import Input, LSTM, Dense, TimeDistributed, Activation, Lambda, Permute, RepeatVector
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.optimizers import RMSprop, adam, SGD
from keras import layers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils import get_custom_objects, multi_gpu_model
from matplotlib import pyplot as plt
import pattern_discovery.tools.param as p_disc_tools_param
from datetime import datetime
from sortedcontainers import SortedDict
import time
from pattern_discovery.display.raster import plot_spikes_raster
from PIL import ImageSequence, ImageDraw
import PIL
from mouse_session_loader import load_mouse_sessions
from shapely import geometry
from scipy import ndimage
from random import shuffle
from keras import backend as K
import os
from pattern_discovery.tools.misc import get_continous_time_periods
import scipy as sci_py
import scipy.io as sio
import sys
import platform
from pattern_discovery.tools.signal import smooth_convolve
from tensorflow.python.client import device_lib
from alt_model_checkpoint import AltModelCheckpoint

device_lib.list_local_devices()


class DataForMs(p_disc_tools_param.Parameters):
    def __init__(self, path_data, result_path, time_str=None):
        if time_str is None:
            self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        else:
            self.time_str = time_str
        super().__init__(path_results=result_path, time_str=self.time_str, bin_size=1)
        self.path_data = path_data
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None


class RasterData:

    def __init__(self, ms, first_frame,
                 window_len, predict_animal_weight, n_augmentations_to_perform=0):
        self.ms = ms
        self.first_frame = first_frame
        self.window_len = window_len
        # weight to apply, use by the model to produce the loss function result
        self.weight = 1

        # number of transformation to perform on this movie
        self.n_augmentations_to_perform = n_augmentations_to_perform
        # set of functions used for data augmentation, one will be selected when copying a movie
        self.data_augmentation_fct_list = []
        # used if a movie_data has been copied
        self.data_augmentation_fct = None
        self.predict_animal_weight = predict_animal_weight

        self.n_available_augmentation_fct = n_augmentations_to_perform
        for n in range(self.n_available_augmentation_fct):
            self.data_augmentation_fct_list.append(lambda raster: np.random.shuffle(raster))

    def get_labels(self):
        if self.predict_animal_weight:
            return self.ms.weight
        else:
            return self.ms.age

    def __eq__(self, other):
        if self.ms.description != other.ms.description:
            return False
        if self.first_frame != self.first_frame:
            return False
        return True

    def copy(self):
        raster_copy = RasterData(ms=self.ms, first_frame=self.first_frame,
                                 window_len=self.window_len, predict_animal_weight=self.predict_animal_weight)

        raster_copy.data_augmentation_fct = self.data_augmentation_fct
        return raster_copy

    def add_n_augmentation(self, n_augmentation):
        self.n_augmentations_to_perform = min(self.n_augmentations_to_perform + n_augmentation,
                                              self.n_available_augmentation_fct)

    def pick_a_transformation_fct(self):
        if len(self.data_augmentation_fct_list) > 0:
            fct = self.data_augmentation_fct_list[0]
            self.data_augmentation_fct_list = self.data_augmentation_fct_list[1:]
            return fct
        return None


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
    # return Lambda(lambda a: K.sigmoid(a) * a)(x)


def box_plot_data_by_age(data_dict, title, filename,
                         y_label, param, colors=None,
                         path_results=None, y_lim=None,
                         x_label=None, with_scatters=True,
                         scatters_with_same_colors=None,
                         scatter_size=20,
                         scatter_alpha=0.5,
                         special_scatters_coords=None,
                         special_scatters_color=None,
                         link_medians=True,
                         color_link_medians="red",
                         link_special_scatters = True,
                         n_sessions_dict=None,
                         background_color="black",
                         labels_color="white", save_formats="pdf"):
    """

    :param data_dict:
    :param n_sessions_dict: should be the same keys as data_dict, value is an int reprenseing the number of sessions
    that gave those data (N), a n will be display representing the number of poins in the boxplots if n != N
    :param title:
    :param filename:
    :param y_label:
    :param y_lim: tuple of int,
    :param scatters_with_same_colors: scatter that have the same index in the data_dict,, will be colors
    with the same colors, using the list of colors given by scatters_with_same_colors
    :param param: Contains a field name colors used to color the boxplot
    :param save_formats:
    :return:
    """
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    colorfull = (colors is not None)

    median_color = background_color if colorfull else labels_color

    ax1.set_facecolor(background_color)

    fig.patch.set_facecolor(background_color)

    labels = []
    data_list = []
    medians_values = []
    special_scatters_x = []
    special_scatters_y = []
    for age, data in data_dict.items():
        data_list.append(data)
        medians_values.append(np.median(data))
        label = age
        if n_sessions_dict is None:
            # label += f"\n(n={len(data)})"
            pass
        else:
            n_sessions = n_sessions_dict[age]
            if n_sessions != len(data):
                label += f"\n(N={n_sessions}, n={len(data)})"
            else:
                label += f"\n(N={n_sessions})"
        labels.append(label)
        if special_scatters_coords is not None:
            n_scatters = len(special_scatters_coords[age])
            special_scatters_x.extend([len(data_list)]*n_scatters)
            special_scatters_y.extend(special_scatters_coords[age])

    bplot = plt.boxplot(data_list, patch_artist=colorfull,
                        labels=labels, sym='', zorder=30)  # whis=[5, 95], sym='+'
    # color=["b", "cornflowerblue"],
    # fill with colors

    # edge_color="silver"

    for element in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(bplot[element], color="white")

    for element in ['means', 'medians']:
        plt.setp(bplot[element], color=median_color)

    if colorfull:
        if colors is None:
            colors = param.colors[:len(data_dict)]
        else:
            while len(colors) < len(data_dict):
                colors.extend(colors)
            colors = colors[:len(data_dict)]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            r, g, b, a = patch.get_facecolor()
            # for transparency purpose
            patch.set_facecolor((r, g, b, 0.8))

    if with_scatters:
        for data_index, data in enumerate(data_list):
            # Adding jitter
            x_pos = [1 + data_index + ((np.random.random_sample() - 0.5) * 0.5) for x in np.arange(len(data))]
            y_pos = data
            font_size = 3
            colors_scatters = []
            if scatters_with_same_colors is not None:
                while len(colors_scatters) < len(y_pos):
                    colors_scatters.extend(scatters_with_same_colors)
            else:
                colors_scatters = [colors[data_index]]
            ax1.scatter(x_pos, y_pos,
                        color=colors_scatters[:len(y_pos)],
                        alpha=scatter_alpha,
                        marker="o",
                        edgecolors=background_color,
                        s=scatter_size, zorder=1)
    if special_scatters_coords is not None:
        ax1.scatter(special_scatters_x, special_scatters_y,
                    color=special_scatters_color,
                    alpha=1,
                    marker="o",
                    edgecolors=background_color,
                    s=scatter_size/2, zorder=35)
        if link_special_scatters:
            ax1.plot(special_scatters_x, special_scatters_y, color=special_scatters_color, linewidth=2)
    if link_medians:
        ax1.plot(np.arange(1, len(medians_values)+1), medians_values, zorder=36, color=color_link_medians, linewidth=2)
    # plt.xlim(0, 100)
    plt.title(title)

    ax1.set_ylabel(f"{y_label}", fontsize=30, labelpad=20)
    if y_lim is not None:
        ax1.set_ylim(y_lim[0], y_lim[1])
    if x_label is not None:
        ax1.set_xlabel(x_label, fontsize=30, labelpad=20)
    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)

    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=5)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)
    xticks = np.arange(1, len(data_dict) + 1)
    ax1.set_xticks(xticks)
    # removing the ticks but not the labels
    ax1.xaxis.set_ticks_position('none')
    # sce clusters labels
    ax1.set_xticklabels(labels)

    # padding between ticks label and  label axis
    # ax1.tick_params(axis='both', which='major', pad=15)
    fig.tight_layout()
    # adjust the space between axis and the edge of the figure
    # https://matplotlib.org/faq/howto_faq.html#move-the-edge-of-an-axes-to-make-room-for-tick-labels
    # fig.subplots_adjust(left=0.2)

    if isinstance(save_formats, str):
        save_formats = [save_formats]

    if path_results is None:
        path_results = param.path_results
    for save_format in save_formats:
        fig.savefig(f'{path_results}/{filename}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())

    plt.close()


class DataGenerator(keras.utils.Sequence):
    """
    Based on an exemple found in https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Feed to keras to generate data
    """

    # 'Generates data for Keras'
    def __init__(self, data_list, raster_generator,
                 batch_size, window_len, n_cells, with_augmentation,
                 is_shuffle=True):
        """

        :param data_list: a list containing the information to get the data. Each element
        is an instance of RasterData
        :param batch_size:
        :param window_len:
        :param with_augmentation:
        :param is_shuffle:
        """
        # 'Initialization'
        self.window_len = window_len
        self.input_shape = (n_cells, self.window_len, 1)
        self.batch_size = batch_size
        self.data_list = data_list
        self.with_augmentation = with_augmentation
        self.raster_generator = raster_generator

        if self.with_augmentation:
            # augment the dict now, adding to the key a str representing the transformation and same for
            # in the value
            self.prepare_augmentation()

        # useful for the shuffling
        self.n_samples = len(self.data_list)
        # self.n_channels = n_channels
        # self.n_classes = n_classes
        self.is_shuffle = is_shuffle
        self.indexes = None
        self.on_epoch_end()

    def prepare_augmentation(self):
        n_samples = len(self.data_list)
        print(f"n_samples before data augmentation: {n_samples}")
        new_data = []

        for index_data in np.arange(n_samples):
            raster_data = self.data_list[index_data]
            # we will do as many transformation as indicated in raster_data.n_augmentations_to_perform
            if raster_data.n_augmentations_to_perform == 0:
                continue
            for t in np.arange(raster_data.n_augmentations_to_perform):
                if t >= raster_data.n_available_augmentation_fct:
                    break
                new_movie = raster_data.copy()
                new_movie.data_augmentation_fct = raster_data.pick_a_transformation_fct()
                new_data.append(new_movie)

        self.data_list.extend(new_data)

        print(f"n_samples after data augmentation: {len(self.data_list)}")

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print(f"len(indexes) {len(indexes)}")
        # Find list of IDs
        data_list_tmp = [self.data_list[k] for k in indexes]

        # Generate data
        data, labels, sample_weights = self.__data_generation(data_list_tmp)

        return data, labels, sample_weights

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        # TODO: each mini-batch should have the same proportion of data (neuropil, transients, fake transients)
        # TODO: create a function that shuffle the index with respect of this information
        if self.is_shuffle:
            np.random.shuffle(self.indexes)
        # self.data_keys = list(self.data_dict.keys())
        # if self.is_shuffle:
        #     shuffle(self.data_keys)

    def __data_generation(self, data_list_tmp):
        # len(data_list_tmp) == self.batch_size
        # 'Generates data containing batch_size samples' # data : (self.batch_size, *dim, n_channels)
        # Initialization

        # data, data_masked, labels = generate_movies_from_metadata(movie_data_list=data_list_tmp,
        #                                                           window_len=self.window_len,
        #                                                           max_width=self.max_width,
        #                                                           max_height=self.max_height,
        #                                                           pixels_around=self.pixels_around,
        #                                                           buffer=self.buffer,
        #                                                           source_profiles_dict=self.source_profiles_dict)
        # print(f"self.raster_generator {len(data_list_tmp)}")
        data_dict, labels = self.raster_generator.generate_raster_from_metadata(raster_data_list=data_list_tmp)
        # print(f"__data_generation data.shape {data.shape}")
        # put more weight to the active frames
        sample_weights = np.ones(labels.shape[0])
        for index_batch, movie_data in enumerate(data_list_tmp):
            sample_weights[index_batch] = movie_data.weight

        return data_dict, labels, sample_weights


class RasterGenerator:
    """
    Used to generate raster, that will be produce for training data during each mini-batch.
    This is an abstract classes that need to have heritage.
    The function generate_raster_from_metadata will be used to produced those rasters, the number
    vary depending on the class instantiated
    """

    def __init__(self, window_len, n_cells, n_classes):
        self.window_len = window_len
        self.n_cells = n_cells
        self.n_classes = n_classes

    # self.n_inputs shouldn't be changed

    def get_nb_inputs(self):
        return self.n_inputs

    def generate_raster_from_metadata(self, raster_data_list, with_labels=True):
        pass


class ClassicRasterGenerator(RasterGenerator):
    """
    Will generate one input being the masked cell (the one we focus on) and the second input
    would be the whole patch with all pixels given
    """

    def __init__(self, window_len, n_cells,
                 n_classes, use_raster_dur=True, age_index_mapping=None,
                 predict_animal_weight=False):
        super().__init__(window_len=window_len, n_cells=n_cells, n_classes=n_classes)
        self.use_raster_dur = use_raster_dur
        self.age_index_mapping = age_index_mapping
        self.predict_animal_weight = predict_animal_weight

    def generate_raster_from_metadata(self, raster_data_list, with_labels=True):
        # print(f"Start generate_raster_from_metadata")
        batch_size = len(raster_data_list)
        # print(f"batch_size {batch_size}")
        data = np.zeros((batch_size, self.n_cells, self.window_len, 1))
        if with_labels:
            if self.n_classes <= 1:
                if self.predict_animal_weight:
                    labels = np.zeros(batch_size)
                else:
                    labels = np.zeros(batch_size, dtype="uint8")
            else:
                labels = np.zeros((batch_size, self.n_classes), dtype="uint8")
        # Generate data
        for index_batch, raster_data in enumerate(raster_data_list):
            ms = raster_data.ms
            first_frame = raster_data.first_frame

            augmentation_fct = raster_data.data_augmentation_fct
            if self.use_raster_dur:
                spike_nums_to_use = ms.spike_struct.spike_nums_dur
            else:
                spike_nums_to_use = ms.spike_struct.spike_nums

            raster = spike_nums_to_use[np.arange(self.n_cells),
                     first_frame:first_frame + self.window_len]

            # doing augmentation if the function exists
            if augmentation_fct is not None:
                raster = raster.copy()
                new_raster = augmentation_fct(raster)
                if new_raster is not None:
                    raster = new_raster

            raster = raster.reshape((raster.shape[0], raster.shape[1], 1))
            data[index_batch] = raster
            if self.n_classes == 1:
                if self.predict_animal_weight:
                    labels[index_batch] = ms.weight
                else:
                    labels[index_batch] = ms.age
            else:
                ms_labels = np.zeros(self.n_classes)
                ms_labels[self.age_index_mapping[ms.age]] = 1
                labels[index_batch] = ms_labels
        # print("End generate_raster_from_metadata")
        if with_labels:
            return {"input_1": data}, labels
        else:
            return {"input_1": data}

    def __str__(self):
        return f"{self.n_inputs} inputs. Raster"


def load_data_for_prediction(ms, window_len, overlap_value, predict_animal_weight,
                             n_augmentations_to_perform=0):
    raster = ms.spike_struct.spike_nums_dur
    if raster is None:
        print(f"{ms.description}: raster is None")
    n_frames = raster.shape[1]
    raster_data_list = []
    frames_step = int(np.ceil(window_len * (1 - overlap_value)))
    indices_frames = np.arange(0, raster.shape[1], frames_step)

    for i, frame_index in enumerate(indices_frames):
        break_it = False
        if (frame_index + window_len) == n_frames:
            break_it = True
        elif (frame_index + window_len) > n_frames:
            # in case the number of frames is not divisible by sliding_window_len
            break

        raster_data = RasterData(ms=ms, first_frame=frame_index,
                                 window_len=window_len,
                                 n_augmentations_to_perform=n_augmentations_to_perform,
                                 predict_animal_weight=predict_animal_weight)

        raster_data_list.append(raster_data)

        # we will do as many transformation as indicated in raster_data.n_augmentations_to_perform
        if raster_data.n_augmentations_to_perform == 0:
            continue
        for t in np.arange(raster_data.n_augmentations_to_perform):
            if t >= raster_data.n_available_augmentation_fct:
                break
            new_raster = raster_data.copy()
            new_raster.data_augmentation_fct = raster_data.pick_a_transformation_fct()
            raster_data_list.append(new_raster)

        if break_it:
            break

    return raster_data_list


def load_data_for_generator(param, split_values,
                            window_len,
                            overlap_value,
                            n_augmentations_to_perform,
                            predict_animal_weight,
                            with_shuffling=False,
                            seed_value=None):
    print("load_data_for_generator")

    use_small_sample = True
    if use_small_sample:
        ms_to_use = ["p5_19_03_25_a000_ms", "p6_18_02_07_a001_ms", "p7_18_02_08_a000_ms", "p7_19_03_05_a000_ms",
                     "p8_18_10_17_a001_ms", "p9_17_12_06_a001_ms", "p9_19_02_20_a001_ms", "p10_19_02_21_a002_ms",
                     "p11_17_11_24_a001_ms", "p12_17_11_10_a002_ms", "p13_18_10_29_a000_ms",
                     "p14_18_10_23_a000_ms", "p16_18_11_01_a002_ms", "p19_19_04_08_a000_ms", "p21_19_04_10_a000_ms"]
    else:
        raise Exception("NOT TODAY")

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False)

    ages = []
    train_data = []
    valid_data = []
    test_data = []
    # split_order will indicated the order in which data_set should be filled (train, validation and test data)
    # it is shuffled so it's change each time
    split_order = np.arange(3)

    for ms_str in ms_to_use:
        print(f"ms_str {ms_str}")
        ms = ms_str_to_ms_dict[ms_str]
        ages.append(ms.age)

        # we make it simple by keeping the last part for test
        n_frames = ms.spike_struct.spike_nums_dur.shape[1]
        index_so_far = 0
        for split_index in split_order:
            if split_index > 0:
                # then we create validation and test dataset with no data transformation
                frames_step = window_len
                if split_index == 1:
                    data_list_to_fill = valid_data
                else:
                    data_list_to_fill = test_data
            else:
                # we create training dataset with overlap
                # then we slide the window
                # frames index of the beginning of each movie
                frames_step = int(np.ceil(window_len * (1 - overlap_value)))
                data_list_to_fill = train_data

            start_index = index_so_far
            end_index = start_index + int(n_frames * split_values[split_index])
            index_so_far = end_index
            n_frames_for_loop = n_frames
            indices_frames = np.arange(start_index, end_index, frames_step)

            for i, frame_index in enumerate(indices_frames):
                break_it = False
                if (frame_index + window_len) == n_frames_for_loop:
                    break_it = True
                elif (frame_index + window_len) > n_frames_for_loop:
                    # in case the number of frames is not divisible by sliding_window_len
                    break

                raster_data = RasterData(ms=ms, first_frame=frame_index,
                                         window_len=window_len,
                                         n_augmentations_to_perform=n_augmentations_to_perform,
                                         predict_animal_weight=predict_animal_weight)
                data_list_to_fill.append(raster_data)
                if break_it:
                    break

    return train_data, valid_data, test_data, ages


def build_dense_layer(input_shape, n_classes, dropout_value=-1, dropout_rate=0.4):
    img_input = layers.Input(shape=input_shape)
    x = layers.Flatten(name='flatten')(img_input)
    x = layers.Dense(128, name='fc1')(x)
    # x = layers.Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dense(256, name='fc2')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = BatchNormalization()(x)
    x = BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dense(512, name='fc3')(x)  # 512
    # x = layers.Dropout(dropout_rate)(x)
    # x = BatchNormalization()(x)
    x = BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dense(1024, name='fc4')(x)
    x = BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if n_classes == 1:
        # try to guess the right answer
        x = layers.Dense(n_classes, activation='linear', name='predictions')(x)
    else:
        x = layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    # Create model.
    model = Model(inputs=img_input, outputs=x, name='dense_layer')
    return model


def build_vgg_16_model(input_shape, n_classes, dropout_value=-1):
    img_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    include_top = True
    pooling = 'avg'
    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(128, activation='relu', name='fc1')(x)
        # x = layers.Dense(4096, activation='relu', name='fc2')(x)
        if n_classes == 1:
            # try to guess the right answer
            x = layers.Dense(n_classes, activation='linear', name='predictions')(x)
        else:
            x = layers.Dense(n_classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        if n_classes == 1:
            # try to guess the right answer
            x = layers.Dense(n_classes, activation='linear', name='predictions')(x)
        else:
            x = layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    # Create model.
    model = Model(inputs=img_input, outputs=x, name='vgg16')
    return model


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_training_and_validation_values(history, key_name, result_path, param):
    history_dict = history.history
    train_values = history_dict[key_name]
    val_values = history_dict['val_' + key_name]
    epochs = range(1, len(val_values) + 1)
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1],
                                         'width_ratios': [1]},
                            figsize=(5, 5))
    ax1.plot(epochs, smooth_curve(train_values), 'bo', label=f'Training {key_name}')
    ax1.plot(epochs, smooth_curve(val_values), 'b', label=f'Validation {key_name}')
    # ax1.plot(epochs, train_values, 'bo', label=f'Training {key_name}')
    # ax1.plot(epochs, val_values, 'b', label=f'Validation {key_name}')
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


def predict_age(ms_to_use, param, predict_animal_weight=False):
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
              '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
              '#74add1', '#4575b4', '#313695']

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False,
                                            for_transient_classifier=True)

    # TODO find window_len from model and do sliding_window
    # save result in a boxplot for each sessoion

    # loading model
    path_to_tc_model = param.path_data + "deep_age_predictor_model/"
    json_file = None
    weights_file = None
    # checking if the path exists
    if os.path.isdir(path_to_tc_model):
        # then we look for the json file (representing the model architecture) and the weights file
        # we will assume there is only one file of each in this directory
        # look for filenames in the first directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(path_to_tc_model):
            for file_name in local_filenames:
                if file_name.endswith(".json"):
                    json_file = path_to_tc_model + file_name
                if "weights" in file_name:
                    weights_file = path_to_tc_model + file_name
            # looking only in the top directory
            break
    if (json_file is None) or (weights_file is None):
        raise Exception("model could not be loaded")

    start_time = time.time()
    index_color= 0
    color_dict_by_age = {}
    color_dict_by_session = {}
    label_value_dict = {}
    # Model reconstruction from JSON file
    with open(json_file, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(weights_file)
    stop_time = time.time()
    print(f"Time for loading model: "
          f"{np.round(stop_time - start_time, 3)} s")
    ages = []
    for ms_str in ms_to_use:
        ms = ms_str_to_ms_dict[ms_str]
        ages.append(ms.age)

    n_ages = len(np.unique(ages))

    # will associate each age to an index, useful if use_mutli_class is True
    age_index_mapping = dict()
    for age_index, age in enumerate(np.unique(ages)):
        age_index_mapping[age] = age_index

    results_dict_by_age = SortedDict()

    for ms_str in ms_to_use:
        ms = ms_str_to_ms_dict[ms_str]

        n_cells = model.layers[0].output_shape[1]
        window_len = model.layers[0].output_shape[2]
        print(f"n_cells {n_cells}, window_len {window_len}")

        # Determining how many classes were used
        if len(model.layers[-1].output_shape) == 2:
            n_classes = 1
        else:
            n_classes = model.layers[-1].output_shape[2]

        raster_generator = ClassicRasterGenerator(window_len=window_len,
                                                  n_cells=n_cells, n_classes=n_classes,
                                                  age_index_mapping=age_index_mapping,
                                                  predict_animal_weight=predict_animal_weight)
        raster_data_list = load_data_for_prediction(ms=ms, window_len=window_len,
                                                    predict_animal_weight=predict_animal_weight,
                                                    overlap_value=0, n_augmentations_to_perform=15)

        raster_data_dict, raster_labels = \
            raster_generator.generate_raster_from_metadata(raster_data_list=raster_data_list)
        predictions = model.predict(raster_data_dict)
        for index, prediction in enumerate(predictions):
            if n_classes == 1:
                label_key = str(raster_labels[index])
                if predict_animal_weight:
                    label_key = label_key + f"-p{ms.age}"
                label_value_dict[label_key] = raster_labels[index]

                if ms.age not in color_dict_by_age:
                    color_dict_by_age[ms.age] = colors[index_color]
                    index_color += 1
                color_dict_by_session[label_key] = color_dict_by_age[ms.age]
                if label_key not in results_dict_by_age:
                    results_dict_by_age[label_key] = []

                prediction = list(prediction)
                results_dict_by_age[label_key].extend(prediction)

                    # results_dict_by_age[str(raster_labels[index])].append(prediction)
                print(f"label: {raster_labels[index]}, "
                      f"prediction {prediction}")
            else:
                print(f"label: {' '.join(map(str, raster_labels[index]))}, "
                      f"prediction {' '.join(map(str, np.round(prediction, 2)))}")
    if predict_animal_weight:
        filename = f"weight_predictor"
        y_label = f"Predicted weight (g)"
    else:
        filename = f"age_predictor"
        y_label = f"Predicted age"
    # print(f"len(results_dict_by_age) {len(results_dict_by_age)}")
    colors_by_age = []
    special_scatters_coords = {}
    for key in results_dict_by_age.keys():
        colors_by_age.append(color_dict_by_session[key])
        special_scatters_coords[key] = [label_value_dict[key]]
    special_scatters_color = "white"
    box_plot_data_by_age(data_dict=results_dict_by_age, title="",
                         filename=filename,
                         y_label=y_label,
                         colors=colors_by_age, with_scatters=True,
                         path_results=param.path_results, scatter_size=200,
                         special_scatters_coords=special_scatters_coords,
                         special_scatters_color=special_scatters_color,
                         param=param, save_formats="pdf")

def guessing_age_game(ms_to_use, param):
    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False,
                                            for_transient_classifier=True)
    window_len = 1000
    n_rasters = 0
    n_cells = 400

    for ms_str in ms_to_use:
        ms = ms_str_to_ms_dict[ms_str]
        raster = ms.spike_struct.spike_nums_dur
        n_frames = raster.shape[1]
        frames_step = window_len
        indices_frames = np.arange(0, raster.shape[1], frames_step)

        for i, frame_index in enumerate(indices_frames):
            if (frame_index + window_len) > n_frames:
                # in case the number of frames is not divisible by sliding_window_len
                break
            n_rasters += 1

    rasters_ids = np.arange(n_rasters)
    np.random.shuffle(rasters_ids)

    raster_id_mapping = SortedDict()
    list_rasters = []
    raster_info = []

    for ms_str in ms_to_use:
        ms = ms_str_to_ms_dict[ms_str]

        raster = ms.spike_struct.spike_nums_dur
        n_frames = raster.shape[1]
        raster_data_list = []
        frames_step = window_len
        indices_frames = np.arange(0, raster.shape[1], frames_step)

        for i, frame_index in enumerate(indices_frames):
            break_it = False
            if (frame_index + window_len) == n_frames:
                break_it = True
            elif (frame_index + window_len) > n_frames:
                # in case the number of frames is not divisible by sliding_window_len
                break
            raster_data = np.copy(raster[:n_cells, frame_index:frame_index + window_len])
            np.random.shuffle(raster_data)
            list_rasters.append(raster_data)
            raster_info.append(ms.description + " " + str(frame_index)
                               + " - " + str(frame_index + window_len-1) + " " +
                               str(ms.weight))

            if break_it:
                break

    shuffle_raster_indices = np.arange(len(list_rasters))
    np.random.shuffle(shuffle_raster_indices)
    for index_raster in shuffle_raster_indices:
        raster = list_rasters[index_raster]
        plot_spikes_raster(spike_nums=raster, param=param,
                               spike_train_format=False,
                               title="",
                               file_name=f"raster_id_{rasters_ids[index_raster]}",
                               y_ticks_labels=np.arange(n_cells),
                               y_ticks_labels_size=2,
                               save_raster=True,
                               show_raster=False,
                               plot_with_amplitude=False,
                               show_sum_spikes_as_percentage=True,
                               span_area_only_on_raster=False,
                               spike_shape='o',
                               spike_shape_size=0.5,
                               without_activity_sum=True,
                               save_formats=["png"],
                               without_time_str_in_file_name=True)
        raster_id_mapping[rasters_ids[index_raster]] = raster_info[index_raster]

    file_name = f'{param.path_results}/raster_id_mapping.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"Rasters id -> ms & frames & weight" + '\n')
        for raster_id, ms_info in raster_id_mapping.items():
            file.write(f"{raster_id}: {ms_info}" + '\n')

def deep_age_predictor_main():
    """

    :return:
    """

    # Multiclass classification: A classification task where each input sample
    # should be categorized into more than two categorie
    """
    model.add(layers.Dense(46, activation='softmax'))
    The best loss function to use in this case is categorical_crossentropy . It measures
    the distance between two probability distributions: here, between the probability dis-
    tribution output by the network and the true distribution of the labels. By minimizing
    the distance between these two distributions, you train the network to output some-
    thing as close as possible to the true labels.
    
    """
    # if the target is a continuous scalar value (Scalar regression) problem predicting a value
    """
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    Note that you compile the network with the mse loss function—mean squared error,
    the square of the difference between the predictions and the targets. This is a widely
    used loss function for regression problems.
    You’re also monitoring a new metric during training: mean absolute error ( MAE ). It’s
    the absolute value of the difference between the predictions and the targets

    """

    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")
    if root_path[-1] == '\n':
        root_path = root_path[:-1]
    path_data = root_path + "data/"
    result_path = root_path + "results_classifier/"
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    result_path = result_path + "/" + time_str
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    param = DataForMs(path_data=path_data, result_path=result_path, time_str=time_str)

    predict_animal_weight = True

    go_predict_age = True
    generate_guessing_age_game = False
    if go_predict_age or generate_guessing_age_game:
        # ms_trained = ["p5_19_03_25_a000_ms", "p6_18_02_07_a001_ms", "p7_18_02_08_a000_ms", "p7_19_03_05_a000_ms",
        #              "p8_18_10_17_a001_ms", "p9_17_12_06_a001_ms", "p9_19_02_20_a001_ms", "p10_19_02_21_a002_ms",
        #              "p11_17_11_24_a001_ms", "p12_17_11_10_a002_ms", "p13_18_10_29_a000_ms",
        #              "p14_18_10_23_a000_ms", "p16_18_11_01_a002_ms", "p19_19_04_08_a000_ms", "p21_19_04_10_a000_ms"]

        # weight available
        ms_with_weights = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms", "p6_18_02_07_a001_ms", "p6_18_02_07_a001_ms",
                          "p6_18_02_07_a002_ms", "p7_18_02_08_a000_ms", "p7_18_02_08_a001_ms", "p7_18_02_08_a002_ms",
                          "p7_18_02_08_a003_ms", "p7_19_03_05_a000_ms", "p7_19_03_27_a000_ms", "p7_19_03_27_a001_ms",
                          "p8_18_10_17_a000_ms","p8_18_10_17_a001_ms", "p8_18_10_24_a005_ms", "p8_19_03_19_a000_ms",

                           "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms", "p9_18_09_27_a003_ms", "p9_19_02_20_a000_ms",
                          "p9_19_02_20_a001_ms", "p9_19_02_20_a002_ms", "p9_19_02_20_a003_ms", "p9_19_03_14_a000_ms",
                          "p9_19_03_14_a001_ms", "p9_19_03_22_a000_ms", "p9_19_03_22_a001_ms", "p10_17_11_16_a003_ms",
                          "p10_19_02_21_a002_ms", "p10_19_02_21_a003_ms", "p10_19_02_21_a005_ms",
                          "p10_19_03_08_a000_ms", "p10_19_03_08_a001_ms", "p11_17_11_24_a000_ms",
                          "p11_17_11_24_a001_ms", "p11_19_02_15_a000_ms", "p12_171110_a000_ms", "p12_17_11_10_a002_ms",
                          "p13_18_10_29_a000_ms",  "p14_18_10_23_a000_ms",
                          "p14_18_10_30_a001_ms", "p16_18_11_01_a002_ms",
                          "p19_19_04_08_a000_ms", "p19_19_04_08_a001_ms", "p21_19_04_10_a000_ms",
                          "p21_19_04_10_a001_ms", "p41_19_04_30_a000_ms"]
        # Less than 400 cells: "p13_18_10_29_a001_ms"
        # fix prediction: ,

        ms_without_weights = ["p7_171012_a000_ms", "p7_17_10_18_a002_ms", "p7_17_10_18_a004_ms",
                              "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms", "p13_19_03_11_a000_ms"]
        ms_for_benchmarks = ["p5_19_03_25_a001_ms", "p6_18_02_07_a002_ms", "p7_19_03_27_a000_ms",
                             "p8_18_10_24_a005_ms", "p8_19_03_19_a000_ms", "p9_19_02_20_a000_ms",
                             "p10_19_02_21_a005_ms", "p11_17_11_24_a000_ms", "p12_171110_a000_ms",
                             "p14_18_10_30_a001_ms", "p19_19_04_08_a001_ms",
                             "p21_19_04_10_a001_ms", "p41_19_04_30_a000_ms"] # "p13_18_10_29_a001_ms"
        ms_for_benchmarks = ms_with_weights
        if generate_guessing_age_game:
            guessing_age_game(ms_to_use=ms_for_benchmarks, param=param)
        else:
            predict_age(ms_to_use=ms_for_benchmarks, param=param, predict_animal_weight=predict_animal_weight)
        return

    ######## PARAMS ######
    n_gpus = 1
    n_epochs = 20
    use_multi_class = False
    # multiplying by the number of gpus used as batches will be distributed to each GPU
    batch_size = 8 * n_gpus
    dropout_value = 0.5
    n_augmentations_to_perform = 10
    pixels_around = 0
    with_augmentation_for_training_data = True
    buffer = 1
    # between training, validation  and test data
    split_values = [0.7, 0.2, 0.1]
    optimizer_choice = "RMSprop"

    with_learning_rate_reduction = True
    learning_rate_reduction_patience = 5
    with_early_stopping = True
    early_stop_patience = 15  # 10
    model_descr = ""
    with_shuffling = True
    seed_value = 42  # use None to not use seed
    workers = 10
    window_len = 1250
    n_cells = 400
    overlap_value = 0.9

    train_data_list, valid_data_list, test_data_list, \
    ages = \
        load_data_for_generator(param,
                                split_values=split_values,
                                window_len=window_len,
                                overlap_value=overlap_value,
                                with_shuffling=with_shuffling,
                                n_augmentations_to_perform=n_augmentations_to_perform,
                                seed_value=seed_value,
                                predict_animal_weight=predict_animal_weight)
    print(f"len(test_data_list) {len(test_data_list)}")

    n_ages = len(np.unique(ages))

    # will associate each age to an index, useful if use_mutli_class is True
    age_index_mapping = dict()
    for age_index, age in enumerate(np.unique(ages)):
        age_index_mapping[age] = age_index

    # multi_class for animal weight
    if predict_animal_weight:
        use_multi_class = False

    if use_multi_class:
        n_classes = n_ages
    else:
        n_classes = 1
    if n_classes == 1:
        loss_fct = 'mse'
        metrics = ['mae']
    else:
        loss_fct = 'categorical_crossentropy'
        metrics = ['accuracy']

    raster_patch_generator_for_training = ClassicRasterGenerator(window_len=window_len,
                                                                 n_cells=n_cells, n_classes=n_classes,
                                                                 age_index_mapping=age_index_mapping,
                                                                 predict_animal_weight=predict_animal_weight)
    raster_patch_generator_for_validation = ClassicRasterGenerator(window_len=window_len,
                                                                   n_cells=n_cells, n_classes=n_classes,
                                                                   age_index_mapping=age_index_mapping,
                                                                   predict_animal_weight=predict_animal_weight)
    raster_patch_generator_for_test = ClassicRasterGenerator(window_len=window_len,
                                                             n_cells=n_cells, n_classes=n_classes,
                                                             age_index_mapping=age_index_mapping,
                                                             predict_animal_weight=predict_animal_weight)

    params_generator = {
        'batch_size': batch_size,
        'window_len': window_len,
        'n_cells': n_cells,
        'is_shuffle': True}
    # Generators
    start_time = time.time()
    training_generator = DataGenerator(train_data_list,
                                       with_augmentation=with_augmentation_for_training_data,
                                       raster_generator=raster_patch_generator_for_training,
                                       **params_generator)
    validation_generator = DataGenerator(valid_data_list, with_augmentation=True,
                                         raster_generator=raster_patch_generator_for_validation,
                                         **params_generator)

    stop_time = time.time()
    print(f"Time to create generator: "
          f"{np.round(stop_time - start_time, 3)} s")
    # (sliding_window_size, max_width, max_height, 1)
    # sliding_window in frames, max_width, max_height: in pixel (100, 25, 25, 1) * n_movie
    input_shape = training_generator.input_shape

    # start_time = time.time()
    if n_gpus == 1:
        # model = build_vgg_16_model(input_shape=input_shape, n_classes=n_classes)
        model = build_dense_layer(input_shape=input_shape, n_classes=n_classes)
        parallel_model = model
    else:
        with tf.device('/cpu:0'):
            model = build_vgg_16_model(input_shape=input_shape, n_classes=n_classes)
        parallel_model = multi_gpu_model(model, gpus=n_gpus)

    print(model.summary())
    data_predicted_descr = "age"
    if predict_animal_weight:
        data_predicted_descr = "weight"
    # Save the model architecture
    with open(
            f'{param.path_results}/{data_predicted_descr}_predictor_model_architecture_{model_descr}_'
            f'{param.time_str}.json',
            'w') as f:
        f.write(model.to_json())

    if optimizer_choice == "adam":
        optimizer = adam(lr=0.001, epsilon=1e-08, decay=0.0)
    elif optimizer_choice == "SGD":
        # default parameters: lr=0.01, momentum=0.0, decay=0.0, nesterov=False
        optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    else:
        # default parameters: lr=0.001, rho=0.9, epsilon=None, decay=0.0
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    parallel_model.compile(optimizer=optimizer,
                           loss=loss_fct,
                           metrics=metrics)

    # Set a learning rate annealer
    # from: https://www.kaggle.com/shahariar/keras-swish-activation-acc-0-996-top-7
    if n_classes == 1:
        monitor = "val_mean_absolute_error"
        mode = "min"
    else:
        monitor = "val_acc"
        mode = "max"
    learning_rate_reduction = ReduceLROnPlateau(monitor=monitor,
                                                patience=learning_rate_reduction_patience,
                                                verbose=1,
                                                factor=0.5,
                                                mode=mode,
                                                min_lr=0.0001)  # used to be: 0.00001

    # callbacks to be execute during training
    # A callback is a set of functions to be applied at given stages of the training procedure.
    callbacks_list = []
    if with_learning_rate_reduction:
        callbacks_list.append(learning_rate_reduction)

    if with_early_stopping:
        callbacks_list.append(EarlyStopping(monitor=monitor, min_delta=0,
                                            patience=early_stop_patience, mode=mode,
                                            restore_best_weights=True))

    with_model_check_point = True
    # not very useful to save best only if we use EarlyStopping
    if with_model_check_point:
        end_file_path = f"_{param.time_str}.h5"
        if predict_animal_weight:
            file_path = param.path_results + "/weight_predictor_weights_{epoch:02d}" + end_file_path
        else:
            file_path = param.path_results + "/age_predictor_weights_{epoch:02d}" + end_file_path
        # callbacks_list.append(ModelCheckpoint(filepath=file_path, monitor="val_acc", save_best_only="True",
        #                                       save_weights_only="True", mode="max"))
        # https://github.com/TextpertAi/alt-model-checkpoint
        callbacks_list.append(AltModelCheckpoint(file_path, model, save_weights_only="True",
                                                 save_best_only="True"))

    history = parallel_model.fit_generator(generator=training_generator,
                                           validation_data=validation_generator,
                                           epochs=n_epochs,
                                           use_multiprocessing=True,
                                           workers=workers,
                                           callbacks=callbacks_list, verbose=1)

    print(f"history.history.keys() {history.history.keys()}")

    history_dict = history.history

    show_plots = True

    if show_plots:
        key_names = ["loss"]
        if n_classes == 1:
            key_names.append("mean_absolute_error")
        else:
            key_names.append("acc")
        for key_name in key_names:
            plot_training_and_validation_values(history=history, key_name=key_name,
                                                result_path=result_path, param=param)

    if len(test_data_list) > 0:

        start_time = time.time()
        test_data_dict, test_labels = \
            raster_patch_generator_for_test.generate_raster_from_metadata(raster_data_list=test_data_list)
        stop_time = time.time()
        print(f"Time for generating test data: "
              f"{np.round(stop_time - start_time, 3)} s")
        # calculating default accuracy by putting all predictions to zero

        # test_labels
        # n_test_frames = 0
        # n_rights = 0
        #
        # for batch_labels in test_labels:
        #     if len(batch_labels.shape) == 1:
        #         n_rights += len(batch_labels) - np.sum(batch_labels)
        #         n_test_frames += len(batch_labels)
        #     else:
        #         n_rights += len(batch_labels) - np.sum(batch_labels[:, 0])
        #         n_test_frames += len(batch_labels)
        #
        # if n_test_frames > 0:
        #     print(f"Default test accuracy {str(np.round(n_rights / n_test_frames, 3))}")

        start_time = time.time()

        test_loss, test_acc = \
            model.evaluate(test_data_dict,
                           test_labels, verbose=1)
        print(f"test_metric {test_acc}")

        # predictions = np.ndarray.flatten(model.predict(test_data_dict))
        # print(f"predictions {' '.join(map(str, np.round(predictions, 2)))}")
        # print(f"test_labels {' '.join(map(str, test_labels))}")
        predictions = model.predict(test_data_dict)
        for index, prediction in enumerate(predictions):
            if n_classes == 1:
                print(f"label: {test_labels[index]}, "
                      f"prediction {prediction}")
            else:
                print(f"label: {' '.join(map(str, test_labels[index]))}, "
                      f"prediction {' '.join(map(str, np.round(prediction, 2)))}")

        stop_time = time.time()
        print(f"Time for evaluating test data: "
              f"{np.round(stop_time - start_time, 3)} s")
