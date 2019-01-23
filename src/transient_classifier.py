import numpy as np
import hdf5storage
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Bidirectional
from keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Activation
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.optimizers import RMSprop, adam
from keras import layers
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
from keras.utils import get_custom_objects
from matplotlib import pyplot as plt
import pattern_discovery.tools.param as p_disc_tools_param
from datetime import datetime
import time
from PIL import ImageSequence, ImageDraw
import PIL
from mouse_session_loader import load_mouse_sessions
from shapely import geometry
from scipy import ndimage
from random import shuffle
from keras import backend as K
import os

import sys
import platform

print(f"sys.maxsize {sys.maxsize}, platform.architecture {platform.architecture()}")


# make a function to build the training and validation set (test)
# same for the training labels and test labels
# function equivalent to
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# with data being vectorizes and labels adequates

# one way will be to split the data using a sliding window with overlap of like 50%, and we label
# the frame during which the cell is active

# another way we give him all the potential transients and he learn which on are true and which ones are false.
# but then it won't learn to fusion some onset/peaks periods for long transient

# we will need a generic fucntion that will take a mouse_session in argument and the matrices containing the onsets
# and peaks, and will return a bunch of data and labels. Then we will concatenate the results from different session
# and then split it between, train and test data

# we need to save the weights

# TimeDistributed: https://keras.io/layers/wrappers/ with BiDirectionnal wrapper just below
# exemple: https://keras.io/getting-started/functional-api-guide/

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


# data augmentation functions
def horizontal_flip(movie):
    """
    movie is a 3D numpy array
    :param movie:
    :return:
    """
    new_movie = np.zeros(movie.shape)
    for frame in np.arange(len(movie)):
        new_movie[frame] = np.fliplr(movie[frame])

    return new_movie


def vertical_flip(movie):
    """
    movie is a 3D numpy array
    :param movie:
    :return:
    """
    new_movie = np.zeros(movie.shape)
    for frame in np.arange(len(movie)):
        new_movie[frame] = np.flipud(movie[frame])

    return new_movie


def v_h_flip(movie):
    """
    movie is a 3D numpy array
    :param movie:
    :return:
    """
    new_movie = np.zeros(movie.shape)
    for frame in np.arange(len(movie)):
        new_movie[frame] = np.fliplr(np.flipud(movie[frame]))

    return new_movie


class DataGenerator(keras.utils.Sequence):
    """
    Based on an exemple found in https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    # 'Generates data for Keras'
    def __init__(self, data_list, batch_size=32, window_len=100, with_augmentation=False,
                 pixels_around=3, buffer=None,
                 is_shuffle=True, max_width=30, max_height=30):
        """

        :param data_list: a list containing the information to get the data. Each element is a list with 3 elements
        MouseSession instance, cell index, frame index
        :param batch_size:
        :param window_len:
        :param with_augmentation:
        :param is_shuffle:
        :param max_width:
        :param max_height:
        """
        # 'Initialization'
        self.max_width = max_width
        self.max_height = max_height
        self.pixels_around = pixels_around
        self.buffer = buffer
        self.window_len = window_len
        self.input_shape = (self.window_len, self.max_height, self.max_width, 1)
        self.batch_size = batch_size
        self.data_list = data_list
        self.with_augmentation = with_augmentation
        # to improve performance, keep in memory the mask_profile of a cell and the coords of the frame surrounding the
        # the cell, the key is a string ms.description + cell
        self.source_profiles_dict = dict()

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
        new_data = []
        # for each keys will create as many new keys as transformation to be done
        # adding the function to do the transformation to the value (list), and will create the same key
        # in labels, copying the original labels
        augmentation_functions = [horizontal_flip, vertical_flip, v_h_flip]
        #
        # augmentation_functions_name = ["horizontal_flip", "vertical_flip", "v_h_flip"]

        for index_data in np.arange(n_samples):
            for fct in augmentation_functions:
                elements = self.data_list[index_data]
                ms = elements[0]
                cell = elements[1]
                frame_index = elements[2]
                new_value = [ms, cell, frame_index, fct]
                new_data.append(new_value)

        self.data_list.extend(new_data)

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
        if self.is_shuffle:
            np.random.shuffle(self.indexes)
        # self.data_keys = list(self.data_dict.keys())
        # if self.is_shuffle:
        #     shuffle(self.data_keys)

    def __data_generation(self, data_list_tmp):
        # len(data_list_tmp) == self.batch_size
        # 'Generates data containing batch_size samples' # data : (self.batch_size, *dim, n_channels)
        # Initialization

        data, data_masked, labels = generate_movies_from_metadata(data_list=data_list_tmp, window_len=self.window_len,
                                                                  max_width=self.max_width, max_height=self.max_height,
                                                                  pixels_around=self.pixels_around,
                                                                  buffer=self.buffer,
                                                                  source_profiles_dict=self.source_profiles_dict)
        # print(f"__data_generation data.shape {data.shape}")
        # put more weight to the active frames
        # TODO: considering to put more weight to fake transient
        # TODO: reshape labels such as shape is (batch_size, window_len, 1) and then use "temporal" mode in compile
        # sample_weights = np.ones(labels.shape)
        # sample_weights[labels == 1] = 5
        sample_weights = np.ones(labels.shape[0])
        for i in np.arange(labels.shape[0]):
            if np.sum(labels[i]) > 0:
                sample_weights[i] = 3

        return {'video_input': data, 'video_input_masked': data_masked}, labels, sample_weights


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


def generate_movies_from_metadata(data_list, window_len, max_width, max_height, pixels_around,
                                  buffer, source_profiles_dict):
    batch_size = len(data_list)
    data = np.zeros((batch_size, window_len, max_height, max_width, 1))
    data_masked = np.zeros((batch_size, window_len, max_height, max_width, 1))
    labels = np.zeros((batch_size, window_len), dtype="uint8")

    # Generate data
    for index_batch, value in enumerate(data_list):
        ms = value[0]
        spike_nums_dur = ms.spike_struct.spike_nums_dur
        cell = value[1]
        frame_index = value[2]
        augmentation_fct = None
        if len(value) == 4:
            augmentation_fct = value[3]

        # now we generate the source profile of the cell for those frames and retrieve it if it has
        # already been generated
        src_profile_key = ms.description + str(cell)
        if src_profile_key in source_profiles_dict:
            mask_source_profile, coords = source_profiles_dict[src_profile_key]
        else:
            mask_source_profile, coords = \
                get_source_profile_param(cell=cell, ms=ms, pixels_around=pixels_around, buffer=buffer,
                                         max_width=max_width, max_height=max_width)
            source_profiles_dict[src_profile_key] = [mask_source_profile, coords]

        frames = np.arange(frame_index, frame_index + window_len)
        # setting labels: active frame or not
        labels[index_batch] = spike_nums_dur[cell, frames]
        # now adding the movie of those frames in this sliding_window
        source_profile_frames = get_source_profile_frames(frames=frames, ms=ms, coords=coords)
        # if i == 0:
        #     print(f"source_profile_frames.shape {source_profile_frames.shape}")
        source_profile_frames_masked = np.copy(source_profile_frames)
        source_profile_frames_masked[:, mask_source_profile] = 0

        profile_fit = np.zeros((len(frames), max_height, max_width))
        profile_fit_masked = np.zeros((len(frames), max_height, max_width))
        # we center the source profile
        y_coord = (profile_fit.shape[1] - source_profile_frames.shape[1]) // 2
        x_coord = (profile_fit.shape[2] - source_profile_frames.shape[2]) // 2
        profile_fit[:, y_coord:source_profile_frames.shape[1] + y_coord,
        x_coord:source_profile_frames.shape[2] + x_coord] = \
            source_profile_frames
        profile_fit_masked[:, y_coord:source_profile_frames.shape[1] + y_coord,
        x_coord:source_profile_frames.shape[2] + x_coord] = \
            source_profile_frames_masked

        # doing augmentation if the function exists
        if augmentation_fct is not None:
            profile_fit = augmentation_fct(profile_fit)
            profile_fit_masked = augmentation_fct(profile_fit_masked)

        profile_fit = profile_fit.reshape((profile_fit.shape[0], profile_fit.shape[1], profile_fit.shape[2], 1))
        profile_fit_masked = profile_fit_masked.reshape((profile_fit_masked.shape[0], profile_fit_masked.shape[1],
                                                         profile_fit_masked.shape[2], 1))

        data[index_batch] = profile_fit
        data_masked[index_batch] = profile_fit_masked

    return data, data_masked, labels


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
                  f"{np.round(stop_time - start_time, 3)} s")
            ms.normalize_movie()
        return True
    return False


def scale_polygon_to_source(poly_gon, minx, miny):
    coords = list(poly_gon.exterior.coords)
    scaled_coords = []
    for coord in coords:
        scaled_coords.append((coord[0] - minx, coord[1] - miny))
    return geometry.Polygon(scaled_coords)


def get_source_profile_param(cell, ms, max_width, max_height, pixels_around=0, buffer=None):
    len_frame_x = ms.tiff_movie[0].shape[1]
    len_frame_y = ms.tiff_movie[0].shape[0]

    # determining the size of the square surrounding the cell so it includes all overlapping cells around
    overlapping_cells = ms.coord_obj.intersect_cells[cell]
    cells_to_display = [cell]
    cells_to_display.extend(overlapping_cells)
    poly_gon = ms.coord_obj.cells_polygon[cell]

    # calculating the bound that will surround all the cells
    minx = None
    maxx = None
    miny = None
    maxy = None

    for cell_to_display in cells_to_display:
        poly_gon = ms.coord_obj.cells_polygon[cell_to_display]

        if minx is None:
            minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
        else:
            tmp_minx, tmp_miny, tmp_maxx, tmp_maxy = np.array(list(poly_gon.bounds)).astype(int)
            minx = min(minx, tmp_minx)
            miny = min(miny, tmp_miny)
            maxx = max(maxx, tmp_maxx)
            maxy = max(maxy, tmp_maxy)

    minx = max(0, minx - pixels_around)
    miny = max(0, miny - pixels_around)
    # we use max_width and max_height to make sure it won't be bigger than the frame used by the network
    maxx = np.min(((len_frame_x - 1), (maxx + pixels_around), (minx + max_height - 1)))
    maxy = np.min(((len_frame_y - 1), (maxy + pixels_around), (miny + max_width - 1)))

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

    # source_profile = np.zeros((len(frames), len_y, len_x))

    # frames_tiff = ms.tiff_movie[frames]
    # source_profile = frames_tiff[:, miny:maxy + 1, minx:maxx + 1]
    # normalized so that value are between 0 and 1
    # source_profile = source_profile / np.max(ms.tiff_movie)

    return mask, (minx, maxx, miny, maxy)


def get_source_profile_frames(ms, frames, coords):
    (minx, maxx, miny, maxy) = coords
    # frames_tiff = ms.tiff_movie_norm_0_1[frames]
    # source_profile = frames_tiff[:, miny:maxy + 1, minx:maxx + 1]
    source_profile = ms.tiff_movie_norm_0_1[frames, miny:maxy + 1, minx:maxx + 1]

    # normalized so that value are between 0 and 1
    # source_profile = source_profile / np.max(ms.tiff_movie)

    return source_profile


def load_data_for_generator(param, split_values, sliding_window_len, overlap_value,
                            movies_shuffling=None, with_shuffling=True):
    # TODO: stratification matters !
    """
    Stratification is the technique to allocate the samples evenly based on sample classes
    so that training set and validation set have similar ratio of classes
    """
    print("load_data_for_generator")
    use_small_sample = False
    if use_small_sample:
        ms_to_use = ["p12_171110_a000_ms"]
        cell_to_load_by_ms = {"p12_171110_a000_ms": np.arange(1)}
    else:
        ms_to_use = ["p12_171110_a000_ms", "p7_171012_a000_ms", "p9_18_09_27_a003_ms"]
        cell_to_load_by_ms = {"p12_171110_a000_ms": np.arange(5), "p7_171012_a000_ms": np.arange(20),
                              "p9_18_09_27_a003_ms": np.arange(15)}
        # max p7: 117, max p9: 30, max p12: 6

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False,
                                            for_transient_classifier=True)

    total_n_cells = 0
    # n_movies = 0

    full_data = []

    for ms_str in ms_to_use:
        ms = ms_str_to_ms_dict[ms_str]
        # print(f"{ms.description}, len ms.cells_to_remove {len(ms.cells_to_remove)}, ms.cells_to_remove {ms.cells_to_remove}")
        # cells_to_load_tmp = cell_to_load_by_ms[ms_str]
        # print(f"len(cell_to_load_by_ms[ms_str]) {len(cell_to_load_by_ms[ms_str])}")
        cells_to_load = np.setdiff1d(cell_to_load_by_ms[ms_str], ms.cells_to_remove)
        # print(f"len(cells_to_load) {len(cells_to_load)}")
        # for cell in cells_to_load_tmp:
        #     if cell not in ms.cells_to_remove[cell]:
        #         cells_to_load.append(cell)
        total_n_cells += len(cells_to_load)
        cells_to_load = np.array(cells_to_load)
        cell_to_load_by_ms[ms_str] = cells_to_load
        n_frames = ms.spike_struct.spike_nums_dur.shape[1]
        # n_movies += int(np.ceil(n_frames / (sliding_window_len * overlap_value))) - 1

        movie_loaded = load_movie(ms)
        if not movie_loaded:
            raise Exception(f"could not load movie of ms {ms.description}")

    movies_descr = []
    movie_count = 0
    for ms_str in ms_to_use:
        ms = ms_str_to_ms_dict[ms_str]
        spike_nums_dur = ms.spike_struct.spike_nums_dur
        n_frames = spike_nums_dur.shape[1]
        for cell in cell_to_load_by_ms[ms_str]:
            # then we slide the window
            # frames index of the beginning of each movie
            frames_step = int(np.ceil(sliding_window_len * (1 - overlap_value)))
            indices_movies = np.arange(0, n_frames, frames_step)

            for i, index_movie in enumerate(indices_movies):
                # if i == (len(indices_movies) - 1):
                #     if (indices_movies[i - 1] + sliding_window_len) == n_frames:
                #         break
                #     else:
                #         full_data.append([ms, cell, n_frames - sliding_window_len])
                #
                # else:
                #     full_data.append([ms, cell, index_movie])
                break_it = False
                first_frame = index_movie
                if (index_movie + sliding_window_len) == n_frames:
                    full_data.append([ms, cell, index_movie])
                    break_it = True
                elif (index_movie + sliding_window_len) > n_frames:
                    # in case the number of frames is not divisible by sliding_window_len
                    full_data.append([ms, cell, n_frames - sliding_window_len])
                    first_frame = n_frames - sliding_window_len
                    break_it = True
                else:
                    full_data.append([ms, cell, index_movie])
                movies_descr.append(f"{ms.description}_cell_{cell}_first_frame_{first_frame}")
                movie_count += 1
                if break_it:
                    break

    print(f"movie_count {movie_count}")
    # cells shuffling
    if movies_shuffling is None:
        movies_shuffling = np.arange(movie_count)
        if with_shuffling:
            np.random.shuffle(movies_shuffling)

    n_movies_for_training = int(movie_count * split_values[0])
    n_movies_for_validation = int(movie_count * split_values[1])
    train_data = []
    for index in movies_shuffling[:n_movies_for_training]:
        train_data.append(full_data[index])
    valid_data = []
    for index in movies_shuffling[n_movies_for_training:n_movies_for_training + n_movies_for_validation]:
        valid_data.append(full_data[index])
    test_data = []
    for index in movies_shuffling[n_movies_for_training + n_movies_for_validation:]:
        test_data.append(full_data[index])

    test_movie_descr = []
    for movie in movies_shuffling[n_movies_for_training + n_movies_for_validation:]:
        test_movie_descr.append(movies_descr[movie])

    return train_data, valid_data, test_data, test_movie_descr, cell_to_load_by_ms


def build_model(input_shape, use_mulimodal_inputs=False, dropout_value=0):
    n_frames = input_shape[0]
    # First, let's define a vision model using a Sequential model.
    # This model will encode an image into a vector.
    vision_model = Sequential()
    get_custom_objects().update({'swish': Swish(swish)})
    # to choose between siwsh and relu
    activation_fct = "swish"
    vision_model.add(Conv2D(64, (3, 3), activation=activation_fct, padding='same', input_shape=input_shape[1:]))
    vision_model.add(Conv2D(64, (3, 3), activation=activation_fct))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Conv2D(128, (3, 3), activation=activation_fct, padding='same'))
    vision_model.add(Conv2D(128, (3, 3), activation=activation_fct))
    vision_model.add(MaxPooling2D((2, 2)))
    if dropout_value > 0:
        vision_model.add(layers.Dropout(dropout_value))
    # vision_model.add(Conv2D(256, (3, 3), activation=activation_fct, padding='same'))
    # vision_model.add(Conv2D(256, (3, 3), activation=activation_fct))
    # vision_model.add(Conv2D(256, (3, 3), activation=activation_fct))
    # vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Flatten())

    video_input = Input(shape=input_shape, name="video_input")
    # This is our video encoded via the previously trained vision_model (weights are reused)
    encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
    # encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector
    encoded_video = Bidirectional(LSTM(128, return_sequences=True))(encoded_frame_sequence)
    # if we put input_shape in Bidirectional, it crashes
    # encoded_video = Bidirectional(LSTM(128, return_sequences=True),
    #                               input_shape=(n_frames, 128))(encoded_frame_sequence)
    encoded_video = Bidirectional(LSTM(256))(encoded_video)

    video_input_masked = Input(shape=input_shape, name="video_input_masked")
    # This is our video encoded via the previously trained vision_model (weights are reused)
    encoded_frame_sequence_masked = TimeDistributed(vision_model)(
        video_input_masked)  # the output will be a sequence of vectors
    # encoded_video_masked = LSTM(256)(encoded_frame_sequence_masked)  # the output will be a vector

    encoded_video_masked = Bidirectional(LSTM(128, return_sequences=True))(encoded_frame_sequence_masked)
    encoded_video_masked = Bidirectional(LSTM(256))(encoded_video_masked)

    # in case we want 2 videos, one with masked, and one with the cell centered
    if use_mulimodal_inputs:
        merged = layers.concatenate([encoded_video, encoded_video_masked])
        # output = TimeDistributed(Dense(1, activation='sigmoid')))
        output = Dense(n_frames, activation='sigmoid')(merged)
        video_model = Model(inputs=[video_input, video_input_masked], outputs=output)
    else:
        # output = TimeDistributed(Dense(1, activation='sigmoid'))(encoded_video)
        output = Dense(n_frames, activation='sigmoid')(encoded_video)
        video_model = Model(inputs=video_input, outputs=output)

    return video_model


def get_source_profile_for_prediction(ms, cell, augmentation_functions=None,
                                      overlap_value=0, max_width=30, max_height=30, sliding_window_len=100):
    n_frames = ms.tiff_movie.shape[0]
    n_augmentation_fct = 0
    if augmentation_functions is not None:
        n_augmentation_fct = len(augmentation_functions)
    # count_is_good = True
    # if (n_frames % sliding_window_len) == 0:
    #     n_movies = n_frames // sliding_window_len
    # else:
    #     n_movies = (n_frames // sliding_window_len) + 1
    #     count_is_good = False

    frames_step = int(np.ceil(sliding_window_len * (1 - overlap_value)))
    # number of indices to remove so index + sliding_window_len won't be superior to number of frames
    n_step_to_remove = 0 if (overlap_value == 0) else int(1 / (1 - overlap_value))
    frame_indices_for_movies = np.arange(0, n_frames, frames_step)
    if n_step_to_remove > 0:
        frame_indices_for_movies = frame_indices_for_movies[:-n_step_to_remove + 1]
    # in case the n_frames wouldn't be divisible by frames_step
    if frame_indices_for_movies[-1] + frames_step > n_frames:
        frame_indices_for_movies[-1] = n_frames - sliding_window_len

    # print(f"frames_step {frames_step}, n_step_to_remove {n_step_to_remove}, "
    #       f"frame_indices_for_movies[-1] {frame_indices_for_movies[-1]}")

    # the number of movies is determined by the overlap and the number of transformation that need to be done
    n_movies = len(frame_indices_for_movies) + (len(frame_indices_for_movies) * n_augmentation_fct)

    full_data = np.zeros((n_movies, sliding_window_len, max_height, max_width))
    full_data_masked = np.zeros((n_movies, sliding_window_len, max_height, max_width))
    full_data_frame_indices = np.zeros(n_movies, dtype="int16")

    # start_time = time.time()
    mask_source_profile, (minx, maxx, miny, maxy) = \
        get_source_profile_param(cell=cell, ms=ms, pixels_around=0, buffer=None, max_width=max_width,
                                 max_height=max_width)
    # stop_time = time.time()
    # print(f"Time to get_source_profile_param: "
    #       f"{np.round(stop_time - start_time, 3)} s")
    # times_for_get_source_profile_frames = []

    for index_movie, frame_index in enumerate(frame_indices_for_movies):
        index_movie = index_movie + (index_movie * n_augmentation_fct)
        frames = np.arange(frame_index, frame_index + sliding_window_len)
        # start_time = time.time()
        source_profile_frames = get_source_profile_frames(frames=frames, ms=ms, coords=(minx, maxx, miny, maxy))
        # stop_time = time.time()
        # times_for_get_source_profile_frames.append(stop_time - start_time)

        # if i == 0:
        #     print(f"source_profile_frames.shape {source_profile_frames.shape}")
        source_profile_frames_masked = np.copy(source_profile_frames)
        source_profile_frames_masked[:, mask_source_profile] = 0

        profile_fit = np.zeros((sliding_window_len, max_height, max_width))
        profile_fit_masked = np.zeros((sliding_window_len, max_height, max_width))
        # we center the source profile
        y_coord = (profile_fit.shape[1] - source_profile_frames.shape[1]) // 2
        x_coord = (profile_fit.shape[2] - source_profile_frames.shape[2]) // 2
        profile_fit[:, y_coord:source_profile_frames.shape[1] + y_coord,
        x_coord:source_profile_frames.shape[2] + x_coord] = \
            source_profile_frames
        profile_fit_masked[:, y_coord:source_profile_frames.shape[1] + y_coord,
        x_coord:source_profile_frames.shape[2] + x_coord] = \
            source_profile_frames_masked

        full_data[index_movie] = profile_fit
        full_data_masked[index_movie] = profile_fit_masked
        full_data_frame_indices[index_movie] = frame_index

        # doing augmentation if the function exists
        if augmentation_functions is not None:
            for i_fct, augmentation_fct in enumerate(augmentation_functions):
                i_fct += 1
                full_data[index_movie + i_fct] = augmentation_fct(profile_fit)
                full_data_masked[index_movie + i_fct] = augmentation_fct(profile_fit_masked)
                full_data_frame_indices[index_movie + i_fct] = frame_index

    # print(f"Avg time to get_source_profile_frames (x {n_movies}): "
    #       f"{np.round(np.mean(times_for_get_source_profile_frames), 10)} s")

    return full_data, full_data_masked, full_data_frame_indices


def predict_transient_from_saved_model(ms, cell, weights_file, json_file):
    # TODO: use data transformation and overlap and average over each
    start_time = time.time()
    # Model reconstruction from JSON file
    with open(json_file, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(weights_file)
    stop_time = time.time()
    print(f"Time for loading model: "
          f"{np.round(stop_time - start_time, 3)} s")

    start_time = time.time()
    n_frames = len(ms.tiff_movie)
    multi_inputs = (model.layers[0].output_shape == model.layers[1].output_shape)
    sliding_window_len = model.layers[0].output_shape[1]
    max_height = model.layers[0].output_shape[2]
    max_width = model.layers[0].output_shape[3]
    augmentation_functions = [horizontal_flip, vertical_flip, v_h_flip]
    overlap_value = 0.8
    data, data_masked, \
    data_frame_indices = get_source_profile_for_prediction(ms=ms, cell=cell,
                                                           sliding_window_len=sliding_window_len,
                                                           max_width=max_width,
                                                           max_height=max_height,
                                                           augmentation_functions=augmentation_functions,
                                                           overlap_value=overlap_value)
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2],
                         data.shape[3], 1))
    data_masked = data_masked.reshape((data_masked.shape[0], data_masked.shape[1], data_masked.shape[2],
                                       data_masked.shape[3], 1))
    stop_time = time.time()
    print(f"Time to get the data: "
          f"{np.round(stop_time - start_time, 3)} s")

    start_time = time.time()
    if multi_inputs:
        predictions = model.predict({'video_input': data,
                                     'video_input_masked': data_masked})
    else:
        predictions = model.predict(data_masked)
    stop_time = time.time()
    print(f"Time to get predictions: "
          f"{np.round(stop_time - start_time, 3)} s")

    # now we want to average each prediction for a given frame
    if (overlap_value > 0) or (augmentation_functions is not None):
        frames_predictions = dict()
        # print(f"predictions.shape {predictions.shape}, data_frame_indices.shape {data_frame_indices.shape}")
        for i, data_frame_index in enumerate(data_frame_indices):
            frames_index = np.arange(data_frame_index, data_frame_index + sliding_window_len)
            predictions_for_frames = predictions[i]
            for j, frame_index in enumerate(frames_index):
                if frame_index not in frames_predictions:
                    frames_predictions[frame_index] = []
                frames_predictions[frame_index].append(predictions_for_frames[j])

        predictions = np.zeros(n_frames)
        for frame_index, prediction_values in frames_predictions.items():
            predictions[frame_index] = np.mean(prediction_values)
    else:
        predictions = np.ndarray.flatten(predictions)

        # now we remove the extra prediction in case the number of frames was not divisible by the window length
        if (n_frames % sliding_window_len) != 0:
            real_predictions = np.zeros(n_frames)
            modulo = n_frames % sliding_window_len
            real_predictions[:len(predictions) - sliding_window_len] = predictions[
                                                                       :len(predictions) - sliding_window_len]
            real_predictions[len(predictions) - sliding_window_len:] = predictions[-modulo:]
            predictions = real_predictions

    if len(predictions) != n_frames:
        print(f"predictions len {len(predictions)}, n_frames {n_frames}")

    return predictions


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


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


def train_model():
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")
    path_data = root_path + "data/"
    result_path = root_path + "results_classifier/"
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    result_path = result_path + "/" + time_str
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    param = DataForMs(path_data=path_data, result_path=result_path, time_str=time_str)

    # 3 options to target the cell
    # 1) put the cell in the middle of the frame
    # 2) put all pixels in the border to 1
    # 3) Give 2 inputs, movie full frame (20x20 pixels) + movie mask non binary or binary

    use_mulimodal_inputs = True
    n_epochs = 10
    batch_size = 16
    window_len = 50
    max_width = 25
    max_height = 25
    overlap_value = 0.9
    dropout_value = 0.5
    pixels_around = 0
    buffer = None
    split_values = (0.7, 0.2)

    params_generator = {
        'batch_size': batch_size,
        'window_len': window_len,
        'max_width': max_width,
        'max_height': max_height,
        'pixels_around': pixels_around,
        'buffer': buffer,
        'is_shuffle': True}

    start_time = time.time()
    train_data_list, valid_data_list, test_data_list, \
    test_movie_descr, cell_to_load_by_ms = load_data_for_generator(param,
                                                                   split_values=split_values,
                                                                   sliding_window_len=window_len,
                                                                   overlap_value=overlap_value,
                                                                   movies_shuffling=None,
                                                                   with_shuffling=False)
    stop_time = time.time()
    print(f"Time for loading data for generator: "
          f"{np.round(stop_time - start_time, 3)} s")

    # Generators
    start_time = time.time()
    training_generator = DataGenerator(train_data_list, with_augmentation=True, **params_generator)
    validation_generator = DataGenerator(valid_data_list, with_augmentation=False, **params_generator)
    stop_time = time.time()
    print(f"Time to create generator: "
          f"{np.round(stop_time - start_time, 3)} s")

    # (sliding_window_size, max_width, max_height, 1)
    # sliding_window in frames, max_width, max_height: in pixel (100, 25, 25, 1) * n_movie
    input_shape = training_generator.input_shape

    print(f"input_shape {input_shape}")
    print(f"training_data n_samples {training_generator.n_samples}")
    print(f"valid_data n_samples {validation_generator.n_samples}")
    # print(f"input_shape {input_shape}")
    print("Data loaded")
    # return
    # building the model
    start_time = time.time()
    model = build_model(input_shape, dropout_value=dropout_value, use_mulimodal_inputs=use_mulimodal_inputs)

    print(model.summary())

    # Define the optimizer
    # from https://www.kaggle.com/shahariar/keras-swish-activation-acc-0-996-top-7
    optimizer = adam(lr=0.001, epsilon=1e-08, decay=0.0)
    # optimizer = 'rmsprop'

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', sensitivity, specificity])
    # sample_weight_mode="temporal",

    # Set a learning rate annealer
    # from: https://www.kaggle.com/shahariar/keras-swish-activation-acc-0-996-top-7
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_sensitivity',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    stop_time = time.time()
    print(f"Time for building and compiling the model: "
          f"{np.round(stop_time - start_time, 3)} s")

    # Train model on dataset
    start_time = time.time()
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=n_epochs,
                                  use_multiprocessing=True,
                                  workers=10,
                                  callbacks=[learning_rate_reduction])
    stop_time = time.time()
    print(f"Time for fitting the model to the data with {n_epochs} epochs: "
          f"{np.round(stop_time - start_time, 3)} s")


    show_plots = True

    if show_plots:
        key_names = ["loss", "acc", "sensitivity", "specificity"]
        for key_name in key_names:
            plot_training_and_validation_values(history=history, key_name=key_name, n_epochs=n_epochs,
                                                result_path=result_path, param=param)

    source_profiles_dict = dict()
    test_data, test_data_masked, test_labels = generate_movies_from_metadata(data_list=test_data_list,
                                                                             window_len=window_len,
                                                                             max_width=max_width,
                                                                             max_height=max_height,
                                                                             pixels_around=pixels_around,
                                                                             buffer=buffer,
                                                                             source_profiles_dict=source_profiles_dict)
    if use_mulimodal_inputs:
        test_loss, test_acc, test_sensitivity, test_specificity = model.evaluate({'video_input': test_data,
                                              'video_input_masked': test_data_masked},
                                             test_labels)
    else:
        test_loss, test_acc, test_sensitivity, test_specificity = model.evaluate(test_data_masked, test_labels)
    print(f"test_acc {test_acc}")

    start_time = time.time()
    model_descr = ""
    # model.save(f'{param.path_results}/transient_classifier_model_{model_descr}_test_acc_{test_acc}_{param.time_str}.h5')

    # saving params in a txt file
    file_name_txt = f'{param.path_results}/stat_model_{param.time_str}.txt'
    round_factor = 1

    with open(file_name_txt, "w", encoding='UTF-8') as file:
        file.write(f"n epochs: {n_epochs}" + '\n')
        file.write(f"batch_size: {batch_size}" + '\n')
        file.write(f"use_mulimodal_inputs: {use_mulimodal_inputs}" + '\n')
        file.write(f"window_len: {window_len}" + '\n')
        file.write(f"max_width: {max_width}" + '\n')
        file.write(f"max_height: {max_height}" + '\n')
        file.write(f"overlap_value: {overlap_value}" + '\n')
        file.write(f"dropout_value: {dropout_value}" + '\n')
        file.write(f"pixels_around: {pixels_around}" + '\n')
        file.write(f"buffer: {'None' if (buffer is None) else buffer}" + '\n')
        file.write(f"split_values: {split_values}" + '\n')
        file.write(f"test_loss: {test_loss}" + '\n')
        file.write(f"test_acc: {test_acc}" + '\n')
        file.write(f"test_sensitivity: {test_sensitivity}" + '\n')
        file.write(f"test_specificity: {test_specificity}" + '\n')

        # cells used
        for ms_str, cells in cell_to_load_by_ms.items():
            file.write(f"{ms_str}: ")
            for cell in cells:
                file.write(f"{cell} ")
            file.write("\n")
        file.write("" + '\n')

    model.save_weights(
        f'{param.path_results}/transient_classifier_weights_{model_descr}_test_acc_{test_acc}_{param.time_str}.h5')
    # Save the model architecture
    with open(
            f'{param.path_results}/transient_classifier_model_architecture_{model_descr}_test_acc_{test_acc}_'
            f'{param.time_str}.json',
            'w') as f:
        f.write(model.to_json())

    stop_time = time.time()
    print(f"Time for saving the model: "
          f"{np.round(stop_time - start_time, 3)} s")

    # if use_mulimodal_inputs:
    #     prediction = model.predict({'video_input': test_data,
    #                                 'video_input_masked': test_data_masked})
    # else:
    #     prediction = model.predict(test_data_masked)
    # print(f"prediction.shape {prediction.shape}")

    # for i, test_label in enumerate(test_labels):
    #     if i > 5:
    #         break
    #     print(f"####### video {i}  ##########")
    #     for j, label in enumerate(test_label):
    #         predict_value = str(round(prediction[i, j], 2))
    #         bonus = ""
    #         if label == 1:
    #             bonus = "# "
    #         print(f"{bonus} f {j}: {predict_value} / {label} ")
    #     print("")
    #     print("")
    #     # predict_value = str(round(predict_value, 2))
    #     # print(f"{i}: : {predict_value} / {test_labels[i]}")
    # print(f"test_acc {test_acc}")
