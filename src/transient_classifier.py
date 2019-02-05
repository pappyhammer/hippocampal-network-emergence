import numpy as np
import hdf5storage
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Bidirectional, BatchNormalization
from keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Activation, Lambda
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.optimizers import RMSprop, adam, SGD
from keras import layers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
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
from pattern_discovery.tools.misc import get_continous_time_periods
import scipy.signal as signal
import scipy.io as sio
import sys
import platform
from pattern_discovery.tools.signal import smooth_convolve

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


class MovieEvent:
    """
    Class that represent an event in a movie, for exemple a transient, neuropil etc...
    """

    def __init__(self):
        self.neuropil = False
        self.real_transient = False
        self.fake_transient = False
        self.movement = False
        # length in frames
        self.length_event = 1
        self.first_frame_event = None
        self.last_frame_event = None


class NeuropilEvent(MovieEvent):

    def __init__(self, frame_index):
        super().__init__()
        self.neuropil = True
        # frame_index could be None, in case we don't care about frame index
        self.first_frame_event = frame_index
        self.last_frame_event = frame_index


class RealTransientEvent(MovieEvent):

    def __init__(self, frames_period, amplitude):
        super().__init__()
        self.real_transient = True
        self.first_frame_event = frames_period[0]
        self.last_frame_event = frames_period[1]
        self.length_event = self.last_frame_event - self.first_frame_event + 1
        self.amplitude = amplitude


class FakeTransientEvent(MovieEvent):

    def __init__(self, frames_period, amplitude):
        super().__init__()
        self.fake_transient = True
        self.first_frame_event = frames_period[0]
        self.last_frame_event = frames_period[1]
        self.length_event = self.last_frame_event - self.first_frame_event + 1
        self.amplitude = amplitude


class MovementEvent(MovieEvent):

    def __init__(self, frames_period):
        super().__init__()
        self.movement = True
        self.first_frame_event = frames_period[0]
        self.last_frame_event = frames_period[1]
        self.length_event = self.last_frame_event - self.first_frame_event + 1


class StratificationCamembert:

    def __init__(self, data_list, description,
                 n_max_transformations, debug_mode=False):
        self.data_list = data_list
        self.n_movie_patch = len(data_list)
        self.description = description
        self.debug_mode = debug_mode
        self.n_max_transformations = n_max_transformations

        ##### for fake and real transients
        self.n_transient_total = dict()
        self.n_transient_total["fake"] = 0
        self.n_transient_total["real"] = 0

        self.n_cropped_transient_total = dict()
        self.n_cropped_transient_total["fake"] = 0
        self.n_cropped_transient_total["real"] = 0

        self.n_full_transient_total = dict()
        self.n_full_transient_total["fake"] = 0
        self.n_full_transient_total["real"] = 0

        self.transient_movies = dict()
        self.transient_movies["fake"] = []
        self.transient_movies["real"] = []

        self.cropped_transient_movies = dict()
        self.cropped_transient_movies["fake"] = []
        self.cropped_transient_movies["real"] = []

        self.min_augmentation_for_transient = dict()
        self.min_augmentation_for_transient["fake"] = 0
        self.min_augmentation_for_transient["real"] = 2

        self.min_augmentation_for_cropped_transient = dict()
        self.min_augmentation_for_cropped_transient["fake"] = 0
        self.min_augmentation_for_cropped_transient["real"] = 0

        # count
        self.n_full_1_transient = dict()
        self.n_full_1_transient["fake"] = 0
        self.n_full_1_transient["real"] = 0

        self.n_full_2p_transient = dict()
        self.n_full_2p_transient["fake"] = 0
        self.n_full_2p_transient["real"] = 0

        # list of movie_data with full transient (1 rep)
        self.full_1_transient = dict()
        self.full_1_transient["fake"] = []
        self.full_1_transient["real"] = []

        self.full_2p_transient = dict()
        self.full_2p_transient["fake"] = []
        self.full_2p_transient["real"] = []

        # contains the indices of the movies (from data_list) in a sorted order (based on amplitude, from low to high)
        self.full_transient_sorted_amplitude = dict()
        self.full_transient_sorted_amplitude["fake"] = []
        self.full_transient_sorted_amplitude["real"] = []


        self.n_transient_dict = dict()
        self.n_transient_dict["fake"] = dict()
        self.n_transient_dict["real"] = dict()

        # perc of full with 1 transient
        self.full_1_transient_perc = dict()
        self.full_1_transient_perc["fake"] = 0
        self.full_1_transient_perc["real"] = 0

        # perc of full with 2 transients or more
        self.full_2p_transient_perc = dict()
        self.full_2p_transient_perc["fake"] = 0
        self.full_2p_transient_perc["real"] = 0

        # % among all transients
        self.full_transient_perc = dict()
        self.full_transient_perc["fake"] = 0
        self.full_transient_perc["real"] = 0

        self.cropped_transient_perc = dict()
        self.cropped_transient_perc["fake"] = 0
        self.cropped_transient_perc["real"] = 0

        self.n_cropped_transient_dict = dict()
        self.n_cropped_transient_dict["fake"] = dict()
        self.n_cropped_transient_dict["real"] = dict()

        self.total_transient_perc = dict()
        self.total_transient_perc["fake"] = 0
        self.total_transient_perc["real"] = 0

        self.transient_lengths = dict()
        self.transient_lengths["fake"] = []
        self.transient_lengths["real"] = []

        self.transient_amplitudes = dict()
        self.transient_amplitudes["fake"] = []
        self.transient_amplitudes["real"] = []

        # MoviePatchData list
        self.neuropil_movies = []

        self.n_only_neuropil = 0
        self.only_neuropil_perc = 0
        self.n_real_and_fake_transient = 0
        # disct with key ms.description and value the number of movies for this session
        self.n_movies_by_session = {}
        # key int representing age, and value the number of movie for this session
        self.n_movies_by_age = {}

        self.compute_slices()

    def compute_slices(self):
        """
        Compute the slices of the camembert
        :return:
        """
        # printing option
        # \x1b
        reset_color = '\033[30m'
        # red: 31, blue: 34
        perc_color = '\033[34m'
        self.n_movie_patch = 0
        for movie_data in self.data_list:
            self.n_movie_patch += (1 + movie_data.n_augmentations_to_perform)

        sorted_amplitudes = dict()
        amplitudes_movie_index = dict()

        for which_ones in ["real", "fake"]:
            # initializing variables
            self.transient_movies[which_ones] = []
            self.cropped_transient_movies[which_ones] = []
            self.full_1_transient[which_ones] = []
            self.full_2p_transient[which_ones] = []
            self.n_transient_dict[which_ones] = dict()
            self.n_cropped_transient_dict[which_ones] = dict()
            self.transient_lengths[which_ones] = []
            self.transient_amplitudes[which_ones] = []
            self.full_transient_sorted_amplitude[which_ones] = []
            sorted_amplitudes[which_ones] = []
            amplitudes_movie_index[which_ones] = []

        # MoviePatchData list
        self.neuropil_movies = []
        self.n_only_neuropil = 0
        self.only_neuropil_perc = 0
        self.n_real_and_fake_transient = 0
        # disct with key ms.description and value the number of movies for this session
        self.n_movies_by_session = {}
        # key int representing age, and value the number of movie for this session
        self.n_movies_by_age = {}

        if self.debug_mode:
            print(f"{'##' * 10}")
            print(f"{self.description}")
            print(f"{'##' * 10}")
        for movie_index, movie_data in enumerate(self.data_list):
            movie_info = movie_data.movie_info
            only_neuropil = True
            with_real_transient = False
            with_cropped_real_transient = False
            with_fake_transient = False
            n_movies = 1 + movie_data.n_augmentations_to_perform
            if "n_transient" in movie_info:
                with_real_transient = True
                only_neuropil = False
                n_transient = movie_info["n_transient"]
                if n_transient == 1:
                    self.full_1_transient["real"].append(movie_data)
                else:
                    self.full_2p_transient["real"].append(movie_data)
                self.transient_movies["real"].append(movie_data)
                self.n_transient_dict["real"][n_transient] = self.n_transient_dict["real"].get(n_transient,
                                                                                               0) + n_movies
                if "transients_amplitudes" in movie_info:
                    self.transient_amplitudes["real"].extend(movie_info["transients_amplitudes"])
                    sorted_amplitudes["real"].append(np.max(movie_info["transients_amplitudes"]))
                    amplitudes_movie_index["real"].append(movie_index)
                if "transients_lengths" in movie_info:
                    self.transient_lengths["real"].extend(movie_info["transients_lengths"])
            if ("n_cropped_transient" in movie_info) and (not with_real_transient):
                only_neuropil = False
                with_cropped_real_transient = True
                self.cropped_transient_movies["real"].append(movie_data)
                n_cropped_transient = movie_info["n_cropped_transient"]
                self.n_cropped_transient_dict["real"][n_cropped_transient] = \
                    self.n_cropped_transient_dict["real"].get(n_cropped_transient, 0) + n_movies
                if "transients_amplitudes" in movie_info:
                    sorted_amplitudes["real"].append(np.max(movie_info["transients_amplitudes"]))
                    amplitudes_movie_index["real"].append(movie_index)
            if ("n_fake_transient" in movie_info) and (not with_real_transient) and (not with_cropped_real_transient):
                only_neuropil = False
                with_fake_transient = True
                self.transient_movies["fake"].append(movie_data)
                n_fake_transient = movie_info["n_fake_transient"]
                if n_fake_transient == 1:
                    self.full_1_transient["fake"].append(movie_data)
                else:
                    self.full_2p_transient["fake"].append(movie_data)
                self.n_transient_dict["fake"][n_fake_transient] = \
                    self.n_transient_dict["fake"].get(n_fake_transient, 0) + n_movies

                if "fake_transients_amplitudes" in movie_info:
                    self.transient_amplitudes["fake"].extend(movie_info["fake_transients_amplitudes"])
                    sorted_amplitudes["fake"].append(np.max(movie_info["fake_transients_amplitudes"]))
                    amplitudes_movie_index["fake"].append(movie_index)
                if "fake_transients_lengths" in movie_info:
                    self.transient_lengths["fake"].extend(movie_info["fake_transients_lengths"])
            if ("n_cropped_fake_transient" in movie_info) and (not with_real_transient) and (not with_fake_transient) \
                    and (not with_cropped_real_transient):
                only_neuropil = False
                self.cropped_transient_movies["fake"].append(movie_data)
                n_cropped_fake_transient = movie_info["n_cropped_fake_transient"]
                self.n_cropped_transient_dict["fake"][n_cropped_fake_transient] = \
                    self.n_cropped_transient_dict["fake"].get(n_cropped_fake_transient, 0) + n_movies
                if "fake_transients_amplitudes" in movie_info:
                    sorted_amplitudes["fake"].append(np.max(movie_info["fake_transients_amplitudes"]))
                    amplitudes_movie_index["fake"].append(movie_index)

            if with_fake_transient and with_real_transient:
                self.n_real_and_fake_transient += n_movies

            if only_neuropil:
                self.n_only_neuropil += n_movies
                self.neuropil_movies.append(movie_data)

            self.n_movies_by_session[movie_data.ms.description] = \
                self.n_movies_by_session.get(movie_data.ms.description, 0) + n_movies
            self.n_movies_by_age[movie_data.ms.age] = self.n_movies_by_age.get(movie_data.ms.age, 0) + n_movies

        # sorting movie_data by amplitude, from the smallest to the biggest
        for which_ones in ["real", "fake"]:
            index_array = np.argsort(sorted_amplitudes[which_ones])
            for index in index_array:
                self.full_transient_sorted_amplitude[which_ones].append(amplitudes_movie_index[which_ones][index])

        self.only_neuropil_perc = (self.n_only_neuropil / self.n_movie_patch) * 100

        if self.debug_mode:
            print(f"{'#' * 10}")
            print(f"{'#' * 10}")
            print(f"{'#' * 10}")
            print(f"len train data {self.n_movie_patch}")
            print("")
            print(f"n_real_and_fake_transient {self.n_real_and_fake_transient}")
            print("")
            print("")

        for which_ones in ["real", "fake"]:
            # calculating the max number of augmentations done
            movies = self.transient_movies[which_ones]
            movies.extend(self.cropped_transient_movies[which_ones])
            transformations_done = []
            for movie in movies:
                transformations_done.append(movie.n_augmentations_to_perform)
            transformations_done = np.array(transformations_done)
            if self.debug_mode and (len(transformations_done) > 0):
                print(f"{which_ones}: transformations, min {np.min(transformations_done)}, "
                      f"max {np.max(transformations_done)}, "
                      f"mean {str(np.round(np.mean(transformations_done), 2))}")
            if self.debug_mode:
                print(f"{which_ones}: n_transient_dict {self.n_transient_dict[which_ones]}")
            self.n_full_transient_total[which_ones] = 0
            self.n_full_1_transient[which_ones] = 0
            self.n_full_2p_transient[which_ones] = 0
            for rep, count in self.n_transient_dict[which_ones].items():
                self.n_full_transient_total[which_ones] += count
                if rep == 1:
                    self.n_full_1_transient[which_ones] += count
                else:
                    self.n_full_2p_transient[which_ones] += count
            if self.n_full_transient_total[which_ones] > 0:
                self.full_1_transient_perc[which_ones] = (self.n_full_1_transient[which_ones] /
                                                          self.n_full_transient_total[which_ones]) * 100
                self.full_2p_transient_perc[which_ones] = (self.n_full_2p_transient[which_ones] /
                                                           self.n_full_transient_total[which_ones]) * 100
                if self.debug_mode:
                    print(perc_color + f"1 full {which_ones} transient perc: "
                    f"{str(np.round(self.full_1_transient_perc[which_ones], 2))} %" + reset_color)
                    print(perc_color + f"2+ full {which_ones} transient perc: "
                    f"{str(np.round(self.full_2p_transient_perc[which_ones], 2))} %" + reset_color)
            if self.debug_mode:
                print(f"{which_ones}: n_cropped_transient_dict {self.n_cropped_transient_dict[which_ones]}")
            self.n_cropped_transient_total[which_ones] = 0
            for rep, count in self.n_cropped_transient_dict[which_ones].items():
                self.n_cropped_transient_total[which_ones] += count

            self.n_transient_total[which_ones] = self.n_cropped_transient_total[which_ones] + \
                                                 self.n_full_transient_total[which_ones]
            self.total_transient_perc[which_ones] = (self.n_transient_total[which_ones] / self.n_movie_patch) * 100

            if self.n_transient_total[which_ones] > 0:
                self.full_transient_perc[which_ones] = (self.n_full_transient_total[which_ones] /
                                                        self.n_transient_total[which_ones]) * 100
                self.cropped_transient_perc[which_ones] = (self.n_cropped_transient_total[which_ones] /
                                                           self.n_transient_total[which_ones]) * 100
                if self.debug_mode:
                    print(perc_color + f"Full {which_ones}: "
                    f"{str(np.round(self.full_transient_perc[which_ones], 2))} %" + reset_color)
                    print(perc_color + f"Cropped {which_ones}: "
                    f"{str(np.round(self.cropped_transient_perc[which_ones], 2))} %" + reset_color)
            if self.debug_mode and (len(self.transient_lengths[which_ones]) > 0):
                print(f"{which_ones}: transient_lengths n {len(self.transient_lengths[which_ones])} / "
                      f"min-max {np.min(self.transient_lengths[which_ones])} - "
                      f"{np.max(self.transient_lengths[which_ones])}")
                print(f"{which_ones}: mean transient_amplitudes {np.mean(self.transient_amplitudes[which_ones])}")
                print("")
                print("")

        if self.debug_mode:
            for which_ones in ["real", "fake"]:
                print(perc_color + f"Total movie with {which_ones} transients {self.n_transient_total[which_ones]}: "
                f"{str(np.round(self.total_transient_perc[which_ones], 2))} %" + reset_color)
            print(perc_color + f"n_only_neuropil {self.n_only_neuropil}: "
            f"{str(np.round(self.only_neuropil_perc, 2))} %" + reset_color)
            print("")
            print("")

        if self.debug_mode:
            print(f"n_movies_by_session {self.n_movies_by_session}")
            for session, count in self.n_movies_by_session.items():
                print(perc_color + f"{session}: {str(np.round((count / self.n_movie_patch) * 100, 2))} %" + reset_color)
            print(f"n_movies_by_age {self.n_movies_by_age}")
            for age, count in self.n_movies_by_age.items():
                print(perc_color + f"p{age}: {str(np.round((count / self.n_movie_patch) * 100, 2))} %" + reset_color)

    def add_augmentation_to_all_patches(self, n_augmentation):
        """
        Add to all movie patches in the camember a given number of augmentation, except neuropil
        :param n_augmentation:
        :return:
        """
        for movie_patch_data in self.data_list:
            if "only_neuropil" not in movie_patch_data.movie_info:
                movie_patch_data.add_n_augmentation(n_augmentation)

    def set_weights(self):
        # first we compute the thresholds
        # print(f"len(self.transient_amplitudes['real']) {len(self.transient_amplitudes['real'])}")
        real_amplitudes = np.unique(self.transient_amplitudes["real"])
        # print(f"real_amplitudes {len(real_amplitudes)}")
        fake_amplitudes = np.unique(self.transient_amplitudes["fake"])
        real_lengths = np.unique(self.transient_lengths["real"])
        fake_lengths = np.unique(self.transient_lengths["fake"])

        if len(real_amplitudes) > 0:
            real_amplitudes_threshold = np.percentile(real_amplitudes, 10)
        else:
            real_amplitudes_threshold = None
        if len(fake_amplitudes) > 0:
            fake_amplitudes_threshold = np.percentile(fake_amplitudes, 90)
        else:
            fake_amplitudes_threshold = None
        if len(real_lengths) > 0:
            real_lengths_threshold = np.percentile(real_lengths, 90)
        else:
            real_lengths_threshold = None

        if len(fake_lengths) > 0:
            fake_lengths_threshold = np.percentile(fake_lengths, 90)
        else:
            fake_lengths_threshold = None

        for movie_data in self.data_list:
            movie_info = movie_data.movie_info
            if "n_transient" in movie_info:
                if movie_info["n_transient"] > 1:
                    movie_data.weight += 5
                if (real_lengths_threshold is not None) and ("transients_lengths" in movie_info):
                    lengths = np.array(movie_info["transients_lengths"])
                    if len(np.where(lengths > real_lengths_threshold)[0]) > 0:
                        # print(f"lengths {lengths}, real_lengths_threshold {real_lengths_threshold}")
                        # means at least a transient length is superior to the 90th percentile
                        movie_data.weight += 3
                if (real_amplitudes_threshold is not None) and ("transients_amplitudes" in movie_info):
                    amplitudes = np.array(movie_info["transients_amplitudes"])
                    if len(np.where(amplitudes < real_amplitudes_threshold)[0]) > 0:
                        # print(f"amplitudes {amplitudes}, real_amplitudes_threshold {real_amplitudes_threshold}")
                        # means at least a transient amplitude is inferior to the 10th percentile
                        movie_data.weight += 3
                continue
            if "n_cropped_transient" in movie_info:
                continue
            if "n_fake_transient" in movie_info:
                movie_data.weight += 50
                if (fake_lengths_threshold is not None) and ("fake_transients_lengths" in movie_info):
                    lengths = np.array(movie_info["fake_transients_lengths"])
                    if len(np.where(lengths > fake_lengths_threshold)[0]) > 0:
                        # print(f"lengths {lengths}, real_lengths_threshold {fake_lengths_threshold}")
                        # means at least a transient length is superior to the 90th percentile
                        movie_data.weight += 5
                if (fake_amplitudes_threshold is not None) and ("fake_transients_amplitudes" in movie_info):
                    amplitudes = np.array(movie_info["fake_transients_amplitudes"])
                    if len(np.where(amplitudes < fake_amplitudes_threshold)[0]) > 0:
                        # means at least a transient amplitude is superior to the 90th percentile
                        movie_data.weight += 10
                continue
            if "n_cropped_fake_transient" in movie_info:
                movie_data.weight += 5

    def balance_all(self, main_ratio_balance,  first_round):
        # if a ratio is put to one, then the class is untouched, but then the sum of other ratio should be equal to
        # 1 or set to -1 as well
        # main_ratio_balance = (0.6, 0.25, 0.15)
        if self.debug_mode:
            print("")
            print(f"$$$$$$$$$$$$$$$$$$$$$$ camembert.balance_all {main_ratio_balance} $$$$$$$$$$$$$$$$$$$$$$")
            print("")

        tolerance = 0.5

        # dealing with the case of the ratio is 0
        if (main_ratio_balance[0] == 0) and first_round:
            # then we delete all real transients
            new_data_list = []
            for movie_data in self.data_list:
                movie_info = movie_data.movie_info
                if "n_transient" in movie_info:
                    continue
                if "n_cropped_transient" in movie_info:
                    continue
                new_data_list.append(movie_data)
            self.data_list = new_data_list
            # updating the stat
            self.compute_slices()

        if (main_ratio_balance[1] == 0) and first_round:
            # then we delete all fake transients
            new_data_list = []
            for movie_data in self.data_list:
                movie_info = movie_data.movie_info
                if "n_transient" in movie_info:
                    new_data_list.append(movie_data)
                    continue
                if "n_cropped_transient" in movie_info:
                    new_data_list.append(movie_data)
                    continue
                if "n_fake_transient" in movie_info:
                    continue
                if "n_cropped_fake_transient" in movie_info:
                    continue
                new_data_list.append(movie_data)
            self.data_list = new_data_list
            # updating the stat
            self.compute_slices()

        if (main_ratio_balance[0] > 0) and (main_ratio_balance[1] > 0):

            ratio_real_fake = main_ratio_balance[0] / main_ratio_balance[1]

            if (self.n_transient_total["real"] > 0) and (self.n_transient_total["fake"] > 0):
                if np.abs((self.n_transient_total["real"] / self.n_transient_total["fake"]) - ratio_real_fake) > \
                        tolerance:
                    if (self.n_transient_total["fake"] * ratio_real_fake) > self.n_transient_total["real"]:
                        # it means they are too many fakes, we need to add real transients or delete fake ones
                        if first_round:
                            # we delete fake, as we don't care about balance in fakes during first_round
                            # we want to change self.data_list
                            n_fake_to_delete = int(self.n_transient_total["fake"] -
                                                   (self.n_transient_total["real"] / ratio_real_fake))
                            delete_low_amplitudes_first = True
                            new_data_list = []
                            if delete_low_amplitudes_first:
                                # but keeping some still, thus keeping one and removing one
                                indices_to_remove = []
                                n_fake_removed = 0
                                sorted_index = 0
                                while n_fake_removed < n_fake_to_delete:
                                    index_data_list = self.full_transient_sorted_amplitude["fake"][sorted_index]
                                    indices_to_remove.append(index_data_list)
                                    movie_data = self.data_list[index_data_list]
                                    n_fake_removed += (1 + movie_data.n_augmentations_to_perform)
                                    sorted_index += 1

                                for index_data_list, movie_data in enumerate(self.data_list):
                                    if index_data_list in indices_to_remove:
                                        continue
                                    new_data_list.append(movie_data)
                            else:
                                n_fake_removed = 0
                                for movie_data in self.data_list:
                                    movie_info = movie_data.movie_info
                                    if "n_transient" in movie_info:
                                        new_data_list.append(movie_data)
                                        continue
                                    if "n_cropped_transient" in movie_info:
                                        new_data_list.append(movie_data)
                                        continue
                                    if "n_fake_transient" in movie_info:
                                        if n_fake_removed < n_fake_to_delete:
                                            n_fake_removed += (1 + movie_data.n_augmentations_to_perform)
                                            continue
                                    if "n_cropped_fake_transient" in movie_info:
                                        if n_fake_removed < n_fake_to_delete:
                                            n_fake_removed += (1 + movie_data.n_augmentations_to_perform)
                                            continue
                                    new_data_list.append(movie_data)
                            self.data_list = new_data_list
                        else:
                            # we add real transients
                            n_real_to_add = (self.n_transient_total["fake"] * ratio_real_fake) - \
                                            self.n_transient_total["real"]
                            print(f"n_real_to_add {n_real_to_add}")
                            # we want to add the same numbers such that we keep the ratio among the real_transients
                            # n_unique_real_transients represents the number of original movie patches, without taking in
                            # consideration the transformations that will be made
                            n_unique_real_transients = len(self.transient_movies["real"])
                            n_unique_real_transients += len(self.cropped_transient_movies["real"])
                            n_augmentations_options = [n_unique_real_transients * x for x in
                                                       np.arange(0, (self.n_max_transformations -
                                                                     self.min_augmentation_for_transient["real"]) + 3)]
                            n_augmentations_options = np.array(n_augmentations_options)
                            idx = (np.abs(n_augmentations_options - n_real_to_add)).argmin()
                            print(f"idx {idx}, len(n_augmentations_options): {len(n_augmentations_options)}")
                            print(f"n_augmentations_options[idx] {n_augmentations_options[idx]}")
                            if idx > 0:
                                n_transients_at_max_before = 0
                                n_transients_at_max_after = 0
                                n_added = 0
                                for movie_patch_data in self.transient_movies["real"]:
                                    augm_before = movie_patch_data.n_augmentations_to_perform
                                    movie_patch_data.add_n_augmentation(n_augmentation=idx)
                                    augm_after = movie_patch_data.n_augmentations_to_perform
                                    if augm_before == augm_after:
                                        n_transients_at_max_before += 1
                                    elif (augm_before + idx) < augm_after:
                                        n_transients_at_max_after += 1
                                    n_added += (augm_after - augm_before)
                                for movie_patch_data in self.cropped_transient_movies["real"]:
                                    augm_before = movie_patch_data.n_augmentations_to_perform
                                    movie_patch_data.add_n_augmentation(n_augmentation=idx)
                                    augm_after = movie_patch_data.n_augmentations_to_perform
                                    if augm_before == augm_after:
                                        n_transients_at_max_before += 1
                                    elif (augm_before + idx) < augm_after:
                                        n_transients_at_max_after += 1
                                    n_added += (augm_after - augm_before)
                                # TODO: if the ratio is not good, it means we couldn't add more transformation, in that
                                # TODO: case we want to remove some fake frames ?
                                if self.debug_mode:
                                    print(f"n_real_to_add {n_real_to_add}, n_added {n_added}")
                                    # print(f"n_transients_at_max_before {n_transients_at_max_before}, "
                                    #       f"n_transients_at_max_after {n_transients_at_max_before}")

                    else:
                        # it means they are too many real, we need to add fake transients
                        n_fake_to_add = (self.n_transient_total["real"] / ratio_real_fake) - \
                                        self.n_transient_total["fake"]
                        # we want to add the same numbers such that we keep the ratio among the real_transients
                        # n_unique_fake_transients represents the number of original movie patches, without taking in
                        # consideration the transformations that will be made
                        n_unique_fake_transients = len(self.transient_movies["fake"])
                        n_unique_fake_transients += len(self.cropped_transient_movies["fake"])
                        n_augmentations_options = [n_unique_fake_transients * x for x in
                                                   np.arange(1, (self.n_max_transformations -
                                                                 self.min_augmentation_for_transient["fake"]) + 2)]
                        n_augmentations_options = np.array(n_augmentations_options)
                        idx = (np.abs(n_augmentations_options - n_fake_to_add)).argmin()
                        if idx > 0:
                            for movie_patch_data in self.transient_movies["fake"]:
                                movie_patch_data.add_n_augmentation(n_augmentation=idx)
                            for movie_patch_data in self.cropped_transient_movies["fake"]:
                                movie_patch_data.add_n_augmentation(n_augmentation=idx)

            # updating the stat
            self.compute_slices()

        if (main_ratio_balance[2] == 0) and first_round:
            # then we delete all neuropil patches
            new_data_list = []
            for movie_data in self.data_list:
                movie_info = movie_data.movie_info
                if "only_neuropil" in movie_info:
                    continue
                new_data_list.append(movie_data)
            self.data_list = new_data_list
        elif (main_ratio_balance[2] > 0) and (self.n_only_neuropil > 0):
            if (self.n_transient_total["real"] > 0) or (self.n_transient_total["fake"] > 0):
                n_transient_total = self.n_transient_total["real"] + self.n_transient_total["fake"]
                ratio_transients_neuropils = (main_ratio_balance[0] + main_ratio_balance[1]) / main_ratio_balance[2]
                if np.abs((n_transient_total / self.n_only_neuropil) -
                          ratio_transients_neuropils) > tolerance:
                    if (self.n_only_neuropil * ratio_transients_neuropils) > n_transient_total:
                        # it means they are too many neuropil, we need to remove some
                        # TODO: See to remove neuropils with the lowest variation
                        n_neuropils_to_remove = int(self.n_only_neuropil -
                                                    (n_transient_total / ratio_transients_neuropils))
                        # print(f"!!!!!!!!!!!!!!!!!! n_neuropils_to_remove {n_neuropils_to_remove}")
                        # we want to change self.data_list
                        n_neuropils_removed = 0
                        new_data_list = []
                        for movie_data in self.data_list:
                            movie_info = movie_data.movie_info
                            if ("only_neuropil" in movie_info) and (n_neuropils_removed < n_neuropils_to_remove):
                                n_neuropils_removed += 1 + movie_data.n_augmentations_to_perform
                            else:
                                new_data_list.append(movie_data)
                        self.data_list = new_data_list

                    else:
                        # it means they are too many transients, we need to add neuropil
                        if self.debug_mode:
                            print(f"=== adding neuropil")
                        neuropil_to_add = (n_transient_total / ratio_transients_neuropils) - self.n_only_neuropil
                        if self.debug_mode:
                            print(f"=== neuropil_to_add {neuropil_to_add}")
                            print(f"=== len(self.neuropil_movies) {len(self.neuropil_movies)}")
                        augmentation_added = 0
                        movie_index = 0
                        while augmentation_added < neuropil_to_add:
                            self.neuropil_movies[movie_index].add_n_augmentation(n_augmentation=1)
                            movie_index = (movie_index + 1) % len(self.neuropil_movies)
                            augmentation_added += 1

        if self.debug_mode:
            print("")
            print(f"***************** After balancing real transients, fake ones and neuropil *****************")
            print("")
        # updating the stat
        self.compute_slices()

    def balance_transients(self, which_ones, crop_non_crop_ratio_balance, non_crop_ratio_balance):
        # if a ratio is put to one, then the class is untouched, but then the sum of other ratio should be equal to
        # 1 or set to -1 as well
        if which_ones not in ["fake", "real"]:
            raise Exception(f"which_ones not in {['fake', 'real']}")

        if self.debug_mode:
            print("")
            print(f"$$$$$$$$$$$$$$$$$$$$$$ camembert.balance_transients {which_ones} $$$$$$$$$$$$$$$$$$$$$$")
            print("")

        if self.n_transient_total[which_ones] == 0:
            return

        tolerance = 0.5
        if self.min_augmentation_for_transient[which_ones] > 0:
            for movie_data in self.transient_movies[which_ones]:
                movie_data.add_n_augmentation(n_augmentation=self.min_augmentation_for_transient[which_ones])
        if self.min_augmentation_for_cropped_transient[which_ones] > 0:
            for movie_data in self.cropped_transient_movies[which_ones]:
                movie_data.add_n_augmentation(n_augmentation=self.min_augmentation_for_cropped_transient[which_ones])

        if self.debug_mode:
            print("")
            print(f"$$ After adding min augmentation $$")
            print("")
        # updating stat
        self.compute_slices()

        if non_crop_ratio_balance[0] == 0:
            # we want to delete all the full 1 transient
            new_data_list = []
            for movie_data in self.data_list:
                movie_info = movie_data.movie_info
                if (which_ones == "fake") and ("n_fake_transient" in movie_info) \
                        and ("n_transient" not in movie_info) and ("n_cropped_transient" not in movie_info):
                    if movie_info["n_fake_transient"] == 1:
                        continue
                if (which_ones == "real") and ("n_transient" in movie_info):
                    if movie_info["n_transient"] == 1:
                        continue
                new_data_list.append(movie_data)
            self.data_list = new_data_list

            # updating stat
            self.compute_slices()
        if non_crop_ratio_balance[1] == 0:
            # we want to delete all the  2p full transient
            new_data_list = []
            for movie_data in self.data_list:
                movie_info = movie_data.movie_info
                if (which_ones == "fake") and ("n_fake_transient" in movie_info) \
                        and ("n_transient" not in movie_info) and ("n_cropped_transient" not in movie_info):
                    if movie_info["n_fake_transient"] > 1:
                        continue
                if (which_ones == "real") and ("n_transient" in movie_info):
                    if movie_info["n_transient"] > 1:
                        continue
                new_data_list.append(movie_data)
            self.data_list = new_data_list

            # updating stat
            self.compute_slices()

        if (non_crop_ratio_balance[0] > 0) and (non_crop_ratio_balance[1] > 0):
            ratio = non_crop_ratio_balance[0] / non_crop_ratio_balance[1]
            if ((self.n_full_2p_transient[which_ones] > 0) and (self.n_full_1_transient[which_ones] > 0)) and \
                    (np.abs(
                        (self.n_full_1_transient[which_ones] / self.n_full_2p_transient[which_ones]) - ratio) > tolerance):
                # we have a 5% tolerance
                if (self.n_full_2p_transient[which_ones] * ratio) > self.n_full_1_transient[which_ones]:
                    # to balance real, we augment them, to balance fake: we remove some
                    # TODO: take in consideration the initial balance and if possible augment fake to balance the session
                    if which_ones == "real":
                        # it means we don't have enough full 1 transient and need to augment self.n_full_1_transient
                        # first we need to determine the difference
                        full_1_to_add = (self.n_full_2p_transient[which_ones] * ratio) - self.n_full_1_transient[which_ones]
                        if self.debug_mode:
                            print(f"n_full_2p_transient[which_ones] {self.n_full_2p_transient[which_ones]}, "
                                  f" ratio {ratio}, n_full_1_transient[which_ones] {self.n_full_1_transient[which_ones]}")
                            print(f"diff {full_1_to_add}")
                        augmentation_added = 0
                        movie_index = 0
                        while augmentation_added < full_1_to_add:
                            self.full_1_transient[which_ones][movie_index].add_n_augmentation(n_augmentation=1)
                            movie_index = (movie_index + 1) % len(self.full_1_transient[which_ones])
                            augmentation_added += 1
                    else:
                        # we have too much full 2p transient, we want to remove some (for fakes ones)
                        # first we need to determine the difference
                        n_full_2p_to_remove = self.n_full_2p_transient[which_ones] - \
                                              (self.n_full_1_transient[which_ones] / ratio)
                        # we want to change self.data_list
                        n_full_2p_removed = 0
                        new_data_list = []
                        for movie_data in self.data_list:
                            movie_info = movie_data.movie_info
                            if "n_transient" in movie_info:
                                new_data_list.append(movie_data)
                                continue
                            if "n_cropped_transient" in movie_info:
                                new_data_list.append(movie_data)
                                continue
                            if "n_fake_transient" in movie_info:
                                n_fake_transient = movie_info["n_fake_transient"]
                                if n_fake_transient == 1:
                                    new_data_list.append(movie_data)
                                    continue
                                elif n_full_2p_removed < n_full_2p_to_remove:
                                    n_full_2p_removed += (1 + movie_data.n_augmentations_to_perform)
                                    continue
                            new_data_list.append(movie_data)
                        self.data_list = new_data_list
                else:
                    if which_ones == "real":
                        # it means we have too many full_1_transient, need to augment self.n_full_2p_transient
                        # first we want to respect the non_crop_ratio_balance
                        full_2p_to_add = (self.n_full_1_transient[which_ones] / ratio) - self.n_full_2p_transient[
                            which_ones]
                        augmentation_added = 0
                        movie_index = 0
                        missed_augm = 0
                        while augmentation_added < full_2p_to_add:
                            movie_patch = self.full_2p_transient[which_ones][movie_index]
                            movie_patch.add_n_augmentation(n_augmentation=1)
                            movie_index = (movie_index + 1) % len(self.full_2p_transient[which_ones])
                            augmentation_added += 1
                    else:
                        # we have too much full 1 transient, we want to remove some (for fakes ones)
                        # first we need to determine the difference
                        n_full_1_to_remove = self.n_full_1_transient[which_ones] - \
                                             (self.n_full_2p_transient[which_ones] * ratio)
                        # we want to change self.data_list
                        n_full_1_removed = 0
                        new_data_list = []
                        for movie_data in self.data_list:
                            movie_info = movie_data.movie_info
                            if "n_transient" in movie_info:
                                new_data_list.append(movie_data)
                                continue
                            if "n_cropped_transient" in movie_info:
                                new_data_list.append(movie_data)
                                continue
                            if "n_fake_transient" in movie_info:
                                n_fake_transient = movie_info["n_fake_transient"]
                                if n_fake_transient > 1:
                                    new_data_list.append(movie_data)
                                    continue
                                elif n_full_1_removed < n_full_1_to_remove:
                                    n_full_1_removed += (1 + movie_data.n_augmentations_to_perform)
                                    continue
                            new_data_list.append(movie_data)
                        self.data_list = new_data_list

        if self.debug_mode:
            print("")
            print(f"$$ After balancing non cropped {which_ones} transients $$")
            print("")
        # updating stat
        self.compute_slices()

        if self.n_cropped_transient_total[which_ones] == 0:
            return

        if crop_non_crop_ratio_balance[0] == 0:
            # we want to delete all the non cropped transients
            new_data_list = []
            for movie_data in self.data_list:
                movie_info = movie_data.movie_info
                if (which_ones == "fake") and ("n_fake_transient" in movie_info) \
                        and ("n_transient" not in movie_info) and ("n_cropped_transient" not in movie_info):
                    continue
                if (which_ones == "real") and ("n_transient" in movie_info):
                    continue
                new_data_list.append(movie_data)
            self.data_list = new_data_list

            # updating stat
            self.compute_slices()

        if crop_non_crop_ratio_balance[1] == 0:
            # we want to delete all the  cropped transients
            new_data_list = []
            for movie_data in self.data_list:
                movie_info = movie_data.movie_info
                if (which_ones == "fake") and ("n_cropped_fake_transient" in movie_info) \
                        and ("n_transient" not in movie_info) and ("n_cropped_transient" not in movie_info):
                    continue
                if (which_ones == "real") and ("n_cropped_transient" in movie_info) and \
                        ("n_transient" not in movie_info):
                    continue
                new_data_list.append(movie_data)
            self.data_list = new_data_list

            # updating stat
            self.compute_slices()

        if (crop_non_crop_ratio_balance[0] > 0) and (crop_non_crop_ratio_balance[1] > 0):
            # now we want to balance cropped and non cropped (full)
            ratio = crop_non_crop_ratio_balance[0] / crop_non_crop_ratio_balance[1]
            if np.abs((self.n_full_transient_total[which_ones] / self.n_cropped_transient_total[which_ones]) - ratio) > \
                    tolerance:
                if (self.n_cropped_transient_total[which_ones] * ratio) > self.n_full_transient_total[which_ones]:
                    if which_ones == "real":
                        # it means we have to many cropped one and we need to augment n_full_transient_total
                        # first we need to determine the difference
                        full_to_add = (self.n_cropped_transient_total[which_ones] * ratio) - \
                                      self.n_full_transient_total[which_ones]
                        # we compute how many full movie we have (self.n_full_transient_total[which_ones] contains the number
                        # of movies included all the transformation that will be perform) we want the number of unique original
                        # movies before transformations
                        n_full = len(self.full_1_transient[which_ones]) + len(self.full_2p_transient[which_ones])
                        # we want to add the same numbers such that we keep the ratio among the full_fake_transients
                        n_movie_patch_options = [n_full * x for x in
                                                 np.arange(0, (self.n_max_transformations -
                                                               self.min_augmentation_for_transient[which_ones]) + 3)]
                        n_movie_patch_options = np.array(n_movie_patch_options)
                        idx = (np.abs(n_movie_patch_options - full_to_add)).argmin()
                        # print(f"//// {which_ones}: self.n_cropped_transient_total[which_ones] "
                        #       f"{self.n_cropped_transient_total[which_ones]}, "
                        #       f" ratio {ratio}, self.n_full_transient_total[which_ones] "
                        #       f"{self.n_full_transient_total[which_ones]}")
                        # print(f"////  full_to_add {full_to_add}")
                        # print(f"////  n_movie_patch_options {n_movie_patch_options}")
                        # print(f"////  idx {idx}")
                        # print(f"////  n_movie_patch_options[idx] {n_movie_patch_options[idx]}")
                        if idx > 0:
                            for movie_patch_data in self.full_1_transient[which_ones]:
                                movie_patch_data.add_n_augmentation(n_augmentation=idx)
                            for movie_patch_data in self.full_2p_transient[which_ones]:
                                movie_patch_data.add_n_augmentation(n_augmentation=idx)
                    else:
                        # it means we have to many cropped one and we need to remove some
                        # first we need to determine the difference
                        n_cropped_to_remove = self.n_cropped_transient_total[which_ones] - \
                                              (self.n_full_transient_total[which_ones] / ratio)
                        # we want to change self.data_list
                        n_cropped_removed = 0
                        new_data_list = []
                        for movie_data in self.data_list:
                            movie_info = movie_data.movie_info
                            if "n_transient" in movie_info:
                                new_data_list.append(movie_data)
                                continue
                            if "n_cropped_transient" in movie_info:
                                new_data_list.append(movie_data)
                                continue
                            if "n_fake_transient" in movie_info:
                                new_data_list.append(movie_data)
                                continue
                            if "n_cropped_fake_transient" in movie_info:
                                if n_cropped_removed < n_cropped_to_remove:
                                    n_cropped_removed += (1 + movie_data.n_augmentations_to_perform)
                                    continue
                            new_data_list.append(movie_data)
                        self.data_list = new_data_list
                else:
                    # for full transient, we don't want to remove some even for fake, otherwise it would unbalance
                    # the full1 and full2
                    # if means we have too many full transient, we need to augment n_cropped_transient_total
                    # first we want to respect the non_crop_ratio_balance
                    n_cropped_to_add = (self.n_full_transient_total[which_ones] / ratio) - \
                                       self.n_cropped_transient_total[which_ones]
                    augmentation_added = 0
                    movie_index = 0
                    print(f"++++++ n_cropped_to_add {n_cropped_to_add}")
                    print(f"n_full_transient_total {which_ones} {self.n_full_transient_total[which_ones]}, "
                          f"n_cropped_transient_total {which_ones}  {self.n_cropped_transient_total[which_ones]}")
                    while augmentation_added < n_cropped_to_add:
                        self.cropped_transient_movies[which_ones][movie_index].add_n_augmentation(n_augmentation=1)
                        movie_index = (movie_index + 1) % len(self.cropped_transient_movies[which_ones])
                        augmentation_added += 1

        if self.debug_mode:
            print("")
            print(f"$$ After balancing cropped and non cropped {which_ones} transients $$")
            print("")
        self.compute_slices()


class StratificationDataProcessor:

    def __init__(self, data_list, description, n_max_transformations, main_ratio_balance=(0.6, 0.25, 0.15),
                 crop_non_crop_ratio_balance=(0.9, 0.1), non_crop_ratio_balance=(0.6, 0.4),
                 debug_mode=False):
        self.data_list = data_list
        self.n_transformations_for_session = n_max_transformations // 3
        self.n_max_transformations = n_max_transformations - self.n_transformations_for_session

        # for each session, we make a camembert of the movie_patches of this session
        # and balance the patches in the session
        # then we will balance the session among themselves by adding the number of augmentation
        # for all the patches of a given session, thus keeping the balance in the data
        self.movie_patches_data_by_session = dict()
        for movie_data in self.data_list:
            if movie_data.ms.description not in self.movie_patches_data_by_session:
                self.movie_patches_data_by_session[movie_data.ms.description] = []
            self.movie_patches_data_by_session[movie_data.ms.description].append(movie_data)

        # just to have the stat
        StratificationCamembert(data_list=self.data_list,
                                description=description,
                                n_max_transformations=self.n_max_transformations,
                                debug_mode=True)

        self.camembert_by_session = dict()
        for session, session_movie_data in self.movie_patches_data_by_session.items():
            self.camembert_by_session[session] = StratificationCamembert(data_list=session_movie_data,
                                                                         description=session,
                                                                         n_max_transformations=self.n_max_transformations,
                                                                         debug_mode=debug_mode)
        # #### First we want to balance each session in itself
        # a first step, would be to first balance the fake transients
        # then see how many transformations are needed to be added to real transients to get the right proportion
        # then look at neuropil and from neuropil decide if we want to delete some or do data augmentation in some of

        # for camembert in self.camembert_by_session.values():
        #     camembert.balance_all(main_ratio_balance=main_ratio_balance)

        # balancing the real transients, fake ones and neuropils among themselves
        for camembert in self.camembert_by_session.values():
            camembert.balance_all(main_ratio_balance=main_ratio_balance, first_round=True)

        # them
        for camembert in self.camembert_by_session.values():
            camembert.balance_transients(which_ones="fake", crop_non_crop_ratio_balance=crop_non_crop_ratio_balance,
                                         non_crop_ratio_balance=non_crop_ratio_balance)

        # balancing the real transients
        for camembert in self.camembert_by_session.values():
            camembert.balance_transients(which_ones="real", crop_non_crop_ratio_balance=crop_non_crop_ratio_balance,
                                         non_crop_ratio_balance=non_crop_ratio_balance)

        # balancing the real transients, fake ones and neuropils among themselves
        for camembert in self.camembert_by_session.values():
            camembert.balance_all(main_ratio_balance=main_ratio_balance, first_round=False)

        # ####  then balance session between themselves
        # taking the sessions with the most movies and using it as exemples
        max_movie_patch = 0
        for camembert in self.camembert_by_session.values():
            max_movie_patch = max(max_movie_patch, camembert.n_movie_patch)

        for camembert in self.camembert_by_session.values():
            if camembert.n_movie_patch == max_movie_patch:
                continue
            # we need to find the multiplicator between 1 and (self.n_transformations_for_session +1)
            # that would give the closest count from the max
            n_movie_patch = camembert.n_movie_patch
            # list of potential movie patches in this session depending on the augmentation factor
            # from 1 (no transformation added) to (self.n_transformations_for_session + 1)
            n_movie_patch_options = [n_movie_patch * x for x in np.arange(1, (self.n_transformations_for_session + 2))]
            n_movie_patch_options = np.array(n_movie_patch_options)
            idx = (np.abs(n_movie_patch_options - max_movie_patch)).argmin()
            if idx > 0:
                camembert.add_augmentation_to_all_patches(n_augmentation=idx)

        # updating the data list, in case some movie patches would have been deleted
        new_data_list = []
        for camembert in self.camembert_by_session.values():
            new_data_list.extend(camembert.data_list)
        self.data_list = new_data_list

        # just to have the stat
        if debug_mode:
            print(f"////////// AFTER balancing sessions //////////////")
        balanced_camembert = StratificationCamembert(data_list=self.data_list,
                                                     description=description+"_balanced",
                                                     n_max_transformations=self.n_max_transformations,
                                                     debug_mode=True)
        # setting the weight based on amplitudes and lengths of the transients
        # also adding weight to fake transients, and multiple real ones
        balanced_camembert.set_weights()

    def get_new_data_list(self):
        return self.data_list


class MoviePatchData:

    def __init__(self, ms, cell, index_movie, max_n_transformations,
                 encoded_frames, decoding_frame_dict,
                 window_len, with_info=False):
        # max_n_transformationsmax number of transofrmations to a movie patch
        # if the number of available function to transform is lower, the lower one would be kept
        self.manual_max_transformation = max_n_transformations
        self.ms = ms
        self.cell = cell
        # index of the first frame of the movie over the whole movie
        self.index_movie = index_movie
        self.last_index_movie = index_movie + window_len - 1
        self.window_len = window_len
        # weight to apply, use by the model to produce the loss function result
        self.weight = 1
        # number of transformation to perform on this movie, information to use if with_info == True
        # otherwise it means the object will be transform with the self.data_augmentation_fct
        self.n_augmentations_to_perform = 0

        # used if a movie_data has been copied
        self.data_augmentation_fct = None

        # set of functions used for data augmentation, one will be selected when copying a movie
        self.data_augmentation_fct_list = list()
        # functions based on rotations and flips
        rot_fct = []
        # adding fct to the set
        flips = [horizontal_flip, vertical_flip, v_h_flip]
        for flip in flips:
            rot_fct.append(flip)
        # 180° angle is the same as same as v_h_flip
        # 10 angles
        rotation_angles = np.array([20, 50, 90, 120, 160, 200, 230, 270, 310, 240])
        shuffle(rotation_angles)
        for angle in rotation_angles:
            rot_fct.append(lambda movie: rotate_movie(movie, angle))
        # 24 shifting transformations combinaison
        x_shift_y_shift_couples = []
        for x_shift in np.arange(-2, 3):
            for y_shift in np.arange(-2, 3):
                if (x_shift == 0) and (y_shift == 0):
                    continue
                x_shift_y_shift_couples.append((x_shift, y_shift))
        shifts_fct = []
        # keeping 11 shifts, from random
        n_shifts = 11
        shift_indices = np.arange(len(x_shift_y_shift_couples))
        if n_shifts < len(shift_indices):
            np.random.shuffle(shift_indices)
            shift_indices = shift_indices[:n_shifts]
        for index in shift_indices:
            x_shift = x_shift_y_shift_couples[index][0]
            y_shift = x_shift_y_shift_couples[index][1]
            shifts_fct.append(lambda movie: shift_movie(movie, x_shift=x_shift, y_shift=x_shift))

        for i in np.arange(max(len(rot_fct), len(shifts_fct))):
            if i < len(rot_fct):
                self.data_augmentation_fct_list.append(rot_fct[i])
            if i < len(shifts_fct):
                self.data_augmentation_fct_list.append(shifts_fct[i])

        self.n_available_augmentation_fct = min(self.manual_max_transformation, len(self.data_augmentation_fct_list))

        # movie_info dict containing the different informations about the movie such as the number of transients etc...
        """
        Keys so far (with value type) -> comments :
        
        n_transient (int)
        transients_lengths (list of int)
        transients_amplitudes (list of float)
        n_cropped_transient (int) -> max value should be 2
        cropped_transients_lengths (list of int)
        n_fake_transient (int)
        n_cropped_fake_transient (int) > max value should be 2
        fake_transients_lengths (list of int)
        fake_transients_amplitudes (list of float)
        inter_neuron (boolean)
        """
        self.movie_info = None
        self.encoded_frames = encoded_frames
        self.decoding_frame_dict = decoding_frame_dict
        if with_info:
            self.movie_info = dict()
            # then we want to know how many transients in this frame etc...
            # each code represent a specific event
            unique_codes = np.unique(encoded_frames[index_movie:index_movie + window_len])
            # print(f"unique_codes {unique_codes},  len {len(unique_codes)}")
            is_only_neuropil = True
            for code in unique_codes:
                event = decoding_frame_dict[code]
                if not event.neuropil:
                    is_only_neuropil = False

                if event.real_transient or event.fake_transient:

                    # we need to determine if it's a cropped one or full one
                    if (event.first_frame_event < index_movie) or (event.last_frame_event > self.last_index_movie):
                        # it's cropped
                        if event.real_transient:
                            key_str = "n_cropped_transient"
                            if "cropped_transients_lengths" not in self.movie_info:
                                self.movie_info["cropped_transients_lengths"] = []
                            self.movie_info["cropped_transients_lengths"].append(event.length_event)
                            if "transients_amplitudes" not in self.movie_info:
                                self.movie_info["transients_amplitudes"] = []
                            self.movie_info["transients_amplitudes"].append(event.amplitude)
                        else:
                            key_str = "n_cropped_fake_transient"
                            if "fake_transients_amplitudes" not in self.movie_info:
                                self.movie_info["fake_transients_amplitudes"] = []
                            self.movie_info["fake_transients_amplitudes"].append(event.amplitude)
                        self.movie_info[key_str] = self.movie_info.get(key_str, 0) + 1
                        continue

                    # means it's a full transient
                    if event.real_transient:
                        key_str = "n_transient"
                        if "transients_lengths" not in self.movie_info:
                            self.movie_info["transients_lengths"] = []
                        self.movie_info["transients_lengths"].append(event.length_event)
                        if "transients_amplitudes" not in self.movie_info:
                            self.movie_info["transients_amplitudes"] = []
                        self.movie_info["transients_amplitudes"].append(event.amplitude)
                    else:
                        key_str = "n_fake_transient"
                        if "fake_transients_lengths" not in self.movie_info:
                            self.movie_info["fake_transients_lengths"] = []
                        self.movie_info["fake_transients_lengths"].append(event.length_event)
                        if "fake_transients_amplitudes" not in self.movie_info:
                            self.movie_info["fake_transients_amplitudes"] = []
                        self.movie_info["fake_transients_amplitudes"].append(event.amplitude)
                    self.movie_info[key_str] = self.movie_info.get(key_str, 0) + 1
            if (ms.spike_struct.inter_neurons is not None) and (cell in ms.spike_struct.inter_neurons):
                self.movie_info["inter_neuron"] = True
            if is_only_neuropil:
                self.movie_info["only_neuropil"] = True

    def __eq__(self, other):
        if self.ms.description != other.ms.description:
            return False
        if self.cell != other.cell:
            return False
        if self.index_movie != self.index_movie:
            return False
        return True

    def copy(self):
        movie_copy = MoviePatchData(ms=self.ms, cell=self.cell, index_movie=self.index_movie,
                                    max_n_transformations=self.manual_max_transformation,
                                    encoded_frames=self.encoded_frames, decoding_frame_dict=self.decoding_frame_dict,
                                    window_len=self.window_len)
        movie_copy.data_augmentation_fct = self.data_augmentation_fct
        return movie_copy

    def add_n_augmentation(self, n_augmentation):
        self.n_augmentations_to_perform = min(self.n_augmentations_to_perform + n_augmentation,
                                              self.n_available_augmentation_fct)

    def pick_a_transformation_fct(self):
        if len(self.data_augmentation_fct_list) > 0:
            fct = self.data_augmentation_fct_list[0]
            self.data_augmentation_fct_list = self.data_augmentation_fct_list[1:]
            return fct
        return None

    def is_only_neuropil(self):
        """

        :return: True if there is only neuropil (no transients), False otherwise
        """
        if self.movie_info is None:
            return False

        if "n_transient" in self.movie_info:
            return False
        if "n_cropped_transient" in self.movie_info:
            return False
        if "n_fake_transient" in self.movie_info:
            return False
        if "n_cropped_fake_transient" in self.movie_info:
            return False

        return True


############################################################################################
############################################################################################
############################### data augmentation functions ###############################
############################################################################################
############################################################################################

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


def rotate_movie(movie, angle):
    """
        movie is a 3D numpy array
        :param movie:
        :return:
        """
    new_movie = np.zeros(movie.shape)
    for frame in np.arange(len(movie)):
        new_movie[frame] = ndimage.rotate(movie[frame], angle=angle, reshape=False, mode='reflect')

    return new_movie


def shift_movie(movie, x_shift, y_shift):
    """
    movie is a 3D numpy array
    :param movie:
    :param x_shift:
    :param y_shift:
    :return:
    """
    if x_shift >= movie.shape[2]:
        raise Exception(f"x_shift {x_shift} >= movie.shape[2] {movie.shape[2]}")
    if y_shift >= movie.shape[1]:
        raise Exception(f"y_shift {y_shift} >= movie.shape[1] {movie.shape[1]}")

    new_movie = np.zeros(movie.shape)

    if (y_shift == 0) and (x_shift == 0):
        new_movie = movie[:, :, :]
    elif (y_shift == 0) and (x_shift > 0):
        for frame in np.arange(len(movie)):
            new_movie[frame, :, :-x_shift] = movie[frame, :, x_shift:]
    elif (y_shift == 0) and (x_shift < 0):
        for frame in np.arange(len(movie)):
            new_movie[frame, :, -x_shift:] = movie[frame, :, :x_shift]
    elif (y_shift > 0) and (x_shift == 0):
        for frame in np.arange(len(movie)):
            new_movie[frame, :-y_shift, :] = movie[frame, y_shift:, :]
    elif (y_shift < 0) and (x_shift == 0):
        for frame in np.arange(len(movie)):
            new_movie[frame, -y_shift:, :] = movie[frame, :y_shift, :]
    elif (y_shift > 0) and (x_shift > 0):
        for frame in np.arange(len(movie)):
            new_movie[frame, :-y_shift, :-x_shift] = movie[frame, y_shift:, x_shift:]
    elif (y_shift < 0) and (x_shift < 0):
        for frame in np.arange(len(movie)):
            new_movie[frame, -y_shift:, -x_shift:] = movie[frame, :y_shift, :x_shift]
    elif (y_shift > 0) and (x_shift < 0):
        for frame in np.arange(len(movie)):
            new_movie[frame, :-y_shift, -x_shift:] = movie[frame, y_shift:, :x_shift]
    elif (y_shift < 0) and (x_shift > 0):
        for frame in np.arange(len(movie)):
            new_movie[frame, -y_shift:, :-x_shift] = movie[frame, :y_shift, x_shift:]

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

        :param data_list: a list containing the information to get the data. Each element
        is an instance of MoviePatchData
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
        print(f"n_samples before data augmentation: {n_samples}")
        new_data = []

        for index_data in np.arange(n_samples):
            movie_data = self.data_list[index_data]
            # we will do as many transformation as indicated in movie_data.n_augmentations_to_perform
            if movie_data.n_augmentations_to_perform == 0:
                continue
            for t in np.arange(movie_data.n_augmentations_to_perform):
                if t >= movie_data.n_available_augmentation_fct:
                    break
                new_movie = movie_data.copy()
                new_movie.data_augmentation_fct = movie_data.pick_a_transformation_fct()
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

        data, data_masked, labels = generate_movies_from_metadata(movie_data_list=data_list_tmp,
                                                                  window_len=self.window_len,
                                                                  max_width=self.max_width,
                                                                  max_height=self.max_height,
                                                                  pixels_around=self.pixels_around,
                                                                  buffer=self.buffer,
                                                                  source_profiles_dict=self.source_profiles_dict)
        # print(f"__data_generation data.shape {data.shape}")
        # put more weight to the active frames
        # TODO: considering to put more weight to fake transient
        # TODO: reshape labels such as shape is (batch_size, window_len, 1) and then use "temporal" mode in compile
        # TODO: otherwise, use the weight in the movie_data in data_list_tmp to apply the corresponding weight
        # sample_weights = np.ones(labels.shape)
        # sample_weights[labels == 1] = 5
        sample_weights = np.ones(labels.shape[0])
        for index_batch, movie_data in enumerate(data_list_tmp):
            sample_weights[index_batch] = movie_data.weight

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
    # return Lambda(lambda a: K.sigmoid(a) * a)(x)


def generate_movies_from_metadata(movie_data_list, window_len, max_width, max_height, pixels_around,
                                  buffer, source_profiles_dict):
    batch_size = len(movie_data_list)
    data = np.zeros((batch_size, window_len, max_height, max_width, 1))
    data_masked = np.zeros((batch_size, window_len, max_height, max_width, 1))
    labels = np.zeros((batch_size, window_len), dtype="uint8")

    # Generate data
    for index_batch, movie_data in enumerate(movie_data_list):
        ms = movie_data.ms
        spike_nums_dur = ms.spike_struct.spike_nums_dur
        cell = movie_data.cell
        frame_index = movie_data.index_movie
        augmentation_fct = movie_data.data_augmentation_fct

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

        # doing augmentation if the function exists
        if augmentation_fct is not None:
            source_profile_frames = augmentation_fct(source_profile_frames)
            source_profile_frames_masked = augmentation_fct(source_profile_frames_masked)

        # then we fit it the frame use by the network, padding the surrounding by zero if necessary
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
    len_frame_x = ms.tiff_movie_normalized[0].shape[1]
    len_frame_y = ms.tiff_movie_normalized[0].shape[0]

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
    # and we crop the frame if necessary
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


    return mask, (minx, maxx, miny, maxy)


def get_source_profile_frames(ms, frames, coords):
    (minx, maxx, miny, maxy) = coords
    source_profile = ms.tiff_movie_normalized[frames, miny:maxy + 1, minx:maxx + 1]

    return source_profile


def find_all_onsets_and_peaks_on_traces(ms, cell, threshold_factor=0.5):
    trace = ms.traces[cell]
    n_frames = trace.shape[0]
    peak_nums = np.zeros(n_frames, dtype="int8")
    peaks, properties = signal.find_peaks(x=trace, distance=2)
    peak_nums[peaks] = 1
    spike_nums = np.zeros(n_frames, dtype="int8")
    onsets = []
    diff_values = np.diff(trace)
    for index, value in enumerate(diff_values):
        if index == (len(diff_values) - 1):
            continue
        if value < 0:
            if diff_values[index + 1] >= 0:
                onsets.append(index + 1)
    # print(f"onsets {len(onsets)}")
    onsets = np.array(onsets)
    spike_nums[onsets] = 1

    threshold = (threshold_factor * np.std(trace)) + np.min(trace)
    peaks_under_threshold_index = peaks[trace[peaks] < threshold]
    # peaks_over_threshold_index = peaks[trace[peaks] >= threshold]
    # removing peaks under threshold and associated onsets
    peak_nums[peaks_under_threshold_index] = 0

    # onsets to remove
    onsets_index = np.where(spike_nums)[0]
    onsets_detected = []
    for peak_time in peaks_under_threshold_index:
        # looking for the peak preceding the onset
        onsets_before = np.where(onsets_index < peak_time)[0]
        if len(onsets_before) > 0:
            onset_to_remove = onsets_index[onsets_before[-1]]
            onsets_detected.append(onset_to_remove)
    spike_nums[np.array(onsets_detected)] = 0

    # now we construct the spike_nums_dur
    spike_nums_dur = np.zeros(n_frames, dtype="int8")

    peaks_index = np.where(peak_nums)[0]
    onsets_index = np.where(spike_nums)[0]

    for onset_index in onsets_index:
        peaks_after = np.where(peaks_index > onset_index)[0]
        if len(peaks_after) == 0:
            continue
        peaks_after = peaks_index[peaks_after]
        peak_after = peaks_after[0]
        if (peak_after - onset_index) > 200:
            print(f"tc: {ms.description} long transient in cell {cell} of "
                  f"duration {peak_after - onset_index} frames at frame {onset_index}")

        spike_nums_dur[onset_index:peak_after + 1] = 1

    return spike_nums_dur


def cell_encoding(ms, cell):
    # so far we need ms.traces
    n_frames = ms.traces.shape[1]
    encoded_frames = np.zeros(n_frames, dtype="int16")
    decoding_frame_dict = dict()
    # zero will be the Neuropil
    decoding_frame_dict[0] = NeuropilEvent(frame_index=None)
    next_code = 1
    if ms.z_score_traces is None:
        # creatin the z_score traces
        ms.normalize_traces()

    # we need spike_nums_dur and trace
    if ms.spike_struct.spike_nums_dur is None:
        if (ms.spike_struct.spike_nums is None) or (ms.spike_struct.peak_nums is None):
            raise Exception(f"{ms.decription} spike_nums and peak_nums should not be None")
        ms.build_spike_nums_dur()

    # first we add the real transient
    transient_periods = get_continous_time_periods(ms.spike_struct.spike_nums_dur[cell])
    # print(f"sum ms.spike_struct.spike_nums_dur[cell] {np.sum(ms.spike_struct.spike_nums_dur[cell])}")
    # list of tuple, first frame and last frame (included) of each transient
    for transient_period in transient_periods:
        amplitude = np.max(ms.z_score_raw_traces[cell, transient_period[0]:transient_period[1] + 1])
        encoded_frames[transient_period[0]:transient_period[1] + 1] = next_code
        event = RealTransientEvent(frames_period=transient_period, amplitude=amplitude)
        decoding_frame_dict[next_code] = event
        next_code += 1

    # then we look for all transient and take the mean + 1 std of transient peak
    # and keep the one that are not real as fake one
    # for that first we need to compute peaks_nums, spike_nums and spike_nums_dur from all onsets
    if cell not in ms.transient_classifier_spike_nums_dur:
        ms.transient_classifier_spike_nums_dur[cell] = \
            find_all_onsets_and_peaks_on_traces(ms=ms, cell=cell, threshold_factor=0.6)
    tc_dur = ms.transient_classifier_spike_nums_dur[cell]
    all_transient_periods = get_continous_time_periods(tc_dur)
    for transient_period in all_transient_periods:
        # checking if it's part of a real transient
        sp_dur = ms.spike_struct.spike_nums_dur[cell]
        if np.sum(sp_dur[transient_period[0]:transient_period[1] + 1]) > 0:
            continue
        amplitude = np.max(ms.z_score_raw_traces[cell, transient_period[0]:transient_period[1] + 1])
        encoded_frames[transient_period[0]:transient_period[1] + 1] = next_code
        event = FakeTransientEvent(frames_period=transient_period, amplitude=amplitude)
        decoding_frame_dict[next_code] = event
        next_code += 1

    return encoded_frames, decoding_frame_dict


def load_data_for_generator(param, split_values, sliding_window_len, overlap_value,
                            max_n_transformations,
                            movies_shuffling=None, with_shuffling=False, main_ratio_balance=(0.6, 0.25, 0.15),
                            crop_non_crop_ratio_balance=(0.9, 0.1),
                            non_crop_ratio_balance=(0.6, 0.4),
                            seed_value=None):
    """
    Stratification is the technique to allocate the samples evenly based on sample classes
    so that training set and validation set have similar ratio of classes
    p7_171012_a000_ms: up to cell 117 included, interesting cells:
    52 (69t), 75 (59t), 81 (50t), 93 (35t), 115 (28t), 83, 53 (51 mvt), 3
    p8_18_10_24_a005_ms: up to cell 22 included
    p9_18_09_27_a003_ms: up to cell 31 included
    p11_17_11_24_a000: 0 to 25 + 29
    p12_171110_a000_ms: up to cell 9 included (10 soon)
    p13_18_10_29_a001: 0, 5, 12, 13, 31, 42, 44, 48, 51, 77, 117

    # p13_18_10_29_a001_GUI_transients_RD.mat
    """
    # TODO: code to count how many transients in total
    print("load_data_for_generator")
    use_small_sample = True
    # used for counting how many cells and transients available
    load_them_all = False
    if load_them_all:
        ms_to_use = ["p7_171012_a000_ms", "p8_18_10_24_a005_ms", "p9_18_09_27_a003_ms", "p11_17_11_24_a000_ms",
                     "p12_171110_a000_ms", "p13_18_10_29_a001_ms"]
        cell_to_load_by_ms = {"p7_171012_a000_ms": np.arange(118), "p8_18_10_24_a005_ms": np.arange(22),
                              "p9_18_09_27_a003_ms": np.arange(32),
                              "p11_17_11_24_a000_ms": np.concatenate((np.arange(26), [29])),
                              "p12_171110_a000_ms": np.arange(10),
                              "p13_18_10_29_a001_ms": np.array([0, 5, 12, 13, 31, 42, 44, 48, 51, 77, 117])}
    elif use_small_sample:
        ms_to_use = ["p7_171012_a000_ms"]
        cell_to_load_by_ms = {"p7_171012_a000_ms": np.array([52, 53, 75, 81, 83, 93, 115])}
        # np.array([3, 52, 53, 75, 81, 83, 93, 115])
        # np.arange(1) np.array([8])
        # np.array([52, 53, 75, 81, 83, 93, 115]
        # ms_to_use = ["p12_171110_a000_ms"]
        # cell_to_load_by_ms = {"p12_171110_a000_ms": np.array([0, 7])}
        # ms_to_use = ["p13_18_10_29_a001_ms"]
        # cell_to_load_by_ms = {"p13_18_10_29_a001_ms": np.array([0, 5, 12, 13, 31, 42, 44, 48, 51])}
    else:
        ms_to_use = ["p7_171012_a000_ms", "p8_18_10_24_a005_ms", "p11_17_11_24_a000_ms",
                     "p12_171110_a000_ms", "p13_18_10_29_a001_ms"]
        #  "p9_18_09_27_a003_ms",
        cell_to_load_by_ms = {"p7_171012_a000_ms":  np.array([52, 53]), # 75, 81
                              "p8_18_10_24_a005_ms": np.array([0, 1]), # 9, 10
                              # "p9_18_09_27_a003_ms": np.array([3, 5]), # 7, 9
                              "p11_17_11_24_a000_ms": np.array([3, 22]), # 24,29
                              "p12_171110_a000_ms": np.array([0, 7]), # 3
                              "p13_18_10_29_a001_ms": np.array([0, 5])} # 12, 13
        # max p7: 117, max p9: 30, max p12: 6 .build_spike_nums_dur()

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=True, load_abf=False,
                                            for_transient_classifier=True)

    total_n_cells = 0
    # n_movies = 0

    full_data = []

    n_transients_available = 0

    # filtering the cells, to keep only the one not removed or with a good source profile according to cell classifier
    for ms_str in ms_to_use:
        ms = ms_str_to_ms_dict[ms_str]
        cells_to_load = np.setdiff1d(cell_to_load_by_ms[ms_str], ms.cells_to_remove)
        if (not load_them_all) and ms.cell_cnn_predictions is not None:
            print(f"Using cnn predictions from {ms.description}")
            # not taking into consideration cells that are not predicted as true from the cell classifier
            cells_predicted_as_false = np.where(ms.cell_cnn_predictions < 0.5)[0]
            cells_to_load = np.setdiff1d(cells_to_load, cells_predicted_as_false)

        total_n_cells += len(cells_to_load)
        cells_to_load = np.array(cells_to_load)
        cell_to_load_by_ms[ms_str] = cells_to_load

        if load_them_all:
            for cell in cells_to_load:
                n_transients_available += len(get_continous_time_periods(ms.spike_struct.spike_nums_dur[cell]))
        else:
            movie_loaded = load_movie(ms)
            if not movie_loaded:
                raise Exception(f"could not load movie of ms {ms.description}")

    if load_them_all:
        print(f"n_sessions {len(ms_to_use)}")
        print(f"total_n_cells {total_n_cells}")
        print(f"n_transients_available {n_transients_available}")
        raise Exception(f"load_them_all")
    if total_n_cells == 0:
        raise Exception(f"No cells loaded")

    print(f"total_n_cells {total_n_cells}")

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
            encoded_frames, decoding_frame_dict = cell_encoding(ms=ms, cell=cell)

            for i, index_movie in enumerate(indices_movies):
                break_it = False
                first_frame = index_movie
                if (index_movie + sliding_window_len) == n_frames:
                    break_it = True
                elif (index_movie + sliding_window_len) > n_frames:
                    # in case the number of frames is not divisible by sliding_window_len
                    first_frame = n_frames - sliding_window_len
                    break_it = True
                movie_data = MoviePatchData(ms=ms, cell=cell, index_movie=first_frame, window_len=sliding_window_len,
                                            max_n_transformations=max_n_transformations,
                                            with_info=True, encoded_frames=encoded_frames,
                                            decoding_frame_dict=decoding_frame_dict)
                full_data.append(movie_data)
                movies_descr.append(f"{ms.description}_cell_{cell}_first_frame_{first_frame}")
                movie_count += 1
                if break_it:
                    break

    print(f"movie_count {movie_count}")
    # cells shuffling
    if movies_shuffling is None:
        movies_shuffling = np.arange(movie_count)
        if with_shuffling:
            if seed_value is not None:
                np.random.seed(seed_value)
            sc = StratificationCamembert(data_list=full_data,
                                    description="FULL DATA",
                                    n_max_transformations=6,
                                    debug_mode=True)
            try_balancing = False
            # shuffle seems to work better
            if try_balancing:
                full_data = []
                index_fake = 0
                index_real = 0
                index_neuropil = 0
                batch_size = 100
                len_fake = len(sc.transient_movies["fake"])
                len_real = len(sc.transient_movies["real"])
                len_neuropil = len(sc.neuropil_movies)
                n_fake_by_batch = int((len_fake / movie_count) * batch_size)
                n_real_by_batch = int((len_real / movie_count) * batch_size)
                n_neuropil_by_batch = int((len_neuropil / movie_count) * batch_size)

                while True:
                    no_patches_added = True
                    if index_real < len_real:
                        end_index = min(index_real + n_real_by_batch, len_real)
                        for i in np.arange(index_real, end_index):
                            full_data.append(sc.transient_movies["real"][i])
                        index_real = end_index
                        no_patches_added = False

                    if index_fake < len_fake:
                        end_index = min(index_fake + n_fake_by_batch, len_fake)
                        for i in np.arange(index_fake, end_index):
                            full_data.append(sc.transient_movies["fake"][i])
                        index_fake = end_index
                        no_patches_added = False

                    if index_neuropil < len_neuropil:
                        end_index = min(index_neuropil + n_neuropil_by_batch, len_neuropil)
                        for i in np.arange(index_neuropil, end_index):
                            full_data.append(sc.neuropil_movies[i])
                        index_neuropil = end_index
                        no_patches_added = False

                    if no_patches_added:
                        break
            else:
                np.random.shuffle(movies_shuffling)

    n_movies_for_training = int(movie_count * split_values[0])
    n_movies_for_validation = int(movie_count * split_values[1])
    train_data = []
    for index in movies_shuffling[:n_movies_for_training]:
        train_data.append(full_data[index])

    valid_data = []
    for index in movies_shuffling[n_movies_for_training:n_movies_for_training + n_movies_for_validation]:
        valid_data.append(full_data[index])

    StratificationCamembert(data_list=valid_data,
                            description="VALIDATION DATA",
                            n_max_transformations=6,
                            debug_mode=True)
    test_data = []
    for index in movies_shuffling[n_movies_for_training + n_movies_for_validation:]:
        test_data.append(full_data[index])

    StratificationCamembert(data_list=test_data,
                            description="TEST DATA",
                            n_max_transformations=6,
                            debug_mode=True)

    test_movie_descr = []
    for movie in movies_shuffling[n_movies_for_training + n_movies_for_validation:]:
        test_movie_descr.append(movies_descr[movie])

    n_max_transformations = train_data[0].n_available_augmentation_fct
    strat_process = StratificationDataProcessor(data_list=train_data, n_max_transformations=n_max_transformations,
                                                description="TRAINING DATA",
                                                debug_mode=False, main_ratio_balance=main_ratio_balance,
                                                crop_non_crop_ratio_balance=crop_non_crop_ratio_balance,
                                                non_crop_ratio_balance=non_crop_ratio_balance)
    train_data = strat_process.get_new_data_list()

    return train_data, valid_data, test_data, test_movie_descr, cell_to_load_by_ms


def build_model(input_shape, lstm_layers_size, activation_fct="relu", use_mulimodal_inputs=False, dropout_rate=0,
                dropout_rnn_rate=0,
                without_bidirectional=False, with_batch_normalization=False):
    n_frames = input_shape[0]
    # First, let's define a vision model using a Sequential model.
    # This model will encode an image into a vector.
    vision_model = Sequential()
    get_custom_objects().update({'swish': Swish(swish)})
    # to choose between swish and relu

    vision_model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape[1:]))
    if activation_fct != "swish":
        vision_model.add(Activation(activation_fct))
    else:
        vision_model.add(Lambda(swish))
    if with_batch_normalization:
        vision_model.add(BatchNormalization())
    vision_model.add(Conv2D(64, (3, 3)))
    if activation_fct != "swish":
        vision_model.add(Activation(activation_fct))
    else:
        vision_model.add(Lambda(swish))
    if with_batch_normalization:
        vision_model.add(BatchNormalization())
    vision_model.add(MaxPooling2D((2, 2)))

    vision_model.add(Conv2D(128, (3, 3), padding='same'))
    if activation_fct != "swish":
        vision_model.add(Activation(activation_fct))
    else:
        vision_model.add(Lambda(swish))
    vision_model.add(Conv2D(128, (3, 3)))
    if activation_fct != "swish":
        vision_model.add(Activation(activation_fct))
    else:
        vision_model.add(Lambda(swish))
    if with_batch_normalization:
        vision_model.add(BatchNormalization())
    vision_model.add(MaxPooling2D((2, 2)))

    # vision_model.add(Conv2D(256, (3, 3), activation=activation_fct, padding='same'))
    # vision_model.add(Conv2D(256, (3, 3), activation=activation_fct))
    # vision_model.add(Conv2D(256, (3, 3), activation=activation_fct))
    # vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Flatten())
    if dropout_rate > 0:
        vision_model.add(layers.Dropout(dropout_rate))

    video_input = Input(shape=input_shape, name="video_input")
    # This is our video encoded via the previously trained vision_model (weights are reused)
    encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
    if without_bidirectional:
        for lstm_index, lstm_size in enumerate(lstm_layers_size):
            if lstm_index == 0:
                encoded_video = LSTM(lstm_size, dropout=dropout_rnn_rate,
                                     recurrent_dropout=dropout_rnn_rate,
                                     return_sequences=True)(encoded_frame_sequence)
            else:
                encoded_video = LSTM(lstm_size, dropout=dropout_rnn_rate, recurrent_dropout=dropout_rnn_rate)(
                    encoded_video)
    else:
        # encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector
        for lstm_index, lstm_size in enumerate(lstm_layers_size):
            if lstm_index == 0:
                encoded_video = Bidirectional(LSTM(lstm_size, dropout=dropout_rnn_rate,
                                                   recurrent_dropout=dropout_rnn_rate,
                                                   return_sequences=True))(encoded_frame_sequence)
            else:
                encoded_video = Bidirectional(LSTM(lstm_size, dropout=dropout_rnn_rate,
                                                   recurrent_dropout=dropout_rnn_rate))(encoded_video)

        # TODO: test if GlobalMaxPool1D +/- dropout is useful here ?
        # encoded_video = GlobalMaxPool1D()(encoded_video)
        # encoded_video = Dropout(0.25)(encoded_video)

        # if we put input_shape in Bidirectional, it crashes
        # encoded_video = Bidirectional(LSTM(128, return_sequences=True),
        #                               input_shape=(n_frames, 128))(encoded_frame_sequence)

    video_input_masked = Input(shape=input_shape, name="video_input_masked")
    # This is our video encoded via the previously trained vision_model (weights are reused)
    encoded_frame_sequence_masked = TimeDistributed(vision_model)(
        video_input_masked)  # the output will be a sequence of vectors
    # encoded_video_masked = LSTM(256)(encoded_frame_sequence_masked)  # the output will be a vector
    if without_bidirectional:
        for lstm_index, lstm_size in enumerate(lstm_layers_size):
            if lstm_index == 0:
                encoded_video_masked = LSTM(lstm_size, dropout=dropout_rnn_rate, recurrent_dropout=dropout_rnn_rate,
                                            return_sequences=True)(encoded_frame_sequence_masked)
            else:
                encoded_video_masked = LSTM(lstm_size, dropout=dropout_rnn_rate,
                                            recurrent_dropout=dropout_rnn_rate)(encoded_video_masked)
    else:
        for lstm_index, lstm_size in enumerate(lstm_layers_size):
            if lstm_index == 0:
                encoded_video_masked = Bidirectional(LSTM(lstm_size, dropout=dropout_rnn_rate,
                                                          recurrent_dropout=dropout_rnn_rate, return_sequences=True))(
                    encoded_frame_sequence_masked)
            else:
                encoded_video_masked = Bidirectional(LSTM(lstm_size, dropout=dropout_rnn_rate,
                                                          recurrent_dropout=dropout_rnn_rate
                                                          ))(encoded_video_masked)

    # in case we want 2 videos, one with masked, and one with the cell centered
    if use_mulimodal_inputs:
        merged = layers.concatenate([encoded_video, encoded_video_masked])
        if with_batch_normalization:
            merged = BatchNormalization()(merged)
        if dropout_rate > 0:
            merged = layers.Dropout(dropout_rate)(merged)
        # TODO: test those 7 lines (https://www.kaggle.com/amansrivastava/exploration-bi-lstm-model)
        # number_dense_units = 1000
        # merged = Dense(number_dense_units, activation=activation_function)(merged)
        # merged = Activation(activation_fct)(merged)
        # if with_batch_normalization:
        #     merged = BatchNormalization()(merged)
        # if dropout_rate > 0:
        #     merged = (layers.Dropout(dropout_rate)(merged)

        # output = TimeDistributed(Dense(1, activation='sigmoid')))(merged)
        output = Dense(n_frames, activation='sigmoid')(merged)
        video_model = Model(inputs=[video_input, video_input_masked], outputs=output)
    else:
        # output = TimeDistributed(Dense(1, activation='sigmoid'))(encoded_video)
        output = Dense(n_frames, activation='sigmoid')(encoded_video)
        video_model = Model(inputs=video_input, outputs=output)

    return video_model


def get_source_profile_for_prediction(ms, cell, augmentation_functions=None,
                                      overlap_value=0, max_width=30, max_height=30, sliding_window_len=100):
    n_frames = ms.tiff_movie_normalized.shape[0]
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


def transients_prediction_from_movie(ms_to_use, param, overlap_value=0.8,
                                     use_data_augmentation=True, cells_to_predict=None):
    # TODO: Pass a dict to predict more than one session at once
    if len(ms_to_use) > 1:
        ms_to_use = list(ms_to_use[0])

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False,
                                            for_transient_classifier=True)

    ms = ms_str_to_ms_dict[ms_to_use[0]]

    n_cells = len(ms.coord)
    if cells_to_predict is None:
        cells_to_load = np.arange(n_cells)
    else:
        cells_to_load = np.array(cells_to_predict)

    cells_to_load = np.setdiff1d(cells_to_load, ms.cells_to_remove)
    using_cnn_predictions = False
    if using_cnn_predictions:
        if ms.cell_cnn_predictions is not None:
            print(f"Using cnn predictions from {ms.description}")
            # not taking into consideration cells that are not predicted as true from the cell classifier
            cells_predicted_as_false = np.where(ms.cell_cnn_predictions < 0.5)[0]
            cells_to_load = np.setdiff1d(cells_to_load, cells_predicted_as_false)

    total_n_cells = len(cells_to_load)
    if total_n_cells == 0:
        raise Exception(f"No cells loaded")

    cells_to_load = np.array(cells_to_load)

    movie_loaded = load_movie(ms)
    if not movie_loaded:
        raise Exception(f"could not load movie of ms {ms.description}")

    n_frames = ms.tiff_movie_normalized.shape[0]
    print(f"transients_prediction_from_movie n_frames {n_frames}")

    spike_nums_dur = np.zeros((n_cells, n_frames), dtype="int8")
    predictions_by_cell = np.zeros((n_cells, n_frames))

    # loading model
    path_to_tc_model = param.path_data + "transient_classifier_model/"
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
    # Model reconstruction from JSON file
    with open(json_file, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(weights_file)
    stop_time = time.time()
    print(f"Time for loading model: "
          f"{np.round(stop_time - start_time, 3)} s")

    start_time = time.time()
    predictions_threshold = 0.4
    for cell in cells_to_load:
        predictions = predict_transient_from_model(ms=ms, cell=cell, model=model, overlap_value=overlap_value,
                                                   use_data_augmentation=use_data_augmentation)
        predictions_by_cell[cell] = predictions
        spike_nums_dur[cell, predictions >= predictions_threshold] = 1

    stop_time = time.time()
    print(f"Time to predict {total_n_cells} cells: "
          f"{np.round(stop_time - start_time, 3)} s")

    file_name = f"/{ms.description}_predictions_{param.time_str}.mat"
    sio.savemat(param.path_results + file_name, {'spike_nums_dur_predicted': spike_nums_dur,
                                                 'predictions': predictions_by_cell})


def predict_transient_from_model(ms, cell, model, overlap_value=0.8,
                                 use_data_augmentation=True):
    start_time = time.time()
    n_frames = len(ms.tiff_movie_normalized)
    multi_inputs = (model.layers[0].output_shape == model.layers[1].output_shape)
    sliding_window_len = model.layers[0].output_shape[1]
    max_height = model.layers[0].output_shape[2]
    max_width = model.layers[0].output_shape[3]
    if use_data_augmentation:
        augmentation_functions = [horizontal_flip, vertical_flip, v_h_flip]
    else:
        augmentation_functions = None
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
    print(f"Time to get predictions for cell {cell}: "
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


def predict_transient_from_saved_model(ms, cell, weights_file, json_file, overlap_value=0.8,
                                       use_data_augmentation=True):
    start_time = time.time()
    # Model reconstruction from JSON file
    with open(json_file, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(weights_file)
    stop_time = time.time()
    print(f"Time for loading model: "
          f"{np.round(stop_time - start_time, 3)} s")

    return predict_transient_from_model(ms=ms, cell=cell, model=model, overlap_value=overlap_value,
                                        use_data_augmentation=use_data_augmentation)


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
    # ax1.plot(epochs, smooth_curve(train_values), 'bo', label=f'Training {key_name}')
    # ax1.plot(epochs, smooth_curve(val_values), 'b', label=f'Validation {key_name}')
    ax1.plot(epochs, train_values, 'bo', label=f'Training {key_name}')
    ax1.plot(epochs, val_values, 'b', label=f'Validation {key_name}')
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


# from: http://www.deepideas.net/unbalanced-classes-machine-learning/
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# from: http://www.deepideas.net/unbalanced-classes-machine-learning/
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


# from https://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy
def single_class_accuracy_precision(interesting_class_id):
    def precision(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return precision


# from https://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy
def single_class_accuracy_recall(interesting_class_id):
    def recall(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_true with class_id_preds for precision here
        accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return recall


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

    go_predict_from_movie = False

    if go_predict_from_movie:
        transients_prediction_from_movie(ms_to_use=["p12_171110_a000_ms"], param=param, overlap_value=0.8,
                                         use_data_augmentation=True,
                                         cells_to_predict=np.arange(10))
        # "p13_18_10_29_a001_ms"
        # np.array([0, 5, 12, 13, 31, 42, 44, 48, 51, 77, 117])
        # p12_171110_a000_ms
        # p7_171012_a000_ms
        return

    # 3 options to target the cell
    # 1) put the cell in the middle of the frame
    # 2) put all pixels in the border to 1
    # 3) Give 2 inputs, movie full frame (20x20 pixels) + movie mask non binary or binary

    """
    Best so far:
    use_mulimodal_inputs = True
    batch_size = 16 (different not tried)
    window_len = 50 (just tried 100 otherwise)
    max_width = 25
    max_height = 25
    overlap_value = 0.9
    dropout_value = 0
    pixels_around = 0
    with_augmentation_for_training_data = True
    buffer = None
    split_values = (0.7, 0.2) (does it matter ?)
    optimizer_choice = "adam"
    activation_fct = "swish"
    with_learning_rate_reduction = True
    without_bidirectional = False
    lstm_layers_size = [128, 256]
    """
    use_mulimodal_inputs = True
    n_epochs = 40
    batch_size = 16
    window_len = 10
    max_width = 25
    max_height = 25
    overlap_value = 0.9
    dropout_value = 0
    dropout_value_rnn = 0
    with_batch_normalization = True
    max_n_transformations = 6
    pixels_around = 0
    with_augmentation_for_training_data = True
    buffer = None
    split_values = (0.6, 0.2)
    optimizer_choice = "RMSprop"  # "SGD"  "RMSprop"  "adam", SGD
    activation_fct = "swish"
    with_learning_rate_reduction = True
    learning_rate_reduction_patience = 3
    without_bidirectional = False
    lstm_layers_size = [256, 512] # 128, 256, 512
    with_early_stopping = True
    early_stop_patience = 25 # 10
    model_descr = ""
    with_shuffling = True
    seed_value = 42  # use None to not use seed
    main_ratio_balance = (0.60, 0.20, 0.20)
    crop_non_crop_ratio_balance = (-1, -1)  # (0.8, 0.2)
    non_crop_ratio_balance = (-1, -1)  # (0.85, 0.15)

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
    test_movie_descr, cell_to_load_by_ms = \
        load_data_for_generator(param,
                                split_values=split_values,
                                sliding_window_len=window_len,
                                overlap_value=overlap_value,
                                movies_shuffling=None,
                                with_shuffling=with_shuffling,
                                main_ratio_balance=main_ratio_balance,
                                crop_non_crop_ratio_balance=crop_non_crop_ratio_balance,
                                non_crop_ratio_balance=non_crop_ratio_balance,
                                max_n_transformations=max_n_transformations,
                                seed_value=seed_value
                                )

    stop_time = time.time()
    print(f"Time for loading data for generator: "
          f"{np.round(stop_time - start_time, 3)} s")

    # Generators
    start_time = time.time()
    training_generator = DataGenerator(train_data_list, with_augmentation=with_augmentation_for_training_data,
                                       **params_generator)
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
    model = build_model(input_shape, activation_fct=activation_fct, dropout_rate=dropout_value,
                        dropout_rnn_rate=dropout_value_rnn,
                        use_mulimodal_inputs=use_mulimodal_inputs, without_bidirectional=without_bidirectional,
                        lstm_layers_size=lstm_layers_size,
                        with_batch_normalization=with_batch_normalization)

    print(model.summary())
    # raise Exception("TOTOOO")

    # Save the model architecture
    with open(
            f'{param.path_results}/transient_classifier_model_architecture_{model_descr}_'
            f'{param.time_str}.json',
            'w') as f:
        f.write(model.to_json())

    # Define the optimizer
    # from https://www.kaggle.com/shahariar/keras-swish-activation-acc-0-996-top-7

    if optimizer_choice == "adam":
        optimizer = adam(lr=0.001, epsilon=1e-08, decay=0.0)
    elif optimizer_choice == "SGD":
        # default parameters: lr=0.01, momentum=0.0, decay=0.0, nesterov=False
        optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    else:
        # default parameters: lr=0.001, rho=0.9, epsilon=None, decay=0.0
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    # optimizer = 'rmsprop'

    # precision = PPV and recall = sensitiviy but in our case just concerning the active frames
    # the sensitivity and specificity otherwise refers to non-active and active frames classifier
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', sensitivity, specificity, precision])
    # sample_weight_mode="temporal",

    # Set a learning rate annealer
    # from: https://www.kaggle.com/shahariar/keras-swish-activation-acc-0-996-top-7
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_precision',
                                                patience=learning_rate_reduction_patience,
                                                verbose=1,
                                                factor=0.5,
                                                mode='max',
                                                min_lr=0.00001)  # used to be: 0.00001

    # callbacks to be execute during training
    # A callback is a set of functions to be applied at given stages of the training procedure.
    callbacks_list = []
    if with_learning_rate_reduction:
        callbacks_list.append(learning_rate_reduction)

    if with_early_stopping:
        callbacks_list.append(EarlyStopping(monitor="val_acc", min_delta=0, patience=early_stop_patience, mode="max",
                                            restore_best_weights=True))

    with_model_check_point = True
    # not very useful to save best only if we use EarlyStopping
    if with_model_check_point:
        end_file_path = f"_{param.time_str}.h5"
        file_path = param.path_results + "/transient_classifier_weights_{epoch:02d}-{val_acc:.4f}" + end_file_path
        # callbacks_list.append(ModelCheckpoint(filepath=file_path, monitor="val_acc", save_best_only="True",
        #                                       save_weights_only="True", mode="max"))

        callbacks_list.append(ModelCheckpoint(filepath=file_path, monitor="val_acc",
                                              save_weights_only="True", mode="max"))

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
                                  callbacks=callbacks_list)

    print(f"history.history.keys() {history.history.keys()}")
    stop_time = time.time()
    print(f"Time for fitting the model to the data with {n_epochs} epochs: "
          f"{np.round(stop_time - start_time, 3)} s")

    show_plots = True

    if show_plots:
        key_names = ["loss", "acc", "sensitivity", "specificity", "precision"]
        for key_name in key_names:
            plot_training_and_validation_values(history=history, key_name=key_name,
                                                result_path=result_path, param=param)

    history_dict = history.history
    start_time = time.time()

    # model.save(f'{param.path_results}/transient_classifier_model_{model_descr}_test_acc_{test_acc}_{param.time_str}.h5')

    # saving params in a txt file
    file_name_txt = f'{param.path_results}/stat_model_{param.time_str}.txt'
    round_factor = 1

    with open(file_name_txt, "w", encoding='UTF-8') as file:
        file.write(f"n epochs: {n_epochs}" + '\n')
        file.write(f"with_augmentation_for_training_data {with_augmentation_for_training_data}" + '\n')
        file.write(f"batch_size: {batch_size}" + '\n')
        file.write(f"with_shuffling: {with_shuffling}" + '\n')
        file.write(f"seed_value: {seed_value}" + '\n')
        file.write(f"with_learning_rate_reduction: {with_learning_rate_reduction}" + '\n')
        file.write(f"without_bidirectional: {without_bidirectional}" + '\n')
        file.write(f"use_mulimodal_inputs: {use_mulimodal_inputs}" + '\n')
        file.write(f"lstm_layers_size: {lstm_layers_size}" + '\n')
        file.write(f"window_len: {window_len}" + '\n')
        file.write(f"max_width: {max_width}" + '\n')
        file.write(f"max_height: {max_height}" + '\n')
        file.write(f"overlap_value: {overlap_value}" + '\n')
        file.write(f"dropout_value: {dropout_value}" + '\n')
        file.write(f"dropout_value_rnn: {dropout_value_rnn}" + '\n')
        file.write(f"with_batch_normalization: {with_batch_normalization}" + '\n')
        file.write(f"pixels_around: {pixels_around}" + '\n')
        file.write(f"buffer: {'None' if (buffer is None) else buffer}" + '\n')
        file.write(f"split_values: {split_values}" + '\n')
        file.write(f"main_ratio_balance: {main_ratio_balance}" + '\n')
        file.write(f"crop_non_crop_ratio_balance: {crop_non_crop_ratio_balance}" + '\n')
        file.write(f"non_crop_ratio_balance: {non_crop_ratio_balance}" + '\n')
        file.write(f"max_n_transformations: {max_n_transformations}" + '\n')
        file.write(f"optimizer_choice: {optimizer_choice}" + '\n')
        file.write(f"activation_fct: {activation_fct}" + '\n')
        file.write(f"train_loss: {history_dict['loss']}" + '\n')
        file.write(f"val_loss: {history_dict['val_loss']}" + '\n')
        file.write(f"train_acc: {history_dict['acc']}" + '\n')
        file.write(f"val_acc: {history_dict['val_acc']}" + '\n')
        file.write(f"train_sensitivity: {history_dict['sensitivity']}" + '\n')
        file.write(f"val_sensitivity: {history_dict['val_sensitivity']}" + '\n')
        file.write(f"train_specificity: {history_dict['specificity']}" + '\n')
        file.write(f"val_specificity: {history_dict['val_specificity']}" + '\n')
        file.write(f"train_precision: {history_dict['precision']}" + '\n')
        file.write(f"val_precision: {history_dict['val_precision']}" + '\n')
        # file.write(f"train_recall: {history_dict['recall']}" + '\n')
        # file.write(f"val_recall: {history_dict['val_recall']}" + '\n')

        # cells used
        for ms_str, cells in cell_to_load_by_ms.items():
            file.write(f"{ms_str}: ")
            for cell in cells:
                file.write(f"{cell} ")
            file.write("\n")
        file.write("" + '\n')

    val_acc = history_dict['val_acc'][-1]
    # model.save_weights(
    #     f'{param.path_results}/transient_classifier_weights_{model_descr}_val_acc_{val_acc}_{param.time_str}.h5')

    stop_time = time.time()
    print(f"Time for saving the model: "
          f"{np.round(stop_time - start_time, 3)} s")

    start_time = time.time()
    source_profiles_dict = dict()
    test_data, test_data_masked, test_labels = generate_movies_from_metadata(movie_data_list=test_data_list,
                                                                             window_len=window_len,
                                                                             max_width=max_width,
                                                                             max_height=max_height,
                                                                             pixels_around=pixels_around,
                                                                             buffer=buffer,
                                                                             source_profiles_dict=source_profiles_dict)
    stop_time = time.time()
    print(f"Time for generating test data: "
          f"{np.round(stop_time - start_time, 3)} s")
    print(f"test_data.shape {test_data.shape}")
    # calculating default accuracy by putting all predictions to zero

    # test_labels
    n_test_frames = 0
    n_rights = 0
    for batch_labels in test_labels:
        n_rights += len(batch_labels) - np.sum(batch_labels)
        n_test_frames += len(batch_labels)
    print(f"Default test accuracy {str(np.round(n_rights / n_test_frames, 3))}")

    start_time = time.time()
    if use_mulimodal_inputs:
        test_loss, test_acc, test_sensitivity, test_specificity, test_precision = \
            model.evaluate({'video_input': test_data,
                            'video_input_masked': test_data_masked},
                           test_labels, verbose=2)
    else:
        test_loss, test_acc, test_sensitivity, test_specificity, test_precision = \
            model.evaluate(test_data_masked, test_labels)

    print(f"test_acc {test_acc}, test_sensitivity {test_sensitivity}, test_specificity {test_specificity}, "
          f"test_precision {test_precision}")

    stop_time = time.time()
    print(f"Time for evaluating test data: "
          f"{np.round(stop_time - start_time, 3)} s")

    # start_time = time.time()
    # if use_mulimodal_inputs:
    #     prediction = model.predict({'video_input': test_data,
    #                                 'video_input_masked': test_data_masked})
    # else:
    #     prediction = model.predict(test_data_masked)
    # print(f"prediction.shape {prediction.shape}")
    # stop_time = time.time()
    # print(f"Time to predict test data: "
    #       f"{np.round(stop_time - start_time, 3)} s")

    # for i in np.arange(prediction.shape[0]):
    #     print(f"Sum {i}: {np.sum(prediction[i])}")
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
