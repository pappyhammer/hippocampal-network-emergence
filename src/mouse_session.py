import matplotlib.cm as cm
from scipy import signal
# important to avoid a bug when using virtualenv
# import matplotlib
# matplotlib.use('TkAgg')
# to comment for mesocentre
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
import tifffile
from matplotlib.figure import SubplotParams
from matplotlib import patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import hdf5storage
import time
import os
import pyabf
import matplotlib.image as mpimg
import networkx as nx
from pattern_discovery.graph.misc import welsh_powell
# to add homemade package, go to preferences, then project interpreter, then click on the wheel symbol
# then show all, then select the interpreter and lick on the more right icon to display a list of folder and
# add the one containing the folder pattern_discovery
import pattern_discovery.tools.misc as tools_misc
from pattern_discovery.tools.misc import get_time_correlation_data
from pattern_discovery.tools.misc import get_continous_time_periods, give_unique_id_to_each_transient_of_raster_dur
from pattern_discovery.display.raster import plot_spikes_raster, plot_with_imshow
from pattern_discovery.display.misc import time_correlation_graph, plot_hist_distribution, plot_scatters, plot_box_plots
from pattern_discovery.display.cells_map_module import CoordClass
from pattern_discovery.clustering.kmean_version.k_mean_clustering import CellAssembliesStruct
import pattern_discovery.cilva.analysis as cilva_analysis
import pattern_discovery.cilva.core as cilva_core
from sortedcontainers import SortedDict
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from hne_spike_structure import HNESpikeStructure
# from mvt_selection_gui import MvtSelectionGui
from pattern_discovery.tools.signal import smooth_convolve
from PIL import ImageSequence
from sklearn.decomposition import PCA
from shapely.geometry import MultiPoint, LineString
from ScanImageTiffReader import ScanImageTiffReader
import hne_animation as hne_anim
import math
import PIL
import pattern_discovery.display.misc as display_misc
import itertools
from pattern_discovery.tools.lfp_analysis_tools import WaveletParameters, spectral_analysis_on_time_segment, \
    butter_bandpass_filter, plot_wavelet_heatmap
import scipy.signal
import scipy.stats as stats
import yaml


class MouseSession:
    def __init__(self, age, session_id, param, sampling_rate=10, weight=None, spike_nums=None, spike_nums_dur=None,
                 percentile_for_low_activity_threshold=1):
        """

        :param age:
        :param session_id:
        :param param:
        :param nb_ms_by_frame:
        :param sampling_rate: in Hz
        :param weight:
        :param spike_nums:
        :param spike_nums_dur:
        :param percentile_for_low_activity_threshold:
        """
        # should be a list of int
        self.param = param
        self.age = age
        self.session_id = str(session_id)
        self.sampling_rate = sampling_rate
        self.description = f"P{self.age}_{self.session_id}"
        # tell when an abf file has been loaded
        self.abf_loaded = False
        self.spike_struct = HNESpikeStructure(mouse_session=self, spike_nums=spike_nums, spike_nums_dur=spike_nums_dur)
        # spike_nums represents the onsets of the neuron spikes
        # bin from 25000 frames caiman "onsets"
        self.caiman_spike_nums = None
        # raster_dur built upon the 25000 frames caiman "onsets"
        self.caiman_spike_nums_dur = None
        self.caiman_active_periods = None
        # will be a dict, see load_suite2p_data() to know the key
        self.suite2p_data = None
        # if True, will use suite2p_data for all computations
        self.use_suite_2p = False
        self.traces = None
        self.raw_traces = None
        self.smooth_traces = None
        self.z_score_traces = None
        self.z_score_raw_traces = None
        self.z_score_smooth_traces = None
        # average of the all frame pixels, array of float same length as the number of frames
        self.global_roi = None
        # two arrays of size n_frames, representing the shift used to correct the motion from the movie
        self.x_shifts = None
        self.y_shifts = None
        # array of boolean indicating at which frame there is some shifting
        self.shift_periods_bool = None
        self.coord = None
        # 1-D array with for each frame the speed as a float value
        self.speed_by_frame = None
        # comes from the gui
        # used by the transient classifier to indicated which frames are doubtful and should not be used for training
        self.doubtful_frames_nums = None
        # an array of int with the index of the cells to remove
        self.cells_to_remove = None
        # if cells have been removed using self.cells_to_remove, then it will be an array of length the original numbers
        # of cells and as value either of positive int representing the new index of the cell or -1 if the cell has been
        # removed
        self.removed_cells_mapping = None
        # 1d array float
        self.lfp_signal = None
        self.lfp_sampling_rate = None
        # array 1d representing the frames in which a sharp is happening, same length as the number of sharp
        self.sharp_frames = None
        # array of float, each index corresponds to a cell and the value is the prediction made by the cell classifier
        self.cell_cnn_predictions = None
        self.load_cnn_cell_classifier_results()
        self.rnn_transients_predictions = None
        # use to load part of a full tiff movie, should be initialized before using rnn
        self.tiffs_for_transient_classifier_path = None
        self.activity_threshold = None
        self.low_activity_threshold_by_percentile = dict()
        self.percentile_for_low_activity_threshold = percentile_for_low_activity_threshold
        self.low_activity_threshold = None
        self.avg_cell_map_img = None
        self.avg_cell_map_img_file_name = None
        self.tif_movie_file_name = None
        # will be a dict, with key the principal component number, and value the cells order
        self.pca_seq_cells_order = None
        # will be use by the cell classifier
        self.tiff_movie = None
        # movie dimensions (x = shape[1] and y = shape[0] for a given frame)
        self.movie_len_x = None
        self.movie_len_y = None
        # will be use by the cell and transient classifier, normalize between 0 and 1
        self.tiff_movie_norm_0_1 = None
        # z_score normalization
        self.tiff_movie_normalized = None
        # mean of the tiff_movie
        self.tiff_movie_mean = None
        # std of the tiff_movie, done to avoid doing it on the clusters
        self.tiff_movie_std = None
        # Pillow image
        self.tiff_movie_image = None
        # list of list of int representing cell indices
        # initiated when loading_cell_assemblies
        self.cell_assemblies = None
        # dict with key a int representing the cell_assembly cluster and and value a list of tuple representing first
        # and last index including of the SCE
        self.sce_times_in_single_cell_assemblies = None
        # list of tuple of int
        self.sce_times_in_multiple_cell_assemblies = None
        # list of tuple of int (gather data from sce_times_in_single_cell_assemblies and
        # sce_times_in_multiple_cell_assemblies)
        self.sce_times_in_cell_assemblies = None
        # for each cell, list of list, each correspond to tuples (first and last index of the SCE in frames)
        # in which the cell is supposed to be active for the single cell assemblie to which it belongs
        self.sce_times_in_cell_assemblies_by_cell = None

        if (self.param is not None) and (self.param.cell_assemblies_data_path is not None):
            self.load_cell_assemblies_data()

        # for seq
        self.best_order_loaded = None
        if (self.param is not None) and (self.param.best_order_data_path is not None):
            self.load_best_order_data()
        self.weight = weight
        self.coord_obj = None

        # dict containing information for Richard data
        # keys will be : Active_Wake_Frames, Quiet_Wake_Frames, REMs_Frames, NREMs_Frames
        self.richard_dict = None

        # list of tuple of int representing first and last frame of a period of z-shift
        self.z_shift_periods = []

        # used for time-correlation graph purpose
        self.time_lags_list = None
        self.correlation_list = None
        self.time_lags_dict = None
        self.correlation_dict = None
        self.time_lags_window = None

        # initialized when loading abf
        self.abf_sampling_rate = None
        self.mvt_frames = None
        self.mvt_frames_periods = None
        # same len as mvt_frames
        self.speed_by_mvt_frame = None
        self.with_run = False
        self.with_piezo = False
        # raw piezo: after removing period not corresponding to the movie, and using absolute value
        self.raw_piezo = None
        self.raw_piezo_without_abs = None
        self.peaks_raw_piezo = None
        # represents the frames such as defined in the abf file, used to match the frames index to the raw_piezo data.
        # for frames f with index 10, x = abf_frames[f] will give us the index of f such that self.raw_piezo[x]
        # represent
        # the piezzo value at frame x
        self.abf_frames = None
        self.abf_times_in_sec = None
        self.threshold_piezo = None
        self.twitches_frames = None
        self.twitches_frames_periods = None
        # dictionnay containing the periods of shift from period_gui_selection from piezzo + imaging shift
        # keys: shift_twitch, shift_long, shift_unclassified
        self.shift_data_dict = None
        # periods (tuples of int)
        self.short_lasting_mvt = None
        self.short_lasting_mvt_frames = None
        self.short_lasting_mvt_frames_periods = None
        self.complex_mvt = None
        self.complex_mvt_frames = None
        self.complex_mvt_frames_periods = None
        self.intermediate_behavourial_events = None
        self.intermediate_behavourial_events_frames = None
        self.intermediate_behavourial_events_frames_periods = None
        self.noise_mvt = None
        self.noise_mvt_frames = None
        self.noise_mvt_frames_periods = None

        # used for transient classifier purpose
        self.transient_classifier_spike_nums_dur = dict()

        self.sce_bool = None
        self.sce_times_numbers = None
        self.SCE_times = None
        # for khazipov method
        self.lowest_std_in_piezo = None

        # TWITCHES:
        """
                groups: 0 : all twitches,
                group 1: those in sce_events
                group 2: those not in sce-events
                group 3: those not in sce-events but with sce-events followed less than one second after
                group 4: those not in sce-events and not followed by sce
         """
        self.twitches_group_title = SortedDict()
        self.twitches_group_title[0] = "all"
        self.twitches_group_title[1] = "in_events"
        self.twitches_group_title[2] = "outside_events"
        self.twitches_group_title[3] = "just_before_events"
        self.twitches_group_title[4] = "nothing_special"
        self.twitches_group_title[5] = "event_in_or_after_twitch"
        self.twitches_group_title[6] = "activity_linked_to_twitch"
        self.twitches_group_title[7] = "significant_events_not_links_to_twitch"
        self.twitches_group_title[8] = "significant_events_not_links_to_twitch_or_mvts"
        self.twitches_group_title[9] = "significant_events_linked_to_twitch_or_mvts"

        # key is a twitch group, value is a list of tuple representing the beginning and the end time (included)
        # for each event
        # 1, 3 and 4
        # 5: regroup sce before and during events
        self.events_by_twitches_group = SortedDict()

    def load_caiman_results(self, path_data):
        """
        Load caiman prediction, used to be displayed in the GUI
        :param path_data:
        :return:
        """
        start_time = time.time()
        if self.raw_traces is None:
            self.load_tiff_movie_in_memory()
            self.raw_traces = self.build_raw_traces_from_movie()
            if self.raw_traces is None:
                print(f"{self.description} no raw_traces, no caiman loading")
                return

        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(self.param.path_data + path_data):
            file_names.extend(local_filenames)
            break
        if len(file_names) == 0:
            return

        caiman_file_name = None
        for file_name in file_names:
            file_name_original = file_name
            file_name = file_name.lower()
            if self.description.lower() not in file_name:
                continue
            if "caiman" not in file_name:
                continue
            if "spikenums" not in file_name:
                continue
            caiman_file_name = os.path.join(self.param.path_data, path_data, file_name_original)

        if caiman_file_name is None:
            return
        data_file = hdf5storage.loadmat(caiman_file_name)
        # print(f'load_caiman_results: {list(data_file.keys())}')
        if "spikenums" in data_file:
            caiman_spike_nums = data_file["spikenums"].astype(int)
        else:
            caiman_spike_nums = data_file["spikenumsPyr"].astype(int)

        spike_nums_bin = np.zeros((caiman_spike_nums.shape[0], caiman_spike_nums.shape[1] // 2),
                                  dtype="int8")

        for cell in np.arange(spike_nums_bin.shape[0]):
            binned_cell = caiman_spike_nums[cell].reshape(-1, 2).mean(axis=1)
            binned_cell[binned_cell > 0] = 1
            spike_nums_bin[cell] = binned_cell.astype("int")

        self.caiman_spike_nums = spike_nums_bin

        n_cells = self.raw_traces.shape[0]
        n_times = self.raw_traces.shape[1]

        # copying traces
        traces = self.raw_traces[:]

        # normalizing it, should be useful only to plot them
        for i in np.arange(n_cells):
            traces[i, :] = (traces[i, :] - np.mean(traces[i, :])) / np.std(traces[i, :])

        # based on diff
        spike_nums_all = np.zeros((n_cells, n_times), dtype="int8")
        for cell in np.arange(n_cells):
            onsets = []
            diff_values = np.diff(traces[cell])
            for index, value in enumerate(diff_values):
                if index == (len(diff_values) - 1):
                    continue
                if value < 0:
                    if diff_values[index + 1] >= 0:
                        onsets.append(index + 1)
            if len(onsets) > 0:
                spike_nums_all[cell, np.array(onsets)] = 1

        peak_nums = np.zeros((n_cells, n_times), dtype="int8")
        for cell in np.arange(n_cells):
            peaks, properties = signal.find_peaks(x=traces[cell])
            peak_nums[cell, peaks] = 1

        spike_nums_dur = np.zeros((n_cells, n_times), dtype="int8")
        for cell in np.arange(n_cells):
            peaks_index = np.where(peak_nums[cell, :])[0]
            onsets_index = np.where(spike_nums_all[cell, :])[0]

            for onset_index in onsets_index:
                peaks_after = np.where(peaks_index > onset_index)[0]
                if len(peaks_after) == 0:
                    continue
                peaks_after = peaks_index[peaks_after]
                peak_after = peaks_after[0]

                spike_nums_dur[cell, onset_index:peak_after + 1] = 1

        caiman_spike_nums_dur = np.zeros((spike_nums_dur.shape[0], spike_nums_dur.shape[1]), dtype="int8")
        caiman_active_periods = dict()
        for cell in np.arange(n_cells):
            caiman_active_periods[cell] = []
            periods = get_continous_time_periods(spike_nums_dur[cell])
            # print(f"{cell}: len(periods) {len(periods)}")
            for period in periods:
                if np.sum(self.caiman_spike_nums[cell, period[0]:period[1] + 1]) > 0:
                    caiman_active_periods[cell].append((period[0], period[1]))
                    caiman_spike_nums_dur[cell, period[0]:period[1] + 1] = 1
            # print(f"{cell}: len(caiman_active_periods[cell]) {len(caiman_active_periods[cell])}")

        # raster_dur built upon the 25000 frames caiman "onsets"
        self.caiman_active_periods = caiman_active_periods
        self.caiman_spike_nums_dur = caiman_spike_nums_dur

        stop_time = time.time()
        print(f"Time for loading caiman results {self.description}: "
              f"{np.round(stop_time - start_time, 3)} s")

    def normalize_movie(self):
        do_01_normalization = False
        do_z_score_normalization = True

        if do_01_normalization:
            # 0 to 1 normalization
            if (self.tiff_movie is not None) and (self.tiff_movie_normalized is None):
                max_value = np.max(self.tiff_movie)
                min_value = np.min(self.tiff_movie)
                print(f"{self.description} max tiff_movie {str(np.round(y_max_value, 3))}, "
                      f"mean tiff_movie {str(np.round(np.mean(self.tiff_movie), 3))}, "
                      f"median tiff_movie {str(np.round(np.median(self.tiff_movie), 3))}")
                # self.tiff_movie_normalized = (self.tiff_movie - y_min_value) / (y_max_value - y_min_value)
                self.tiff_movie_normalized = self.tiff_movie / max_value

        if do_z_score_normalization:
            # z-score standardization
            if (self.tiff_movie is not None) and (self.tiff_movie_normalized is None):
                print("normalizing the movie")
                # if (self.tiff_movie_mean is not None) and (self.tiff_movie_std is not None):
                #     # using loaded mean and std
                #     print("mean and std of tiff mmovie are loaded, no need to normalized the movie now")
                #     return
                #     # self.tiff_movie_normalized = (self.tiff_movie - self.tiff_movie_mean) / self.tiff_movie_std
                # else:
                self.tiff_movie_normalized = self.tiff_movie - np.mean(self.tiff_movie)
                print("movie normalization almost done")
                self.tiff_movie_normalized = self.tiff_movie_normalized / np.std(self.tiff_movie)
                # self.tiff_movie_normalized = np.copy(self.tiff_movie)
                print("movie normalization done")

    def create_smooth_traces(self):
        if self.raw_traces is None:
            print(f"create_smooth_traces for {self.description}, raw_traces is None")
            return
        if self.smooth_traces is not None:
            return

        self.smooth_traces = np.copy(self.raw_traces)
        # smoothing the trace
        windows = ['hanning', 'hamming', 'bartlett', 'blackman']
        i_w = 1
        window_length = 11
        for i in np.arange(self.raw_traces.shape[0]):
            smooth_signal = smooth_convolve(x=self.smooth_traces[i], window_len=window_length,
                                            window=windows[i_w])
            beg = (window_length - 1) // 2
            self.smooth_traces[i] = smooth_signal[beg:-beg]

    def normalize_traces(self):
        n_cells = self.raw_traces.shape[0]
        self.z_score_traces = np.zeros(self.raw_traces.shape)
        self.z_score_raw_traces = np.zeros(self.raw_traces.shape)
        if self.smooth_traces is None:
            self.create_smooth_traces()
        self.z_score_smooth_traces = np.zeros(self.smooth_traces.shape)
        # z_score traces
        for i in np.arange(n_cells):
            if self.traces is not None:
                self.z_score_traces[i, :] = (self.traces[i, :] - np.mean(self.traces[i, :])) / np.std(self.traces[i, :])
            if self.raw_traces is not None:
                self.z_score_raw_traces[i, :] = (self.raw_traces[i, :] - np.mean(self.raw_traces[i, :])) \
                                                / np.std(self.raw_traces[i, :])
            self.z_score_smooth_traces[i, :] = (self.smooth_traces[i, :] - np.mean(self.smooth_traces[i, :])) \
                                               / np.std(self.smooth_traces[i, :])

    def plot_psth_over_event_time_correlation_graph_style(self, event_str, time_around_events=10,
                                                          ax_to_use=None, color_to_use=None,
                                                          ax_to_use_total_events=None, color_to_use_total_events=None,
                                                          ax_to_use_total_spikes=None, color_to_use_total_spikes=None,
                                                          ax_to_use_for_scatter=None, color_to_use_for_scatter=None):
        """
        Same shape as a time-correlation graph but the zero will correpsond the event time, the celle will be
        correlated with itself
        :return:
        """
        if (self.shift_data_dict is None) or (self.spike_struct.spike_nums is None):
            return

        # results = get_time_correlation_data(spike_nums=self.spike_struct.spike_nums,
        #                                     events_times=self.twitches_frames_periods, time_around_events=10)
        # self.time_lags_list, self.correlation_list, \
        # self.time_lags_dict, self.correlation_dict, self.time_lags_window, cells_list = results
        spike_nums = self.spike_struct.spike_nums

        # events_times = self.twitches_frames_periods
        events_times = get_continous_time_periods(self.shift_data_dict[event_str].astype("int8"))
        n_twitches = len(events_times)
        print(f"{self.description}: number of twitches {n_twitches}")
        time_around_events = time_around_events

        nb_neurons = len(spike_nums)
        n_times = len(spike_nums[0, :])
        # values for each cell
        time_lags_dict = dict()
        correlation_dict = dict()
        # for ploting
        time_lags_list = []
        correlation_list = []
        cells_list = []
        nb_spikes_by_cell = dict()

        # first determining what is the maximum duration of an event, for array dimension purpose
        # max_duration_event = 0
        # for times in events_times:
        #     max_duration_event = np.max((max_duration_event, times[1] - times[0]))

        # time_window = int(np.ceil((max_duration_event + (time_around_events * 2)) / 2))
        # time_window = (time_around_events*2) + 1

        ratio_cells = []
        ratio_spike_twitch_total_twitches = []
        ratio_spike_twitch_total_spikes = []

        for neuron in np.arange(nb_neurons):
            # look at onsets
            neuron_spikes, = np.where(spike_nums[neuron, :])

            if len(neuron_spikes) == 0:
                continue

            spike_nums_to_use = spike_nums

            # time_window by 4
            distribution_array = np.zeros(((time_around_events * 2) + 1),
                                          dtype="int16")

            mask = np.ones(nb_neurons, dtype="bool")
            mask[neuron] = False

            # event_index = time_window
            # looping on each spike of the main neuron
            for n, event_times in enumerate(events_times):
                # only taking in consideration events that are not too close from bottom range or upper range
                # event_time = (event_times[0] + event_times[1]) // 2
                # taking beginning of event
                event_time = event_times[0]
                min_limit = max(0, (event_time - time_around_events))
                max_limit = min(event_time + 1 + time_around_events, n_times)
                # min((peak_time + time_window), (n_times - 1))
                if np.sum(spike_nums[neuron, min_limit:max_limit]) == 0:
                    continue
                # see to consider the case in which the cell spikes 2 times around a peak during the tim_window
                neuron_spike_time = spike_nums[neuron, min_limit:max_limit]
                # if np.sum(neuron_spike_time) > 1:
                # print(f"np.sum(neuron_spike_time) {np.sum(neuron_spike_time)} {np.where(neuron_spike_time)[0]}")
                # spikes_indices = np.where(spike_nums_to_use[:, min_limit:max_limit])
                beg_index = 0
                if (event_time - time_around_events) < 0:
                    beg_index = 0 - (event_time - time_around_events)
                len_result = max_limit - min_limit
                # print(f"spikes_indices {spikes_indices}")
                # copy_of_neuron_distrib = np.copy(distribution_array_2_d[neuron, :])
                distribution_array[beg_index:beg_index + len_result] += neuron_spike_time
                # distribution_array_2_d[neuron, :] = copy_of_neuron_distrib
            total_spikes = np.sum(distribution_array)
            total_spikes_after = np.sum(distribution_array[time_around_events - 1:])
            adding_this_neuron = total_spikes > 0

            # if np.sum(distribution_array[time_around_events - 2:]) < 2:
            #     adding_this_neuron = False
            #
            # if np.sum(distribution_array[time_around_events - 2:]) <= 2 * np.sum(
            #         distribution_array[:time_around_events - 2]):
            #     adding_this_neuron = False

            # if np.sum(distribution_array[time_around_events - 2:]) > 1:
            if adding_this_neuron:
                # nb_spikes = np.sum(distribution_array[time_around_events - 2:])
                # nb_spikes = total_spikes
                nb_spikes_by_cell[neuron] = total_spikes
                nb_spikes = total_spikes_after
                # print(f"cell {neuron}: distribution_array {distribution_array}")
                # TODO: plot distribution of % ratio
                # TODO: plot piezzo for top cells in distribution
                ratio_cells.append(neuron)
                ratio_spike_twitch_total_twitches.append((nb_spikes / n_twitches) * 100)
                ratio_spike_twitch_total_spikes.append((nb_spikes / np.sum(spike_nums[neuron, :])) * 100)
                # print(f"ratio spikes after twitch / n_twitches: "
                #       f"{np.round(nb_spikes / n_twitches, 3)}")
                # print(f"ratio spikes after twitch / n_spikes: "
                #       f"{np.round(nb_spikes / np.sum(spike_nums[neuron, :]), 3)}")

            # adding the cell only if it has at least a spike around peak times
            if adding_this_neuron:
                bin_distrib = np.copy(distribution_array)
                # joining 2 frames
                bin_size = 2
                if len(bin_distrib) % 2 != 0:
                    bin_distrib = bin_distrib[:-1]
                # elif len(bin_distrib) % 3 == 0:
                #     bin_size = 3

                bin_distrib = np.reshape(bin_distrib, (bin_size, len(bin_distrib) // bin_size))
                bin_distrib = np.sum(bin_distrib, axis=0)
                bin_distrib = np.reshape(bin_distrib, (bin_size, len(bin_distrib) // bin_size))
                bin_distrib = np.sum(bin_distrib, axis=0)
                correlation_value = np.max(bin_distrib) / total_spikes
                # array_to_average = np.zeros(np.sum(distribution_array))
                # start = 0
                # for index, time_lag in enumerate(np.arange(-time_window * 2, time_window * 2 + 1)):
                #     n_spike_for_this_time_lag = distribution_array[index]
                #     array_to_average[start:(start+n_spike_for_this_time_lag)] = time_lag
                #     start += n_spike_for_this_time_lag
                # avg_time_lag = np.mean(array_to_average)
                # other way:
                time_lags_range = np.arange(-time_around_events, time_around_events + 1)
                distribution_array = distribution_array * time_lags_range
                avg_time_lag = np.sum(distribution_array) / total_spikes
                time_lags_dict[neuron] = avg_time_lag
                correlation_dict[neuron] = correlation_value

        for cell, time_lag in time_lags_dict.items():
            time_lags_list.append(time_lag)
            correlation_list.append(correlation_dict[cell])
            cells_list.append(cell)

        ratio_spike_twitch_total_twitches = np.array(ratio_spike_twitch_total_twitches)
        ratio_spike_twitch_total_spikes = np.array(ratio_spike_twitch_total_spikes)
        ratio_cells = np.array(ratio_cells)

        # plot both ratio on the same plot
        plot_scatters(ratio_spike_twitch_total_twitches, ratio_spike_twitch_total_spikes, size_scatter=30,
                      ax_to_use=ax_to_use_for_scatter, color_to_use=color_to_use_for_scatter,
                      legend_str=self.description,
                      xlabel="spikes in twitches vs total twitches (%)",
                      ylabel="spikes in twitches vs total spikes (%)",
                      filename_option="total_twitches_vs_total_spikes",
                      save_formats="pdf")

        if self.spike_struct.inter_neurons is not None:
            inter_neurons = np.array(self.spike_struct.inter_neurons)
            # keeping only interneurons that are among the selected cells
            inter_neurons = np.intersect1d(ratio_cells, inter_neurons)
        else:
            inter_neurons = []
        values_to_scatter = []
        non_inter_neurons = np.setdiff1d(ratio_cells, inter_neurons)
        ratio_interneurons = []
        for inter_neuron in inter_neurons:
            index = np.where(ratio_cells == inter_neuron)[0]
            ratio_interneurons.append(ratio_spike_twitch_total_spikes[index])
        ratio_non_interneurons = []
        for non_inter_neuron in non_inter_neurons:
            index = np.where(ratio_cells == non_inter_neuron)[0]
            ratio_non_interneurons.append(ratio_spike_twitch_total_spikes[index])
        labels = []
        scatter_shapes = []
        colors = []
        # values_to_scatter.append(np.mean(ratio_spike_twitch_total_spikes))
        # labels.extend(["mean"])
        # scatter_shapes.extend(["o"])
        # colors.extend(["white"])
        if len(ratio_non_interneurons) > 0:
            values_to_scatter.append(np.mean(ratio_non_interneurons))
            labels.extend(["mean"])
            scatter_shapes.extend(["o"])
            colors.extend(["white"])
        # if len(ratio_interneurons) > 0:
        #     values_to_scatter.append(np.mean(ratio_interneurons))
        #     values_to_scatter.extend(ratio_interneurons)
        #     labels.extend(["mean", f"interneuron (x{len(inter_neurons)})"])
        #     scatter_shapes.extend(["o"])
        #     scatter_shapes.extend(["*"] * len(inter_neurons))
        #     colors.extend(["red"])
        #     colors.extend(["red"] * len(inter_neurons))

        plot_hist_distribution(distribution_data=ratio_spike_twitch_total_spikes,
                               description=f"{self.description}",
                               values_to_scatter=np.array(values_to_scatter),
                               labels=labels,
                               scatter_shapes=scatter_shapes,
                               colors=colors,
                               twice_more_bins=True,
                               xlabel=f"spikes in {event_str} vs total spikes (%)",
                               param=self.param,
                               ax_to_use=ax_to_use_total_spikes,
                               color_to_use=color_to_use_total_spikes)

        # keeping only interneurons that are among the selected cells
        inter_neurons = np.intersect1d(ratio_cells, inter_neurons)
        values_to_scatter = []
        non_inter_neurons = np.setdiff1d(ratio_cells, inter_neurons)
        ratio_interneurons = []
        for inter_neuron in inter_neurons:
            index = np.where(ratio_cells == inter_neuron)[0]
            ratio_interneurons.append(ratio_spike_twitch_total_twitches[index])
        ratio_non_interneurons = []
        for non_inter_neuron in non_inter_neurons:
            index = np.where(ratio_cells == non_inter_neuron)[0]
            ratio_non_interneurons.append(ratio_spike_twitch_total_twitches[index])
        labels = []
        scatter_shapes = []
        colors = []
        # values_to_scatter.append(np.mean(ratio_spike_twitch_total_spikes))
        # labels.extend(["mean"])
        # scatter_shapes.extend(["o"])
        # colors.extend(["white"])
        if len(ratio_non_interneurons) > 0:
            values_to_scatter.append(np.mean(ratio_non_interneurons))
            labels.extend(["mean"])
            scatter_shapes.extend(["o"])
            colors.extend(["white"])
        # if len(ratio_interneurons) > 0:
        #     values_to_scatter.append(np.mean(ratio_interneurons))
        #     values_to_scatter.extend(ratio_interneurons)
        #     labels.extend(["mean", f"interneuron (x{len(inter_neurons)})"])
        #     scatter_shapes.extend(["o"])
        #     scatter_shapes.extend(["*"] * len(inter_neurons))
        #     colors.extend(["red"])
        #     colors.extend(["red"] * len(inter_neurons))

        plot_hist_distribution(distribution_data=ratio_spike_twitch_total_twitches,
                               description=f"{self.description}",
                               values_to_scatter=np.array(values_to_scatter),
                               labels=labels,
                               scatter_shapes=scatter_shapes,
                               colors=colors,
                               twice_more_bins=True,
                               xlabel=f"spikes in {event_str} vs total {event_str} (%)",
                               param=self.param,
                               ax_to_use=ax_to_use_total_events,
                               color_to_use=color_to_use_total_events)
        #

        cells_groups = []
        groups_colors = []

        # if (self.spike_struct.inter_neurons is not None) and (len(self.spike_struct.inter_neurons) > 0):
        #     cells_groups.append(self.spike_struct.inter_neurons)
        #     groups_colors.append("red")

        if self.cell_assemblies is not None:
            n_assemblies = len(self.cell_assemblies)
            cells_groups = self.cell_assemblies
            for i in np.arange(n_assemblies):
                groups_colors.append(cm.nipy_spectral(float(i + 1) / (n_assemblies + 1)))

        time_window = time_around_events // 2

        show_percentiles = None
        # show_percentiles = [99]
        # first plotting each individual time-correlation graph with the same x-limits
        time_correlation_graph(time_lags_list=time_lags_list,
                               correlation_list=correlation_list,
                               time_lags_dict=time_lags_dict,
                               correlation_dict=correlation_dict,
                               n_cells=self.spike_struct.n_cells,
                               time_window=time_window,
                               plot_cell_numbers=True,
                               value_to_text_in_cell=nb_spikes_by_cell,
                               cells_groups=cells_groups,
                               groups_colors=groups_colors,
                               data_id=self.description + f"_{event_str}_{n_twitches}",
                               param=self.param,
                               set_x_limit_to_max=True,
                               time_stamps_by_ms=0.01,
                               ms_scale=200,
                               size_cells_in_groups=150,
                               show_percentiles=show_percentiles,
                               ax_to_use=ax_to_use, color_to_use=color_to_use)

        # now we take the cells that are the most correlated to twitches

    def get_n_frames_from_movie_file_without_loading_it(self):
        if self.tif_movie_file_name is not None:
            start_time = time.time()
            im = PIL.Image.open(self.tif_movie_file_name)
            n_frames = len(list(ImageSequence.Iterator(im)))
            stop_time = time.time()
            print(f"Time to get movie n_frames: "
                  f"{np.round(stop_time - start_time, 3)} s")
            return n_frames
        print("No movie file_name, n_frames couldn't be obtained")
        return None

    def load_tiff_movie_in_memory(self):
        if self.tif_movie_file_name is not None:
            if self.tiff_movie is None:
                print(f"Loading movie for {self.description}")
                # using_scan_image_tiff = False
                # if using_scan_image_tiff:
                try:
                    start_time = time.time()
                    tiff_movie = ScanImageTiffReader(self.tif_movie_file_name).data()
                    stop_time = time.time()
                    print(f"Time for loading movie with scan_image_tiff: "
                          f"{np.round(stop_time - start_time, 3)} s")
                except Exception as e:
                    start_time = time.time()
                    im = PIL.Image.open(self.tif_movie_file_name)
                    n_frames = len(list(ImageSequence.Iterator(im)))
                    dim_x, dim_y = np.array(im).shape
                    print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")
                    tiff_movie = np.zeros((n_frames, dim_x, dim_y), dtype="uint16")
                    for frame, page in enumerate(ImageSequence.Iterator(im)):
                        tiff_movie[frame] = np.array(page)
                    stop_time = time.time()
                    print(f"Time for loading movie: "
                          f"{np.round(stop_time - start_time, 3)} s")

                self.tiff_movie = tiff_movie

    def load_tiff_movie_mean_and_std_(self, path):
        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(self.param.path_data, path)):
            file_names.extend(local_filenames)
            break
        if len(file_names) == 0:
            return

        for file_name in file_names:
            file_name_original = file_name
            file_name = file_name.lower()

            if file_name.endswith(".npy") and ("mean" in file_name.lower()):
                # valid only for shifts data so far
                self.tiff_movie_mean = np.load(os.path.join(self.param.path_data, path, file_name))
            elif file_name.endswith(".npy") and ("std" in file_name.lower()):
                # valid only for shifts data so far
                self.tiff_movie_std = np.load(os.path.join(self.param.path_data, path, file_name))

    def load_movie_dimensions(self):
        """
        Will load the dimension of the movie in self.movie_len_x and self.movie_len_y
        :return:
        """
        if (self.movie_len_x is not None) and (self.movie_len_y is not None):
            return

        if self.tiff_movie is None:
            if self.tif_movie_file_name is None:
                return
            im = PIL.Image.open(self.tif_movie_file_name)
            im_array = np.array(im)
            self.movie_len_x = im_array.shape[1]
            self.movie_len_y = im_array.shape[0]
        else:
            self.movie_len_x = self.tiff_movie[0].shape[1]
            self.movie_len_y = self.tiff_movie[0].shape[0]
        print(f"{self.description}: self.movie_len_x {self.movie_len_x}, self.movie_len_y {self.movie_len_y}")

    def produce_roi_shift_animation_with_cell_assemblies(self):
        if (self.global_roi is None) or (self.x_shifts is None):
            return
        if (self.cell_assemblies is None) or (self.coord_obj is None):
            return
        n_assemblies = len(self.cell_assemblies)
        cells_groups_colors = []
        for i in np.arange(n_assemblies):
            # print(f"cm.nipy_spectral(float(i + 1) / (n_assemblies + 1)) "
            #       f"{cm.nipy_spectral(float(i + 1) / (n_assemblies + 1))}")
            cells_groups_colors.append(cm.nipy_spectral(float(i + 1) / (n_assemblies + 1)))
        # print(f"cells_groups_colors {cells_groups_colors}")
        # self.coord_obj.compute_center_coord(cells_groups=self.cell_assemblies,
        #                                     cells_groups_colors=cells_groups_colors,
        #                                     dont_fill_cells_not_in_groups=True)

        animation = hne_anim.HNEAnimation(n_frames=12500, n_rows=3, n_cols=1)
        raw_movie_box = hne_anim.RawMovieBox(tiff_file_name=self.tif_movie_file_name, zoom_factor=2,
                                             cells_groups_to_color=self.cell_assemblies,
                                             colors_for_cells_groups=cells_groups_colors,
                                             coord_obj=self.coord_obj, cells_groups_alpha=None,
                                             raster_dur=self.spike_struct.spike_nums_dur
                                             )
        animation.add_box(row=0, col=0, box=raw_movie_box)

        # raw_movie_box.width
        roi = self.global_roi
        # roi = (roi - np.mean(roi)) / np.std(roi)
        # roi += np.abs(np.min(roi))

        shifts = np.sqrt(self.x_shifts ** 2 + self.y_shifts ** 2)
        # normalization
        # shifts = (shifts - np.mean(shifts)) / np.std(shifts)
        # shifts += np.abs(np.min(shifts))
        roi_box = hne_anim.PlotBox(width=raw_movie_box.width, height=80,  # raw_movie_box.width
                                   values_array=roi,
                                   n_frames_to_display=100)
        animation.add_box(row=1, col=0, box=roi_box)
        shift_box = hne_anim.PlotBox(width=raw_movie_box.width, height=80,
                                     values_array=shifts, color_past_and_present="cornflowerblue",
                                     color_future="white",
                                     n_frames_to_display=100)
        animation.add_box(row=2, col=0, box=shift_box)
        animation.produce_animation(path_results=self.param.path_results,
                                    file_name=f"test_raw_movie_{self.description}",
                                    save_formats=["tiff"],  # , "avi"
                                    frames_to_display=np.arange(4000, 4600))
        # p10 2000 to 4000

    def produce_roi_shift_animation(self):
        if (self.global_roi is None) or (self.x_shifts is None):
            return
        animation = hne_anim.HNEAnimation(n_frames=12500, n_rows=2, n_cols=1)
        # raw_movie_box = hne_anim.RawMovieBox(tiff_file_name=self.tif_movie_file_name, zoom_factor=1)
        # animation.add_box(row=0, col=0, box=raw_movie_box)

        # raw_movie_box.width
        roi = self.global_roi
        # roi = (roi - np.mean(roi)) / np.std(roi)
        # roi += np.abs(np.min(roi))

        shifts = np.sqrt(self.x_shifts ** 2 + self.y_shifts ** 2)
        # normalization
        # shifts = (shifts - np.mean(shifts)) / np.std(shifts)
        # shifts += np.abs(np.min(shifts))
        roi_box = hne_anim.PlotBox(width=178, height=80,  # raw_movie_box.width
                                   values_array=roi,
                                   n_frames_to_display=800)
        animation.add_box(row=0, col=0, box=roi_box)
        shift_box = hne_anim.PlotBox(width=178, height=80,
                                     values_array=shifts, color_past_and_present="cornflowerblue",
                                     color_future="white",
                                     n_frames_to_display=800)
        animation.add_box(row=1, col=0, box=shift_box)
        animation.produce_animation(path_results=self.param.path_results,
                                    file_name=f"test_raw_movie_{self.description}",
                                    save_formats=["tiff"],  # , "avi"
                                    frames_to_display=np.arange(200, 1600))
        # P5: width 178, range: 501 to 2001
        # P9: 0 3100
        # p9_19_03_22_a001: 2000 3400
        # p5 oriens: 2600 4000
        # p10 oriens: 8000 - 10500
        # P12_19_02_08_a000 (oriens): 200 1600
        # P12_17_11_10_a000: 3600 - 4800

    def produce_animation(self):
        # self.load_tiff_movie_in_memory()
        animation = hne_anim.HNEAnimation(n_frames=12500, n_rows=2, n_cols=1)
        raw_movie_box = hne_anim.RawMovieBox(tiff_file_name=self.tif_movie_file_name)
        animation.add_box(row=0, col=0, box=raw_movie_box)
        # raw_movie_box.width
        sum_spikes = tools_misc.get_activity_sum(raster=self.spike_struct.spike_nums_dur,
                                                 get_sum_spikes_as_percentage=True)
        activity_box = hne_anim.PlotBox(width=raw_movie_box.width, height=80,
                                        values_array=sum_spikes,
                                        n_frames_to_display=100)
        animation.add_box(row=1, col=0, box=activity_box)
        animation.produce_animation(path_results=self.param.path_results,
                                    file_name=f"test_raw_movie_{self.description}",
                                    save_formats=["tiff"],  # , "avi"
                                    frames_to_display=np.arange(12400, 12499))

    def build_raw_traces_from_movie(self):
        if self.tiff_movie is None:
            return
        print(f"{self.description} build_raw_traces_from_movie")
        raw_traces = np.zeros((self.coord_obj.n_cells, self.tiff_movie.shape[0]))
        for cell in np.arange(self.coord_obj.n_cells):
            mask = self.coord_obj.get_cell_mask(cell=cell,
                                                dimensions=(self.tiff_movie.shape[1], self.tiff_movie.shape[2]))
            raw_traces[cell, :] = np.mean(self.tiff_movie[:, mask], axis=1)
        return raw_traces

    def plot_time_correlation_graph_over_events(self, event_str, time_around_events=10, ax_to_use=None,
                                                color_to_use=None):
        if (self.shift_data_dict is None) or (self.spike_struct.spike_nums is None):
            return
        twitches_frames_periods = get_continous_time_periods(self.shift_data_dict[event_str].astype("int8"))
        results = get_time_correlation_data(spike_nums=self.spike_struct.spike_nums,
                                            events_times=twitches_frames_periods, time_around_events=time_around_events)
        self.time_lags_list, self.correlation_list, \
        self.time_lags_dict, self.correlation_dict, self.time_lags_window, cells_list = results

        cells_groups = []
        groups_colors = []

        # if (self.spike_struct.inter_neurons is not None) and (len(self.spike_struct.inter_neurons) > 0):
        #     cells_groups.append(self.spike_struct.inter_neurons)
        #     groups_colors.append("red")

        if self.cell_assemblies is not None:
            n_assemblies = len(self.cell_assemblies)
            cells_groups = self.cell_assemblies
            for i in np.arange(n_assemblies):
                groups_colors.append(cm.nipy_spectral(float(i + 1) / (n_assemblies + 1)))

        time_window = self.time_lags_window

        show_percentiles = None
        # show_percentiles = [99]
        # first plotting each individual time-correlation graph with the same x-limits
        time_correlation_graph(time_lags_list=self.time_lags_list,
                               correlation_list=self.correlation_list,
                               time_lags_dict=self.time_lags_dict,
                               correlation_dict=self.correlation_dict,
                               n_cells=self.spike_struct.n_cells,
                               time_window=time_window,
                               plot_cell_numbers=False,
                               cells_groups=cells_groups,
                               groups_colors=groups_colors,
                               data_id=self.description + "_" + event_str,
                               param=self.param,
                               set_x_limit_to_max=True,
                               time_stamps_by_ms=0.01,
                               ms_scale=200,
                               size_cells_in_groups=150,
                               show_percentiles=show_percentiles,
                               ax_to_use=ax_to_use,
                               color_to_use=color_to_use)

    def define_twitches_events(self):
        if self.twitches_frames_periods is None:
            return

        spike_nums_dur = self.spike_struct.spike_nums_dur
        # spike_nums = self.spike_struct.spike_nums
        spike_nums_to_use = spike_nums_dur
        n_cells = len(spike_nums_dur)
        n_times = len(spike_nums_dur[0, :])

        twitches_times, twitches_periods = self.get_twitches_times_by_group(sce_bool=self.sce_bool,
                                                                            sce_periods=self.SCE_times,
                                                                            twitches_group=7)
        self.events_by_twitches_group[7] = twitches_periods

        periods = self.get_events_independent_of_mvt()
        self.events_by_twitches_group[8] = periods

        periods = self.get_events_independent_of_mvt(reverse_it=True)
        self.events_by_twitches_group[9] = periods

        # 5: regroup sce before and during events
        twitches_group = 1
        self.events_by_twitches_group[1] = []
        self.events_by_twitches_group[5] = []
        self.events_by_twitches_group[6] = []

        twitches_times, twitches_periods = self.get_twitches_times_by_group(sce_bool=self.sce_bool,
                                                                            twitches_group=twitches_group)

        for twitch_index, twitch_period in enumerate(twitches_periods):
            sce_index = self.sce_times_numbers[twitch_period[0]:twitch_period[1] + 1]
            # taking the first sce following the
            sce_index = sce_index[np.where(sce_index >= 0)[0][0]]
            self.events_by_twitches_group[1].append(self.SCE_times[sce_index])
            self.events_by_twitches_group[5].append(self.SCE_times[sce_index])
            self.events_by_twitches_group[6].append(self.SCE_times[sce_index])
        # n_twitches = len(twitches_times)

        # then group 3, twitches before sce_event, taking the sce after as event
        twitches_group = 3
        self.events_by_twitches_group[3] = []
        twitches_times, twitches_periods = self.get_twitches_times_by_group(sce_bool=self.sce_bool,
                                                                            twitches_group=twitches_group)

        for twitch_index, twitch_period in enumerate(twitches_periods):
            # print(f"twitch_period {twitch_period}")
            end_time = np.min((twitch_period[1] + 1 + 10, len(self.sce_times_numbers)))
            sce_index = self.sce_times_numbers[twitch_period[1]:end_time]
            # print(f"sce_index {sce_index}")
            # taking the first sce following the
            sce_index = sce_index[np.where(sce_index >= 0)[0][0]]
            self.events_by_twitches_group[3].append(self.SCE_times[sce_index])
            if self.SCE_times[sce_index] in self.events_by_twitches_group[5]:
                print(f"self.SCE_times[sce_index] {self.SCE_times[sce_index]}, "
                      f"self.events_by_twitches_group[5] {self.events_by_twitches_group[5]}")
                print("something wrong self.SCE_times[sce_index] in self.events_by_twitches_group[5]")
            self.events_by_twitches_group[5].append(self.SCE_times[sce_index])
            self.events_by_twitches_group[6].append(self.SCE_times[sce_index])

        twitches_group = 4
        self.events_by_twitches_group[4] = []
        twitches_times, twitches_periods = self.get_twitches_times_by_group(sce_bool=self.sce_bool,
                                                                            twitches_group=twitches_group)
        twitch_mean_sum_cells = []
        for twitch_period in twitches_periods:
            twitch_mean_sum_cells.append(np.mean(np.sum(spike_nums_dur[:, twitch_period[0]:twitch_period[1] + 1],
                                                        axis=0)))
        for twitch_index, twitch_period in enumerate(twitches_periods):

            # prendre les cellules dont la somme est supérieur à la moyenne au moment du Twitch.
            # continuer jusqu'à ce que ça descende sous cette moyenne ou 2 sec max
            beg_index = None
            end_index = None
            len_max = 20
            for time_index in np.arange(twitch_period[1] + 1, np.min((n_times, twitch_period[1] + 1 + len_max))):
                if np.sum(spike_nums_dur[:, time_index]) >= twitch_mean_sum_cells[twitch_index]:
                    if beg_index is None:
                        beg_index = time_index
                else:
                    if beg_index is not None:
                        # end_index include the event detected
                        end_index = time_index - 1
                        break
            if beg_index is None:
                continue
            if end_index is None:
                # mean the for loop went to the end
                end_index = np.min((n_times, twitch_period[1] + 1 + len_max)) - 1
            self.events_by_twitches_group[4].append((beg_index, end_index))
            self.events_by_twitches_group[6].append((beg_index, end_index))

        # spike_sum_of_sum_at_time_dict, spikes_sums_at_time_dict, \
        # spikes_at_time_dict = self.get_spikes_by_time_around_a_time(twitches_times, spike_nums_to_use, 25)

    def plot_source_profile_with_transients_profiles(self, profiles_dict, path_results):

        def get_new_figure(n_columns, n_lines):
            fig = plt.figure(figsize=(8, 14),
                             subplotpars=SubplotParams(hspace=0, wspace=0),
                             dpi=500)
            fig_patch = fig.patch
            fig_patch.set_facecolor("white")
            # plt.subplots_adjust(left=0.01, right=0.01, top=0.01, bottom=0.01)
            # now adding as many suplots as need, depending on how many overlap has the cell
            # n_columns = 10
            width_ratios = [100 // n_columns] * n_columns
            # n_lines = (((n_source_profile_max - 1) // n_columns) + 1) * 2
            height_ratios = [100 // n_lines] * n_lines
            grid_spec = gridspec.GridSpec(n_lines, n_columns, width_ratios=width_ratios,
                                          height_ratios=height_ratios,
                                          figure=fig, wspace=0.1, hspace=0.1)
            ax_grid = dict()
            for line_gs in np.arange(n_lines):
                for col_gs in np.arange(n_columns):
                    ax_grid[(line_gs, col_gs)] = \
                        fig.add_subplot(grid_spec[line_gs, col_gs])
                    ax = ax_grid[(line_gs, col_gs)]
                    # ax_source_profile_by_cell[cell_to_display].set_facecolor("black")
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.get_yaxis().set_visible(False)
                    ax.get_xaxis().set_visible(False)
                    ax.set_frame_on(False)
            return ax_grid, fig

        def save_figure(path_results, fig, index):
            save_formats = ["png", "pdf"]
            if not os.path.isdir(path_results):
                os.mkdir(path_results)

            if isinstance(save_formats, str):
                save_formats = [save_formats]
            for save_format in save_formats:
                fig.savefig(f'{path_results}/'
                            f'{self.description}_source_profiles_classifier_vs_caiman_{index}'
                            f'.{save_format}',
                            format=f"{save_format}",
                            facecolor=sources_profile_fig.get_facecolor(), edgecolor='none', bbox_inches='tight')

        print(f'profiles_dict.keys() {list(profiles_dict.keys())}')
        n_lines = 20
        n_columns = 10
        lw_contour = 1
        ax_grid, sources_profile_fig = get_new_figure(n_columns=n_columns, n_lines=n_lines)
        # n_figure plotted
        index_fig = 0
        source_profile_fig_index = 0

        for main_cell, profiles_list in profiles_dict.items():
            print(f"main_cell {main_cell}: len(profiles_list) {len(profiles_list)}")
            if source_profile_fig_index == n_lines:
                # we need to save the figure and create a new one
                save_figure(path_results=path_results, fig=sources_profile_fig, index=index_fig)
                index_fig += 1
                # we need to create a new figure
                ax_grid, sources_profile_fig = get_new_figure(n_columns=n_columns, n_lines=n_lines)
                source_profile_fig_index = 0

            # intersect_cells used to display cells contours in the source profile
            intersect_cells = self.coord_obj.intersect_cells[main_cell]
            cells_color = dict()
            cells_color[main_cell] = "red"
            cells_to_display = [main_cell]
            for index, cell_inter in enumerate(intersect_cells):
                cells_color[cell_inter] = cm.nipy_spectral(float(index + 1) / (len(intersect_cells) + 1))

            cells_to_display.extend(intersect_cells)

            # calculating the bound that will surround all the cells
            minx = None
            maxx = None
            miny = None
            maxy = None
            corr_by_cell = dict()
            for cell_to_display in cells_to_display:
                poly_gon = self.coord_obj.cells_polygon[cell_to_display]

                if minx is None:
                    minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
                else:
                    tmp_minx, tmp_miny, tmp_maxx, tmp_maxy = np.array(list(poly_gon.bounds)).astype(int)
                    minx = min(minx, tmp_minx)
                    miny = min(miny, tmp_miny)
                    maxx = max(maxx, tmp_maxx)
                    maxy = max(maxy, tmp_maxy)
            bounds = (minx, miny, maxx, maxy)

            column_to_aim = 0
            source_profile, minx, miny, mask_source_profile = self.coord_obj.get_source_profile(cell=main_cell,
                                                                                                tiff_movie=self.tiff_movie,
                                                                                                traces=self.raw_traces,
                                                                                                peak_nums=self.spike_struct.peak_nums,
                                                                                                spike_nums=self.spike_struct.spike_nums,
                                                                                                pixels_around=3,
                                                                                                bounds=bounds)
            xy_source = self.coord_obj.get_cell_new_coord_in_source(cell=main_cell, minx=minx, miny=miny)
            ax = ax_grid[(source_profile_fig_index, column_to_aim)]
            img_src_profile = ax.imshow(source_profile, cmap=plt.get_cmap('gray'))
            # xy_source = self.coord_obj.get_cell_new_coord_in_source(cell=main_cell, minx=minx, miny=miny)
            contour_cell = patches.Polygon(xy=xy_source,
                                           fill=False,
                                           edgecolor=cells_color[main_cell],
                                           zorder=15, lw=lw_contour)
            ax.add_patch(contour_cell)
            column_to_aim += 1

            if len(profiles_list) == 0:
                source_profile_fig_index += 1
                continue

            for profile_info in profiles_list:
                transient = profile_info[0]
                classifier_error = profile_info[1]
                caiman_error = profile_info[2]
                if column_to_aim == n_columns:
                    # we need to go to next line
                    source_profile_fig_index += 1
                    if source_profile_fig_index == n_lines:
                        # we need to save the figure and create a new one
                        save_figure(path_results=path_results, fig=sources_profile_fig, index=index_fig)
                        index_fig += 1
                        # we need to create a new figure
                        ax_grid, sources_profile_fig = get_new_figure(n_columns=n_columns, n_lines=n_lines)
                        source_profile_fig_index = 0
                    column_to_aim = 1
                ax = ax_grid[(source_profile_fig_index, column_to_aim)]
                transient_profile, minx, miny = self.coord_obj.get_transient_profile(cell=cell_to_display,
                                                                                     tiff_movie=self.tiff_movie,
                                                                                     traces=self.raw_traces,
                                                                                     transient=transient,
                                                                                     pixels_around=3, bounds=bounds)
                img_t_profile = ax.imshow(transient_profile, cmap=plt.get_cmap('gray'))

                for cell in cells_to_display:
                    xy_source = self.coord_obj.get_cell_new_coord_in_source(cell=cell, minx=minx, miny=miny)
                    cell_color = cells_color[cell]
                    contour_cell = patches.Polygon(xy=xy_source,
                                                   fill=False,
                                                   edgecolor=cell_color,
                                                   zorder=15, lw=lw_contour)
                    ax.add_patch(contour_cell)

                ax.set_frame_on(True)
                # 3 range of color
                if classifier_error:
                    frame_color = "red"
                else:
                    frame_color = "green"

                for spine in ax.spines.values():
                    spine.set_edgecolor(frame_color)
                    spine.set_linewidth(3)
                    if caiman_error:
                        spine.set_linestyle("dashed")
                    else:
                        spine.set_linestyle("solid")
                column_to_aim += 1
            source_profile_fig_index += 1
        # we need to save the figure and create a new one
        save_figure(path_results=path_results, fig=sources_profile_fig, index=index_fig)

    def evaluate_overlaps_accuracy(self, path_data, path_results, with_figures=True):
        """
        Based on transient and source profiles, evaluate which transients are due to overlaps and give the number
        of transients that were actually not detected as overlap
        Returns:

        """

        print('Starting evaluate_overlaps_accuracy')
        # just to test how many cells have predictions
        cells_predicted = 0
        print(f"len(self.spike_struct.spike_nums_dur) {len(self.spike_struct.spike_nums_dur)}")
        for spikes in self.spike_struct.spike_nums_dur:
            if np.sum(spikes) > 0:
                cells_predicted += 1
        print(f"cells_predicted {cells_predicted}")

        # n_errors / n_transients_due_to_overlaps will give the percentage of erros concerning overlaps
        n_transients_due_to_overlaps = 0
        n_errors = 0
        n_errors_caiman = 0
        # we can add another raster_dur
        n_errors_other = 0
        # count the number of cells with overlaps
        n_cells_with_overlaps = 0
        # if True, we use onsets/spikes and not spike_nums_dur
        using_caiman_spikes = True
        pixels_around = 1
        print(f"using_caiman_spikes {using_caiman_spikes}")
        print(f"sum self.caiman_spike_nums {np.sum(self.caiman_spike_nums)}")

        other_raster_dur = None

        add_other = False
        if add_other:
            file_name = "p12/p12_17_11_10_a000/predictions/p12_17_11_10_a000_predictions_caiman_meso_v26_epoch_5.npz"
            predictions = np.load(os.path.join(self.param.path_data, file_name))["predictions"]
            # data = hdf5storage.loadmat(os.path.join(self.param.path_data, file_name))
            # predictions = data["predictions"]
            prediction_threshold = 0.5
            predicted_raster_dur_dict = np.zeros((len(predictions), len(predictions[0])), dtype="int8")
            for cell in np.arange(len(predictions)):
                pred = predictions[cell]
                # predicted_raster_dur_dict[cell, pred >= predictions_threshold] = 1
                if len(pred.shape) == 1:
                    predicted_raster_dur_dict[cell, pred >= prediction_threshold] = 1
                elif (len(pred.shape) == 2) and (pred.shape[1] == 1):
                    pred = pred[:, 0]
                    predicted_raster_dur_dict[cell, pred >= prediction_threshold] = 1
                elif (len(pred.shape) == 2) and (pred.shape[1] == 3):
                    # real transient, fake ones, other (neuropil, decay etc...)
                    # keeping predictions about real transient when superior
                    # to other prediction on the same frame
                    max_pred_by_frame = np.max(pred, axis=1)
                    real_transient_frames = (pred[:, 0] == max_pred_by_frame)
                    predicted_raster_dur_dict[cell, real_transient_frames] = 1
                elif pred.shape[1] == 2:
                    # real transient, fake ones
                    # keeping predictions about real transient superior to the threshold
                    # and superior to other prediction on the same frame
                    max_pred_by_frame = np.max(pred, axis=1)
                    real_transient_frames = np.logical_and((pred[:, 0] >= prediction_threshold),
                                                           (pred[:, 0] == max_pred_by_frame))
                    predicted_raster_dur_dict[cell, real_transient_frames] = 1
            other_raster_dur = predicted_raster_dur_dict
            # cre_meso_v1_caiman_epoch_15

        overlap_area_threshold = 0.15
        low_correlation_threshold = 0.2
        high_correlation_threshold = 0.70

        # first key is a tuple of 2 cells, and the value is a list of tuple of 2 int representing the transient
        # times
        errors_dict = dict()

        coord_obj = self.coord_obj
        n_cells = self.coord_obj.n_cells
        explored_pairs = []
        self.load_tiff_movie_in_memory()
        self.create_smooth_traces()

        traces = self.smooth_traces
        n_times = traces.shape[1]

        # ---- for figure -----
        # counter
        # will be used to print the cell profiles of cells with overlaps
        # and how the transient profile associated to each overlap
        profiles_dict = dict()

        # now we want to compute all potential peak and onsets
        # based on diff
        spike_nums_all = np.zeros((n_cells, n_times), dtype="int8")
        for cell in np.arange(n_cells):
            onsets = []
            diff_values = np.diff(traces[cell])
            for index, value in enumerate(diff_values):
                if index == (len(diff_values) - 1):
                    continue
                if value < 0:
                    if diff_values[index + 1] >= 0:
                        onsets.append(index + 1)
            if len(onsets) > 0:
                spike_nums_all[cell, np.array(onsets)] = 1

        peak_nums = np.zeros((n_cells, n_times), dtype="int8")
        for cell in np.arange(n_cells):
            peaks, properties = signal.find_peaks(x=traces[cell])
            peak_nums[cell, peaks] = 1

        spike_nums_dur_artificial = np.zeros((n_cells, n_times), dtype="int8")
        for cell in np.arange(n_cells):
            peaks_index = np.where(peak_nums[cell, :])[0]
            onsets_index = np.where(spike_nums_all[cell, :])[0]

            for onset_index in onsets_index:
                peaks_after = np.where(peaks_index > onset_index)[0]
                if len(peaks_after) == 0:
                    continue
                peaks_after = peaks_index[peaks_after]
                peak_after = peaks_after[0]

                spike_nums_dur_artificial[cell, onset_index:peak_after + 1] = 1

        # then we look for all pairs of overlapping cells
        for cell in np.arange(n_cells):
            if cell % 1 == 0:
                print(f"cell {cell} / {n_cells - 1}")

            # if the cell as no peaks or onsets, then we won't be able to compute the source profile
            if np.sum(self.spike_struct.spike_nums_dur[cell]) == 0:
                continue

            intersect_cells = coord_obj.intersect_cells[cell]
            # only looking for cells with potential overlaps
            if len(intersect_cells) == 0:
                continue
            intersect_cells_to_explore = []
            # then we want to keep intersecting cells with at least a minimum of interesections
            poly_1 = coord_obj.cells_polygon[cell]
            for intersect_cell in intersect_cells:
                if (cell, intersect_cell) in explored_pairs or (intersect_cell, cell) in explored_pairs:
                    continue
                explored_pairs.append((cell, intersect_cell))
                poly_2 = coord_obj.cells_polygon[intersect_cell]
                poly_inter = poly_1.intersection(poly_2)
                # keeping the cell if the intersecting area is superior to overlap_area_threshold % of the biggest area
                biggest_area = max(poly_1.area, poly_2.area)
                if poly_inter.area > (overlap_area_threshold * biggest_area):
                    intersect_cells_to_explore.append(intersect_cell)
            if len(intersect_cells_to_explore) == 0:
                continue
            else:
                print(f"N intersect cells of {cell}: {len(intersect_cells_to_explore)}")

            n_cells_with_overlaps += 1
            poly_gon = coord_obj.cells_polygon[cell]

            # Correlation test
            bounds_corr = np.array(list(poly_gon.bounds)).astype(int)
            source_profile_corr, minx_corr, \
            miny_corr, mask_source_profile = coord_obj.get_source_profile(cell=cell, tiff_movie=self.tiff_movie,
                                                                          traces=self.raw_traces,
                                                                          bounds=bounds_corr,
                                                                          peak_nums=self.spike_struct.peak_nums,
                                                                          spike_nums=self.spike_struct.spike_nums,
                                                                          pixels_around=pixels_around, buffer=1)
            # normalizing
            source_profile_corr_norm = source_profile_corr - np.mean(source_profile_corr)
            # we want the mask to be at ones over the cell
            mask_source_profile = (1 - mask_source_profile).astype(bool)

            # to save memory
            transient_profiles_dict = dict()

            for cell_2 in intersect_cells_to_explore:
                if np.sum(self.spike_struct.spike_nums_dur[cell_2]) == 0:
                    continue
                n_cells_with_overlaps += 1

                poly_gon_2 = coord_obj.cells_polygon[cell_2]

                # Correlation test
                bounds_corr_2 = np.array(list(poly_gon_2.bounds)).astype(int)

                source_profile_corr_2, minx_corr_2, \
                miny_corr_2, mask_source_profile_2 = coord_obj.get_source_profile(cell=cell_2,
                                                                                  tiff_movie=self.tiff_movie,
                                                                                  bounds=bounds_corr_2,
                                                                                  traces=self.raw_traces,
                                                                                  peak_nums=self.spike_struct.peak_nums,
                                                                                  spike_nums=self.spike_struct.spike_nums,
                                                                                  pixels_around=pixels_around, buffer=1)
                # normalizing
                source_profile_corr_2_norm = source_profile_corr_2 - np.mean(source_profile_corr_2)
                # we want the mask to be at ones over the cell
                mask_source_profile_2 = (1 - mask_source_profile_2).astype(bool)

                transients = get_continous_time_periods(spike_nums_dur_artificial[cell])
                # transients_2 = get_continous_time_periods(spike_nums_dur[cell_2])
                # we loop all transients and check for overlaps
                for transient in transients:

                    # now we compute the correlation for each transient profile with its source
                    # cell 1
                    transient_profile_corr, minx_corr_tp, miny_corr_tp = \
                        coord_obj.get_transient_profile(cell=cell, tiff_movie=self.tiff_movie,
                                                        traces=self.raw_traces,
                                                        transient=transient,
                                                        pixels_around=pixels_around,
                                                        bounds=bounds_corr)
                    transient_profile_corr_norm = transient_profile_corr - np.mean(transient_profile_corr)

                    pearson_corr, pearson_p_value = stats.pearsonr(source_profile_corr_norm[mask_source_profile],
                                                                   transient_profile_corr_norm[mask_source_profile])

                    # cell 2
                    transient_profile_corr_2, minx_corr_tp_2, miny_corr_tp_2 = \
                        coord_obj.get_transient_profile(cell=cell_2, tiff_movie=self.tiff_movie,
                                                        traces=self.raw_traces,
                                                        transient=transient,
                                                        pixels_around=pixels_around,
                                                        bounds=bounds_corr_2)
                    transient_profile_corr_2_norm = transient_profile_corr_2 - np.mean(transient_profile_corr_2)

                    pearson_corr_2, pearson_p_value = stats.pearsonr(source_profile_corr_2_norm[mask_source_profile_2],
                                                                     transient_profile_corr_2_norm[
                                                                         mask_source_profile_2])

                    # if we have corr values that are opposed, we can suppose that overlap is responsible for the
                    # transient augmentation
                    n_frames_in_transient = transient[1] - transient[0] + 1
                    threshold_n_transients = max(1, (n_frames_in_transient // 3))
                    if (pearson_corr < low_correlation_threshold) and (pearson_corr_2 > high_correlation_threshold):
                        classifier_error = False
                        caiman_error = False
                        # then it means the transient in cell is fake and due to overlap
                        # we check if the spike_nums_dur say it is indeed False
                        if np.sum(self.spike_struct.spike_nums_dur[cell, transient[0]:transient[1] + 1]) >= \
                                threshold_n_transients:
                            # then it's not right
                            print(f"## Correlations {cell} {cell_2}: {np.round(pearson_corr, 2)} & "
                                  f"{np.round(pearson_corr_2, 2)} {transient}")
                            n_errors += 1
                            classifier_error = True
                            if (cell, cell_2) not in errors_dict:
                                errors_dict[(cell, cell_2)] = []
                            errors_dict[(cell, cell_2)].append(transient)
                        if using_caiman_spikes:
                            if self.caiman_spike_nums is not None:
                                if np.sum(self.caiman_spike_nums[cell, transient[0]:transient[1] + 1]) > 0:
                                    print(f"## Correlations {cell} {cell_2}: {np.round(pearson_corr, 2)} & "
                                          f"{np.round(pearson_corr_2, 2)} {transient} (CaImAn)")
                                    n_errors_caiman += 1
                                    caiman_error = True
                        elif self.caiman_spike_nums_dur is not None:
                            if np.sum(self.caiman_spike_nums_dur[cell, transient[0]:transient[1] + 1]) > \
                                    threshold_n_transients:
                                print(f"## Correlations {cell} {cell_2}: {np.round(pearson_corr, 2)} & "
                                      f"{np.round(pearson_corr_2, 2)} {transient} (CaImAn)")
                                n_errors_caiman += 1
                                caiman_error = True
                        if other_raster_dur is not None:
                            if np.sum(other_raster_dur[cell, transient[0]:transient[1] + 1]) >= \
                                    threshold_n_transients:
                                # then it's not right
                                print(f"## Correlations {cell} {cell_2}: {np.round(pearson_corr, 2)} & "
                                      f"{np.round(pearson_corr_2, 2)} {transient} (other)")
                                n_errors_other += 1
                        n_transients_due_to_overlaps += 1
                        # to display sources and transients profiles
                        if cell not in profiles_dict:
                            profiles_dict[cell] = []
                        # then we add the new transient profile
                        # profiles_dict[cell].append([transient_profile_corr, minx_corr_tp, miny_corr_tp,
                        #                             classifier_error, caiman_error])
                        profiles_dict[cell].append([transient, classifier_error, caiman_error])
                    elif (pearson_corr > high_correlation_threshold) and (pearson_corr_2 < low_correlation_threshold):
                        classifier_error = False
                        caiman_error = False
                        # then it means the transient in cell_2 is fake and due to overlap
                        # we check if the spike_nums_dur say it is indeed False
                        if np.sum(self.spike_struct.spike_nums_dur[cell_2, transient[0]:transient[1] + 1]) > \
                                threshold_n_transients:
                            # then it's not right
                            print(f"## Correlations {cell} {cell_2}: {np.round(pearson_corr, 2)} & "
                                  f"{np.round(pearson_corr_2, 2)} {transient}")
                            n_errors += 1
                            classifier_error = True
                            if (cell, cell_2) not in errors_dict:
                                errors_dict[(cell, cell_2)] = []
                            errors_dict[(cell, cell_2)].append(transient)
                        if using_caiman_spikes:
                            if self.caiman_spike_nums is not None:
                                if np.sum(self.caiman_spike_nums[cell_2, transient[0]:transient[1] + 1]) > 0:
                                    print(f"## Correlations {cell} {cell_2}: {np.round(pearson_corr, 2)} & "
                                          f"{np.round(pearson_corr_2, 2)} {transient} (CaImAn)")
                                    n_errors_caiman += 1
                                    caiman_error = True
                        elif self.caiman_spike_nums_dur is not None:
                            if np.sum(self.caiman_spike_nums_dur[cell_2, transient[0]:transient[1] + 1]) > \
                                    threshold_n_transients:
                                print(f"## Correlations {cell} {cell_2}: {np.round(pearson_corr, 2)} & "
                                      f"{np.round(pearson_corr_2, 2)} {transient} (CaImAn)")
                                n_errors_caiman += 1
                                caiman_error = True

                        if other_raster_dur is not None:
                            if np.sum(other_raster_dur[cell_2, transient[0]:transient[1] + 1]) >= \
                                    threshold_n_transients:
                                # then it's not right
                                print(f"## Correlations {cell} {cell_2}: {np.round(pearson_corr, 2)} & "
                                      f"{np.round(pearson_corr_2, 2)} {transient} (other)")
                                n_errors_other += 1

                        n_transients_due_to_overlaps += 1
                        # to display sources and transients profiles
                        if cell_2 not in profiles_dict:
                            profiles_dict[cell_2] = []
                            # profiles_dict[cell_2].append([source_profile_corr_2, minx_corr_2, miny_corr_2])
                        # then we add the new transient profile
                        # profiles_dict[cell_2].append([transient_profile_corr_2, minx_corr_tp_2, miny_corr_tp_2,
                        #                             classifier_error, caiman_error])
                        profiles_dict[cell_2].append([transient, classifier_error, caiman_error])

                    else:
                        continue
            if n_transients_due_to_overlaps > 0:
                print(f"n_errors {n_errors}, n_transients_due_to_overlaps {n_transients_due_to_overlaps}, "
                      f"rate {np.round((n_errors / n_transients_due_to_overlaps) * 100, 3)} %")

                print(f"n_errors_caiman {n_errors_caiman}, "
                      f"n_transients_due_to_overlaps {n_transients_due_to_overlaps}, "
                      f"rate {np.round((n_errors_caiman / n_transients_due_to_overlaps) * 100, 3)} %")

                print(f"n_errors_other {n_errors_other}, "
                      f"n_transients_due_to_overlaps {n_transients_due_to_overlaps}, "
                      f"rate {np.round((n_errors_other / n_transients_due_to_overlaps) * 100, 3)} %")
            else:
                print("No transients_due_to_overlaps yet")

        print(f"n_transients_due_to_overlaps {n_transients_due_to_overlaps} from {len(profiles_dict)} cells")
        if n_transients_due_to_overlaps > 0:
            meso_false_ratio = (n_errors / n_transients_due_to_overlaps) * 100
            meso_positive_ratio = 100 - meso_false_ratio
            print(f"Correctly predicted overlaps meso: "
                  f"{np.round(meso_positive_ratio, 3)} %")
            print(f"Correctly predicted overlaps other: "
                  f"{np.round(100 - ((n_errors_other / n_transients_due_to_overlaps) * 100), 3)} %")
            print(f"Correctly predicted overlaps caiman: "
                  f"{np.round(100 - ((n_errors_caiman / n_transients_due_to_overlaps) * 100), 3)} %")

        else:
            print("No transients_due_to_overlaps yet")

        self.plot_source_profile_with_transients_profiles(profiles_dict, path_results=path_results)
        # with open(os.path.join(path_results, f'errors_overlaps_{self.description}.yaml'), 'w') as outfile:
        #     yaml.safe_dump(errors_dict, outfile, default_flow_style=False)

    def get_spikes_by_time_around_a_time(self, twitches_times, spike_nums_to_use, time_around):

        # key is an int which reprensent the sum of spikes at a certain distance (in frames) of the event,
        # negative or positive
        spike_sum_of_sum_at_time_dict = SortedDict()
        spikes_sums_at_time_dict = SortedDict()
        spikes_at_time_dict = SortedDict()

        n_times = len(spike_nums_to_use[0, :])
        n_cells = len(spike_nums_to_use)

        if len(twitches_times) == 0:
            return

        for twitch_id, twitch_time in enumerate(twitches_times):

            beg_time = np.max((0, twitch_time - time_around))
            end_time = np.min((n_times, twitch_time + time_around + 1))

            # before the event
            sum_spikes = np.sum(spike_nums_to_use[:, beg_time:twitch_time], axis=0)
            # print(f"before time_spikes {time_spikes}")
            time_spikes = np.arange(-(twitch_time - beg_time), 0)
            for i, time_spike in enumerate(time_spikes):
                spike_sum_of_sum_at_time_dict[time_spike] = spike_sum_of_sum_at_time_dict.get(time_spike, 0) + \
                                                            sum_spikes[i]
                if time_spike not in spikes_sums_at_time_dict:
                    spikes_sums_at_time_dict[time_spike] = []
                spikes_sums_at_time_dict[time_spike].append(sum_spikes[i])

                if time_spike not in spikes_at_time_dict:
                    spikes_at_time_dict[time_spike] = []
                spikes_at_time_dict[time_spike].append(np.where(spike_nums_to_use[:, beg_time + i])[0])

            # after the event
            sum_spikes = np.sum(spike_nums_to_use[:, twitch_time:end_time], axis=0)
            time_spikes = np.arange(0, end_time - twitch_time)
            for i, time_spike in enumerate(time_spikes):
                spike_sum_of_sum_at_time_dict[time_spike] = spike_sum_of_sum_at_time_dict.get(time_spike, 0) + \
                                                            sum_spikes[i]
                if time_spike not in spikes_sums_at_time_dict:
                    spikes_sums_at_time_dict[time_spike] = []
                spikes_sums_at_time_dict[time_spike].append(sum_spikes[i])

                if time_spike not in spikes_at_time_dict:
                    spikes_at_time_dict[time_spike] = []
                spikes_at_time_dict[time_spike].append(np.where(spike_nums_to_use[:, twitch_time + i])[0])
        return spike_sum_of_sum_at_time_dict, spikes_sums_at_time_dict, spikes_at_time_dict

    def get_spikes_duration_by_time_around_a_time(self, twitches_times, spike_nums_to_use, time_around):

        # key is an int which reprensent the sum of spikes's duration at a certain distance (in ms) of the event,
        # negative or positive
        spike_sum_of_sum_at_time_dict = SortedDict()
        # same as before, but not normalized
        spike_sum_of_sum_non_norm_at_time_dict = SortedDict()
        spikes_sums_at_time_dict = SortedDict()
        spikes_at_time_dict = SortedDict()

        n_times = len(spike_nums_to_use[0, :])
        n_cells = len(spike_nums_to_use)

        if len(twitches_times) == 0:
            return

        for twitch_id, twitch_time in enumerate(twitches_times):

            beg_time = np.max((0, twitch_time - time_around))
            end_time = np.min((n_times, twitch_time + time_around + 1))

            # before the event
            spike_nums_duration = np.copy(spike_nums_to_use).astype("int16")
            # we put at each time the duration of the transient instead of the ones
            for cell in np.arange(len(spike_nums_to_use)):
                periods = get_continous_time_periods(spike_nums_to_use[cell])
                for period in periods:
                    # in ms
                    duration = int(((period[1] - period[0] + 1) / self.sampling_rate) * 1000)
                    spike_nums_duration[cell, period[0]:period[1] + 1] = duration
            duration_spikes = np.sum(spike_nums_duration[:, beg_time:twitch_time], axis=0)
            sum_spikes = np.sum(spike_nums_to_use[:, beg_time:twitch_time], axis=0)

            # print(f"before time_spikes {time_spikes}")
            time_spikes = np.arange(-(twitch_time - beg_time), 0)
            for i, time_spike in enumerate(time_spikes):
                # normalizaing by the number of cells
                if sum_spikes[i] > 0:
                    spike_sum_of_sum_at_time_dict[time_spike] = spike_sum_of_sum_at_time_dict.get(time_spike, 0) + \
                                                                (duration_spikes[i] / sum_spikes[i])
                else:
                    spike_sum_of_sum_at_time_dict[time_spike] = spike_sum_of_sum_at_time_dict.get(time_spike, 0) + \
                                                                duration_spikes[i]
                spike_sum_of_sum_non_norm_at_time_dict[time_spike] = \
                    spike_sum_of_sum_non_norm_at_time_dict.get(time_spike, 0) + \
                    duration_spikes[i]
                if time_spike not in spikes_sums_at_time_dict:
                    spikes_sums_at_time_dict[time_spike] = []
                if sum_spikes[i] > 0:
                    spikes_sums_at_time_dict[time_spike].append((duration_spikes[i] / sum_spikes[i]))
                else:
                    spikes_sums_at_time_dict[time_spike].append(0)

                if time_spike not in spikes_at_time_dict:
                    spikes_at_time_dict[time_spike] = []
                spikes_at_time_dict[time_spike].append(np.where(spike_nums_to_use[:, beg_time + i])[0])

            # after the event
            duration_spikes = np.sum(spike_nums_duration[:, twitch_time:end_time], axis=0)
            sum_spikes = np.sum(spike_nums_to_use[:, twitch_time:end_time], axis=0)

            time_spikes = np.arange(0, end_time - twitch_time)
            for i, time_spike in enumerate(time_spikes):
                if sum_spikes[i] > 0:
                    spike_sum_of_sum_at_time_dict[time_spike] = spike_sum_of_sum_at_time_dict.get(time_spike, 0) + \
                                                                (duration_spikes[i] / sum_spikes[i])
                else:
                    spike_sum_of_sum_at_time_dict[time_spike] = spike_sum_of_sum_at_time_dict.get(time_spike, 0) + \
                                                                duration_spikes[i]

                spike_sum_of_sum_non_norm_at_time_dict[time_spike] = \
                    spike_sum_of_sum_non_norm_at_time_dict.get(time_spike, 0) + \
                    duration_spikes[i]

                if time_spike not in spikes_sums_at_time_dict:
                    spikes_sums_at_time_dict[time_spike] = []
                if sum_spikes[i] > 0:
                    spikes_sums_at_time_dict[time_spike].append((duration_spikes[i] / sum_spikes[i]))
                else:
                    spikes_sums_at_time_dict[time_spike].append(0)

                if time_spike not in spikes_at_time_dict:
                    spikes_at_time_dict[time_spike] = []
                spikes_at_time_dict[time_spike].append(np.where(spike_nums_to_use[:, twitch_time + i])[0])
        return spike_sum_of_sum_at_time_dict, spike_sum_of_sum_non_norm_at_time_dict, spikes_sums_at_time_dict, \
               spikes_at_time_dict

    def detect_twitches(self):
        """
        Detecting twitches based on the mvt periods. A twitch should be short movement with no movement
        around
        :return:
        """
        if self.mvt_frames_periods is None:
            return

        self.twitches_frames_periods = []
        self.twitches_frames = []

        # 500 ms
        max_twitch_duration = 5
        space_before_twitch = 30

        for index, period in enumerate(self.mvt_frames_periods):
            if (period[1] - period[0]) > max_twitch_duration:
                continue
            if (index == 0) or (period[0] - self.mvt_frames_periods[index - 1][1]) >= space_before_twitch:
                self.twitches_frames_periods.append(period)
                self.twitches_frames.extend(list(np.arange(period[0], period[1] + 1)))
        # print(f"{self.description} {len(self.twitches_frames_periods)} twitches: {self.twitches_frames_periods}")
        self.twitches_frames_periods = np.array(self.twitches_frames_periods)
        self.twitches_frames = np.array(self.twitches_frames)

    def get_events_independent_of_mvt(self, reverse_it=False):
        # reverse_it is True, then we return event links to mvt
        events_periods = []
        sce_times_related_to_mvt = np.zeros(len(self.sce_bool), dtype="bool")
        for mvt_period in self.mvt_frames_periods:
            is_in_sce = np.any(self.sce_bool[mvt_period[0]: mvt_period[1] + 1])
            if is_in_sce:
                indices = np.where(self.sce_bool[mvt_period[0]: mvt_period[1] + 1])[0] + mvt_period[0]
                sce_times_related_to_mvt[indices] = True
            # looking if there is a sce less than a second after
            end_time = np.min((mvt_period[1] + 1 + 10, len(self.sce_bool)))
            sce_after = np.any(self.sce_bool[mvt_period[1]:end_time])

            if sce_after:
                indices = np.where(self.sce_bool[mvt_period[1]:end_time])[0] + mvt_period[1]
                sce_times_related_to_mvt[indices] = True

        for sce_period in self.SCE_times:
            if reverse_it:
                if np.any(sce_times_related_to_mvt[sce_period[0]:sce_period[1] + 1]):
                    events_periods.append((sce_period[0], sce_period[1]))
            else:
                if not np.any(sce_times_related_to_mvt[sce_period[0]:sce_period[1] + 1]):
                    events_periods.append((sce_period[0], sce_period[1]))
        return events_periods

    def get_twitches_times_by_group(self, sce_bool=None, sce_periods=None, twitches_group=0):
        twitches_times = []
        twitches_periods = []

        # not using twitches_groups anymore
        twitches_frames_periods = get_continous_time_periods(self.shift_data_dict["shift_twitch"].astype("int8"))
        for twitch_period in twitches_frames_periods:
            if (sce_bool is None) or (twitches_group == 0):
                # twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
                # taking beginning of twitch
                twitches_times.append(twitch_period[0])
                twitches_periods.append((twitch_period[0], twitch_period[1]))

        # if twitches_group == 7:
        #     sce_times_related_to_twitch = np.zeros(len(sce_bool), dtype="bool")
        #     for twitch_period in self.twitches_frames_periods:
        #         is_in_sce = np.any(sce_bool[twitch_period[0]: twitch_period[1] + 1])
        #         if is_in_sce:
        #             indices = np.where(sce_bool[twitch_period[0]: twitch_period[1] + 1])[0] + twitch_period[0]
        #             sce_times_related_to_twitch[indices] = True
        #         # looking if there is a sce less than a second after
        #         end_time = np.min((twitch_period[1] + 1 + 10, len(sce_bool)))
        #         sce_after = np.any(sce_bool[twitch_period[1]:end_time])
        #
        #         if sce_after:
        #             indices = np.where(sce_bool[twitch_period[1]:end_time])[0] + twitch_period[1]
        #             sce_times_related_to_twitch[indices] = True
        #
        #     for sce_period in sce_periods:
        #         if not np.any(sce_times_related_to_twitch[sce_period[0]:sce_period[1] + 1]):
        #             twitches_times.append((sce_period[0] + sce_period[1]) // 2)
        #             twitches_periods.append((sce_period[0], sce_period[1]))
        #
        # for twitch_period in self.twitches_frames_periods:
        #     if (sce_bool is None) or (twitches_group == 0):
        #         twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
        #         twitches_periods.append((twitch_period[0], twitch_period[1]))
        #         continue
        #     is_in_sce = np.any(sce_bool[twitch_period[0]: twitch_period[1] + 1])
        #     if twitches_group == 1:
        #         if is_in_sce:
        #             twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
        #             twitches_periods.append((twitch_period[0], twitch_period[1]))
        #         continue
        #
        #     if twitches_group == 2:
        #         if not is_in_sce:
        #             twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
        #             twitches_periods.append((twitch_period[0], twitch_period[1]))
        #         continue
        #
        #     if is_in_sce:
        #         continue
        #     # looking if there is a sce less than a second after
        #     end_time = np.min((twitch_period[1] + 1 + 10, len(sce_bool)))
        #     sce_after = np.any(sce_bool[twitch_period[1]:end_time])
        #     if twitches_group == 3:
        #         if sce_after:
        #             twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
        #             twitches_periods.append((twitch_period[0], twitch_period[1]))
        #             continue
        #     if twitches_group == 4:
        #         if not sce_after:
        #             twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
        #             twitches_periods.append((twitch_period[0], twitch_period[1]))
        #             continue
        return twitches_times, twitches_periods

    def get_spikes_values_around_long_mvt(self, time_around=100, low_percentile=25, high_percentile=75):
        if self.spike_struct.spike_nums_dur is None:
            return

        spike_nums_dur = self.spike_struct.spike_nums_dur
        spike_nums_to_use = spike_nums_dur

        n_cells = len(spike_nums_dur)

        # frames on which to center the ptsth
        mvt_times = []

        mvt_frames_periods = get_continous_time_periods(self.shift_data_dict["shift_long"].astype("int8"))
        for mvt_period in mvt_frames_periods:
            mvt_times.append(mvt_period[0])

        n_mvts = len(mvt_times)

        results = self.get_spikes_by_time_around_a_time(mvt_times, spike_nums_to_use, time_around)
        if results is None:
            return

        spike_sum_of_sum_at_time_dict, spikes_sums_at_time_dict, \
        spikes_at_time_dict = results

        distribution = []
        mean_values = []
        median_values = []
        low_values = []
        high_values = []
        std_values = []
        time_x_values = np.arange(-1 * time_around, time_around + 1)
        for time, nb_spikes_at_time in spike_sum_of_sum_at_time_dict.items():
            # print(f"time {time}")
            distribution.extend([time] * nb_spikes_at_time)
            # mean percentage of cells at each twitch
        for time_value in time_x_values:
            if time_value in spike_sum_of_sum_at_time_dict:
                mean_values.append((np.mean(spikes_sums_at_time_dict[time_value]) / n_cells) * 100)
                median_values.append((np.median(spikes_sums_at_time_dict[time_value]) / n_cells) * 100)
                std_values.append((np.std(spikes_sums_at_time_dict[time_value]) / n_cells) * 100)
                low_values.append((np.percentile(spikes_sums_at_time_dict[time_value], low_percentile) / n_cells) * 100)
                high_values.append(
                    (np.percentile(spikes_sums_at_time_dict[time_value], high_percentile) / n_cells) * 100)
            else:
                print(f"time {time_value} not there")
                mean_values.append(0)
                std_values.append(0)
                median_values.append(0)
                low_values.append(0)
                high_values.append(0)
        return n_mvts, time_x_values, np.array(mean_values), \
               np.array(median_values), np.array(low_values), np.array(high_values), np.array(std_values)

    def get_spikes_values_around_twitches(self, sce_bool=None, time_around=100,
                                          twitches_group=0, low_percentile=25, high_percentile=75,
                                          use_traces=False):
        if self.spike_struct.spike_nums_dur is None:
            return

        spike_nums_dur = self.spike_struct.spike_nums_dur
        # spike_nums = self.spike_struct.spike_nums
        if use_traces:
            # print(f"get_spikes_values_around_twitches use_traces")
            spike_nums_to_use = self.raw_traces
        else:
            spike_nums_to_use = spike_nums_dur

        n_cells = len(spike_nums_dur)

        # frames on which to center the ptsth
        twitches_times, twitches_periods = self.get_twitches_times_by_group(sce_bool=sce_bool,
                                                                            twitches_group=twitches_group)

        n_twitches = len(twitches_times)

        results = self.get_spikes_by_time_around_a_time(twitches_times, spike_nums_to_use, time_around)
        if results is None:
            return

        spike_sum_of_sum_at_time_dict, spikes_sums_at_time_dict, \
        spikes_at_time_dict = results

        distribution = []
        mean_values = []
        median_values = []
        low_values = []
        high_values = []
        std_values = []
        time_x_values = np.arange(-1 * time_around, time_around + 1)
        for time, nb_spikes_at_time in spike_sum_of_sum_at_time_dict.items():
            # print(f"time {time}")
            distribution.extend([time] * int(nb_spikes_at_time))
            # mean percentage of cells at each twitch
        for time_value in time_x_values:
            if time_value in spike_sum_of_sum_at_time_dict:
                mean_values.append((np.mean(spikes_sums_at_time_dict[time_value]) / n_cells) * 100)
                median_values.append((np.median(spikes_sums_at_time_dict[time_value]) / n_cells) * 100)
                std_values.append((np.std(spikes_sums_at_time_dict[time_value]) / n_cells) * 100)
                low_values.append((np.percentile(spikes_sums_at_time_dict[time_value], low_percentile) / n_cells) * 100)
                high_values.append(
                    (np.percentile(spikes_sums_at_time_dict[time_value], high_percentile) / n_cells) * 100)
            else:
                print(f"time {time_value} not there")
                mean_values.append(0)
                std_values.append(0)
                median_values.append(0)
                low_values.append(0)
                high_values.append(0)
        return n_twitches, time_x_values, np.array(mean_values), \
               np.array(median_values), np.array(low_values), np.array(high_values), np.array(std_values)

    def get_spikes_duration_values_around_twitches(self, sce_bool=None, time_around=100,
                                                   twitches_group=0, low_percentile=25, high_percentile=75):
        # UGLY CODE
        if self.spike_struct.spike_nums_dur is None:
            return

        spike_nums_dur = self.spike_struct.spike_nums_dur
        # spike_nums = self.spike_struct.spike_nums
        spike_nums_to_use = spike_nums_dur

        n_cells = len(spike_nums_dur)

        # frames on which to center the ptsth
        twitches_times, twitches_periods = self.get_twitches_times_by_group(sce_bool=sce_bool,
                                                                            twitches_group=twitches_group)

        n_twitches = len(twitches_times)

        results = self.get_spikes_duration_by_time_around_a_time(twitches_times, spike_nums_to_use, time_around)
        if results is None:
            return

        # sum of duration, already normalized by the number of active cells
        spike_sum_of_sum_at_time_dict, spike_sum_of_sum_non_norm_at_time_dict, spikes_sums_at_time_dict, \
        spikes_at_time_dict = results

        distribution = []
        mean_values = []
        median_values = []
        low_values = []
        high_values = []
        std_values = []
        time_x_values = np.arange(-1 * time_around, time_around + 1)
        for time, duration_spikes_at_time in spike_sum_of_sum_non_norm_at_time_dict.items():
            # print(f"time {time}")
            distribution.extend([time] * duration_spikes_at_time)
            # mean percentage of cells at each twitch
        for time_value in time_x_values:
            if time_value in spike_sum_of_sum_at_time_dict:
                mean_values.append(np.mean(spikes_sums_at_time_dict[time_value]))
                median_values.append(np.median(spikes_sums_at_time_dict[time_value]))
                std_values.append(np.std(spikes_sums_at_time_dict[time_value]))
                low_values.append(np.percentile(spikes_sums_at_time_dict[time_value], low_percentile) / n_cells)
                high_values.append(np.percentile(spikes_sums_at_time_dict[time_value], high_percentile) / n_cells)
            else:
                print(f"time {time_value} not there")
                mean_values.append(0)
                std_values.append(0)
                median_values.append(0)
                low_values.append(0)
                high_values.append(0)
        return n_twitches, time_x_values, np.array(mean_values), \
               np.array(median_values), np.array(low_values), np.array(high_values), np.array(std_values)

    def plot_psth_long_mvt(self, time_around=100, line_mode=False, ax_to_use=None, put_mean_line_on_plt=False,
                           color_to_use=None, duration_option=False, save_formats="pdf"):
        """

        :param line_mode:
        :param ax_to_use:
        :param put_mean_line_on_plt:
        :param color_to_use:
        :param duration_option:
        :param save_formats:
        :return:
        """
        if self.shift_data_dict is None:
            return

        # sce_bool = self.sce_bool
        sce_bool = None

        # if duration_option:
        #     results = \
        #         self.get_spikes_duration_values_around_twitches(sce_bool=sce_bool, time_around=time_around,
        #                                                         twitches_group=twitches_group)
        # else:
        results = \
            self.get_spikes_values_around_long_mvt(time_around=time_around)

        if results is None:
            return
        n_mvts, time_x_values, mean_values, median_values, low_values, high_values, std_values = results

        n_cells = len(self.spike_struct.spike_nums_dur)
        # activity_threshold_percentage = (self.activity_threshold / n_cells) * 100

        hist_color = "blue"
        edge_color = "white"
        # bar chart

        for mean_version in [True]:  # False
            max_value = 0
            if ax_to_use is None:
                fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                        gridspec_kw={'height_ratios': [1]},
                                        figsize=(15, 10))
                fig.patch.set_facecolor("black")

                ax1.set_facecolor("black")
            else:
                ax1 = ax_to_use
            if line_mode:
                ms_to_plot = [self]
                for index_ms, ms in enumerate(ms_to_plot):
                    weight_str = ""
                    if ms.weight is not None:
                        weight_str = f" ({ms.weight} g)"
                    ms_mean_values = mean_values
                    ms_std_values = std_values
                    ms_median_values = median_values
                    ms_low_values = low_values
                    ms_high_values = high_values

                    if color_to_use is not None:
                        color = color_to_use
                    else:
                        color = hist_color

                    if mean_version:
                        ax1.plot(time_x_values,
                                 ms_mean_values, color=color, lw=2, label=f"{ms.description}{weight_str} {n_mvts} mvt")
                        if put_mean_line_on_plt:
                            plt.plot(time_x_values,
                                     ms_mean_values, color=color, lw=2)
                        ax1.fill_between(time_x_values, ms_mean_values - ms_std_values,
                                         ms_mean_values + ms_std_values,
                                         alpha=0.5, facecolor=color)
                        max_value = np.max((max_value, np.max(ms_mean_values + ms_std_values)))
                    else:
                        ax1.plot(time_x_values,
                                 ms_median_values, color=color, lw=2, label=f"{ms.description}{weight_str} {n_mvts} "
                                                                            f"mvt")
                        ax1.fill_between(time_x_values, ms_low_values, ms_high_values, alpha=0.5, facecolor=color)
                        max_value = np.max((max_value, np.max(ms_high_values)))
            else:
                if color_to_use is not None:
                    hist_color = color_to_use
                    edge_color = "white"
                weight_str = ""
                if ms.weight is not None:
                    weight_str = f" ({ms.weight} g)"
                ax1.bar(time_x_values,
                        mean_values, color=hist_color, edgecolor=edge_color,
                        label=f"{self.description}{weight_str} {n_mvts} mvt")
                max_value = np.max((max_value, np.max(mean_values)))
            ax1.vlines(0, 0,
                       np.max(mean_values), color="white", linewidth=2,
                       linestyles="dashed")
            if put_mean_line_on_plt:
                plt.vlines(0, 0,
                           np.max(mean_values), color="white", linewidth=2,
                           linestyles="dashed")
            # ax1.hlines(activity_threshold_percentage, -1 * time_around, time_around,
            #            color="white", linewidth=1,
            #            linestyles="dashed")

            ax1.tick_params(axis='y', colors="white")
            ax1.tick_params(axis='x', colors="white")

            ax1.legend()

            extra_info = ""
            if line_mode:
                extra_info = "lines_"
            if mean_version:
                extra_info += "mean_"
            else:
                extra_info += "median_"

            descr = self.description

            # ax1.title(f"{descr} {n_mvts} twitches bar chart {title_option} {extra_info}")
            if duration_option:
                ax1.set_ylabel(f"Duration (ms)")
            else:
                ax1.set_ylabel(f"Spikes (%)")
            ax1.set_xlabel("time (frames)")
            ax1.set_ylim(0, max_value + 1)
            # ax1.set_ylim(0, np.max((activity_threshold_percentage, max_value)) + 1)

            ax1.xaxis.label.set_color("white")
            ax1.yaxis.label.set_color("white")
            if ax_to_use is None:
                if isinstance(save_formats, str):
                    save_formats = [save_formats]
                for save_format in save_formats:
                    fig.savefig(f'{self.param.path_results}/{descr}_bar_chart_'
                                f'{n_mvts}_mvt'
                                f'_{extra_info}{self.param.time_str}.{save_format}',
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())

                plt.close()

    def plot_psth_twitches(self, time_around=100,
                           twitches_group=0, line_mode=False,
                           ax_to_use=None, put_mean_line_on_plt=False,
                           color_to_use=None,
                           with_other_ms=None,
                           duration_option=False,
                           use_traces=False,
                           save_formats="pdf"):
        """
        Not using groups anymore
        :param sce_bool:
        :param only_in_sce:
        :param time_around:
        :param twitches_group: 5 groups: 0 : all twitches,
        group 1: those in sce_events
        group 2: those not in sce-events
        group 3: those not in sce-events but with sce-events followed less than one second after
        group 4: those not in sce-events and not followed by sce
        :param save_formats:
        :return:
        """

        # if self.twitches_frames_periods is None:
        #     return
        if self.shift_data_dict is None:
            return

        # sce_bool = self.sce_bool
        sce_bool = None

        if with_other_ms is not None:
            line_mode = True
        if duration_option:
            results = \
                self.get_spikes_duration_values_around_twitches(sce_bool=sce_bool, time_around=time_around,
                                                                twitches_group=twitches_group)
        else:
            results = \
                self.get_spikes_values_around_twitches(sce_bool=sce_bool, time_around=time_around,
                                                       twitches_group=twitches_group, use_traces=use_traces)

        if results is None:
            return
        n_twitches, time_x_values, mean_values, median_values, low_values, high_values, std_values = results

        n_cells = len(self.spike_struct.spike_nums_dur)
        # activity_threshold_percentage = (self.activity_threshold / n_cells) * 100

        hist_color = "blue"
        edge_color = "white"
        # bar chart

        """
        groups: 0 : all twitches,
        group 1: those in sce_events
        group 2: those not in sce-events
        group 3: those not in sce-events but with sce-events followed less than one second after
        group 4: those not in sce-events and not followed by sce
        """

        title_option = self.twitches_group_title[twitches_group]

        for mean_version in [True]:  # False
            max_value = 0
            if ax_to_use is None:
                fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                        gridspec_kw={'height_ratios': [1]},
                                        figsize=(15, 10))
                fig.patch.set_facecolor("black")

                ax1.set_facecolor("black")
            else:
                ax1 = ax_to_use
            if line_mode:
                ms_to_plot = [self]
                if with_other_ms is not None:
                    ms_to_plot.extend(with_other_ms)
                for index_ms, ms in enumerate(ms_to_plot):
                    if ms.description == self.description:
                        ms_mean_values = mean_values
                        ms_std_values = std_values
                        ms_median_values = median_values
                        ms_low_values = low_values
                        ms_high_values = high_values
                    else:
                        line_mode = True
                        if duration_option:
                            results = \
                                ms.get_spikes_duration_values_around_twitches(sce_bool=sce_bool,
                                                                              time_around=time_around,
                                                                              twitches_group=twitches_group)
                        else:
                            results = \
                                ms.get_spikes_values_around_twitches(sce_bool=ms.sce_bool, time_around=time_around,
                                                                     twitches_group=twitches_group)

                        if results is None:
                            continue
                        ms_n_twitches, ms_time_x_values, ms_mean_values, ms_median_values, \
                        ms_low_values, ms_high_values, ms_std_values = results

                    if color_to_use is not None:
                        color = color_to_use
                    elif with_other_ms is None:
                        color = hist_color
                    else:
                        color = cm.nipy_spectral(float(index_ms + 1) / (len(with_other_ms) + 2))
                    if mean_version:
                        ax1.plot(time_x_values,
                                 ms_mean_values, color=color, lw=2, label=f"{ms.description} {n_twitches} twitches")
                        if put_mean_line_on_plt:
                            plt.plot(time_x_values,
                                     ms_mean_values, color=color, lw=2)
                        if with_other_ms is None:
                            ax1.fill_between(time_x_values, ms_mean_values - ms_std_values,
                                             ms_mean_values + ms_std_values,
                                             alpha=0.5, facecolor=color)
                        max_value = np.max((max_value, np.max(ms_mean_values + ms_std_values)))
                    else:
                        ax1.plot(time_x_values,
                                 ms_median_values, color=color, lw=2, label=f"{ms.description} {n_twitches} twitches")
                        if with_other_ms is None:
                            ax1.fill_between(time_x_values, ms_low_values, ms_high_values,
                                             alpha=0.5, facecolor=color)
                        max_value = np.max((max_value, np.max(ms_high_values)))
            else:
                if color_to_use is not None:
                    hist_color = color_to_use
                    edge_color = "white"
                ax1.bar(time_x_values,
                        mean_values, color=hist_color, edgecolor=edge_color,
                        label=f"{self.description} {n_twitches} twitches")
                max_value = np.max((max_value, np.max(mean_values)))
            ax1.vlines(0, 0,
                       np.max(mean_values), color="white", linewidth=2,
                       linestyles="dashed")
            if put_mean_line_on_plt:
                plt.vlines(0, 0,
                           np.max(mean_values), color="white", linewidth=2,
                           linestyles="dashed")
            # ax1.hlines(activity_threshold_percentage, -1 * time_around, time_around,
            #            color="white", linewidth=1,
            #            linestyles="dashed")

            ax1.tick_params(axis='y', colors="white")
            ax1.tick_params(axis='x', colors="white")

            # if with_other_ms is not None:
            ax1.legend()

            extra_info = ""
            if line_mode:
                extra_info = "lines_"
            if mean_version:
                extra_info += "mean_"
            else:
                extra_info += "median_"

            descr = self.description
            if with_other_ms is not None:
                descr = f"p{self.age}"

            # ax1.title(f"{descr} {n_twitches} twitches bar chart {title_option} {extra_info}")
            if duration_option:
                ax1.set_ylabel(f"Duration (ms)")
            else:
                ax1.set_ylabel(f"Spikes (%)")
            ax1.set_xlabel("time (frames)")
            ax1.set_ylim(0, max_value + 1)
            # ax1.set_ylim(0, np.max((activity_threshold_percentage, max_value)) + 1)

            ax1.xaxis.label.set_color("white")
            ax1.yaxis.label.set_color("white")
            # xticks = np.arange(0, len(data_dict))
            # ax1.set_xticks(xticks)
            # # sce clusters labels
            # ax1.set_xticklabels(labels)
            if duration_option:
                title_option += "_duration"
            if ax_to_use is None:
                if isinstance(save_formats, str):
                    save_formats = [save_formats]
                for save_format in save_formats:
                    fig.savefig(f'{self.param.path_results}/{descr}_bar_chart_'
                                f'{n_twitches}_twitches_{title_option}'
                                f'_{extra_info}{self.param.time_str}.{save_format}',
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())

                plt.close()

        # if len(distribution) == 0:
        #     continue
        # distribution = np.array(distribution)
        # max_range = np.max((np.max(distribution), time_around))
        # min_range = np.min((np.min(distribution), -time_around))
        # weights = (np.ones_like(distribution) / (len(distribution))) * 100
        #
        # fig, ax1 = plt.subplots(nrows=1, ncols=1,
        #                         gridspec_kw={'height_ratios': [1]},
        #                         figsize=(15, 10))
        # ax1.set_facecolor("black")
        # # as many bins as time
        # bins = (max_range - min_range) // 2
        # # bins = int(np.sqrt(len(distribution)))
        # hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins, range=(min_range, max_range),
        #                                             facecolor=hist_color,
        #                                             edgecolor=edge_color,
        #                                             weights=weights, log=False)
        # ax1.vlines(0, 0,
        #            np.max(hist_plt), color="white", linewidth=2,
        #            linestyles="dashed")
        # plt.title(f"{self.description} {n_twitches} twitches psth")
        # ax1.set_ylabel(f"Probability distribution (%)")
        # ax1.set_xlabel("time (frames)")
        # ax1.set_ylim(0, np.max(hist_plt)+1)
        # # xticks = np.arange(0, len(data_dict))
        # # ax1.set_xticks(xticks)
        # # # sce clusters labels
        # # ax1.set_xticklabels(labels)
        #
        # if isinstance(save_formats, str):
        #     save_formats = [save_formats]
        # for save_format in save_formats:
        #     fig.savefig(f'{self.param.path_results}/{self.description}_psth_'
        #                 f'{n_twitches}_twitches'
        #                 f'_{self.param.time_str}.{save_format}',
        #                 format=f"{save_format}")
        #
        # plt.close()

    def load_best_order_data(self):
        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(self.param.best_order_data_path):
            file_names.extend(local_filenames)
            break
        if len(file_names) == 0:
            return

        for file_name in file_names:
            file_name_original = file_name
            file_name = file_name.lower()
            descr = self.description.lower()
            if descr not in file_name:
                continue

            with open(self.param.best_order_data_path + file_name_original, "r", encoding='UTF-8') as file:
                for nb_line, line in enumerate(file):
                    if line.startswith("best_order"):
                        line_list = line.split(':')
                        cells = line_list[1].split(" ")
                        self.best_order_loaded = np.array([int(cell) for cell in cells])
                        # print(f"{self.description} {len(self.best_order_loaded)} :self.best_order_loaded {self.best_order_loaded}")
                        # raise Exception()

    def load_cell_assemblies_data(self):
        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(self.param.cell_assemblies_data_path):
            file_names.extend(local_filenames)
            break
        if len(file_names) == 0:
            return

        for file_name in file_names:
            file_name_original = file_name
            file_name = file_name.lower()
            descr = self.description.lower()
            if not file_name.startswith(descr):
                continue
            # list of list, each list correspond to one cell assemblie
            self.cell_assemblies = []
            # key is the CA index, eachkey is a list correspond to tuples
            # (first and last index of the SCE in frames)
            self.sce_times_in_single_cell_assemblies = dict()
            self.sce_times_in_multiple_cell_assemblies = []
            # list of list, each list correspond to tuples (first and last index of the SCE in frames)
            self.sce_times_in_cell_assemblies = []
            # for each cell, list of list, each correspond to tuples (first and last index of the SCE in frames)
            # in which the cell is supposed to be active for the single cell assemblie to which it belongs
            self.sce_times_in_cell_assemblies_by_cell = dict()

            with open(self.param.cell_assemblies_data_path + file_name_original, "r", encoding='UTF-8') as file:
                param_section = False
                cell_ass_section = False
                for nb_line, line in enumerate(file):
                    if line.startswith("#PARAM#"):
                        param_section = True
                        continue
                    if line.startswith("#CELL_ASSEMBLIES#"):
                        cell_ass_section = True
                        param_section = False
                        continue
                    if cell_ass_section:
                        if line.startswith("SCA_cluster"):
                            cells = []
                            line_list = line.split(':')
                            cells = line_list[2].split(" ")
                            self.cell_assemblies.append([int(cell) for cell in cells])
                        elif line.startswith("single_sce_in_ca"):
                            line_list = line.split(':')
                            ca_index = int(line_list[1])
                            self.sce_times_in_single_cell_assemblies[ca_index] = []
                            couples_of_times = line_list[2].split("#")
                            for couple_of_time in couples_of_times:
                                times = couple_of_time.split(" ")
                                self.sce_times_in_single_cell_assemblies[ca_index].append([int(t) for t in times])
                                self.sce_times_in_cell_assemblies.append([int(t) for t in times])
                        elif line.startswith("multiple_sce_in_ca"):
                            line_list = line.split(':')
                            sces_times = line_list[1].split("#")
                            for sce_time in sces_times:
                                times = sce_time.split(" ")
                                self.sce_times_in_multiple_cell_assemblies.append([int(t) for t in times])
                                self.sce_times_in_cell_assemblies.append([int(t) for t in times])
                        elif line.startswith("cell"):
                            line_list = line.split(':')
                            cell = int(line_list[1])
                            self.sce_times_in_cell_assemblies_by_cell[cell] = []
                            sces_times = line_list[2].split("#")
                            for sce_time in sces_times:
                                times = sce_time.split()
                                self.sce_times_in_cell_assemblies_by_cell[cell].append([int(t) for t in times])

                # print(f"self.sce_times_in_single_cell_assemblies {self.sce_times_in_single_cell_assemblies}")
                # print(f"self.sce_times_in_multiple_cell_assemblies {self.sce_times_in_multiple_cell_assemblies}")
                # print(f"self.sce_times_in_cell_assemblies {self.sce_times_in_cell_assemblies}")
                # raise Exception("titi")

    def plot_raw_traces_around_twitches(self):
        if self.twitches_frames_periods is None:
            return
        twitches_periods = self.twitches_frames_periods
        twitches_frames = []
        for twitch_period in twitches_periods:
            twitches_frames.append(int((twitch_period[0] + twitch_period[1]) // 2))
        self.plot_raw_traces_around_frames(frames_indices=np.array(twitches_frames), data_descr="twitches")

    def plot_raw_traces_around_each_sce_for_each_cell(self):
        sce_times = self.SCE_times
        # percentage of SCE in which the cell participate
        participation_to_sce_by_cell = dict()
        n_sces = len(sce_times[3:])
        for cell in np.arange(self.spike_struct.n_cells):
            spikes = np.where(self.spike_struct.spike_nums_dur[cell, :])[0]
            if len(spikes) > 0:
                n_sces_with_this_cell = len(np.unique(self.sce_times_numbers[spikes]))
                p = np.round((n_sces_with_this_cell / n_sces) * 100, 1)
                participation_to_sce_by_cell[cell] = f" ({p} %)"
            else:
                participation_to_sce_by_cell[cell] = " (0 %)"

        for sce_index, sce_period in enumerate(sce_times[3:]):
            sce_index += 3
            index_peak = sce_period[0] + np.argmax(np.sum(
                self.spike_struct.spike_nums_dur[:, sce_period[0]:sce_period[1] + 1], axis=0))
            cells_to_color = []
            colors_for_cells = []
            cells_to_color.append(np.where(np.sum(
                self.spike_struct.spike_nums_dur[:, sce_period[0]:sce_period[1] + 1], axis=1))[0])
            colors_for_cells.append("red")
            v_lines = []
            v_lines_colors = []
            if index_peak != sce_period[0]:
                v_lines.append(sce_period[0] - index_peak)
                v_lines_colors.append("white")
            if index_peak != sce_period[1]:
                v_lines.append(sce_period[1] - index_peak)
                v_lines_colors.append("white")
            # print(f"index_peak {index_peak}, sce_period[0] {sce_period[0]}, sce_period[1] {sce_period[1]}, "
            #       f"v_lines {v_lines}")
            self.plot_raw_traces_around_frame_for_each_cell(frame_index=index_peak,
                                                            data_descr=f"SCE_{sce_index}",
                                                            cells_to_color=cells_to_color,
                                                            colors_for_cells=colors_for_cells,
                                                            range_in_frames=100,
                                                            v_lines=v_lines,
                                                            extra_info_by_cell=participation_to_sce_by_cell,
                                                            v_lines_colors=v_lines_colors)

            # plot map
            plot_cells_map = True
            if plot_cells_map:
                cells_groups_colors = ["red"]
                cells_groups = cells_to_color

                # self.coord_obj.compute_center_coord(cells_groups=cells_groups,
                #                                     cells_groups_colors=cells_groups_colors)

                self.coord_obj.plot_cells_map(param=self.param,
                                              data_id=self.description, show_polygons=False,
                                              title_option=f"cells_in_sce_{sce_index}",
                                              with_cell_numbers=True,
                                              cells_groups=cells_groups,
                                              cells_groups_colors=cells_groups_colors,
                                              background_color="black", default_cells_color="blue")

    def plot_raw_traces_around_frame_for_each_cell(self, frame_index, data_descr, show_plot=False, range_in_frames=50,
                                                   cells_to_color=None, colors_for_cells=None,
                                                   v_lines=None, v_lines_colors=None, extra_info_by_cell=None,
                                                   save_formats="pdf"):
        traces = self.raw_traces
        if traces is None:
            return
        n_cells = len(traces)
        n_times = len(traces[0, :])

        grouped_values = []
        n_lines = 10
        n_col = 10
        n_plots_by_fig = n_lines * n_col

        for cell in np.arange(n_cells):

            len_plot = int((range_in_frames * 2) + 1)
            x_times = np.arange(-range_in_frames, range_in_frames + 1)
            all_values = np.zeros(len_plot)

            beg_time = np.max((0, frame_index - range_in_frames))
            end_time = np.min((n_times, frame_index + range_in_frames + 1))
            len_data = end_time - beg_time
            if frame_index - range_in_frames >= 0:
                value_beg = 0
            else:
                value_beg = 0 - (frame_index - range_in_frames)

            all_values[value_beg:value_beg + len_data] = traces[cell, beg_time:end_time]

            # mean_values = np.mean(all_values, axis=0)
            # std_values = np.std(all_values, axis=0)

            grouped_values.append(all_values)
            # grouped_std_values.append(std_values)

            # plt.title(f"trace around {data_descr} of {self.description} {frame_index}")

            if ((cell + 1) % n_plots_by_fig == 0) or (cell == (n_cells - 1)):
                n_cells_to_plot = n_cells
                if ((cell + 1) % n_plots_by_fig == 0):
                    first_cell = cell - n_plots_by_fig + 1
                else:
                    first_cell = cell - ((cell + 1) % n_plots_by_fig) + 1

                if (cell == (n_cells - 1)):
                    n_cells_to_plot = len(grouped_values)

                fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                                         gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                                         figsize=(30, 20))
                fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
                axes = axes.flatten()
                for ax_index, ax in enumerate(axes):
                    if (ax_index + 1) > n_cells_to_plot:
                        break
                    ax.set_facecolor("black")
                    color_to_use = "blue"
                    for index_group, cells in enumerate(cells_to_color):
                        if (ax_index + first_cell) in cells:
                            color_to_use = colors_for_cells[index_group]
                            break
                    extra_info = ""
                    if extra_info_by_cell is not None:
                        if (ax_index + first_cell) in extra_info_by_cell:
                            extra_info = extra_info_by_cell[ax_index + first_cell]
                    ax.plot(x_times,
                            grouped_values[ax_index], color=color_to_use,
                            lw=2, label=f"n° {ax_index + first_cell}{extra_info}")
                    # ax.fill_between(x_times, grouped_mean_values[ax_index] - grouped_std_values[ax_index],
                    #                 grouped_mean_values[ax_index] + grouped_std_values[ax_index],
                    #                 alpha=0.5, facecolor="blue")
                    ax.legend()
                    ax.vlines(0, 0,
                              np.max(grouped_values[ax_index]), color="white",
                              linewidth=1,
                              linestyles="dashed")
                    for v_line_index, v_line in enumerate(v_lines):
                        ax.vlines(v_line, 0,
                                  np.max(grouped_values[ax_index]), color=v_lines_colors[v_line_index],
                                  linewidth=1,
                                  linestyles="dashed")
                    xticks = np.arange(-range_in_frames, range_in_frames + 1, 10)
                    xticks_labels = np.arange(-(range_in_frames // 10), (range_in_frames // 10) + 1)
                    ax.set_xticks(xticks)
                    # sce clusters labels
                    ax.set_xticklabels(xticks_labels)

                if isinstance(save_formats, str):
                    save_formats = [save_formats]
                for save_format in save_formats:
                    fig.savefig(f'{self.param.path_results}/{self.description}_cells_{first_cell}-{cell}_'
                                f'traces_around_{frame_index}_{data_descr}_'
                                f'{range_in_frames}_frames'
                                f'_{self.param.time_str}.{save_format}',
                                format=f"{save_format}")

                if show_plot:
                    plt.show()
                plt.close()

                grouped_values = []

    def plot_raw_traces_around_frames(self, frames_indices, data_descr, show_plot=False, range_in_frames=50,
                                      save_formats="pdf"):
        if self.traces is None:
            return
        n_cells = len(self.traces)
        n_times = len(self.traces[0, :])

        grouped_mean_values = []
        grouped_std_values = []
        n_lines = 10
        n_col = 5
        n_plots_by_fig = n_lines * n_col

        for cell in np.arange(n_cells):

            len_plot = int((range_in_frames * 2) + 1)
            x_times = np.arange(-range_in_frames, range_in_frames + 1)
            all_values = np.zeros((len(frames_indices), len_plot))

            for frame_count, frame_index in enumerate(frames_indices):
                beg_time = np.max((0, frame_index - range_in_frames))
                end_time = np.min((n_times, frame_index + range_in_frames + 1))
                len_data = end_time - beg_time
                if frame_index - range_in_frames >= 0:
                    value_beg = 0
                else:
                    value_beg = 0 - (frame_index - range_in_frames)

                # print(f"piezo_beg_time {piezo_beg_time}, piezo_end_time {piezo_end_time}, value_beg {value_beg}, "
                #       f"sce_time {sce_time}")

                all_values[frame_count, value_beg:value_beg + len_data] = self.traces[cell, beg_time:end_time]

            mean_values = np.mean(all_values, axis=0)
            std_values = np.std(all_values, axis=0)

            grouped_mean_values.append(mean_values)
            grouped_std_values.append(std_values)

            plt.title(f"trace around {data_descr} of {self.description} {len(frames_indices)} {data_descr}")

            if ((cell + 1) % n_plots_by_fig == 0) or (cell == (n_cells - 1)):
                n_cells_to_plot = n_cells
                if ((cell + 1) % n_plots_by_fig == 0):
                    first_cell = cell - n_plots_by_fig + 1
                else:
                    first_cell = cell - ((cell + 1) % n_plots_by_fig) + 1

                if (cell == (n_cells - 1)):
                    n_cells_to_plot = len(grouped_mean_values)

                fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                                         gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                                         figsize=(20, 15))
                fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
                axes = axes.flatten()
                for ax_index, ax in enumerate(axes):
                    if (ax_index + 1) > n_cells_to_plot:
                        break
                    ax.set_facecolor("black")

                    ax.plot(x_times,
                            grouped_mean_values[ax_index], color="blue", lw=2, label=f"cell {ax_index + first_cell}")
                    ax.fill_between(x_times, grouped_mean_values[ax_index] - grouped_std_values[ax_index],
                                    grouped_mean_values[ax_index] + grouped_std_values[ax_index],
                                    alpha=0.5, facecolor="blue")
                    ax.legend()
                    ax.vlines(0, np.min(grouped_mean_values[ax_index] - grouped_std_values[ax_index]),
                              np.max(grouped_mean_values[ax_index] + grouped_std_values[ax_index]), color="white",
                              linewidth=1,
                              linestyles="dashed")
                    xticks = np.arange(-range_in_frames, range_in_frames + 1, 10)
                    xticks_labels = np.arange(-(range_in_frames // 10), (range_in_frames // 10) + 1)
                    ax.set_xticks(xticks)
                    # sce clusters labels
                    ax.set_xticklabels(xticks_labels)

                if isinstance(save_formats, str):
                    save_formats = [save_formats]
                for save_format in save_formats:
                    fig.savefig(f'{self.param.path_results}/{self.description}_cells_{first_cell}-{cell}_'
                                f'traces_around_{len(frames_indices)}_{data_descr}_'
                                f'{range_in_frames}_frames'
                                f'_{self.param.time_str}.{save_format}',
                                format=f"{save_format}")

                if show_plot:
                    plt.show()
                plt.close()

                grouped_mean_values = []
                grouped_std_values = []

    def plot_piezo_around_event(self, show_plot=False, range_in_sec=2, save_formats="pdf"):
        print(f"plot_piezo_around_event: {self.description}")
        if self.raw_piezo is None:
            print(f"{self.description} has no raw piezo")
            return
        sce_periods = self.SCE_times

        n_time_by_sec = int(len(self.abf_times_in_sec) // self.abf_times_in_sec[-1])
        # print(f"n_time_by_sec {n_time_by_sec}")
        len_plot = int((range_in_sec * n_time_by_sec * 2) + 1)
        x_times = np.linspace((-n_time_by_sec * range_in_sec), ((n_time_by_sec * range_in_sec) + 1), len_plot)
        all_values = np.zeros((len(sce_periods), len_plot))
        # mean_values = np.zeros(len_plot)
        # std_values = np.zeros(len_plot)

        for sce_index, sce_period in enumerate(sce_periods):
            index_peak = sce_period[0] + np.argmax(np.sum(
                self.spike_struct.spike_nums_dur[:, sce_period[0]:sce_period[1] + 1], axis=0))
            sce_time = self.abf_frames[index_peak]
            piezo_beg_time = np.max((0, sce_time - (range_in_sec * n_time_by_sec)))
            piezo_end_time = np.min((len(self.raw_piezo), sce_time + (range_in_sec * n_time_by_sec) + 1))
            len_data = piezo_end_time - piezo_beg_time
            if (sce_time - (range_in_sec * n_time_by_sec)) >= 0:
                value_beg = 0
            else:
                value_beg = 0 - (sce_time - (range_in_sec * n_time_by_sec))

            # print(f"piezo_beg_time {piezo_beg_time}, piezo_end_time {piezo_end_time}, value_beg {value_beg}, "
            #       f"sce_time {sce_time}")

            all_values[sce_index, value_beg:value_beg + len_data] = self.raw_piezo[piezo_beg_time:piezo_end_time]

        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(20, 8))

        ax.set_facecolor("black")

        plt.plot(x_times,
                 mean_values, color="blue", lw=2)
        ax.fill_between(x_times, mean_values - std_values,
                        mean_values + std_values,
                        alpha=0.5, facecolor="blue")

        plt.title(f"piezo around events of {self.description}")

        xticks = np.arange((-n_time_by_sec * range_in_sec), ((n_time_by_sec * range_in_sec) + 1), n_time_by_sec)
        xticks_labels = np.arange(-range_in_sec, range_in_sec + 1)
        ax.set_xticks(xticks)
        # sce clusters labels
        ax.set_xticklabels(xticks_labels)

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{self.param.path_results}/{self.description}_piezo_around_events'
                        f'_{self.param.time_str}.{save_format}',
                        format=f"{save_format}")

        if show_plot:
            plt.show()
        plt.close()

    def plot_piezo_with_extra_info(self, show_plot=True, save_formats="pdf"):
        if (self.raw_piezo is None) or (self.abf_frames is None):
            return

        print(f"plot_piezo_with_extra_info {self.description}")

        span_areas_coords = []
        span_area_colors = []
        span_areas_coords.append(self.mvt_frames_periods)
        span_area_colors.append('red')
        # if with_cell_assemblies_sce and self.sce_times_in_cell_assemblies is not None:
        #     span_areas_coords.append(self.sce_times_in_cell_assemblies)
        #     span_area_colors.append('green')
        # else:
        span_areas_coords.append(self.SCE_times)
        span_area_colors.append('green')
        # span_areas_coords.append(self.twitches_frames_periods)
        # span_area_colors.append('blue')

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(20, 8))
        plt.plot(self.abf_times_in_sec, self.raw_piezo, lw=.5, color="black")

        if span_areas_coords is not None:
            for index, span_area_coord in enumerate(span_areas_coords):
                for coord in span_area_coord:
                    color = span_area_colors[index]
                    coord_0 = self.abf_times_in_sec[self.abf_frames[coord[0]]]
                    coord_1 = self.abf_times_in_sec[self.abf_frames[coord[1]] + 1]
                    ax.axvspan(coord_0, coord_1, alpha=0.5,
                               facecolor=color, zorder=1)
        for twitch_index, twitch_period in enumerate(self.twitches_frames_periods):
            twitch_time_beg = self.abf_times_in_sec[self.abf_frames[twitch_period[0]]]
            twitch_time_end = self.abf_times_in_sec[self.abf_frames[twitch_period[1]]]
            # print(f"twitch {twitch_index} beg-end "
            #       f"{np.round(twitch_time_beg, 2)}-"
            #       f"{np.round(twitch_time_end, 2)}")
            pos = (twitch_time_beg + twitch_time_end) / 2

            y = np.max(self.raw_piezo[self.abf_frames[twitch_period[0]]:
                                      self.abf_frames[twitch_period[1]] + 1])
            # y = np.percentile(self.raw_piezo, 99)
            if twitch_index == 0:
                ax.scatter(x=pos, y=y,
                           marker="*",
                           color=["blue"], s=10, zorder=20, label="twitch")
            else:
                ax.scatter(x=pos, y=y,
                           marker="*",
                           color=["blue"], s=10, zorder=20)

        sce_periods = self.SCE_times

        for sce_index, sce_period in enumerate(sce_periods):
            sce_time_beg = self.abf_times_in_sec[self.abf_frames[sce_period[0]]]
            sce_time_end = self.abf_times_in_sec[self.abf_frames[sce_period[1]]]
            pos = (sce_time_beg + sce_time_end) / 2
            y = np.max(self.raw_piezo[self.abf_frames[sce_period[0]]:
                                      self.abf_frames[sce_period[1]] + 1])
            label = "sce"
            if sce_index == 0:
                ax.scatter(x=pos, y=y,
                           marker="o",
                           color=["lightgreen"], s=20, zorder=10, label=label)
            else:
                ax.scatter(x=pos, y=y,
                           marker="o",
                           color=["lightgreen"], s=20, zorder=10)

        if self.sce_times_in_cell_assemblies is not None:
            sce_periods = self.sce_times_in_cell_assemblies

            for sce_index, sce_period in enumerate(sce_periods):
                sce_time_beg = self.abf_times_in_sec[self.abf_frames[sce_period[0]]]
                sce_time_end = self.abf_times_in_sec[self.abf_frames[sce_period[1]]]
                pos = (sce_time_beg + sce_time_end) / 2
                y = np.max(self.raw_piezo[self.abf_frames[sce_period[0]]:
                                          self.abf_frames[sce_period[1]] + 1])
                # y = np.percentile(self.raw_piezo, 99)
                label = "sce in cell assemblies"
                if sce_index == 0:
                    ax.scatter(x=pos, y=y,
                               marker="o",
                               color=["cornflowerblue"], s=20, zorder=15, label=label)
                else:
                    ax.scatter(x=pos, y=y,
                               marker="o",
                               color=["cornflowerblue"], s=20, zorder=15)

        if self.threshold_piezo is not None:
            ax.hlines(self.threshold_piezo, 0, np.max(self.abf_times_in_sec),
                      color="orange", linewidth=1,
                      linestyles="dashed")

        plt.title(f"piezo {self.description} threshold {self.threshold_piezo}")

        plt.legend()

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            extra_str = ""
            if self.sce_times_in_cell_assemblies is not None:
                extra_str = "_with_cell_assemblies_sce"
            fig.savefig(f'{self.param.path_results}/{self.description}_raw_piezo_{extra_str}'
                        f'_{self.param.time_str}.{save_format}',
                        format=f"{save_format}")

        if show_plot:
            plt.show()
        plt.close()

    def load_and_analyse_cilva_results(self, f, s, path_cilva_data):
        print("load_and_analyse_cilva_results")
        file_names = []

        # look for filenames in the first directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(self.param.path_data + path_cilva_data):
            file_names.extend(dirnames)
            break

        cilva_file_name = None
        for file_name in file_names:
            if "cilva" not in file_name:
                continue
            if self.description.lower() not in file_name.lower():
                continue
            cilva_file_name = file_name
            print(f"cilva_file_name {cilva_file_name}")
            break

        if cilva_file_name is None:
            return False

        plot_ea_vs_sa = True
        plot_fit = True

        #### Load and compare model fits
        alpha, beta, w, b, x, sigma, tau_r, tau_d, gamma, L = cilva_analysis.load_fit(
            os.path.join(self.param.path_data, path_cilva_data, cilva_file_name),
            'train')

        N, T = f.shape

        n_cells = N

        if len(w.shape) == 1:
            # print(f'w {w}')
            w = np.reshape(w, [w.shape[0], 1])
            # print(f'w_reshape {w}')

        kernel = cilva_core.calcium_kernel(tau_r, tau_d, T)
        f_hat = cilva_analysis.reconstruction(alpha, beta, w, b, x, kernel, s)
        print(f"f_hat.shape {f_hat.shape}")
        corr_coefs = np.array([np.corrcoef(f[n], f_hat[n])[0, 1] for n in range(N)])
        inds = np.argsort(corr_coefs)[::-1]

        n_cells_by_plot = 30
        max_n_lines = 10

        if plot_fit:
            # TODO: Compute pearson correlation between the fit and the original traces and plot the distribution
            for index_first_cell in range(0, n_cells, n_cells_by_plot):
                n_cells_in_this_plot = min(n_cells_by_plot, n_cells - index_first_cell)
                n_lines = n_cells_in_this_plot if n_cells_in_this_plot <= max_n_lines else max_n_lines
                n_col = math.ceil(n_cells_in_this_plot / n_lines)
                # for histogram all events
                fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                                         gridspec_kw={'width_ratios': [1] * n_col,
                                                      'height_ratios': [1] * n_lines},
                                         figsize=(30, 25))
                fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
                # fig.patch.set_facecolor(background_color)
                axes = axes.flatten()
                for cell_index, cell in enumerate(np.arange(index_first_cell, index_first_cell + n_cells_in_this_plot)):
                    axes[cell_index].plot(f[inds[cell]], color='k', linewidth=1)
                    axes[cell_index].plot(f_hat[inds[cell]], color='g', linewidth=0.8)
                    pearson_corr = stats.pearsonr(f[inds[cell]], f_hat[inds[cell]])[0]
                    axes[cell_index].text(x=len(f[inds[cell]]) - 500,
                                          y=np.mean(f[inds[cell]]) + 6 * np.std(f[inds[cell]]),
                                          s=f"r={np.round(corr_coefs[inds[cell]], 2)}", color="black", zorder=22,
                                          ha='center', va="center", fontsize=10, fontweight='bold')
                    axes[cell_index].set_xlim([0, T])
                    axes[cell_index].set_frame_on(False)

                    # axes[cell_index].get_xaxis().set_visible(True)
                    # axes[cell_index].get_yaxis().set_visible(False)

                save_formats = "pdf"
                if isinstance(save_formats, str):
                    save_formats = [save_formats]

                for save_format in save_formats:
                    fig.savefig(f'{self.param.path_results}/{self.description}_cilva_trace_fit_'
                                f'cell_{index_first_cell}-{index_first_cell + n_cells_in_this_plot}'
                                f'_{self.param.time_str}.{save_format}',
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())

        ##### Correlation coefficient vs count
        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(2, 2))
        ax.hist(corr_coefs, color='firebrick')
        plt.xlim([-0.15, 1])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel('Correlation coefficient')
        plt.ylabel('Count')
        save_formats = "pdf"
        if isinstance(save_formats, str):
            save_formats = [save_formats]

        for save_format in save_formats:
            fig.savefig(f'{self.param.path_results}/{self.description}_cilva_corr_coeff_vs_count'
                        f'_{self.param.time_str}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())

        ##### Decouple evoked and (low dimensional) spontaneous components
        f_evoked, f_spont = cilva_analysis.decouple_traces(alpha, beta, w, b, x, kernel, s)

        if plot_ea_vs_sa:
            span_area_coords = None
            span_area_colors = None
            if self.shift_data_dict is not None:
                colors = ["red", "green", "blue", "pink", "orange"]
                i = 0
                span_area_coords = []
                span_area_colors = []
                for name_period, period in self.shift_data_dict.items():
                    span_area_coords.append(get_continous_time_periods(period.astype("int8")))
                    span_area_colors.append(colors[i % len(colors)])
                    print(f"Period {name_period} -> {colors[i]}")
                    i += 1

            for index_first_cell in range(0, n_cells, n_cells_by_plot):
                n_cells_in_this_plot = min(n_cells_by_plot, n_cells - index_first_cell)
                n_lines = n_cells_in_this_plot if n_cells_in_this_plot <= max_n_lines else max_n_lines
                n_col = math.ceil(n_cells_in_this_plot / n_lines)
                # for histogram all events
                fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                                         gridspec_kw={'width_ratios': [1] * n_col,
                                                      'height_ratios': [1] * n_lines},
                                         figsize=(30, 25))
                fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
                # fig.patch.set_facecolor(background_color)
                axes = axes.flatten()
                alpha_span_area = 0.5

                for cell_index, cell in enumerate(np.arange(index_first_cell, index_first_cell + n_cells_in_this_plot)):
                    axes[cell_index].plot(f[inds[cell]], color='k', linewidth=0.8)
                    axes[cell_index].plot(f_spont[inds[cell]], color='C0', linewidth=0.7, ls='--')
                    axes[cell_index].plot(f_evoked[inds[cell]], color='firebrick', linewidth=0.6)
                    axes[cell_index].set_xlim([0, T])
                    axes[cell_index].set_frame_on(False)
                    for index, span_area_coord in enumerate(span_area_coords):
                        for coord in span_area_coord:
                            if span_area_colors is not None:
                                color = span_area_colors[index]
                            else:
                                color = "lightgrey"
                            axes[cell_index].axvspan(coord[0], coord[1],
                                                     alpha=alpha_span_area, facecolor=color, zorder=1)
                save_formats = "pdf"
                if isinstance(save_formats, str):
                    save_formats = [save_formats]

                for save_format in save_formats:
                    fig.savefig(f'{self.param.path_results}/{self.description}_cilva_trace_EA_vs_SA_'
                                f'cell_{index_first_cell}-{index_first_cell + n_cells_in_this_plot}'
                                f'_{self.param.time_str}.{save_format}',
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
                plt.close()
        ##### Tuning curves

        kmax = np.max(kernel)
        tuning_curves = (kmax * alpha[:, None] * w)  # First two stimuli not presented
        print(f"tuning_curves {tuning_curves}")
        fig, axes = plt.subplots(figsize=(4, 4), sharex=True, sharey=False, ncols=4, nrows=4)
        axes = axes.flatten()
        for n in range(16):
            # print(f'{n}: tuning_curves[n, :] {tuning_curves[n, :]}')
            axes[n].plot(tuning_curves[n, :], color='firebrick', linewidth=1)
            # axes[n].scatter([0], tuning_curves[n, :], color='firebrick')
            axes[n].get_xaxis().set_visible(False)
            axes[n].get_yaxis().set_visible(False)
        fig.text(0.5, 0.04, 'Stimulus', ha='center')
        fig.text(0.04, 0.5, 'Response', va='center', rotation='vertical')
        save_formats = "pdf"
        if isinstance(save_formats, str):
            save_formats = [save_formats]

        for save_format in save_formats:
            fig.savefig(f'{self.param.path_results}/{self.description}_cilva_tuning_curves'
                        f'_{self.param.time_str}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())
        plt.close()

        # cross-corr between tuning curves
        tuning_curves_matrix = np.zeros((n_cells, n_cells))
        for cell in np.arange(n_cells):
            for cell_2 in np.arange(n_cells):
                if cell == cell_2:
                    tuning_curves_matrix[cell, cell_2] = 1
                tuning_curves_matrix[cell, cell_2] = stats.pearsonr(tuning_curves[cell], tuning_curves[cell_2])[0]

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(2, 2))
        ax = sns.heatmap(tuning_curves_matrix)
        # fig = ax.get_figure()

        save_formats = ["pdf"]
        if isinstance(save_formats, str):
            save_formats = [save_formats]

        for save_format in save_formats:
            fig.savefig(
                f'{self.param.path_results}/{self.description}_cilva_tuning_curves_heatmap'
                f'_{self.param.time_str}.{save_format}',
                format=f"{save_format}",
                facecolor=fig.get_facecolor())
        plt.close()

        # DO HDBSCAN ON DISTANCES MATRIX - CONSIDER PRECOMPUTED DISTANCES
        clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                    gen_min_span_tree=False, leaf_size=40,
                                    metric='precomputed', min_cluster_size=2, min_samples=None, p=None)
        # metric='precomputed' euclidean

        clusterer.fit(tuning_curves_matrix)

        labels = clusterer.labels_
        # print(f"labels.shape: {labels.shape}")
        print(f"N clusters hdbscan: {labels.max() + 1}")
        print(f"labels: {labels}")
        print(f"With no clusters hdbscan: {len(np.where(labels == -1)[0])}")
        n_clusters = 0
        if labels.max() + 1 > 0:
            n_clusters = labels.max() + 1

        if n_clusters > 0:
            n_epoch_by_cluster = [[len(np.where(labels == x)[0])] for x in np.arange(n_clusters)]
            print(f"Number of epochs by clusters hdbscan: {' '.join(map(str, n_epoch_by_cluster))}")

        tuning_curves_matrix_order = np.copy(tuning_curves_matrix)
        labels_indices_sorted = np.argsort(labels)
        tuning_curves_matrix_order = tuning_curves_matrix_order[labels_indices_sorted, :]
        tuning_curves_matrix_order = tuning_curves_matrix_order[:, labels_indices_sorted]

        # Generate figure: dissimilarity matrice ordered by cluster
        # Replace Inf values by NaN for better visualization
        # tuning_curves_matrix_order[np.where(np.isinf(tuning_curves_matrix_order))] = np.nan
        # svm = sns.heatmap(distances_order, annot=True)  # if you want the value
        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(2, 2))
        svm = sns.heatmap(tuning_curves_matrix_order)
        svm.set_yticklabels(labels_indices_sorted)
        svm.set_xticklabels(labels_indices_sorted)
        # fig = svm.get_figure()
        # plt.show()
        save_formats = ["pdf"]
        if isinstance(save_formats, str):
            save_formats = [save_formats]

        for save_format in save_formats:
            fig.savefig(
                f'{self.param.path_results}/{self.description}_cilva_tuning_curves_heatmap_cluster'
                f'_{self.param.time_str}.{save_format}',
                format=f"{save_format}",
                facecolor=fig.get_facecolor())
        plt.close()

        '''

            Factor loading matrix

        '''

        # Sort neurons to maximise visual modularity
        b_order = []
        ams = np.argmax(b, 1)
        L = L.astype(int)
        for l in range(L):
            nrns = np.where(ams == l)[0]
            b_order.append(nrns[np.argsort(b[nrns, l])[::-1]])
        b_order = np.concatenate(b_order)
        b = b[b_order, :]

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(2, 2))
        plt.imshow(b, aspect='auto')
        plt.colorbar()
        plt.xticks(range(L))
        plt.gca().set_xticklabels(range(1, L + 1))
        plt.xlabel('Factors')
        plt.ylabel('Neurons')
        save_formats = "pdf"
        if isinstance(save_formats, str):
            save_formats = [save_formats]

        for save_format in save_formats:
            fig.savefig(f'{self.param.path_results}/{self.description}_factor_loading_matrix'
                        f'_{self.param.time_str}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())

        '''

            Decomposition of variance

        '''

        var_total = np.var(f_hat, 1)
        var_evoked = np.var(f_evoked, 1)
        var_spont = np.var(f_spont, 1)
        var_cov = (var_total - var_spont - var_evoked) / 2
        var_f = np.var(f, 1) - sigma ** 2  # Correction for imaging noise variance

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(15, 2))
        plt.plot([], [], color='firebrick', linewidth=5)
        plt.plot([], [], color='C0', linewidth=5)
        plt.plot([], [], color='C1', linewidth=5)

        plt.legend(['Evoked variance', 'Spontaneous variance', 'Covariance'], frameon=False, ncol=3, loc=(0.25, 0.9))

        sns.barplot(np.arange(N), var_evoked / var_f, color='firebrick', linewidth=0)
        sns.barplot(np.arange(N), var_spont / var_f, bottom=var_evoked / var_f, linewidth=0, color='C0')
        sns.barplot(np.arange(N), np.abs(var_cov) / var_f, bottom=(var_evoked + var_spont) / var_f, linewidth=0,
                    color='C1')

        plt.xlabel('Neuron')
        plt.ylabel('Proportion\nvariance')
        plt.xticks(np.arange(0, N, 10))
        plt.gca().set_xticklabels(np.arange(0, N, 10))
        plt.xlim([-1, N])
        plt.ylim([0, 1])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        save_formats = "pdf"
        if isinstance(save_formats, str):
            save_formats = [save_formats]

        for save_format in save_formats:
            fig.savefig(f'{self.param.path_results}/{self.description}_decomposition_of_variance'
                        f'_{self.param.time_str}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())

        return True

    def run_cilva(self, path_cilva_data="cilva_data"):
        """
        Run cilva analysis https://github.com/GoodhillLab/CILVA
        :return:
        """

        if self.shift_data_dict is None:
            print("run_cilva no twitches informations")
            return False
        if self.raw_traces is None:
            print("run_cilva no raw_traces")
            return False

        # variances = []
        raw_traces = np.zeros(self.raw_traces.shape)
        # z_score traces
        for i in np.arange(self.raw_traces.shape[0]):
            raw_traces[i, :] = self.raw_traces[i, :] - np.median(self.raw_traces[i, :])
            raw_traces[i, raw_traces[i, :] < 0] = 0
            # raw_traces[i, :] = self.raw_traces[i, :] - np.min(self.raw_traces[i, :])
            raw_traces[i, :] = raw_traces[i, :] / np.max(raw_traces[i, :])
            # variances.append(np.var(raw_traces[i, :]))

        # plot_hist_distribution(distribution_data=variances,
        #                        description=f"{self.description}_traces_variances",
        #                        param=self.param, tight_x_range=True)
        # print(f"n_cells <= 2 {len(np.where(np.sum(self.spike_struct.spike_nums, axis=1) <= 0)[0])}")
        # raise Exception("STOP")

        # n_times = raw_traces.shape[1]
        n_times = 2000

        remove_low_firing_cells = True
        if remove_low_firing_cells:
            print(f"n cells before filtering: {len(raw_traces)}")
            cells_to_keep = np.where(np.sum(self.spike_struct.spike_nums, axis=1) > 2)[0]
            raw_traces = raw_traces[cells_to_keep][100:300, 0:n_times]
            print(f"n cells after filtering: {len(raw_traces)}")

        shift_data_binary = self.shift_data_dict["shift_twitch"][:n_times]

        twitches_frames_periods = get_continous_time_periods(shift_data_binary.astype("int8"))
        # n_stimulus * n_times, 1 if the stimulus is present at that time
        twitches_onsets = np.zeros((len(twitches_frames_periods), n_times))
        # twitches_onsets = np.zeros((1, n_times))
        for period_index, period in enumerate(twitches_frames_periods):
            # twitches_onsets[0, period[0]:period[1]+1] = 1
            twitches_onsets[period_index, period[0]] = 1
            # twitches_onsets[0, period[0]] = 1

        try_fit_kernel_time_constants = False

        if try_fit_kernel_time_constants:
            tau_r, tau_d, errs = cilva_core.fit_kernel_time_constants(f=raw_traces, s=twitches_onsets,
                                                                      N=raw_traces.shape[0],
                                                                      T=raw_traces.shape[1],
                                                                      K=len(twitches_frames_periods),
                                                                      eta=0.1, num_iters=5, return_errs=True)
            print(f"tau_r {tau_r}, tau_d {tau_d}, errs {errs}")
            return

        # look if cilva has been run before
        # return True is it's the case and the analysis went trough
        cilva_loaded = self.load_and_analyse_cilva_results(path_cilva_data=path_cilva_data, f=raw_traces,
                                                           s=twitches_onsets)
        if cilva_loaded:
            return

        imrate = self.sampling_rate
        # from cilva.run.py
        # Default parameter values
        L = 3
        # used to be 40 and 40
        num_iters = 40
        iters_per_altern = 60
        gamma = 1.00
        # Note: default rise and decay time constants are appropriate for our GCaMP6s zebrafish larvae.
        # They may not be suitable for other indicators or animal models.
        # default_tau_r = 5.681 / imrate
        # default_tau_d = 11.551 / imrate
        # tau_r and tau_d are in seconds
        tau_r = 25 / imrate
        tau_d = 150 / imrate  # up to 200
        convert_stim = False

        # Importing numpy and model must be performed *after* multithreading is configured.
        import pattern_discovery.cilva.model as cilva_model

        # Train model
        print('Fitting the Calcium Imaging Latent Variable Analysis model...')
        learned_params = cilva_model.train(raw_traces, twitches_onsets, convert_stim,
                                           L, num_iters, iters_per_altern, gamma, tau_r,
                                           tau_d, imrate)
        learned_params += [[tau_r], [tau_d], [gamma], [L]]
        param_names = ['alpha', 'beta', 'w', 'b', 'x', 'sigma', 'tau_r', 'tau_d', 'gamma', 'L']

        # Save learned params to file
        print('Saving parameters to file...')
        path = os.path.join(self.param.path_results, f'cilva_{self.description}_' +
                            '_L_{}_num_iters_{}_iters_per_altern_{}_gamma_{:.2f}_tau_r_{:.2f}_tau_d_{:.2f}_imrate_{:.4f}/'.format(
                                L, num_iters, iters_per_altern, gamma, tau_r, tau_d, imrate))
        if not os.path.isdir(path):
            os.mkdir(path)
        for i, param_name in enumerate(param_names):
            np.savetxt(path + param_name, learned_params[i], fmt='%1.6e')

        # Estimate factor activity on held-out test data
        # if args.test:
        #     cvd_param_names = ['alpha', 'beta', 'w', 'b', 'sigma']
        #     cvd_params = []
        #     for param_name in cvd_param_names:
        #         # Collect parameters from learned_params
        #         cvd_params += [learned_params[param_names.index(param_name)]]
        #
        #     x_test = model.cvd(args.test, L, convert_stim, cvd_params, num_iters * iters_per_altern, gamma, tau_r,
        #                        tau_d)
        #     np.savetxt(path + 'x_test', x_test, fmt='%1.6e')
        print('Model-fitting complete.')

    def analyse_lfp(self):
        wp = self.create_wavelet_param()
        print(f"analyse_lfp on {self.description}")
        lfp_signal = self.lfp_signal
        lfp_sampling_rate = self.lfp_sampling_rate
        print(f"len(lfp_signal) {len(lfp_signal)}")
        print(f"lfp_sampling_rate {lfp_sampling_rate}")
        if lfp_sampling_rate > 1000:
            decimate_factor = 10
            lfp_signal = scipy.signal.decimate(lfp_signal, decimate_factor)
            print(f"After decimate: len(lfp_signal) {len(lfp_signal)}")
            lfp_sampling_rate = int(lfp_sampling_rate / decimate_factor)
        apply_band_pass = False
        if apply_band_pass:
            lowcut = 1
            highcut = 300
            lfp_signal = butter_bandpass_filter(lfp_signal, lowcut, highcut,
                                                lfp_sampling_rate, order=3)
        file_name = f"{self.description}_lfp_spectrogram"

        save_formats = "png"
        dpi = 400
        n_times = len(lfp_signal)
        num_levels_contourf = 50
        window_len_in_s = 60
        n_times = len(lfp_signal)
        window_len_in_times = window_len_in_s * lfp_sampling_rate

        colors_shift = ["red", "green", "blue", "pink", "orange"]

        raster_dur = self.spike_struct.spike_nums_dur
        n_cells = raster_dur.shape[1]
        spikes_frames = np.zeros(0)
        for time_index, beg_time in enumerate(np.arange(0, n_times, window_len_in_times)):
            wav_outcome = spectral_analysis_on_time_segment(beg_time=beg_time,
                                                            lfp_signal=lfp_signal, sampling_rate=lfp_sampling_rate,
                                                            n_times=n_times, window_len_in_s=window_len_in_s,
                                                            save_formats="png", wavelet_param=wp,
                                                            file_name=file_name,
                                                            param=self.param, dpi=dpi,
                                                            save_spectrogram=False, keep_dbconverted=True)

            beg_time_s = beg_time // lfp_sampling_rate
            if wp.using_median_for_threshold:
                threshold = np.median(wav_outcome.dbconverted)
            else:
                threshold = np.mean(wav_outcome.dbconverted) + np.std(wav_outcome.dbconverted)

            max_color_heatmap = np.max(wav_outcome.dbconverted)
            file_name_segment = file_name + f"_{beg_time_s}s_win_{window_len_in_s}"
            n_lines = 3
            n_col = 1
            background_color = "black"
            fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                                     gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [0.45, 0.1, 0.45]},
                                     figsize=(30, 20))
            fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
            fig.patch.set_facecolor(background_color)
            axes = axes.flatten()
            beg_time_raster = int((beg_time / lfp_sampling_rate) * self.sampling_rate)
            end_time_raster = int(((beg_time + window_len_in_times) / lfp_sampling_rate) * self.sampling_rate)
            print(f"beg_time_raster-end_time_raster {beg_time_raster} - {end_time_raster}")

            spikes_array = np.array(wav_outcome.spikes).astype(float)
            spikes_array = (((spikes_array + beg_time) / lfp_sampling_rate) * self.sampling_rate).astype("int16")
            spikes_frames = np.concatenate((spikes_frames, spikes_array))

            frames_to_display = np.arange(beg_time_raster, end_time_raster)

            if self.speed_by_frame is not None:
                binary_speed = np.zeros(len(self.speed_by_frame), dtype="int8")
                binary_speed[self.speed_by_frame > 0] = 1
                speed_periods_tmp = get_continous_time_periods(binary_speed)
                speed_periods = []
                for speed_period in speed_periods_tmp:
                    if (speed_period[0] not in frames_to_display) and (speed_period[1] not in frames_to_display):
                        continue
                    elif (speed_period[0] in frames_to_display) and (speed_period[1] in frames_to_display):
                        speed_periods.append((speed_period[0] - beg_time_raster, speed_period[1] - beg_time_raster))
                    elif speed_period[0] in frames_to_display:
                        speed_periods.append(
                            (speed_period[0] - beg_time_raster, frames_to_display[-1] - beg_time_raster))
                    else:
                        speed_periods.append((0, speed_period[1] - beg_time_raster))

            # colors for movement periods
            span_area_coords = None
            span_area_colors = None
            with_mvt_periods = True

            if with_mvt_periods:
                colors = ["red", "green", "blue", "pink", "orange"]
                i = 0
                span_area_coords = []
                span_area_colors = []
                periods_dict = self.shift_data_dict
                if periods_dict is not None:
                    print(f"{self.description}:")
                    for name_period, period in periods_dict.items():
                        mvt_periods_tmp = get_continous_time_periods(period.astype("int8"))
                        mvt_periods = []
                        for mvt_period in mvt_periods_tmp:
                            if (mvt_period[0] not in frames_to_display) and (
                                    mvt_period[1] not in frames_to_display):
                                continue
                            elif (mvt_period[0] in frames_to_display) and (mvt_period[1] in frames_to_display):
                                mvt_periods.append((mvt_period[0] - beg_time_raster, mvt_period[1] - beg_time_raster))
                            elif mvt_period[0] in frames_to_display:
                                mvt_periods.append(
                                    (mvt_period[0] - beg_time_raster, frames_to_display[-1] - beg_time_raster))
                            else:
                                mvt_periods.append((0, mvt_period[1] - beg_time_raster))
                        span_area_coords.append(mvt_periods)
                        # print(f"span_area_coords {span_area_coords}")
                        span_area_colors.append(colors[i % len(colors)])
                        print(f"  Period {name_period} -> {colors[i]}")
                        i += 1
                elif self.speed_by_frame is not None:
                    span_area_coords = []
                    span_area_colors = []
                    span_area_coords.append(speed_periods)
                    span_area_colors.append("cornflowerblue")
                else:
                    print(f"no mvt info for {self.description}")

            x_ticks_labels = [x for x in frames_to_display if x % 50 == 0]
            x_ticks = [x * 50 for x in np.arange(len(x_ticks_labels))]
            x_ticks_labels_size = 10
            plot_spikes_raster(spike_nums=raster_dur[:, frames_to_display],
                               param=self.param,
                               display_spike_nums=True,
                               axes_list=[axes[0], axes[1]],
                               x_ticks_labels=x_ticks_labels,
                               x_ticks_labels_size=x_ticks_labels_size,
                               x_ticks=x_ticks,
                               display_traces=False,
                               spike_train_format=False,
                               y_ticks_labels=np.arange(n_cells),
                               y_ticks_labels_size=2,
                               save_raster=False,
                               show_raster=False,
                               span_area_coords=span_area_coords,
                               span_area_colors=span_area_colors,
                               alpha_span_area=0.4,
                               plot_with_amplitude=False,
                               # raster_face_color="white",
                               hide_x_labels=True,
                               without_activity_sum=False,
                               show_sum_spikes_as_percentage=True,
                               span_area_only_on_raster=False,
                               spike_shape='|',
                               spike_shape_size=5,
                               save_formats="pdf")

            ax_to_use = axes[2]

            plot_wavelet_heatmap(threshold=threshold,
                                 num_levels_contourf=num_levels_contourf,
                                 log_y=wp.log_y,
                                 display_EEG=True,
                                 sampling_rate=lfp_sampling_rate,
                                 wp=wp,
                                 wav_outcome=wav_outcome, max_v_colorbar=max_color_heatmap,
                                 param=self.param,
                                 file_name=file_name_segment,
                                 plot_freq_band_detection=wp.show_freq_bands,
                                 dpi=dpi, ax_to_use=ax_to_use,
                                 first_x_tick_label=beg_time_s,
                                 levels=None, red_v_bars=None, save_formats=save_formats)

            if isinstance(save_formats, str):
                save_formats = [save_formats]

            for save_format in save_formats:
                fig.savefig(f'{self.param.path_results}/{file_name_segment}'
                            f'_{self.param.time_str}.{save_format}',
                            format=f"{save_format}",
                            facecolor=fig.get_facecolor())
        np.save(f"p{self.age}/{self.description.lower()}/{self.description}_sharp_frames.npy", np.array(spikes_frames),
                allow_pickle=True)

    def create_wavelet_param(self):
        """
        Object used to keep parameters for wavelet analysis on lfp
        :return:
        """

        # Avoid to compute wavelet for the main game
        only_compute_baseline_stats = False
        # compute_freq_bands matters only if bosc_method is at False
        compute_freq_bands = False
        # baseline not used for now
        baseline_duration = 120
        max_baseline_duration = False
        use_derivative = False
        # use_spike_removal = False
        # load spikes if file is available
        # load_spikes = False
        detect_spikes = True

        bosc_method = False
        if bosc_method:
            min_freq = 1
            # used to be 60
            max_freq = 60
            # used to be 0.5
            freq_steps = 0.1
            wav_cycles = 7
            log_y = False
        else:
            # numbers of freq to apply by the wavelets (between min_freq and max-Freq) used to be 1
            min_freq = 1
            # used to be 60
            max_freq = 60
            freq_steps = 0.1
            # used to be 16 then 8
            wav_cycles = 7
            log_y = False

        num_frex = int((max_freq - min_freq) / freq_steps)

        # boolean : define if the number of cycles or the wavelet should increase as the frequency do
        # if True, increase from min_cycle to max_cycle
        # a larger number of cycles gives you better frequency precision at the cost of worse temporal precision,
        # and a smaller number of cycles gives you better temporal precision
        # at the cost of worse frequency precision
        increase_gaussian_cycles = False

        # nb of cycles used if increase_gaussian_cycles = False
        min_cycle = 6
        max_cycle = 15
        welch_method = False

        # 3 baseline_mode choice : "decibel_conversion", "percentage_change", "z_score_normalization"
        baseline_modes = ["decibel_conversion", "percentage_change", "z_score_normalization"]
        baseline_mode = baseline_modes[0]

        # If False, mean + std will be used
        using_median_for_threshold = True
        # Hz gap to remove redondant freq band episode that overlap
        hz_gap_fb_ep = 1
        # it means the algorihtm will go to freq from freq, so 2 times freq_steps
        step_fb_ep = int(1 / freq_steps)
        if bosc_method:
            # spike data
            # how percentage of the height should be superior at the treshold specified
            spike_percentage = 0.85
            # by how should be multiplied the threshold used to display wavelet or identify freq_band in order to identify
            # spike
            spike_threshold_ratio = 1.5
            low_freq_spike_detector = 3
            high_freq_spike_detector = 25
        else:
            spike_percentage = 0.90  # 0.75
            # by how should be multiplied the threshold used to display wavelet or identify freq_band in order to identify
            # spike
            spike_threshold_ratio = 2.5  # 1.05
            low_freq_spike_detector = 5
            high_freq_spike_detector = 50
        # in sec
        spike_removal_time_after = 1
        spike_removal_time_before = 0.5

        show_freq_bands = False

        # frequency band to explore, for corresponding Hz, see Game.TYPE_OF_FREQ_BAND
        freq_band_to_explore = ('delta', 'theta', 'delta_theta', 'low_theta', 'high_theta',
                                'alpha', 'alpha1', 'alpha2', 'theta_alpha',
                                'beta', 'gamma', 'slow_gamma', 'mid_gamma', 'fast_gamma')

        wp = WaveletParameters(baseline_duration=baseline_duration,
                               max_baseline_duration=max_baseline_duration,
                               min_freq=min_freq, max_freq=max_freq,
                               num_frex=num_frex, freq_steps=freq_steps,
                               use_derivative=use_derivative,
                               detect_spikes=detect_spikes,
                               show_freq_bands=show_freq_bands,
                               low_freq_spike_detector=low_freq_spike_detector,
                               high_freq_spike_detector=high_freq_spike_detector,
                               increase_gaussian_cycles=increase_gaussian_cycles,
                               min_cycle=min_cycle, max_cycle=max_cycle, wav_cycles=wav_cycles,
                               baseline_mode=baseline_mode, log_y=log_y,
                               hz_gap_fb_ep=hz_gap_fb_ep, step_fb_ep=step_fb_ep,
                               using_median_for_threshold=using_median_for_threshold,
                               compute_freq_bands=compute_freq_bands,
                               freq_band_to_explore=freq_band_to_explore,
                               spike_percentage=spike_percentage,
                               spike_threshold_ratio=spike_threshold_ratio,
                               spike_removal_time_after=spike_removal_time_after,
                               spike_removal_time_before=spike_removal_time_before,
                               bosc_method=bosc_method,
                               welch_method=welch_method,
                               only_compute_baseline_stats=only_compute_baseline_stats)

        return wp

    def pca_on_suite2p_spks(self):
        if self.suite2p_data is None:
            return

        print(f"starting pca_on_suite2p_spks")
        n_components = 10
        spks = self.suite2p_data["spks"]
        is_cell = self.suite2p_data["is_cell"]
        clean_spks = np.zeros((self.coord_obj.n_cells, spks.shape[1]))
        true_cell_index = 0
        for cell_index in np.arange(spks.shape[0]):
            if is_cell[cell_index][0]:
                clean_spks[true_cell_index] = spks[cell_index]
                true_cell_index += 1
        spks = clean_spks
        n_cells = self.coord_obj.n_cells
        # first we remove the fake cell of spks
        normalize_0_1 = False
        normalize_z_score = True
        if normalize_0_1:
            spks_0_1 = np.zeros(spks.shape)
            for cell_index, spk in enumerate(spks):
                spks_0_1[cell_index] = self.norm01(spk)
            spks = spks_0_1
            vmin = 0
            vmax = 0.5
        elif normalize_z_score:
            spks_z = np.zeros(spks.shape)
            for cell_index, spk in enumerate(spks):
                spks_z[cell_index] = (spk - np.mean(spk)) / np.std(spk)
            spks = spks_z
            spks += np.min(spks)
            vmin = 0.1
            vmax = 3
        else:
            vmin = 0.4
            vmax = 20

        print(f"spks: {np.min(spks)} {np.max(spks)} {np.mean(spks)} {np.std(spks)}")

        pca = PCA(n_components=n_components)  #
        pca_result = pca.fit_transform(spks)

        plot_with_imshow(raster=spks,
                         n_subplots=1,
                         hide_x_labels=True,
                         y_ticks_labels_size=2,
                         y_ticks_labels=np.arange(n_cells),
                         show_color_bar=False,
                         values_to_plot=None, cmap="hot",
                         without_ticks=True,
                         vmin=vmin, vmax=vmax,
                         reverse_order=False,
                         speed_array=None,
                         path_results=self.param.path_results,
                         file_name=f"{self.description}_spks_suite2p",
                         save_formats="pdf"
                         )
        for component in np.arange(n_components):
            # sorted_raster = np.copy(spike_nums_dur)
            indices_sorted = np.argsort(pca_result[:, component])[::-1]
            plot_with_imshow(raster=spks[indices_sorted],
                             n_subplots=1,
                             hide_x_labels=True,
                             y_ticks_labels_size=2,
                             y_ticks_labels=np.arange(n_cells)[indices_sorted],
                             show_color_bar=False,
                             values_to_plot=None, cmap="hot",
                             without_ticks=True,
                             vmin=vmin, vmax=vmax,
                             reverse_order=False,
                             speed_array=None,
                             path_results=self.param.path_results,
                             file_name=f"{self.description}_spks_suite2p_pc_{component + 1}",
                             save_formats="pdf"
                             )

            file_name = f'{self.param.path_results}/{self.description}_spks_suite2p_pc_{component + 1}.npy'

            with open(file_name, "w", encoding='UTF-8') as file:
                for cell in indices_sorted:
                    file.write(f"{indices_sorted[cell]}")

                    if cell != indices_sorted[-1]:
                        file.write(' ')
                # file.write('\n')
            file_name = f'{self.param.path_results}/{self.description}_spks_suite2p_cells_order_pc_{component + 1}.npy'
            np.save(file_name, indices_sorted)

    def pca_on_raster(self, span_area_coords=None, span_area_colors=None):
        print(f"starting PCA")
        # save_formats = ["pdf", "png"]
        save_formats = ["png"]
        use_original_raster = True
        do_pca_on_trace = True
        n_components = 20
        n_cells = self.spike_struct.spike_nums_dur.shape[0]
        n_times = self.spike_struct.spike_nums_dur.shape[1]
        spike_nums_dur_original = np.copy(self.spike_struct.spike_nums_dur)
        # self.normalize_traces()
        traces_0_1 = np.zeros((n_cells, n_times))
        for cell in np.arange(n_cells):
            max_value = np.max(self.raw_traces[cell])
            min_value = np.min(self.raw_traces[cell])
            traces_0_1[cell] = (self.raw_traces[cell] - min_value) / (max_value - min_value)
            # traces_0_1[cell] *= 10000
            # traces_0_1[cell] = self.raw_traces[cell]
        if use_original_raster:
            spike_nums_dur = spike_nums_dur_original
        else:
            spike_nums_dur = np.zeros((n_cells, n_times))
            use_duration = False
            for cell in np.arange(n_cells):
                periods = get_continous_time_periods(np.copy(self.spike_struct.spike_nums_dur[cell]))
                if use_duration:
                    # version with length of transients
                    for period in periods:
                        duration = period[1] - period[0] + 1
                        # if duration == 0:
                        #     print(f"pca_on_raster: duration is O")
                        # if duration >= 8:
                        #     spike_nums_dur[cell, period[0]:period[1]+1] = duration

                        if duration >= 10:
                            duration = 4
                        elif duration < 3:
                            duration = 1
                        elif duration < 6:
                            duration = 2
                        else:
                            duration = 3

                        spike_nums_dur[cell, period[0]:period[1] + 1] = duration
                else:
                    # spike_times = np.where(self.spike_struct.spike_nums_dur[cell])[0]
                    # # spike_nums_dur[cell, spike_times] = self.z_score_traces[cell, spike_times]
                    # spike_nums_dur[cell, spike_times] = traces_0_1[cell, spike_times]
                    for period in periods:
                        duration = period[1] - period[0]
                        # spike_nums_dur[cell, period[0]:period[1] + 1] = np.max(traces_0_1[cell, period[0]:period[1] + 1])
                        spike_nums_dur[cell, period[0]:period[1] + 1] = traces_0_1[cell, period[0]:period[1] + 1]
                        # * duration
        # if use_duration:
        #     # ordering cells so all the one active are at the beginning
        #     index_free = 0
        #     for cell in np.arange(n_cells):
        #         if np.sum(spike_nums_dur[cell]) > 0:
        #             tmp = np.copy(spike_nums_dur[index_free])
        #             spike_nums_dur[index_free] = spike_nums_dur[cell]
        #             spike_nums_dur[cell] = tmp
        #             index_free += 1
        pca = PCA(n_components=n_components)  #
        if do_pca_on_trace:
            pca_result = pca.fit_transform(self.raw_traces)
        else:
            pca_result = pca.fit_transform(spike_nums_dur)
        print(f"pca_result.shape {pca_result.shape}")
        # print(f"pca_result[0,:] {pca_result[0,:]}")
        # print(f"pca_result[1,:] {pca_result[1,:]}")
        print(f"pca.explained_variance_ratio_ {pca.explained_variance_ratio_}")

        plot_spikes_raster(spike_nums=spike_nums_dur, param=self.param,
                           title=f"{self.description}_spike_nums",
                           spike_train_format=False,
                           file_name=f"{self.description}_spike_nums",
                           save_raster=True,
                           show_raster=False,
                           sliding_window_duration=1,
                           show_sum_spikes_as_percentage=False,
                           plot_with_amplitude=(not use_original_raster),
                           save_formats=save_formats,
                           spike_shape="o",
                           span_area_coords=span_area_coords,
                           span_area_colors=span_area_colors,
                           spike_shape_size=0.5,
                           spike_nums_for_activity_sum=spike_nums_dur_original,
                           without_activity_sum=False,
                           size_fig=(15, 6))

        # raw_traces_normlized = np.copy(self.raw_traces)
        # for i in np.arange(len(raw_traces_normlized)):
        #     raw_traces_normlized[i] = (raw_traces_normlized[i] -
        #                                np.mean(raw_traces_normlized[i]) / np.std(raw_traces_normlized[i]))
        #     raw_traces_normlized[i] = self.norm01(raw_traces_normlized[i]) * 5

        for component in np.arange(n_components):
            # sorted_raster = np.copy(spike_nums_dur)
            indices_sorted = np.argsort(pca_result[:, component])[::-1]
            plot_spikes_raster(spike_nums=spike_nums_dur[indices_sorted, :], param=self.param,
                               title=f"{self.description}_spike_nums_pc_{component}_ordered",
                               spike_train_format=False,
                               file_name=f"{self.description}_spike_nums_pc_{component}_ordered",
                               y_ticks_labels=indices_sorted,
                               save_raster=True,
                               show_raster=False,
                               sliding_window_duration=1,
                               show_sum_spikes_as_percentage=False,
                               plot_with_amplitude=(not use_original_raster),
                               save_formats=save_formats,
                               spike_shape="o",
                               span_area_coords=span_area_coords,
                               span_area_colors=span_area_colors,
                               spike_shape_size=0.5,
                               spike_nums_for_activity_sum=self.spike_struct.spike_nums_dur[indices_sorted, :],
                               without_activity_sum=False,
                               size_fig=(15, 6), dpi=300)
            n_cells_to_display = 40
            cells_groups_colors = [(1, 0, 0, 1)]
            cells_groups = []
            cells_groups.append(list(indices_sorted[:n_cells_to_display]))
            # cells_groups.append(list(indices_sorted[n_cells_to_display:]))

            self.coord_obj.plot_cells_map(param=self.param,
                                          data_id=self.description, show_polygons=False,
                                          fill_polygons=False,
                                          title_option=f"pc_{component}", connections_dict=None,
                                          cells_groups=cells_groups,
                                          cells_groups_colors=cells_groups_colors,
                                          dont_fill_cells_not_in_groups=False,
                                          with_cell_numbers=True, save_formats=save_formats)

            # plot_spikes_raster(spike_nums=spike_nums_dur[indices_sorted[:n_cells_to_display], :], param=self.param,
            #                    traces=raw_traces_normlized[indices_sorted[:n_cells_to_display], :],
            #                    display_traces=True,
            #                    spike_train_format=False,
            #                    without_activity_sum=True,
            #                    title="pc_{component} traces raster",
            #                    file_name="pc_{component} traces raster",
            #                    y_ticks_labels=np.arange(n_cells_to_display),
            #                    y_ticks_labels_size=2,
            #                    save_raster=True,
            #                    show_raster=False,
            #                    plot_with_amplitude=False,
            #                    # raster_face_color="white",
            #                    # vertical_lines=SCE_times,
            #                    # vertical_lines_colors=['white'] * len(SCE_times),
            #                    # vertical_lines_sytle="solid",
            #                    # vertical_lines_linewidth=[0.2] * len(SCE_times),
            #                    span_area_only_on_raster=False,
            #                    spike_shape_size=0.5,
            #                    save_formats="png")

    def norm01(self, data):
        min_value = np.min(data)
        max_value = np.max(data)

        difference = max_value - min_value

        data -= min_value

        if difference > 0:
            data = data / difference

        return data

    def plot_raster_with_cells_assemblies_events_and_mvts(self):
        if self.sce_times_in_cell_assemblies is None:
            return

        cells_to_highlight = []
        cells_to_highlight_colors = []

        n_cell_assemblies = len(self.cell_assemblies)
        for cell_assembly_index, cell_assembly in enumerate(self.cell_assemblies):
            color = cm.nipy_spectral(float(cell_assembly_index + 1) / (n_cell_assemblies + 1))
            cell_indices_to_color = []
            for cell in cell_assembly:
                cell_indices_to_color.append(cell)
            cells_to_highlight.extend(cell_indices_to_color)
            cells_to_highlight_colors.extend([color] * len(cell_indices_to_color))

        span_areas_coords = []
        span_area_colors = []
        show_z_shift = True
        if show_z_shift and len(self.z_shift_periods) > 0:
            span_areas_coords.append(self.z_shift_periods)
            span_area_colors.append("orange")
        else:
            span_areas_coords.append(self.mvt_frames_periods)
            span_area_colors.append('red')
            span_areas_coords.append(self.sce_times_in_cell_assemblies)
            span_area_colors.append('green')
            span_areas_coords.append(self.twitches_frames_periods)
            span_area_colors.append('blue')

        ## ratio
        for cell_assembly_index in np.arange(-1, len(self.cell_assemblies)):
            if self.mvt_frames_periods is None:
                continue

            if cell_assembly_index == -1:
                this_sce_times_in_cell_assemblies = self.sce_times_in_cell_assemblies
            else:
                this_sce_times_in_cell_assemblies = self.sce_times_in_single_cell_assemblies[cell_assembly_index]

            n_times = len(self.spike_struct.spike_nums_dur[0, :])
            mvt_frames_bool = np.zeros(n_times, dtype="bool")
            twtich_frames_bool = np.zeros(n_times, dtype="bool")

            n_sce_in_mvt = 0
            n_sce_in_twitch = 0
            n_sce = len(this_sce_times_in_cell_assemblies)
            n_sce_rest = 0

            for mvt_period in self.mvt_frames_periods:
                mvt_frames_bool[mvt_period[0]:mvt_period[1] + 1] = True

            for twitch_period in self.twitches_frames_periods:
                twtich_frames_bool[twitch_period[0]:twitch_period[1] + 1] = True

            for sce_period in this_sce_times_in_cell_assemblies:
                if np.any(mvt_frames_bool[sce_period[0]:sce_period[1] + 1]):
                    if np.any(twtich_frames_bool[sce_period[0]:sce_period[1] + 1]):
                        n_sce_in_twitch += 1
                    else:
                        n_sce_in_mvt += 1
                else:
                    n_sce_rest += 1
            bonus_str = "for all cell assemblies"
            if cell_assembly_index > -1:
                bonus_str = f" for cell_assembly n° {cell_assembly_index}"
            print(f"Cell assemblies sce in {self.description}{bonus_str}")
            print(f"n_sce {n_sce}")
            print(f"n_sce_rest {n_sce_rest}")
            print(f"n_sce_in_mvt {n_sce_in_mvt}")
            print(f"n_sce_in_twitch {n_sce_in_twitch}")

        labels = np.arange(len(self.spike_struct.spike_nums_dur))
        plot_spikes_raster(spike_nums=self.spike_struct.spike_nums_dur, param=self.param,
                           title=f"{self.description}_spike_nums_with_mvt_and_cell_assemblies_events",
                           spike_train_format=False,
                           file_name=f"{self.description}_spike_nums_with_mvt_and_cell_assemblies_events",
                           y_ticks_labels=labels,
                           save_raster=True,
                           show_raster=False,
                           sliding_window_duration=1,
                           show_sum_spikes_as_percentage=True,
                           plot_with_amplitude=False,
                           save_formats="pdf",
                           activity_threshold=self.activity_threshold,
                           # cells_to_highlight=cells_to_highlight,
                           # cells_to_highlight_colors=cells_to_highlight_colors,
                           spike_shape="o",
                           spike_shape_size=1,
                           span_area_coords=span_areas_coords,
                           span_area_colors=span_area_colors,
                           span_area_only_on_raster=False,
                           without_activity_sum=False,
                           size_fig=(15, 6))

    def plot_traces_with_shifts(self, organized_by_cell_assemblies=True):
        if self.sce_times_in_cell_assemblies is None:
            organized_by_cell_assemblies = False

        new_cell_order = np.arange(len(self.raw_traces))
        y_ticks_labels_color = "white"
        if organized_by_cell_assemblies:
            y_ticks_labels_color = []
            n_cells = len(self.spike_struct.spike_nums_dur)
            n_cell_assemblies = len(self.cell_assemblies)
            n_cells_in_assemblies = 0
            for cell_assembly in self.cell_assemblies:
                n_cells_in_assemblies += len(cell_assembly)

            new_cell_order = np.zeros(n_cells, dtype="uint16")

            cells_in_assemblies = []
            last_group_index = 0
            for cell_assembly_index, cell_assembly in enumerate(self.cell_assemblies):
                color = cm.nipy_spectral(float(cell_assembly_index + 1) / (n_cell_assemblies + 1))
                y_ticks_labels_color.extend([color] * len(cell_assembly))

                new_cell_order[last_group_index:last_group_index + len(cell_assembly)] = \
                    np.array(cell_assembly).astype("uint16")
                last_group_index += len(cell_assembly)
                cells_in_assemblies.extend(list(cell_assembly))

            other_cells = np.setdiff1d(np.arange(n_cells), cells_in_assemblies)
            y_ticks_labels_color.extend([(1, 1, 1, 1.0)] * len(other_cells))
            new_cell_order[last_group_index:] = other_cells

        shifts = np.abs(self.x_shifts) + np.abs(self.y_shifts)
        # shifts = signal.detrend(shifts)
        # normalization
        shifts = (shifts - np.mean(shifts)) / np.std(shifts)
        if np.min(shifts) < 0:
            shifts -= np.min(shifts)

        def norm01(data):
            min_value = np.min(data)
            max_value = np.max(data)

            difference = max_value - min_value

            data -= min_value

            if difference > 0:
                data = data / difference

            return data

        raw_traces = np.copy(self.raw_traces)
        for i in np.arange(len(raw_traces)):
            raw_traces[i] = (raw_traces[i] - np.mean(raw_traces[i]) / np.std(raw_traces[i]))
            raw_traces[i] = norm01(raw_traces[i])
            raw_traces[i] = norm01(raw_traces[i]) * 5

        plot_spikes_raster(param=self.param,
                           spike_train_format=False,
                           display_spike_nums=False,
                           traces=raw_traces,
                           display_traces=True,
                           y_ticks_labels_color=y_ticks_labels_color,
                           use_brewer_colors_for_traces=True,
                           file_name=f"{self.description}_traces_shift",
                           y_ticks_labels=new_cell_order,
                           y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           without_activity_sum=False,
                           spikes_sum_to_use=shifts,
                           span_area_only_on_raster=False,
                           traces_lw=0.1,
                           save_formats=["pdf", "png"])

    def plot_traces_on_raster(self, spike_nums_to_use=None, sce_times=None, with_run=True,
                              display_spike_nums=False, cellsinpeak=None,
                              order_with_cell_assemblies=False):
        # cellsinpeak: 2d array, binary, 1st dim matches the
        # number of cells, 2nd dim matches the size of sce_times
        def norm01(data):
            min_value = np.min(data)
            max_value = np.max(data)

            difference = max_value - min_value

            data -= min_value

            if difference > 0:
                data = data / difference

            return data

        if spike_nums_to_use is None:
            display_spike_nums = False
        if not display_spike_nums:
            without_activity_sum = True
        else:
            without_activity_sum = False

        raw_traces = self.raw_traces

        for i in np.arange(len(raw_traces)):
            raw_traces[i] = (raw_traces[i] - np.mean(raw_traces[i]) / np.std(raw_traces[i]))
            raw_traces[i] = norm01(raw_traces[i]) * 5

        n_cells = len(raw_traces)

        new_cell_order = np.zeros(n_cells, dtype="uint16")
        if order_with_cell_assemblies and (self.cell_assemblies is not None):
            cells_in_assemblies = []
            last_group_index = 0
            for cell_assembly_index, cell_assembly in enumerate(self.cell_assemblies):
                new_cell_order[last_group_index:last_group_index + len(cell_assembly)] = \
                    np.array(cell_assembly).astype("uint16")
                last_group_index += len(cell_assembly)
                cells_in_assemblies.extend(list(cell_assembly))

            other_cells = np.setdiff1d(np.arange(n_cells), cells_in_assemblies)
            new_cell_order[last_group_index:] = other_cells
        else:
            new_cell_order = np.arange(n_cells)

        raw_traces = raw_traces[new_cell_order]
        spike_nums_to_use = spike_nums_to_use[new_cell_order]
        if cellsinpeak is not None:
            cellsinpeak = cellsinpeak[new_cell_order]

        span_area_coords = []
        span_area_colors = []
        vertical_lines = None
        vertical_lines_colors = None
        vertical_lines_sytle = None
        vertical_lines_linewidth = None
        if sce_times is not None:
            vertical_lines = sce_times
            vertical_lines_colors = ['white'] * len(sce_times)
            vertical_lines_sytle = "solid"
            vertical_lines_linewidth = [0.2] * len(sce_times)
        if (self.speed_by_frame is not None) and with_run:
            binary_speed = np.zeros(len(self.speed_by_frame), dtype="int8")
            binary_speed[self.speed_by_frame > 1] = 1
            speed_periods = get_continous_time_periods(binary_speed)
            span_area_coords.append(speed_periods)
            span_area_colors.append("red")

        scatters_on_traces = None
        if (cellsinpeak is not None) and (sce_times is not None):
            scatters_on_traces = np.zeros(raw_traces.shape, dtype="int8")
            for sce_index, sce_period in enumerate(sce_times):
                frame_index = (sce_period[0] + sce_period[1]) // 2
                scatters_on_traces[:, frame_index] = cellsinpeak[:, sce_index]

        plot_spikes_raster(spike_nums=spike_nums_to_use, param=self.param,
                           display_spike_nums=display_spike_nums,
                           traces_lw=0.1,
                           traces=raw_traces,
                           display_traces=True,
                           spike_train_format=False,
                           title="traces raster",
                           file_name="traces raster",
                           y_ticks_labels=np.arange(len(raw_traces)),
                           y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           span_area_coords=span_area_coords,
                           span_area_colors=span_area_colors,
                           vertical_lines=vertical_lines,
                           vertical_lines_colors=vertical_lines_colors,
                           vertical_lines_sytle=vertical_lines_sytle,
                           vertical_lines_linewidth=vertical_lines_linewidth,
                           plot_with_amplitude=False,
                           raster_face_color="black",
                           show_sum_spikes_as_percentage=True,
                           span_area_only_on_raster=False,
                           spike_shape_size=0.05,
                           scatters_on_traces=scatters_on_traces,
                           scatters_on_traces_marker="*",
                           scatters_on_traces_size=0.2,
                           without_activity_sum=without_activity_sum,
                           save_formats=["png", "pdf"], dpi=500)
        # raise Exception("NOT TODAY")

    def square_coord_around_cell(self, cell, size_square, x_len_max, y_len_max):
        """
        For a given cell, give the coordinates of the square surrounding the cell.

        :param cell:
        :param size_square:
        :param x_len_max:
        :param y_len_max:
        :return: (x_beg, x_end, y_beg, y_end)
        """
        c_x, c_y = self.coord_obj.center_coord[cell]
        # c_y correspond to
        c_y = int(c_y)
        c_x = int(c_x)
        # print(f"len_x {len_x} len_y {len_y}")
        # print(f"c_x {c_x} c_y {c_y}")
        # limit of the new frame, should make a square
        x_beg_movie = max(0, c_x - (size_square // 2))
        x_end_movie = min(x_len_max, c_x + (size_square // 2) + 1)
        # means the cell is near a border
        if (x_end_movie - x_beg_movie) < (size_square + 1):
            if (c_x - x_beg_movie) < (x_end_movie - c_x - 1):
                x_end_movie += ((size_square + 1) - (x_end_movie - x_beg_movie))
            else:
                x_beg_movie -= ((size_square + 1) - (x_end_movie - x_beg_movie))

        y_beg_movie = max(0, c_y - (size_square // 2))
        y_end_movie = min(y_len_max, c_y + (size_square // 2) + 1)
        if (y_end_movie - y_beg_movie) < (size_square + 1):
            if (c_y - y_beg_movie) < (y_end_movie - c_y - 1):
                y_end_movie += ((size_square + 1) - (y_end_movie - y_beg_movie))
            else:
                y_beg_movie -= ((size_square + 1) - (y_end_movie - y_beg_movie))

        return x_beg_movie, x_end_movie, y_beg_movie, y_end_movie

    def produce_cell_assemblies_verification(self):
        """
        Produce a movie for each sce
        :return:
        """
        self.load_tiff_movie_in_memory()
        len_x = self.tiff_movie.shape[2]
        len_y = self.tiff_movie.shape[1]
        size_square = 80
        n_times = self.tiff_movie.shape[0]

        produce_movies = False
        produce_cell_transients_profile = True
        if produce_movies:
            # number of frames to display before and after SCE
            time_around = 10
            path_results = self.param.path_results
            # plotting cells map
            cells_groups = []
            cells_groups_edge_colors = []
            cells_groups_alpha = []
            cells_groups_colors = []
            cells_groups_alpha.append(0.5)
            # white, http://doc.instantreality.org/tools/color_calculator/
            # white: 1, 1, 1
            # red: 1, 0, 0
            cells_groups_edge_colors.append((1, 1, 1, 1.0))
            cells_groups_colors.append((1, 0, 0, 1.0))
            cells_groups.append(list(self.sce_times_in_cell_assemblies_by_cell.keys()))
            avg_cell_map_img = np.mean(self.tiff_movie, axis=0)
            fig = self.coord_obj.plot_cells_map(param=self.param,
                                                data_id="", show_polygons=False,
                                                fill_polygons=False,
                                                connections_dict=None,
                                                cells_groups=cells_groups,
                                                img_on_background=avg_cell_map_img,
                                                cells_groups_colors=cells_groups_colors,
                                                cells_groups_edge_colors=cells_groups_edge_colors,
                                                with_edge=True, cells_groups_alpha=cells_groups_alpha,
                                                dont_fill_cells_not_in_groups=True,
                                                with_cell_numbers=True, save_formats=["png", "pdf"],
                                                save_plot=True, return_fig=False)

            for cell, sce_times_list in self.sce_times_in_cell_assemblies_by_cell.items():
                x_beg_movie, x_end_movie, y_beg_movie, y_end_movie = \
                    self.square_coord_around_cell(cell=cell, size_square=size_square,
                                                  x_len_max=len_x, y_len_max=len_y)
                for sce_times in sce_times_list:
                    file_name = os.path.join(path_results, f"ca_{cell}_{sce_times[0]}_{sce_times[1]}.tiff")
                    first_frame = max(0, sce_times[0] - time_around)
                    last_frame = min(sce_times[1] + time_around + 1, n_times)
                    with tifffile.TiffWriter(file_name) as tiff_writer:
                        for frame in np.arange(first_frame, last_frame):
                            frame_tiff = self.tiff_movie[frame]
                            tiff_array = frame_tiff[y_beg_movie:y_end_movie,
                                         x_beg_movie:x_end_movie]
                            tiff_writer.save(tiff_array, compress=0)

        if produce_cell_transients_profile:
            c_map = plt.get_cmap('gray')
            # if key_cmap is not None:
            #     if key_cmap is "P":
            #         c_map = self.parula_map
            #     if key_cmap is "B":
            #         c_map = plt.get_cmap('Blues')
            cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
                       [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
                       [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952,
                                                              0.779247619], [0.1252714286, 0.3242428571, 0.8302714286],
                       [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238,
                                                                    0.8819571429],
                       [0.0059571429, 0.4086142857, 0.8828428571],
                       [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571,
                                                              0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429],
                       [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667,
                                                                    0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
                       [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571,
                                                                   0.8262714286],
                       [0.0640571429, 0.5569857143, 0.8239571429],
                       [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524,
                                                                    0.819852381], [0.0265, 0.6137, 0.8135],
                       [0.0238904762, 0.6286619048,
                        0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
                       [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381,
                                                                    0.7607190476],
                       [0.0383714286, 0.6742714286, 0.743552381],
                       [0.0589714286, 0.6837571429, 0.7253857143],
                       [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
                       [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429,
                                                                    0.6424333333],
                       [0.2178285714, 0.7250428571, 0.6192619048],
                       [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619,
                                                                    0.5711857143],
                       [0.3481666667, 0.7424333333, 0.5472666667],
                       [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524,
                                                              0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905],
                       [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476,
                                                                    0.4493904762],
                       [0.609852381, 0.7473142857, 0.4336857143],
                       [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
                       [0.7184095238, 0.7411333333, 0.3904761905],
                       [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667,
                                                              0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
                       [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
                       [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857,
                                                                    0.2886428571],
                       [0.9738952381, 0.7313952381, 0.266647619],
                       [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857,
                                                                   0.2164142857],
                       [0.9955333333, 0.7860571429, 0.196652381],
                       [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
                       [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
                       [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333,
                                                              0.0948380952], [0.9661, 0.9514428571, 0.0755333333],
                       [0.9763, 0.9831, 0.0538]]

            parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
            c_map = parula_map
            all_correlations = []
            for cell, sce_times_list in self.sce_times_in_cell_assemblies_by_cell.items():
                sources_profile_fig = plt.figure(figsize=(20, 20),
                                                 subplotpars=SubplotParams(hspace=0, wspace=0))
                fig_patch = sources_profile_fig.patch
                rgba = c_map(0)
                fig_patch.set_facecolor(rgba)

                # now adding as many suplots as needed, depending on how many transients has the cell
                n_profiles = len(sce_times_list) + 1
                n_profiles_by_row = 4
                n_columns = n_profiles_by_row
                width_ratios = [100 // n_columns] * n_columns
                n_lines = (((n_profiles - 1) // n_columns) + 1) * 2
                height_ratios = [100 // n_lines] * n_lines
                grid_spec = gridspec.GridSpec(n_lines, n_columns, width_ratios=width_ratios,
                                              height_ratios=height_ratios,
                                              figure=sources_profile_fig, wspace=0, hspace=0)
                profiles_to_display = []
                correlations = []
                source_profile, minx, miny, mask_source_profile = \
                    self.coord_obj.get_source_profile(tiff_movie=self.tiff_movie, traces=self.raw_traces,
                                                      peak_nums=self.spike_struct.peak_nums,
                                                      spike_nums=self.spike_struct.spike_nums,
                                                      cell=cell,
                                                      pixels_around=1,
                                                      bounds=None)
                # normalizing
                source_profile = source_profile - np.mean(source_profile)
                profiles_to_display.append(source_profile)
                # we want the mask to be at ones over the cell
                mask_source_profile = (1 - mask_source_profile).astype(bool)

                xy_source = self.coord_obj.get_cell_new_coord_in_source(cell=cell,
                                                                        minx=minx, miny=miny)
                for sce_times in sce_times_list:
                    sce_times[0] = max(0, sce_times[0] - 2)
                    sce_times[1] = min(n_times, sce_times[1] + 2)
                    transient_profile, minx, miny = self.coord_obj.get_transient_profile(cell=cell, transient=sce_times,
                                                                                         tiff_movie=self.tiff_movie,
                                                                                         traces=self.raw_traces,
                                                                                         pixels_around=1, bounds=None)

                    transient_profile = transient_profile - np.mean(transient_profile)
                    profiles_to_display.append(transient_profile)
                    pearson_corr, pearson_p_value = stats.pearsonr(source_profile[mask_source_profile],
                                                                   transient_profile[mask_source_profile])
                    correlations.append(pearson_corr)
                if np.NaN not in correlations:
                    all_correlations.extend(correlations)
                for index_profile, profile_to_display in enumerate(profiles_to_display):
                    line_gs = (index_profile // n_columns) * 2
                    col_gs = index_profile % n_columns

                    ax = sources_profile_fig.add_subplot(grid_spec[line_gs, col_gs])
                    # ax_source_profile_by_cell[cell_to_display].set_facecolor("black")
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.get_yaxis().set_visible(False)
                    ax.get_xaxis().set_visible(False)
                    ax.set_frame_on(False)

                    img_src_profile = ax.imshow(profile_to_display, cmap=c_map)

                    lw = 0.2
                    contour_cell = patches.Polygon(xy=xy_source,
                                                   fill=False,
                                                   edgecolor="red",
                                                   zorder=15, lw=lw)
                    ax.add_patch(contour_cell)
                    if index_profile > 0:
                        ax.text(x=0.5, y=0.5, s=f"{np.round(correlations[index_profile - 1], 2)}",
                                color="red", zorder=20,
                                ha='center', va="center", fontsize=12,
                                fontweight='bold')

                save_formats = ["pdf"]
                file_name = f"source_transient_profiles_{cell}"
                for save_format in save_formats:
                    sources_profile_fig.savefig(f'{self.param.path_results}/'
                                                f'{self.description}_{file_name}'
                                                f'.{save_format}',
                                                format=f"{save_format}",
                                                facecolor=sources_profile_fig.get_facecolor(), edgecolor='none')
                plt.close()

            plot_box_plots(data_dict={"": all_correlations}, title="",
                           filename=f"{self.description}_source_transient_profiles_{len(all_correlations)}correlations_in_cell_assemblies",
                           y_label=f"correlation", y_log=False,
                           x_labels_rotation=45,
                           colors="cornflowerblue", with_scatters=True,
                           path_results=self.param.path_results, scatter_size=80,
                           param=self.param, save_formats="pdf")

    def plot_raster_with_cells_assemblies_and_shifts(self, only_cell_assemblies=False):
        if self.sce_times_in_cell_assemblies is None:
            return

        cells_to_highlight = []
        cells_to_highlight_colors = []
        n_cells = len(self.spike_struct.spike_nums_dur)
        n_cell_assemblies = len(self.cell_assemblies)
        n_cells_in_assemblies = 0
        for cell_assembly in self.cell_assemblies:
            n_cells_in_assemblies += len(cell_assembly)

        # if only_cell_assemblies:
        #     new_cell_order = np.zeros(n_cells_in_assemblies, dtype="uint16")
        # else:
        #     new_cell_order = np.zeros(n_cells, dtype="uint16")
        new_cell_order = np.zeros(n_cells, dtype="uint16")

        cells_in_assemblies = []
        last_group_index = 0
        for cell_assembly_index, cell_assembly in enumerate(self.cell_assemblies):
            color = cm.nipy_spectral(float(cell_assembly_index + 1) / (n_cell_assemblies + 1))
            cell_indices_to_color = []
            new_cell_order[last_group_index:last_group_index + len(cell_assembly)] = \
                np.array(cell_assembly).astype("uint16")
            cell_indices_to_color = list(range(last_group_index, last_group_index + len(cell_assembly)))
            cells_to_highlight.extend(cell_indices_to_color)
            cells_to_highlight_colors.extend([color] * len(cell_indices_to_color))
            last_group_index += len(cell_assembly)
            cells_in_assemblies.extend(list(cell_assembly))

        other_cells = np.setdiff1d(np.arange(n_cells), cells_in_assemblies)
        new_cell_order[last_group_index:] = other_cells
        spike_nums_dur = self.spike_struct.spike_nums_dur[new_cell_order, :]
        if only_cell_assemblies:
            spike_nums_dur = spike_nums_dur[:n_cells_in_assemblies]

        span_area_coords = None
        span_area_colors = None
        if self.SCE_times is not None:
            span_area_coords = []
            span_area_colors = []
            span_area_coords.append(self.SCE_times)
            span_area_colors.append('red')

        shifts = np.abs(self.x_shifts) + np.abs(self.y_shifts)
        # shifts = signal.detrend(shifts)
        # normalization
        shifts = (shifts - np.mean(shifts)) / np.std(shifts)
        if np.min(shifts) < 0:
            shifts -= np.min(shifts)
        if only_cell_assemblies:
            labels = new_cell_order[:n_cells_in_assemblies]
        else:
            labels = new_cell_order

        show_z_shift = True
        if show_z_shift and len(self.z_shift_periods) > 0:
            if span_area_coords is None:
                span_area_coords = []
                span_area_colors = []
            span_area_coords.append(self.z_shift_periods)
            span_area_colors.append("orange")

        plot_spikes_raster(spike_nums=spike_nums_dur, param=self.param,
                           title=f"{self.description}_spike_nums_with_mvt_and_cell_assemblies_events",
                           spike_train_format=False,
                           file_name=f"{self.description}_spike_nums_with_mvt_and_cell_assemblies_events",
                           y_ticks_labels=labels,
                           save_raster=True,
                           show_raster=False,
                           sliding_window_duration=1,
                           show_sum_spikes_as_percentage=True,
                           plot_with_amplitude=False,
                           save_formats=["pdf", "png"],
                           cells_to_highlight=cells_to_highlight,
                           cells_to_highlight_colors=cells_to_highlight_colors,
                           span_area_coords=span_area_coords,
                           span_area_colors=span_area_colors,
                           spike_shape="o",
                           spike_shape_size=1,
                           span_area_only_on_raster=False,
                           without_activity_sum=False,
                           spikes_sum_to_use=shifts,
                           size_fig=(15, 5))

    def plot_raster_with_periods(self, periods_dict, bonus_title="",
                                 with_periods=True, with_cell_assemblies=True, only_cell_assemblies=False):
        if self.spike_struct.spike_nums_dur is None:
            return

        if self.sce_times_in_cell_assemblies is None:
            with_cell_assemblies = False

        cells_to_highlight = []
        cells_to_highlight_colors = []
        if with_cell_assemblies:
            n_cells = len(self.spike_struct.spike_nums_dur)
            print(f"n_cells {n_cells}")
            n_cell_assemblies = len(self.cell_assemblies)
            n_cells_in_assemblies = 0
            for cell_assembly in self.cell_assemblies:
                n_cells_in_assemblies += len(cell_assembly)

            # if only_cell_assemblies:
            #     new_cell_order = np.zeros(n_cells_in_assemblies, dtype="uint16")
            # else:
            #     new_cell_order = np.zeros(n_cells, dtype="uint16")
            new_cell_order = np.zeros(n_cells, dtype="uint16")

            cells_in_assemblies = []
            last_group_index = 0
            for cell_assembly_index, cell_assembly in enumerate(self.cell_assemblies):
                color = cm.nipy_spectral(float(cell_assembly_index + 1) / (n_cell_assemblies + 1))
                new_cell_order[last_group_index:last_group_index + len(cell_assembly)] = \
                    np.array(cell_assembly).astype("uint16")
                cell_indices_to_color = list(range(last_group_index, last_group_index + len(cell_assembly)))
                cells_to_highlight.extend(cell_indices_to_color)
                cells_to_highlight_colors.extend([color] * len(cell_indices_to_color))
                last_group_index += len(cell_assembly)
                cells_in_assemblies.extend(list(cell_assembly))

            other_cells = np.setdiff1d(np.arange(n_cells), cells_in_assemblies)
            new_cell_order[last_group_index:] = other_cells
            spike_nums_dur = self.spike_struct.spike_nums_dur[new_cell_order, :]
            if only_cell_assemblies:
                spike_nums_dur = spike_nums_dur[:n_cells_in_assemblies]
                labels = new_cell_order[:n_cells_in_assemblies]
            else:
                labels = new_cell_order
        else:
            labels = np.arange(len(self.spike_struct.spike_nums_dur))
            spike_nums_dur = self.spike_struct.spike_nums_dur

        # span_area_coords = None
        # span_area_colors = None
        if self.speed_by_frame is not None:
            binary_speed = np.zeros(len(self.speed_by_frame), dtype="int8")
            binary_speed[self.speed_by_frame > 1] = 1
            speed_periods = get_continous_time_periods(binary_speed)

        # colors for movement periods
        span_area_coords = None
        span_area_colors = None
        with_mvt_periods = True

        if with_mvt_periods:
            colors = ["red", "green", "blue", "pink", "orange"]
            i = 0
            span_area_coords = []
            span_area_colors = []

            if self.speed_by_frame is not None:
                span_area_coords = []
                span_area_colors = []
                span_area_coords.append(speed_periods)
                span_area_colors.append("cornflowerblue")
            elif with_periods and periods_dict is not None:
                print(f"{self.description}:")
                for name_period, period in periods_dict.items():
                    span_area_coords.append(get_continous_time_periods(period.astype("int8")))
                    span_area_colors.append(colors[i % len(colors)])
                    print(f"  Period {name_period} -> {colors[i]}")
                    i += 1
            else:
                print(f"no mvt info for {self.description}")

        # colors = ["red", "green", "blue", "pink", "orange"]
        # i = 0
        # span_area_coords = []
        # span_area_colors = []
        # if with_periods and (periods_dict is not None):
        #     for name_period, period in periods_dict.items():
        #         span_area_coords.append(get_continous_time_periods(period.astype("int8")))
        #         span_area_colors.append(colors[i % len(colors)])
        #         print(f"Period {name_period} -> {colors[i]}")
        #         i += 1

        if len(spike_nums_dur) < 200:
            spike_shape_size = 0.3
        elif len(spike_nums_dur) < 500:
            spike_shape_size = 0.05
        else:
            spike_shape_size = 0.02
        plot_spikes_raster(spike_nums=spike_nums_dur, param=self.param,
                           title=f"{self.description}_spike_nums_periods",
                           spike_train_format=False,
                           file_name=f"{self.description}_spike_nums_periods{bonus_title}",
                           y_ticks_labels=labels,
                           save_raster=True,
                           show_raster=False,
                           sliding_window_duration=1,
                           show_sum_spikes_as_percentage=False,
                           plot_with_amplitude=False,
                           save_formats=["pdf", "png"],
                           cells_to_highlight=cells_to_highlight,
                           cells_to_highlight_colors=cells_to_highlight_colors,
                           span_area_coords=span_area_coords,
                           span_area_colors=span_area_colors,
                           spike_shape="o",
                           spike_shape_size=spike_shape_size,
                           span_area_only_on_raster=False,
                           without_activity_sum=False,
                           activity_threshold=self.activity_threshold,
                           size_fig=(15, 5))

    def plot_each_inter_neuron_connect_map(self):
        # plot n_in and n_out map of the interneurons
        inter_neurons = self.spike_struct.inter_neurons
        n_inter_neurons = len(inter_neurons)
        if n_inter_neurons == 0:
            return

        for inter_neuron in inter_neurons:
            self.plot_connectivity_maps_of_a_cell(cell_to_map=inter_neuron, cell_descr="inter_neuron")

    def load_suite2p_data(self, data_path, with_coord=False):
        if not os.path.isdir(os.path.join(self.param.path_data, data_path)):
            print(f"Suite 2p data could not be loaded for {self.description}, "
                  f"the directory {os.path.join(self.param.path_data, data_path)}: doesn't exist")
            return

        if self.suite2p_data is not None:
            # already loaded
            return

        self.suite2p_data = dict()
        # commented due to mesocentre
        # if os.path.isfile(os.path.join(self.param.path_data, data_path, 'F.npy')):
        #     f = np.load(os.path.join(self.param.path_data, data_path, 'F.npy'), allow_pickle=True)
        #     self.suite2p_data["F"] = f
        # if os.path.isfile(os.path.join(self.param.path_data, data_path, 'Fneu.npy')):
        #     f_neu = np.load(os.path.join(self.param.path_data, data_path, 'Fneu.npy'), allow_pickle=True)
        #     self.suite2p_data["Fneu"] = f_neu
        # if os.path.isfile(os.path.join(self.param.path_data, data_path, 'spks.npy')):
        #     spks = np.load(os.path.join(self.param.path_data, data_path, 'spks.npy'), allow_pickle=True)
        #     self.suite2p_data["spks"] = spks
        # print(f"spks.shape {spks}")

        stat = np.load(os.path.join(self.param.path_data, data_path, 'stat.npy'), allow_pickle=True)
        self.suite2p_data["stat"] = stat
        # print(f"len(stat) {len(stat)}")
        # stat = stat[0]

        # if os.path.isfile(os.path.join(self.param.path_data, data_path, 'ops.npy')):
        #     ops = np.load(os.path.join(self.param.path_data, data_path, 'ops.npy'), allow_pickle=True)
        #     ops = ops.item()
        #     self.suite2p_data["ops"] = ops

        is_cell = np.load(os.path.join(self.param.path_data, data_path, 'iscell.npy'), allow_pickle=True)
        self.suite2p_data["is_cell"] = is_cell
        # print(f"len(is_cell) {len(is_cell)}")

        # print(f"stat.keys() {list(stat.keys())}")
        # print(f"stat['lam'] {len(stat['lam'])}: {stat['lam']}")
        # print(f"stat['lam'] : {stat['footprint']}")

        use_first_version = False
        if use_first_version:
            coord = []
            for cell in np.arange(len(stat)):
                # print(f"is_cell[cell] {is_cell[cell]}")
                if is_cell[cell][0] == 0:
                    continue
                x_list = []
                y_list = []
                npx = stat[cell]["npix"]
                x_unique = np.unique(stat[cell]["xpix"])
                y_unique = np.unique(stat[cell]["ypix"])
                indices_selected = []
                for x in x_unique:
                    x_indices = np.where(stat[cell]["xpix"] == x)[0]
                    y_pos = stat[cell]["ypix"][x_indices]
                    index_max = x_indices[np.argmax(y_pos)]
                    indices_selected.append(index_max)
                    x_list.append(stat[cell]["xpix"][index_max])
                    y_list.append(stat[cell]["ypix"][index_max])
                    index_min = x_indices[np.argmin(y_pos)]
                    if index_min != index_max:
                        indices_selected.append(index_min)
                        x_list.append(stat[cell]["xpix"][index_min])
                        y_list.append(stat[cell]["ypix"][index_min])

                for y in y_unique:
                    y_indices = np.where(stat[cell]["ypix"] == y)[0]
                    x_pos = stat[cell]["xpix"][y_indices]
                    index_max = y_indices[np.argmax(x_pos)]
                    # indices_selected.append(index_max)
                    if index_max not in indices_selected:
                        indices_selected.append(index_max)
                        y_list.append(stat[cell]["ypix"][index_max])
                        x_list.append(stat[cell]["xpix"][index_max])
                    index_min = y_indices[np.argmin(x_pos)]
                    if (index_min not in indices_selected) and (index_min != index_max):
                        indices_selected.append(index_max)
                        # indices_selected.append(index_min)
                        y_list.append(stat[cell]["ypix"][index_min])
                        x_list.append(stat[cell]["xpix"][index_min])
                coord_vector = np.zeros((2, len(x_list)), dtype="int16")
                coord_vector[0] = np.array(x_list)
                coord_vector[1] = np.array(y_list)
                coord.append(coord_vector)
        else:
            coord = []
            for cell in np.arange(len(stat)):
                if is_cell[cell][0] == 0:
                    continue
                list_points_coord = [(x, y) for x, y in zip(stat[cell]["xpix"], stat[cell]["ypix"])]
                convex_hull = MultiPoint(list_points_coord).convex_hull
                if isinstance(convex_hull, LineString):
                    coord_shapely = MultiPoint(list_points_coord).convex_hull.coords
                else:
                    coord_shapely = MultiPoint(list_points_coord).convex_hull.exterior.coords
                coord.append(np.array(coord_shapely).transpose())
        self.suite2p_data["coord"] = coord
        compare_suite_2p_and_caiman = False
        if compare_suite_2p_and_caiman:
            # test trying to findout to which caiman cell those cells correspond
            suite2p_coord_obj = CoordClass(coord=coord, nb_col=self.movie_len_x,
                                           nb_lines=self.movie_len_y, from_suite_2p=True)
            # for each suite2p cell, will give the estimate of the caiman cell index, will put -1 if no cell is found
            caiman_suite2p_mapping = self.coord_obj.match_cells_indices(suite2p_coord_obj, param=self.param,
                                                                        plot_title_opt=f"{self.description}_suite2p_vs_caiman")
            np.save(os.path.join(self.param.path_results, f"{self.description}_suite2p_vs_caiman.npy"),
                    caiman_suite2p_mapping)

        if with_coord:
            self.coord = coord
            # print(f"self.coord len: {len(self.coord)}")
            self.coord_obj = CoordClass(coord=self.coord, nb_col=self.movie_len_x,
                                        nb_lines=self.movie_len_y, from_suite_2p=True)

        bonus = ""
        if with_coord:
            bonus = " with coord"
        print(f"suite2p data loaded for {self.description}{bonus}")

    def plot_connectivity_maps_of_a_cell(self, cell_to_map, cell_descr, not_in=False,
                                         cell_color="red", links_cell_color="cornflowerblue"):
        if self.coord_obj is None:
            return
        color_each_cells_link_to_cell = True

        connections_dict_in = dict()
        connections_dict_out = dict()
        n_in_matrix = self.spike_struct.n_in_matrix
        n_out_matrix = self.spike_struct.n_out_matrix
        at_least_on_in_link = False
        at_least_on_out_link = False

        connections_dict_in[cell_to_map] = dict()
        connections_dict_out[cell_to_map] = dict()

        for cell in np.where(n_in_matrix[cell_to_map, :])[0]:
            at_least_on_in_link = True
            connections_dict_in[cell_to_map][cell] = 1

        for cell in np.where(n_out_matrix[cell_to_map, :])[0]:
            at_least_on_out_link = True
            connections_dict_out[cell_to_map][cell] = 1

        cells_groups_colors = [cell_color]
        cells_groups = [[cell_to_map]]
        if at_least_on_in_link and color_each_cells_link_to_cell:
            links_cells = list(connections_dict_in[cell_to_map].keys())
            # removing fellow inter_neurons, code could be use to colors some cell in another color
            # links_cells = np.setdiff1d(np.array(links_cells), np.array(inter_neurons))
            if len(links_cells) > 0:
                cells_groups.append(list(connections_dict_in[cell_to_map].keys()))
                cells_groups_colors.append(links_cell_color)

        # self.coord_obj.compute_center_coord(cells_groups=cells_groups,
        #                                     cells_groups_colors=cells_groups_colors)
        if not not_in:
            self.coord_obj.plot_cells_map(param=self.param,
                                          data_id=self.description, show_polygons=False,
                                          title_option=f"n_in_{cell_descr}_{cell_to_map}",
                                          connections_dict=connections_dict_in,
                                          with_cell_numbers=False,
                                          cells_groups=cells_groups,
                                          dont_fill_cells_not_in_groups=True,
                                          cells_groups_colors=cells_groups_colors,
                                          background_color="white", default_cells_color="black",
                                          link_connect_color="black")

        cells_groups_colors = [cell_color]
        cells_groups = [[cell_to_map]]
        if at_least_on_out_link and color_each_cells_link_to_cell:
            links_cells = list(connections_dict_out[cell_to_map].keys())
            # removing fellow inter_neurons
            links_cells = np.setdiff1d(np.array(links_cells), np.array(cell_to_map))
            if len(links_cells) > 0:
                cells_groups.append(list(connections_dict_out[cell_to_map].keys()))
                cells_groups_colors.append(links_cell_color)

        # self.coord_obj.compute_center_coord(cells_groups=cells_groups,
        #                                     cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      title_option=f"n_out_{cell_descr}_{cell_to_map}",
                                      connections_dict=connections_dict_out,
                                      with_cell_numbers=False,
                                      cells_groups=cells_groups,
                                      dont_fill_cells_not_in_groups=True,
                                      cells_groups_colors=cells_groups_colors,
                                      background_color="black", default_cells_color="white",
                                      link_connect_color="white", save_formats=["png", "eps"]
                                      )

    def plot_all_inter_neurons_connect_map(self):
        # plot n_in and n_out map of the interneurons
        inter_neurons = self.spike_struct.inter_neurons
        n_inter_neurons = len(inter_neurons)
        if n_inter_neurons == 0:
            return

        color_each_cells_link_to_interneuron = True

        connections_dict_in = dict()
        connections_dict_out = dict()
        n_in_matrix = self.spike_struct.n_in_matrix
        n_out_matrix = self.spike_struct.n_out_matrix
        at_least_on_in_link = False
        at_least_on_out_link = False
        for inter_neuron in inter_neurons:
            connections_dict_in[inter_neuron] = dict()
            connections_dict_out[inter_neuron] = dict()

            for cell in np.where(n_in_matrix[inter_neuron, :])[0]:
                at_least_on_in_link = True
                connections_dict_in[inter_neuron][cell] = 1

            for cell in np.where(n_out_matrix[inter_neuron, :])[0]:
                at_least_on_out_link = True
                connections_dict_out[inter_neuron][cell] = 1

        cells_groups_colors = ["red"]
        cells_groups = [inter_neurons]
        if at_least_on_in_link and color_each_cells_link_to_interneuron:
            for index_inter_neuron, inter_neuron in enumerate(inter_neurons):
                links_cells = list(connections_dict_in[inter_neuron].keys())
                # removing fellow inter_neurons
                links_cells = np.setdiff1d(np.array(links_cells), np.array(inter_neurons))
                if len(links_cells) > 0:
                    cells_groups.append(list(connections_dict_in[inter_neuron].keys()))
                    cells_groups_colors.append(cm.nipy_spectral(float(index_inter_neuron + 1) / (n_inter_neurons + 1)))

        # self.coord_obj.compute_center_coord(cells_groups=cells_groups,
        #                                     cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      title_option=f"n_in_interneurons_x_{n_inter_neurons}",
                                      connections_dict=connections_dict_in,
                                      cells_groups=cells_groups,
                                      cells_groups_colors=cells_groups_colors,
                                      with_cell_numbers=True)

        cells_groups_colors = ["red"]
        cells_groups = [inter_neurons]
        if at_least_on_out_link and color_each_cells_link_to_interneuron:
            for index_inter_neuron, inter_neuron in enumerate(inter_neurons):
                links_cells = list(connections_dict_out[inter_neuron].keys())
                # removing fellow inter_neurons
                links_cells = np.setdiff1d(np.array(links_cells), np.array(inter_neurons))
                if len(links_cells) > 0:
                    cells_groups.append(list(connections_dict_out[inter_neuron].keys()))
                    cells_groups_colors.append(cm.nipy_spectral(float(index_inter_neuron + 1) / (n_inter_neurons + 1)))

        # self.coord_obj.compute_center_coord(cells_groups=cells_groups,
        #                                     cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      title_option=f"n_out_interneurons_x_{n_inter_neurons}",
                                      connections_dict=connections_dict_out,
                                      cells_groups=cells_groups,
                                      cells_groups_colors=cells_groups_colors,
                                      with_cell_numbers=True)

    def plot_all_cells_on_map_with_avg_on_bg(self, save_plot=True, return_fig=False):
        if self.coord_obj is None:
            return
        # we want to color cells that overlap with different colors
        n_cells = len(self.coord_obj.coord)

        if self.tiff_movie is None:
            self.load_tiff_movie_in_memory()
        avg_cell_map_img = np.mean(self.tiff_movie, axis=0)
        fig = self.coord_obj.plot_cells_map(param=self.param,
                                            data_id=self.description, show_polygons=False,
                                            fill_polygons=True,
                                            dont_fill_cells_not_in_groups=False,
                                            default_cells_color=(1, 0, 0, 0.3),
                                            default_edge_color="red",
                                            with_edge=True,
                                            title_option="all cells", connections_dict=None,
                                            img_on_background=avg_cell_map_img,
                                            with_cell_numbers=False, save_formats=["png", "pdf"],
                                            save_plot=save_plot, return_fig=return_fig)
        if return_fig:
            return fig

    def plot_all_cells_on_map(self, save_plot=True, return_fig=False, with_avg_on_bg=False):
        if self.coord_obj is None:
            return
        # we want to color cells that overlap with different colors
        n_cells = len(self.coord_obj.coord)
        # white, http://doc.instantreality.org/tools/color_calculator/
        # white: 1, 1, 1
        # red: 1, 0, 0
        isolated_cell_color = (1, 0, 0, 1.0)
        isolated_group = []
        cells_groups_colors = []
        cells_groups_edge_colors = []
        cells_groups_alpha = []
        cells_groups = []

        # building networkx graph
        graphs = []
        cells_added = []
        for cell in np.arange(n_cells):
            if cell in cells_added:
                continue
            # welsh_powell
            n_intersect = len(self.coord_obj.intersect_cells[cell])
            if n_intersect == 0:
                isolated_group.append(cell)
                cells_added.append(cell)
            else:
                graph = nx.Graph()
                cells_to_expend = [cell]
                edges = set()
                while len(cells_to_expend) > 0:
                    if cells_to_expend[0] not in cells_added:
                        cells_added.append(cells_to_expend[0])
                        n_intersect = len(self.coord_obj.intersect_cells[cells_to_expend[0]])
                        if n_intersect > 0:
                            for inter_cell in self.coord_obj.intersect_cells[cells_to_expend[0]]:
                                min_c = min(inter_cell, cells_to_expend[0])
                                max_c = max(inter_cell, cells_to_expend[0])
                                edges.add((min_c, max_c))
                                cells_to_expend.append(inter_cell)
                    cells_to_expend = cells_to_expend[1:]
                graph.add_edges_from(list(edges))
                graphs.append(graph)
        cells_by_color_code = dict()
        max_color_code = 0
        for graph in graphs:
            # dict that give for each cell a color code
            col_val = welsh_powell(graph)
            for cell, color_code in col_val.items():
                if color_code not in cells_by_color_code:
                    cells_by_color_code[color_code] = []
                cells_by_color_code[color_code].append(cell)
                max_color_code = max(max_color_code, color_code)

        for color_code, cells in cells_by_color_code.items():
            if len(cells) == 0:
                continue
            cells_groups.append(cells)
            cells_groups_colors.append(cm.nipy_spectral(float(color_code + 1) / (max_color_code + 1)))
            cells_groups_edge_colors.append("white")
            cells_groups_alpha.append(0.8)
        cells_groups.append(isolated_group)
        cells_groups_colors.append(isolated_cell_color)
        cells_groups_alpha.append(1)
        cells_groups_edge_colors.append("white")
        avg_cell_map_img = None
        if with_avg_on_bg:
            if self.tiff_movie is None:
                self.load_tiff_movie_in_memory()
            avg_cell_map_img = np.mean(self.tiff_movie, axis=0)
        fig = self.coord_obj.plot_cells_map(param=self.param,
                                            data_id=self.description, show_polygons=False,
                                            fill_polygons=False,
                                            title_option="all cells", connections_dict=None,
                                            cells_groups=cells_groups,
                                            img_on_background=avg_cell_map_img,
                                            cells_groups_colors=cells_groups_colors,
                                            cells_groups_edge_colors=cells_groups_edge_colors,
                                            with_edge=True, cells_groups_alpha=cells_groups_alpha,
                                            dont_fill_cells_not_in_groups=False,
                                            with_cell_numbers=True, save_formats=["png", "pdf"],
                                            save_plot=save_plot, return_fig=return_fig)
        if return_fig:
            return fig

    def plot_cell_assemblies_on_map(self, save_formats=["pdf"]):
        if (self.cell_assemblies is None) or (self.coord_obj is None):
            return

        n_assemblies = len(self.cell_assemblies)
        cells_groups_colors = []
        cells_groups_edge_colors = []
        cells_groups_alpha = []
        for i in np.arange(n_assemblies):
            # print(f"cm.nipy_spectral(float(i + 1) / (n_assemblies + 1)) "
            #       f"{cm.nipy_spectral(float(i + 1) / (n_assemblies + 1))}")
            color = cm.nipy_spectral(float(i + 1) / (n_assemblies + 1))
            # if i == 0:
            #     color = (100 / 255, 215 / 255, 247 / 255, 1)  # #64D7F7"
            # else:
            #     color = (213 / 255, 38 / 255, 215 / 255, 1)  # #D526D7
            cells_groups_colors.append(color)
            cells_groups_edge_colors.append(color)
            cells_groups_alpha.append(0.3)
        # print(f"cells_groups_colors {cells_groups_colors}")
        # self.coord_obj.compute_center_coord(cells_groups=self.cell_assemblies,
        #                                     cells_groups_colors=cells_groups_colors,
        #                                     dont_fill_cells_not_in_groups=True)
        avg_cell_map_img = mpimg.imread("/media/julien/Not_today/hne_not_today/data/p9/p9_18_09_27_a003/p9_18_09_27_a003_AVG.png")

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      fill_polygons=False,
                                      title_option="cell_assemblies", connections_dict=None,
                                      with_edge=True,
                                      cells_groups=self.cell_assemblies,
                                      cells_groups_colors=cells_groups_colors,
                                      img_on_background=avg_cell_map_img,
                                      dont_fill_cells_not_in_groups=False,
                                      cells_groups_alpha=cells_groups_alpha,
                                      default_cells_color=(1, 1, 1, 0.3),
                                      default_edge_color="white",
                                      with_cell_numbers=False, save_formats=save_formats)
        # self.coord_obj.plot_cells_map(param=self.param,
        #                               data_id=self.description, show_polygons=True,
        #                               fill_polygons=True,
        #                               title_option="cell_assemblies", connections_dict=None,
        #                               with_edge=True,
        #                               cells_groups=self.cell_assemblies,
        #                               cells_groups_colors=cells_groups_colors,
        #                               dont_fill_cells_not_in_groups=True,
        #                               with_cell_numbers=False, save_formats=save_formats)

    def set_low_activity_threshold(self, threshold, percentile_value):
        self.low_activity_threshold_by_percentile[percentile_value] = threshold
        if self.percentile_for_low_activity_threshold in self.low_activity_threshold_by_percentile:
            self.low_activity_threshold = \
                self.low_activity_threshold_by_percentile[self.percentile_for_low_activity_threshold]

    def set_inter_neurons(self, inter_neurons):
        self.spike_struct.inter_neurons = np.array(inter_neurons).astype(int)

    def save_sum_spikes_dur_in_npy_file(self):
        if self.spike_struct.spike_nums_dur is None:
            print(f"{self.description}: spike_nmus_dur None, not able to save sum activity")
            return

        np.save(os.path.join(self.param.path_data, f"p{self.age}", f"{self.description.lower()}",
                             f"{self.description}_sum_activity.npy"),
                np.sum(self.spike_struct.spike_nums_dur, axis=0))
        # np.save(os.path.join(self.param.path_results, f"{self.description}_sum_activity.npy"),
        #         np.sum(self.spike_struct.spike_nums_dur, axis=0))
        print(f"{self.description}: sum of spike_nums_dur saved")

    def plot_cell_assemblies_clusters(self, cellsinpeak):
        """
        :param cellsinpeak: binary 2d array, is a cell in a SCE, cells are lines.
        :return:
        """
        # TODO: doesn't work, TO FINISH
        if (self.cell_assemblies is None) or (self.SCE_times is None):
            return

        # list of list of int representing cell indices
        # initiated when loading_cell_assemblies
        # self.cell_assemblies = None
        # # dict with key a int representing the cell_assembly cluster and and value a list of tuple representing first
        # # and last index including of the SCE
        # self.sce_times_in_single_cell_assemblies = None
        # # list of tuple of int
        # self.sce_times_in_multiple_cell_assemblies = None
        # # list of tuple of int (gather data from sce_times_in_single_cell_assemblies and
        # # sce_times_in_multiple_cell_assemblies)
        # self.sce_times_in_cell_assemblies = None
        n_cells = self.spike_struct.n_cells
        cas = CellAssembliesStruct(data_id=self.description,
                                   sce_clusters_labels=None,
                                   cellsinpeak=cellsinpeak,
                                   sce_clusters_id=None, n_clusters=None,
                                   cluster_with_best_silhouette_score=None,
                                   param=self.param, neurons_labels=np.arange(n_cells),
                                   sliding_window_duration=1,
                                   n_surrogate_k_mean=None, SCE_times=self.SCE_times)
        cas.activity_threshold = self.activity_threshold
        # novel order to display cells organized by cell assembly cluster, last cells being the one without cell assembly
        cells_in_ca_indices = np.array(list(itertools.chain.from_iterable(self.cell_assemblies)))
        cas.cells_indices = cells_in_ca_indices
        n_cells_in_cell_assemblies_clusters = [len(n) for n in self.cell_assemblies]
        cas.n_cells_in_cell_assemblies_clusters = n_cells_in_cell_assemblies_clusters
        cas.n_cells_not_in_cell_assemblies = n_cells - np.sum(n_cells_in_cell_assemblies_clusters)
        # value to one represent the cells spikes without assembly, then number 2 represent the cell assembly 0, etc...
        cellsinpeak_ordered = np.zeros(cellsinpeak.shape, dtype="int8")
        cellsinpeak_ordered[:len(cells_in_ca_indices)] = cellsinpeak[cells_in_ca_indices, :]
        # now we want the last ones
        all_cells = np.arange(n_cells)
        left_cells_indices = np.setdiff1d(all_cells, cells_in_ca_indices)
        cellsinpeak_ordered[len(cells_in_ca_indices):] = cellsinpeak[left_cells_indices, :]
        cas.cellsinpeak_ordered = cellsinpeak_ordered

        self.n_cells_in_multiple_cell_assembly_sce_cl = None
        # give the number of sce in the no-assembly-sce, single-assembly and multiple-assembly groups respectively
        n_sces = len(self.SCE_times)
        cas.n_sce_in_assembly = np.zeros(3, dtype="uint16")
        cas.n_sce_in_assembly[0] = n_sces - len(self.sce_times_in_cell_assemblies)
        cas.n_sce_in_assembly[1] = len(self.sce_times_in_single_cell_assemblies)
        cas.n_sce_in_assembly[2] = len(self.sce_times_in_multiple_cell_assemblies)
        # contains the nb of sces in sce single cell assembly cluster
        cas.n_cells_in_single_cell_assembly_sce_cl = []
        # TO COMPLETE
        # contains the nb of sces in sce multiple cell assembly cluster
        cas.n_cells_in_multiple_cell_assembly_sce_cl = []
        # TO COMPLETE
        cas.plot_cell_assemblies(data_descr=self.description, spike_nums=self.spike_struct.spike_nums_dur,
                                 SCE_times=self.SCE_times, activity_threshold=self.activity_threshold,
                                 with_cells_in_cluster_seq_sorted=False,
                                 sce_times_bool=self.sce_bool,
                                 display_only_cell_assemblies_on_raster=False,
                                 save_formats=["pdf", "png"])

    def load_abf_file(self, path_abf_data=None, abf_file_name=None, threshold_piezo=None,
                      frames_channel=0, piezo_channel=None, run_channel=None, lfp_channel=None, threshold_ratio=2,
                      sampling_rate=50000, offset=None, just_load_npz_file=False):
        # run_channel is usually 2
        """

        :param path_abf_data:
        :param abf_file_name:
        :param threshold_piezo:
        :param frames_channel:
        :param current_channel:
        :param run_channel: if not None, means there is run. Otherwise no run analysis will be performed
        :param threshold_ratio:
        :param sampling_rate:
        :param offset:
        :param just_load_npz_file:
        :return:
        """

        if (path_abf_data is None) and (abf_file_name is None):
            print(f"{self.description}: path_abf_data and abf_file_name are not defined")
            return

        print(f"abf: ms {self.description}")
        self.abf_sampling_rate = sampling_rate
        self.threshold_piezo = threshold_piezo

        use_old_version = False

        if use_old_version:
            # first checking if the data has been saved in a file before
            index_reverse = abf_file_name[::-1].find("/")
            path_abf_data = abf_file_name[:len(abf_file_name) - index_reverse]
            file_names = []

            npz_loaded = False
            # look for filenames in the first directory, if we don't break, it will go through all directories
            for (dirpath, dirnames, local_filenames) in os.walk(self.param.path_data + path_abf_data):
                file_names.extend(local_filenames)
                break
            if len(file_names) > 0:
                for file_name in file_names:
                    if file_name.endswith(".npz") and (not file_name.startswith(".")):
                        if file_name.find("abf") > -1:
                            do_detect_twitches = True
                            # loading data
                            npz_loaded = True
                            npzfile = np.load(self.param.path_data + path_abf_data + file_name)
                            if "mvt_frames" in npzfile:
                                self.mvt_frames = npzfile['mvt_frames']
                                self.mvt_frames_periods = tools_misc.find_continuous_frames_period(self.mvt_frames)
                            if "speed_by_mvt_frame" in npzfile:
                                self.speed_by_mvt_frame = npzfile['speed_by_mvt_frame']
                            # if "raw_piezo" in npzfile:
                            #     self.raw_piezo = npzfile['raw_piezo']
                            # if "abf_frames" in npzfile:
                            #     self.abf_frames = npzfile['abf_frames']
                            # if "abf_times_in_sec" in npzfile:
                            #     self.abf_times_in_sec = npzfile['abf_times_in_sec']
                            if "twitches_frames" in npzfile:
                                self.twitches_frames = npzfile['twitches_frames']
                                self.twitches_frames_periods = tools_misc.find_continuous_frames_period(
                                    self.twitches_frames)
                                do_detect_twitches = False
                            if "short_lasting_mvt_frames" in npzfile:
                                self.short_lasting_mvt_frames = npzfile['short_lasting_mvt_frames']
                                self.short_lasting_mvt_frames_periods = \
                                    tools_misc.find_continuous_frames_period(self.short_lasting_mvt_frames)
                            if "complex_mvt_frames" in npzfile:
                                self.complex_mvt_frames = npzfile['complex_mvt_frames']
                                self.complex_mvt_frames_periods = \
                                    tools_misc.find_continuous_frames_period(self.complex_mvt_frames)
                            if "intermediate_behavourial_events_frames" in npzfile:
                                self.intermediate_behavourial_events_frames = npzfile[
                                    'intermediate_behavourial_events_frames']
                                self.intermediate_behavourial_events_frames_periods = \
                                    tools_misc.find_continuous_frames_period(
                                        self.intermediate_behavourial_events_frames)
                            if "noise_mvt_frames" in npzfile:
                                self.noise_mvt_frames = npzfile['noise_mvt_frames']
                                self.noise_mvt_frames_periods = \
                                    tools_misc.find_continuous_frames_period(self.noise_mvt_frames)
            if just_load_npz_file:
                return

        if abf_file_name is not None:
            # 50000 Hz
            try:
                abf = pyabf.ABF(self.param.path_data + abf_file_name)
            except (FileNotFoundError, OSError) as e:
                print(f"Abf file not found: {abf_file_name}")
                return
        else:
            abf = None
            # look for filenames in the first directory, if we don't break, it will go through all directories
            for (dirpath, dirnames, local_filenames) in os.walk(self.param.path_data + path_abf_data):
                for file_name in local_filenames:
                    if file_name.endswith("abf"):
                        abf = pyabf.ABF(os.path.join(self.param.path_data, path_abf_data, file_name))
                        break
                break
            if abf is None:
                print(f"{self.description} no abf file found in {path_abf_data}")
                return

        print(f"{abf}")
        #  1024 cycle = 1 tour de roue (= 2 Pi 1.5) -> Vitesse (cm / temps pour 1024 cycles).
        #
        # if first channel in advance from 2nd, the mouse goes forward, otherwise it goes backward
        # if with_run:
        #     for channel in np.arange(0, 1):
        #         abf.setSweep(sweepNumber=0, channel=channel)
        #         times_in_sec = abf.sweepX
        #         fig, ax = plt.subplots(nrows=1, ncols=1,
        #                                gridspec_kw={'height_ratios': [1]},
        #                                figsize=(20, 8))
        #         plt.plot(times_in_sec, abf.sweepY, lw=.5)
        #         plt.title(f"channel {channel} {self.description}")
        #         plt.show()
        #         plt.close()

        abf.setSweep(sweepNumber=0, channel=frames_channel)
        times_in_sec = abf.sweepX
        frames_data = abf.sweepY
        original_time_in_sec = times_in_sec
        original_frames_data = frames_data
        # first frame
        first_frame_index = np.where(frames_data < 0.01)[0][0]

        # if lfp_channel is not None:
        for current_channel in np.arange(1, abf.channelCount):
            times_in_sec = np.copy(original_time_in_sec)
            frames_data = np.copy(original_frames_data)
            if (run_channel is not None) and (current_channel == run_channel):
                continue
            if (run_channel is not None) and (current_channel == 4) and (lfp_channel is None):
                lfp_channel = 4
            if (run_channel is None) and (current_channel == 3) and (lfp_channel is None):
                lfp_channel = 3
            abf.setSweep(sweepNumber=0, channel=current_channel)
            mvt_data = abf.sweepY
            if offset is not None:
                mvt_data = mvt_data + offset
            # self.channelCount-1
            mvt_data = mvt_data[first_frame_index:]
            times_in_sec = times_in_sec[:-first_frame_index]
            frames_data = frames_data[first_frame_index:]
            threshold_value = 0.02
            if self.abf_sampling_rate < 50000:
                # frames_data represent the content of the abf channel that contains the frames
                # the index stat at the first frame recorded, meaning the first value where the
                # value is < 0.01
                mask_frames_data = np.ones(len(frames_data), dtype="bool")
                # we need to detect the frames manually, but first removing data between movies
                selection = np.where(frames_data >= threshold_value)[0]
                mask_selection = np.zeros(len(selection), dtype="bool")
                pos = np.diff(selection)
                # looking for continuous data between movies
                to_keep_for_removing = np.where(pos == 1)[0] + 1
                mask_selection[to_keep_for_removing] = True
                selection = selection[mask_selection]
                # we remove the "selection" from the frames data
                mask_frames_data[selection] = False
                frames_data = frames_data[mask_frames_data]
                # len_frames_data_in_s = np.round(len(frames_data) / self.abf_sampling_rate, 3)
                mvt_data = mvt_data[mask_frames_data]
                times_in_sec = times_in_sec[:-len(np.where(mask_frames_data == 0)[0])]
                active_frames = np.linspace(0, len(frames_data), 12500).astype(int)
                mean_diff_active_frames = np.mean(np.diff(active_frames)) / self.abf_sampling_rate
                # print(f"mean diff active_frames {np.round(mean_diff_active_frames, 3)}")
                if mean_diff_active_frames < 0.09:
                    raise Exception("mean_diff_active_frames < 0.09")
            else:
                binary_frames_data = np.zeros(len(frames_data), dtype="int8")
                binary_frames_data[frames_data >= threshold_value] = 1
                binary_frames_data[frames_data < threshold_value] = 0
                # +1 due to the shift of diff
                # contains the index at which each frame from the movie is matching the abf signal
                # length should be 12500
                active_frames = np.where(np.diff(binary_frames_data) == 1)[0] + 1

            # correspond of the variation of the piezo
            mvt_data_without_abs = mvt_data
            mvt_data = np.abs(mvt_data)
            # useful is the piezo channel is known
            # if (run_channel is not None):
            #     self.raw_piezo = mvt_data
            #     self.raw_piezo_without_abs = mvt_data_without_abs

            self.abf_times_in_sec = times_in_sec
            # active_frames = np.concatenate(([0], active_frames))
            # print(f"active_frames {active_frames}")
            nb_frames = len(active_frames)
            self.abf_frames = active_frames
            # print(f"nb_frames {nb_frames}")
            # print(f"len(mvt_data_without_abs) {len(mvt_data_without_abs)}")
            # print(f"self.abf_frames {self.abf_frames[-50:]}")
            # print(f'Saving abf_frames for {self.description}')
            # np.save(self.param.path_data + path_abf_data + self.description +
            #         f"_abf_frames_channel_{current_channel}.npy", self.abf_frames)
            # down sampling rate: 50 for piezzo, 1000 for LFP
            if (lfp_channel is not None) and lfp_channel == current_channel:
                down_sampling_hz = 1000
            elif (piezo_channel is not None) and (piezo_channel == current_channel):
                down_sampling_hz = 50
            elif current_channel <= 2:
                down_sampling_hz = 50
            else:
                down_sampling_hz = 1000
            sampling_step = int(self.abf_sampling_rate / down_sampling_hz)
            # np.save(self.param.path_data + path_abf_data + self.description +
            #         f"_abf_12500_channel_{current_channel}.npy",
            #         mvt_data_without_abs[self.abf_frames])
            # first we want to keep piezzo data only for the active movie, removing the time between imaging session
            # to do so we concatenate the time between frames
            piezzo_shift = np.zeros(0)
            for i in np.arange(0, 12500, 2500):
                last_abf_frame = self.abf_frames[i + 2499]
                # mvt_data_without_abs represents the piezzo values without taking the absolute value
                if self.abf_frames[i + 2499] == len(mvt_data_without_abs):
                    last_abf_frame -= 1
                # sampling_step is produce according to a down_sampling_hz that changes
                # according to the channel (lfp, piezzo etc...)
                new_data = mvt_data_without_abs[np.arange(self.abf_frames[i],
                                                          last_abf_frame, sampling_step)]
                piezzo_shift = np.concatenate((piezzo_shift, new_data,
                                               np.array([mvt_data_without_abs[last_abf_frame]])))
            if current_channel == lfp_channel:
                np.save(self.param.path_data + path_abf_data + self.description +
                        f"_abf_lfp_channel_{current_channel}_{down_sampling_hz}hz.npy",
                        piezzo_shift)
            else:
                np.save(self.param.path_data + path_abf_data + self.description +
                        f"_abf_HR_channel_{current_channel}.npy",
                        piezzo_shift)

        if run_channel is not None:
            # defining active_frames
            times_in_sec = original_time_in_sec
            frames_data = original_frames_data
            times_in_sec = times_in_sec[:-first_frame_index]
            frames_data = frames_data[first_frame_index:]
            threshold_value = 0.02
            if self.abf_sampling_rate < 50000:
                # frames_data represent the content of the abf channel that contains the frames
                # the index stat at the first frame recorded, meaning the first value where the
                # value is < 0.01
                mask_frames_data = np.ones(len(frames_data), dtype="bool")
                # we need to detect the frames manually, but first removing data between movies
                selection = np.where(frames_data >= threshold_value)[0]
                mask_selection = np.zeros(len(selection), dtype="bool")
                pos = np.diff(selection)
                # looking for continuous data between movies
                to_keep_for_removing = np.where(pos == 1)[0] + 1
                mask_selection[to_keep_for_removing] = True
                selection = selection[mask_selection]
                # we remove the "selection" from the frames data
                mask_frames_data[selection] = False
                frames_data = frames_data[mask_frames_data]
                # len_frames_data_in_s = np.round(len(frames_data) / self.abf_sampling_rate, 3)
                mvt_data = mvt_data[mask_frames_data]
                times_in_sec = times_in_sec[:-len(np.where(mask_frames_data == 0)[0])]
                active_frames = np.linspace(0, len(frames_data), 12500).astype(int)
                mean_diff_active_frames = np.mean(np.diff(active_frames)) / self.abf_sampling_rate
                # print(f"mean diff active_frames {np.round(mean_diff_active_frames, 3)}")
                if mean_diff_active_frames < 0.09:
                    raise Exception("mean_diff_active_frames < 0.09")
            else:
                binary_frames_data = np.zeros(len(frames_data), dtype="int8")
                binary_frames_data[frames_data >= threshold_value] = 1
                binary_frames_data[frames_data < threshold_value] = 0
                # +1 due to the shift of diff
                # contains the index at which each frame from the movie is matching the abf signal
                # length should be 12500
                active_frames = np.where(np.diff(binary_frames_data) == 1)[0] + 1

            abf.setSweep(sweepNumber=0, channel=run_channel)
            run_data = abf.sweepY
            mvt_periods, speed_during_mvt_periods, speed_by_time = \
                self.detect_run_periods(mvt_data=run_data, min_speed=0.5)
            speed_by_time = speed_by_time[active_frames]
            self.speed_by_frame = speed_by_time
            # matfiledata = {}  # make a dictionary to store the MAT data in
            # matfiledata['Speed'] = speed_by_time
            # hdf5storage.write(matfiledata, os.path.join(self.param.path_data, path_abf_data),
            #                   f'speed_{self.description}.mat', matlab_compatible=True)
            print(f"Saving speed for {self.description}")
            np.save(os.path.join(self.param.path_data, path_abf_data, f'speed_{self.description}.npy'), speed_by_time)

        self.abf_loaded = True
        # manual selection deactivated
        do_manual_selection = False  # not npz_loaded
        if not do_manual_selection:
            return

        # if (not with_run) and (threshold_piezo is None):
        #     fig, ax = plt.subplots(nrows=1, ncols=1,
        #                            gridspec_kw={'height_ratios': [1]},
        #                            figsize=(20, 8))
        #     plt.plot(times_in_sec, mvt_data, lw=.5)
        #     plt.title(f"piezo {self.description}")
        #     plt.show()
        #     plt.close()
        #     return

        if with_run:
            mvt_periods, speed_during_mvt_periods, speed_by_time = \
                self.detect_run_periods(mvt_data=mvt_data, min_speed=0.5)
            # else:
            #     mvt_periods = self.detect_mvt_periods_with_piezo_and_diff(piezo_data=mvt_data,
            #                                                               piezo_threshold=threshold_piezo,
            #                                                               min_time_between_periods=2 * sampling_rate)
            # print(f"len(mvt_periods) {len(mvt_periods)}")
            # print(f"len(mvt_data) {len(mvt_data)}")
            self.mvt_frames = []
            self.mvt_frames_periods = []
            if with_run:
                self.speed_by_mvt_frame = []
            for mvt_period_index, mvt_period in enumerate(mvt_periods):
                frames = np.where(np.logical_and(active_frames >= mvt_period[0], active_frames <= mvt_period[1]))[0]
                if len(frames) > 0:
                    self.mvt_frames.extend(frames)
                    self.mvt_frames_periods.append((frames[0], frames[-1]))
                    if with_run:
                        for frame in frames:
                            frame_index = active_frames[frame]
                            index_from_beg_mvt_period = frame_index - mvt_period[0]
                            # 100 ms
                            range_around = int(0.1 * self.abf_sampling_rate)
                            speed_during_mvt_period = speed_during_mvt_periods[mvt_period_index]
                            # print(f"len(speed_during_mvt_period) {len(speed_during_mvt_period)}")
                            beg_pos = np.max((0, (index_from_beg_mvt_period - range_around)))
                            end_pos = np.min((len(speed_during_mvt_period),
                                              (index_from_beg_mvt_period + range_around + 1)))
                            # print(f"beg_pos {beg_pos}, end_pos {end_pos}")
                            # taking the mean speed around the frame, with a 100 ms range
                            speed = np.mean(speed_during_mvt_period[beg_pos:end_pos])
                            # print(f"speed: {np.round(speed, 2)}")
                            self.speed_by_mvt_frame.append(speed)
            self.mvt_frames = np.array(self.mvt_frames)
            # if not with_run:
            #     self.detect_twitches()
        else:
            self.analyse_piezo_the_khazipov_way(threshold_ratio=threshold_ratio)

        # plotting the result
        check_piezo_threshold = False
        if check_piezo_threshold:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   gridspec_kw={'height_ratios': [1]},
                                   figsize=(20, 8))
            plt.plot(times_in_sec, mvt_data, lw=.5, color="black")
            if mvt_periods is not None:
                for mvt_period in mvt_periods:
                    color = "red"
                    ax.axvspan(mvt_period[0] / self.abf_sampling_rate, mvt_period[1] / self.abf_sampling_rate,
                               alpha=0.1, facecolor=color, zorder=1)
            plt.title(f"piezo {self.description} threshold {threshold_piezo}")
            plt.show()
            plt.close()

        # savinf the npz file, that will be loaded directly at the next execution
        if with_run:
            np.savez(self.param.path_data + path_abf_data + self.description + "_mvts_from_abf.npz",
                     mvt_frames=self.mvt_frames, speed_by_mvt_frame=self.speed_by_mvt_frame)
        else:
            khazipow_way = True
            if khazipow_way:
                np.savez(self.param.path_data + path_abf_data + self.description + "_mvts_from_abf.npz",
                         twitches_frames=self.twitches_frames,
                         short_lasting_mvt_frames=self.short_lasting_mvt_frames,
                         complex_mvt_frames=self.complex_mvt_frames,
                         intermediate_behavourial_events_frames=self.intermediate_behavourial_events_frames,
                         noise_mvt_frames=self.noise_mvt_frames)
            else:
                if threshold_piezo is not None:
                    np.savez(self.param.path_data + path_abf_data + self.description + "_mvts_from_abf.npz",
                             mvt_frames=self.mvt_frames, raw_piezo=self.raw_piezo, abf_frames=self.abf_frames,
                             abf_times_in_sec=times_in_sec)

    def analyse_piezo_the_khazipov_way(self, threshold_ratio):
        """
        Using self.raw_piezo will determine different variables:
        self.twitch_periods: periods of less than 600 ms separated of 1 sec from other events
        self.short_lasting_mvt: mvt less than 600 ms but with other mvt in less than 1 sec range
        self.complex_mvt: mvt than last more than 900 ms
        Using a window of 10 ms to determine the std and base on that threshold, detecting local maximum
        :return:
        """
        lowest_std = None
        window_in_ms = 10 ** 5
        piezo = self.raw_piezo
        times_for_1_ms = self.abf_sampling_rate / 1000
        window_in_times = int(times_for_1_ms * window_in_ms)
        n_times = len(piezo)
        # windows have a 50% overlap
        for index in np.arange(0, n_times - window_in_times, window_in_times // 2):
            std_value = np.std(piezo[index:index + window_in_times])
            if index == 0:
                lowest_std = std_value
            lowest_std = min(lowest_std, std_value)
        print(f"threshold_ratio {threshold_ratio}, window {window_in_ms}: "
              f"lowest_std {np.round(lowest_std, 3)}")
        self.lowest_std_in_piezo = lowest_std * threshold_ratio

        mvt_periods = self.detect_mvt_periods_with_piezo_and_diff(piezo_data=self.raw_piezo,
                                                                  piezo_threshold=self.lowest_std_in_piezo,
                                                                  min_time_between_periods=times_for_1_ms * 300)
        first_derivative = np.diff(piezo) / np.diff(np.arange(n_times))
        # extending periods with first_derivative_period
        for mvt_period in mvt_periods:
            zero_times = \
                np.where(first_derivative[mvt_period[0] - (self.abf_sampling_rate // 2):mvt_period[0] + 1] == 0)[0]
            if len(zero_times) > 0:
                zero_times += (mvt_period[0] - (self.abf_sampling_rate // 2))
                # print(f"mvt_period[0]-zero_times[-1] {np.round(mvt_period[0]-zero_times[-1], 3)}")
                mvt_period[0] = zero_times[-1]
            zero_times = np.where(first_derivative[mvt_period[1]:mvt_period[1] + (self.abf_sampling_rate // 2)] == 0)[0]
            if len(zero_times) > 0:
                zero_times += mvt_period[1]
                # print(f"zero_times[0]-mvt_period[1] {np.round(zero_times[0]-mvt_period[1], 3)}")
                mvt_period[1] = zero_times[0]
        peaks = self.peaks_raw_piezo

        # classifying mvts
        self.twitches_frames_periods = []
        self.short_lasting_mvt = []
        self.noise_mvt = []
        self.complex_mvt = []
        self.intermediate_behavourial_events = []
        interval_before_twitch = times_for_1_ms * 700
        interval_after_twitch = times_for_1_ms * 400
        min_interval = times_for_1_ms * 100
        last_one_is_noise = False
        last_one_is_complex_mvt = False
        for mvt_index, mvt_period in enumerate(mvt_periods):
            duration = mvt_period[1] - mvt_period[0]
            if duration >= (times_for_1_ms * 900):
                self.complex_mvt.append(mvt_period)
                last_one_is_complex_mvt = True
                last_one_is_noise = False
                continue
            elif duration <= (times_for_1_ms * 600):
                it_is_a_twitch = True
                it_is_noise = False
                if mvt_index > 0:
                    last_mvt_period = mvt_periods[mvt_index - 1]
                    if (mvt_period[0] - last_mvt_period[1]) < interval_before_twitch:
                        it_is_a_twitch = False
                    # if (not last_one_is_noise) and ((mvt_period[0] - last_mvt_period[1]) < min_interval):
                    #     it_is_noise = True
                if mvt_index < (len(mvt_periods) - 1):
                    next_mvt_period = mvt_periods[mvt_index + 1]
                    if (next_mvt_period[0] - mvt_period[1]) < interval_after_twitch:
                        it_is_a_twitch = False
                    # if (next_mvt_period[0] - mvt_period[1]) < min_interval:
                    #     it_is_noise = True
                if it_is_a_twitch:
                    self.twitches_frames_periods.append(mvt_period)
                    # self.twitches_frames.extend(list(np.arange(mvt_period[0], mvt_period[1] + 1)))
                elif it_is_noise:
                    # not keeping those noise events
                    # self.noise_mvt.append(mvt_period)
                    last_one_is_noise = True
                    last_one_is_complex_mvt = False
                    continue
                else:
                    self.short_lasting_mvt.append(mvt_period)
            else:
                # if last_one_is_complex_mvt:
                #     last_mvt_period = mvt_periods[mvt_index - 1]
                #     if ((mvt_period[0] - last_mvt_period[1]) < min_interval):
                #         # then it becomes a complex mvt
                #         self.complex_mvt.append(mvt_period)
                #         last_one_is_complex_mvt = True
                #         last_one_is_noise = False
                #         continue
                self.intermediate_behavourial_events.append(mvt_period)
            last_one_is_noise = False
            last_one_is_complex_mvt = False

        # print("before find_peaks")
        # peaks, _ = find_peaks(piezo, height=(lowest_std_in_piezo, None), prominence=(1, None))
        # peaks = signal.find_peaks_cwt(piezo, np.arange(int(times_for_1_ms*10), int(times_for_1_ms*100), 10))
        # print("after find_peaks")
        display_all_piezo_with_mvts = False
        if display_all_piezo_with_mvts:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   gridspec_kw={'height_ratios': [1]},
                                   figsize=(20, 8))
            plt.plot(self.abf_times_in_sec, piezo, lw=.5, color="black")
            # plt.plot(self.abf_times_in_sec[:-1], np.abs(first_derivative), lw=.5, zorder=10, color="green")
            ax.hlines(self.lowest_std_in_piezo, 0,
                      self.abf_times_in_sec[-1], color="red", linewidth=1,
                      linestyles="dashed", zorder=1)
            # plt.scatter(x=self.abf_times_in_sec[peaks], y=piezo[peaks], marker="*",
            #             color=["black"], s=5, zorder=15)

            # periods_to_color = [self.twitches_frames_periods, self.short_lasting_mvt, self.complex_mvt,
            #                     self.intermediate_behavourial_events, self.noise_mvt]
            # periods_to_color_names = ["twitches", "short lasting mvt", "complex mvt", "intermediate behavourial events",
            #                           "noise mvt"]
            # colors = ["blue", "cornflowerblue", "red", "green", "grey"]
            periods_to_color = [self.complex_mvt]
            periods_to_color_names = ["complex mvt"]
            colors = ["red"]
            for period_index, period_to_color in enumerate(periods_to_color):
                for mvt_period in period_to_color:
                    ax.axvspan(mvt_period[0] / self.abf_sampling_rate, mvt_period[1] / self.abf_sampling_rate,
                               alpha=0.5, facecolor=colors[period_index], zorder=1)
            for twitch_period in self.twitches_frames_periods:
                # twith_time = (twitch_period[0] + twitch_period[1]) // 2
                pos = twitch_period[0] + np.argmax(piezo[twitch_period[0]:twitch_period[1] + 1])
                plt.scatter(x=self.abf_times_in_sec[pos], y=piezo[pos], marker="o",
                            color=["blue"], s=10, zorder=20)

            plt.title(f"piezo {self.description}")

            legend_elements = []
            # [Line2D([0], [0], color='b', lw=4, label='Line')
            for period_index, periods_to_color_name in enumerate(periods_to_color_names):
                n_events = len(periods_to_color[period_index])
                legend_elements.append(Patch(facecolor=colors[period_index],
                                             edgecolor='black', label=f'{periods_to_color_name} x{n_events}'))

            # legend_elements.append(Line2D([0], [0], marker="*", color="w", lw=0, label="peaks",
            #                                   markerfacecolor='black', markersize=10))

            legend_elements.append(Line2D([0], [0], marker="o", color="w", lw=0, label="twitches",
                                          markerfacecolor='blue', markersize=10))

            ax.legend(handles=legend_elements)

            plt.show()
            plt.close()

        gui = MvtSelectionGui(mouse_session=self)
        # getting the results from the gui
        self.twitches_frames_periods = []
        self.short_lasting_mvt = []
        self.noise_mvt = []
        self.intermediate_behavourial_events = []
        for period_times, category in gui.mvt_categories.items():
            if category == gui.categories_code["twitches"]:
                self.twitches_frames_periods.append(period_times)
            elif category == gui.categories_code["short lasting mvt"]:
                self.short_lasting_mvt.append(period_times)
            elif category == gui.categories_code["noise"]:
                self.noise_mvt.append(period_times)
            elif category == gui.categories_code["behavourial events"]:
                self.intermediate_behavourial_events.append(period_times)

        periods_to_frames = []
        self.twitches_frames = []
        self.short_lasting_mvt_frames = []
        self.complex_mvt_frames = []
        self.intermediate_behavourial_events_frames = []
        self.noise_mvt_frames = []
        periods_to_frames.append((self.twitches_frames_periods, self.twitches_frames))
        periods_to_frames.append((self.short_lasting_mvt, self.short_lasting_mvt_frames))
        periods_to_frames.append((self.complex_mvt, self.complex_mvt_frames))
        periods_to_frames.append((self.intermediate_behavourial_events, self.intermediate_behavourial_events_frames))
        periods_to_frames.append((self.noise_mvt, self.noise_mvt_frames))
        # now with put all mvt periods in frame indices and as np.array to save it in a numpy file
        for mvt_periods, frames_list in periods_to_frames:
            for mvt_period in mvt_periods:
                frames = np.where(np.logical_and(self.abf_frames >= mvt_period[0], self.abf_frames <= mvt_period[1]))[0]
                if len(frames) > 0:
                    frames_list.extend(frames)

        self.twitches_frames = np.array(self.twitches_frames)
        self.short_lasting_mvt_frames = np.array(self.short_lasting_mvt_frames)
        self.complex_mvt_frames = np.array(self.complex_mvt_frames)
        self.intermediate_behavourial_events_frames = np.array(self.intermediate_behavourial_events_frames)
        self.noise_mvt_frames = np.array(self.noise_mvt_frames)

    def detect_run_periods(self, mvt_data, min_speed):
        nb_period_by_wheel = 500
        wheel_diam_cm = 2 * math.pi * 1.75
        cm_by_period = wheel_diam_cm / nb_period_by_wheel
        binary_mvt_data = np.zeros(len(mvt_data), dtype="int8")
        speed_by_time = np.zeros(len(mvt_data))
        is_running = np.zeros(len(mvt_data), dtype="int8")
        # print(f"len(mvt_data) {len(mvt_data)}")
        binary_mvt_data[mvt_data >= 4] = 1
        d_times = np.diff(binary_mvt_data)
        pos_times = np.where(d_times == 1)[0] + 1
        for index, pos in enumerate(pos_times[1:]):
            run_duration = pos - pos_times[index - 1]
            run_duration_s = run_duration / self.abf_sampling_rate
            # in cm/s
            speed = cm_by_period / run_duration_s
            # if speed < 1:
            #     print(f"#### speed_i {index}: {np.round(speed, 3)}")
            # else:
            #     print(f"speed_i {index}: {np.round(speed, 3)}")
            if speed >= min_speed:
                speed_by_time[pos_times[index - 1]:pos] = speed
                is_running[pos_times[index - 1]:pos] = 1
                # print(f"is_running {index}: {pos_times[index-1]} to {pos}")

        #  1024 cycle = 1 tour de roue (= 2 Pi 1.5) -> Vitesse (cm / temps pour 1024 cycles).
        # the period of time between two 1 represent a run
        mvt_periods = get_continous_time_periods(is_running)
        mvt_periods = self.merging_time_periods(time_periods=mvt_periods,
                                                min_time_between_periods=0.5 * self.abf_sampling_rate)
        print(f"mvt_periods {mvt_periods}")
        speed_during_mvt_periods = []
        for period in mvt_periods:
            speed_during_mvt_periods.append(speed_by_time[period[0]:period[1] + 1])
        return mvt_periods, speed_during_mvt_periods, speed_by_time

    # 50000 * 2

    def detect_mvt_periods_with_piezo_and_diff(self, piezo_data, piezo_threshold, min_time_between_periods,
                                               debug_mode=False):
        binary_piezo_data = np.zeros(len(piezo_data), dtype="int8")
        binary_piezo_data[piezo_data >= piezo_threshold] = 1
        time_periods = get_continous_time_periods(binary_piezo_data)
        return self.merging_time_periods(time_periods=time_periods,
                                         min_time_between_periods=min_time_between_periods, use_peaks=False,
                                         data=piezo_data)

    def merging_time_periods(self, time_periods, min_time_between_periods, use_peaks=False, data=None):
        n_periods = len(time_periods)
        # print(f"n_periods {n_periods}")
        # for i, time_period in enumerate(time_periods):
        #     print(f"time_period {i}: {np.round(time_period[0]/50000, 2)} - {np.round(time_period[1]/50000, 2)}")
        merged_time_periods = []
        index = 0
        self.peaks_raw_piezo = []
        while index < n_periods:
            time_period = time_periods[index]
            if len(merged_time_periods) == 0:
                merged_time_periods.append([time_period[0], time_period[1]])
                index += 1
                continue
            # we check if the time between both is superior at min_time_between_periods
            last_time_period = time_periods[index - 1]
            beg_time = last_time_period[1]
            end_time = time_period[0]
            if use_peaks:
                beg_time = last_time_period[0] + np.argmax(data[last_time_period[0]:last_time_period[1] + 1])
                if index == 1:
                    self.peaks_raw_piezo.append(beg_time)
                end_time = time_period[0] + np.argmax(data[time_period[0]:time_period[1] + 1])
                self.peaks_raw_piezo.append(end_time)
            if (end_time - beg_time) <= min_time_between_periods:
                # then we merge them
                merged_time_periods[-1][1] = time_period[1]
                index += 1
                continue
            else:
                merged_time_periods.append([time_period[0], time_period[1]])
            index += 1
        return merged_time_periods

    # TODO: A second one with sliding window (200 ms) with low percentile threshold
    def detect_mvt_periods_with_sliding_window(self, piezo_data, window_duration, piezo_threshold,
                                               min_time_between_periods,
                                               debug_mode=False):
        """
        Use a sliding window to detect sce (define as peak of activity > perc_threshold percentile after
        randomisation during a time corresponding to window_duration)
        :param spike_nums: 2D array, lines=cells, columns=time
        :param window_duration:
        :param perc_threshold:
        :param no_redundancy: if True, then when using the sliding window, a second spike of a cell is not taking into
        consideration when looking for a new SCE
        :return: ** one array (mask, boolean) containing True for indices (times) part of an SCE,
        ** a list of tuple corresponding to the first and last index of each SCE, (last index being included in the SCE)
        ** sce_nums: a new spike_nums with in x axis the SCE and in y axis the neurons, with 1 if
        active during a given SCE.
        ** an array of len n_times, that for each times give the SCE number or -1 if part of no cluster
        ** activity_threshold

        """
        window_threshold = window_duration * piezo_threshold
        n_times = len(piezo_data)
        start_period = -1
        # mvt_periods_bool = np.zeros(n_times, dtype="bool")
        mvt_periods_tuples = []
        # mvt_periods_times_numbers = np.ones(n_times, dtype="int16")
        # mvt_periods_times_numbers *= -1
        if debug_mode:
            print(f"n_times {n_times}")
        for t in np.arange(0, (n_times - window_duration)):
            if debug_mode:
                if t % 10 ** 6 == 0:
                    print(f"t {t}")
            sum_value = np.sum(piezo_data[t:(t + window_duration)])
            if sum_value > window_threshold:
                if start_period == -1:
                    start_period = t
            else:
                if start_period > -1:
                    # then a new SCE is detected
                    # mvt_periods_bool[start_period:t] = True
                    mvt_periods_tuples.append((start_period, (t + window_duration) - 2))
                    # sce_tuples.append((start_period, t-1))
                    # mvt_periods_times_numbers[start_period:t] = len(mvt_periods_tuples) - 1
                    start_period = -1

        # print(f"number of sce {len(sce_tuples)}")
        return self.merging_time_periods(time_periods=mvt_periods_tuples,
                                         min_time_between_periods=min_time_between_periods)
        # return mvt_periods_bool, mvt_periods_tuples, mvt_periods_times_numbers

    def set_avg_cell_map_tif(self, file_name):
        if os.path.isfile(self.param.path_data + file_name):
            self.avg_cell_map_img = mpimg.imread(self.param.path_data + file_name)
            # self.avg_cell_map_img = PIL.Image.open(self.param.path_data + file_name)
            # self.avg_cell_map_img = np.array(self.avg_cell_map_img).astype("uint8")
            self.avg_cell_map_img_file_name = self.param.path_data + file_name

    def load_cnn_cell_classifier_results(self):

        path_to_results = self.param.path_data + "cell_classifier_results_txt/"

        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(path_to_results):
            file_names.extend(local_filenames)
            break
        if len(file_names) == 0:
            return

        for file_name in file_names:
            original_file_name = file_name
            file_name = file_name.lower()
            if (not file_name.startswith(".")) and file_name.endswith(".txt") and ('cnn' in file_name) and \
                    (self.description.lower() in file_name):
                self.cell_cnn_predictions = []
                with open(path_to_results + '/' + original_file_name, "r", encoding='UTF-8') as file:
                    for nb_line, line in enumerate(file):
                        line_list = line.split()
                        cells_list = [float(i) for i in line_list]
                        self.cell_cnn_predictions.extend(cells_list)
                self.cell_cnn_predictions = np.array(self.cell_cnn_predictions)
                return

    def load_seq_pca_results(self, path):
        """
        Load matlab results of the search of seq using PCA
        :param path:
        :return:
        """
        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(self.param.path_data, path)):
            file_names.extend(local_filenames)
            break
        if len(file_names) == 0:
            return
        dc_dict = dict()
        dc_shift_dict = dict()
        x_del_dict = dict()
        for file_name in file_names:
            # first we get the pc_number
            if ("Comp" in file_name) and (file_name.endswith(".mat")):
                index_comp = file_name.index("Comp")
                index_mat = file_name.index(".mat")
                pc_number = int(file_name[index_comp + 4:index_mat])
            else:
                continue
            if ("DC" in file_name) and (self.description.lower() in file_name.lower()) \
                    and (file_name.endswith(".mat")) and ("shift" not in file_name.lower()):
                dc_var = hdf5storage.loadmat(os.path.join(self.param.path_data,
                                                          path, file_name))
                dc_dict[pc_number] = dc_var["DC"][0]

            if ("dcshift" in file_name.lower()) and (self.description.lower() in file_name.lower()) \
                    and (file_name.endswith(".mat")):
                data = hdf5storage.loadmat(os.path.join(self.param.path_data,
                                                        path, file_name))
                dc_dict[pc_number] = data["ShiftDC"][0]
            if ("xdel" in file_name.lower()) and (self.description.lower() in file_name.lower()) \
                    and (file_name.endswith(".mat")):
                data = hdf5storage.loadmat(os.path.join(self.param.path_data,
                                                        path, file_name))
                x_del_dict[pc_number] = data["xDel"][0]

        if (dc_dict is None) or (x_del_dict is None):
            return
        for pc_number, dc_array in dc_dict.items():
            # -1 because it's coming from matlab
            cells_order = dc_array[x_del_dict[pc_number] - 1] - 1
            if self.pca_seq_cells_order is None:
                self.pca_seq_cells_order = dict()
            self.pca_seq_cells_order[pc_number] = cells_order

    def load_tif_movie(self, path):
        """
        Don't load the tif movie in memory but will look for the tiff in the path, the tiff should have the name
        of the mouse session defined by self.description(), unsensitive of the case
        :param path:
        :param non_corrected: if True, load movie from "non_corrected" dir
        :return:
        """
        if self.tif_movie_file_name is not None:
            return
        file_names = []


        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(self.param.path_data, path)):
            file_names.extend(local_filenames)
            break
        if len(file_names) == 0:
            return

        for file_name in file_names:
            file_name_original = file_name
            file_name = file_name.lower()
            descr = self.description.lower() + ".tif"
            descr_mot_corr = self.description.lower() + "_motcorr" + ".tif"
            descr_ff = self.description.lower() + ".tiff"
            if (descr != file_name) and (descr_ff != file_name) and (descr_mot_corr != file_name):
                continue
            self.tif_movie_file_name = os.path.join(self.param.path_data, path, file_name_original)
            # print(f"self.tif_movie_file_name {self.tif_movie_file_name}")
        self.load_movie_dimensions()
        # print(f"{self.description}: self.movie_len_x {self.movie_len_x}, self.movie_len_y {self.movie_len_y}")
        # raise Exception("JOJO")

    def load_raster_dur_from_predictions(self, prediction_threshold, variables_mapping, file_name=None,
                                         path_name=None, prediction_keys=None, use_filtered_version=False):
        """
        Loader raster_dur from either a prediction file using threshold at 0.5 to keep the predictions
        or will load if available to a file with also prediction_key on it but with the name
        predicted_raster_dur that would have been produce by this function in order to remove
        potential false_transient
        :param prediction_threshold:
        :param variables_mapping:
        :param file_name:
        :param path_name: if given, will look for a file in this directory with key_prediction on it
        :param prediction_keys: could be a string or a list of string, in that case priority to the first one then next
        etc.. until a file with prediction_key is found
        :param use_filtered_version: if True, will filter the predicted raster_dur created base on predictions
        in order to remove fake transients that could have been missed using co-activation and source transient profile
        correlation. if use_filtered_version, path_name must be incicated
        :return:
        """

        if (file_name is None) and (path_name is None):
            print(f"{self.description}: load_raster_dur_from_predictions no file_name or path_name")
            return

        if use_filtered_version and (path_name is None):
            print(f"{self.description} For using use_filtered_version you need to indicated a path_name")
            return

        if isinstance(prediction_keys, str):
            prediction_keys = [prediction_keys]

        filtered_version_loaded = False
        if use_filtered_version:
            # first we look for a saved version of a filtered version
            data = None
            # loading predictions
            file_names = []
            # look for filenames in the fisrst directory, if we don't break, it will go through all directories
            for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(self.param.path_data,
                                                                             path_name)):
                file_names.extend(local_filenames)
                break

            if len(file_names) > 0:
                for file_name in file_names:
                    if (prediction_keys[0] in file_name) and ("filtered_predicted_raster_dur" in file_name):
                        self.spike_struct.spike_nums_dur = np.load(os.path.join(self.param.path_data,
                                                                                path_name, file_name),
                                                                   allow_pickle=True)
                        print(f"{self.description}: filtered_version_loaded")
                        filtered_version_loaded = True
                        break
        predictions = None
        if path_name is not None:
            data = None
            # loading predictions
            file_names = []
            # look for filenames in the fisrst directory, if we don't break, it will go through all directories
            for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(self.param.path_data,
                                                                             path_name)):
                file_names.extend(local_filenames)
                break

            if len(file_names) > 0:
                for prediction_key in prediction_keys:
                    for file_name in file_names:
                        if (prediction_key in file_name) and ("filtered_predicted_raster_dur" not in file_name) \
                                and (file_name.endswith(".mat")):
                            data = hdf5storage.loadmat(os.path.join(self.param.path_data,
                                                                    path_name, file_name))
                        elif (prediction_key in file_name) and ("filtered_predicted_raster_dur" not in file_name) \
                                and (file_name.endswith(".npy")):
                            predictions = np.load(os.path.join(self.param.path_data,
                                                               path_name, file_name), allow_pickle=True)
                        elif (prediction_key in file_name) and ("filtered_predicted_raster_dur" not in file_name) \
                                and (file_name.endswith(".npz")):
                            data = np.load(os.path.join(self.param.path_data,
                                                        path_name, file_name), allow_pickle=True)
                        else:
                            continue
                        break
                    if (data is not None) or (predictions is not None):
                        break
            if (data is None) and (predictions is None):
                print(f"load_raster_dur_from_predictions no file_name with {prediction_keys} found in "
                          f"{os.path.join(self.param.path_data, path_name)}")
                return
        else:
            try:
                data = hdf5storage.loadmat(os.path.join(self.param.path_data, file_name))
            except (FileNotFoundError, OSError) as e:
                print(f"Load_raster_dur_from_predictions File not fount: {file_name}")
                return

        if (self.tiff_movie is None) and use_filtered_version and (not filtered_version_loaded):
            self.load_tiff_movie_in_memory()
            if self.tiff_movie is None:
                raise Exception(f"{self.description}, load_raster_dur_from_predictions: movie could not be loaded")
            self.normalize_movie()

        if (predictions is not None) or ("predictions" in variables_mapping):
            # predictions might already be loaded if we use a npy file
            if predictions is None:
                predictions = data[variables_mapping["predictions"]]
            self.rnn_transients_predictions = predictions
            if not filtered_version_loaded:
                # then we produce the raster dur based on the predictions using threshold the prediction_threshold
                predicted_raster_dur_dict = np.zeros((len(predictions), len(predictions[0])), dtype="int8")
                for cell in np.arange(len(predictions)):
                    pred = predictions[cell]
                    # predicted_raster_dur_dict[cell, pred >= predictions_threshold] = 1
                    if len(pred.shape) == 1:
                        predicted_raster_dur_dict[cell, pred >= prediction_threshold] = 1
                    elif (len(pred.shape) == 2) and (pred.shape[1] == 1):
                        pred = pred[:, 0]
                        predicted_raster_dur_dict[cell, pred >= prediction_threshold] = 1
                    elif (len(pred.shape) == 2) and (pred.shape[1] == 3):
                        # real transient, fake ones, other (neuropil, decay etc...)
                        # keeping predictions about real transient when superior
                        # to other prediction on the same frame
                        max_pred_by_frame = np.max(pred, axis=1)
                        real_transient_frames = (pred[:, 0] == max_pred_by_frame)
                        predicted_raster_dur_dict[cell, real_transient_frames] = 1
                    elif pred.shape[1] == 2:
                        # real transient, fake ones
                        # keeping predictions about real transient superior to the threshold
                        # and superior to other prediction on the same frame
                        max_pred_by_frame = np.max(pred, axis=1)
                        real_transient_frames = np.logical_and((pred[:, 0] >= prediction_threshold),
                                                               (pred[:, 0] == max_pred_by_frame))
                        predicted_raster_dur_dict[cell, real_transient_frames] = 1
                self.spike_struct.spike_nums_dur = predicted_raster_dur_dict
        else:
            raise Exception("load_raster_dur_from_predictions no predicions variable")

        print(f"Predictions {prediction_key} are loaded for {self.description}")
        self.spike_struct.n_cells = len(self.spike_struct.spike_nums_dur)
        if self.spike_struct.labels is None:
            self.spike_struct.labels = np.arange(len(self.spike_struct.spike_nums_dur))
        if self.spike_struct.n_in_matrix is None:
            self.spike_struct.n_in_matrix = np.zeros((self.spike_struct.n_cells, self.spike_struct.n_cells))
            self.spike_struct.n_out_matrix = np.zeros((self.spike_struct.n_cells, self.spike_struct.n_cells))

        if use_filtered_version and (not filtered_version_loaded):
            if self.raw_traces is None:
                self.raw_traces = self.build_raw_traces_from_movie()
            # we need peak_nums and spike_nums
            # it should be called again at the end of this function
            # after filtering the spike_nums_dur
            # usually done in mouse_session_loader
            self.spike_struct.build_spike_nums_and_peak_nums()

            n_cells = self.spike_struct.n_cells
            n_frames = self.spike_struct.spike_nums_dur.shape[1]
            spike_nums_dur_numbers = \
                give_unique_id_to_each_transient_of_raster_dur(raster_dur=self.spike_struct.spike_nums_dur)

            # then we create a filter version, to remove fake transient due to overlap that would have
            # been wrongly classified and save the prediction
            # we will modify directly self.spike_struct.spike_nums_dur

            # we go cell by cell, identfy their overlapping cells, list the co-activated transient
            # then calculate the source profile of the cells and the profile of each transient,
            # and the correlation
            # between the sources and transients
            overlapping_cells = self.coord_obj.intersect_cells
            source_profile_dict = dict()
            n_co_active_transients_detected = 0
            source_profile_corr_dict = dict()
            # to plot distribution
            n_fake_transients_by_cell = np.zeros(n_cells, dtype="int16")
            fake_transients_periods_by_cell_dict = dict()
            # keep tuple of int to check if the co-active transient of those 2 cells have been verified already
            pair_of_cells_already_checked_dict = dict()
            for cell in np.arange(n_cells):
                intersect_cells = overlapping_cells[cell]
                # first we check if at least a cell has a transient in common with an overlapping cell
                if len(intersect_cells) == 0:
                    continue
                cell_active_frames = np.where(self.spike_struct.spike_nums_dur[cell])[0]
                # key: cells with co_active transients, and value is a list of tuple (onset, peak)
                co_active_frames_dict = dict()
                for intersect_cell in intersect_cells:
                    intersect_cell_active_frames = np.where(self.spike_struct.spike_nums_dur[intersect_cell])[0]
                    common_active_frames = np.intersect1d(cell_active_frames, intersect_cell_active_frames)
                    if len(common_active_frames) == 0:
                        continue
                    binary_frames = np.zeros(n_frames, dtype="int8")
                    binary_frames[common_active_frames] = 1
                    # produce a list of tuple representing the onset and frames of each transient
                    co_active_transients = get_continous_time_periods(binary_frames)
                    if intersect_cell not in co_active_frames_dict:
                        co_active_frames_dict[intersect_cell] = []
                    co_active_frames_dict[intersect_cell].extend(co_active_transients)
                    n_co_active_transients_detected += len(co_active_transients)

                if len(co_active_frames_dict) == 0:
                    # it means no co-active transients
                    continue

                # print(f"{cell} {len(co_active_frames_dict)}-> {co_active_frames_dict}")
                co_active_cells = list(co_active_frames_dict.keys())
                all_cells = list(co_active_cells)
                all_cells.append(cell)
                # main_source_profile = None
                for cell_for_profile in all_cells:
                    if cell_for_profile not in source_profile_dict:
                        source_profile, minx, miny, mask_source_profile = \
                            self.coord_obj.get_source_profile(tiff_movie=self.tiff_movie, traces=self.raw_traces,
                                                              peak_nums=self.spike_struct.peak_nums,
                                                              spike_nums=self.spike_struct.spike_nums,
                                                              cell=cell_for_profile,
                                                              pixels_around=1,
                                                              bounds=None)
                        xy_source = self.coord_obj.get_cell_new_coord_in_source(cell=cell_for_profile,
                                                                                minx=minx, miny=miny)
                        source_profile_dict[cell_for_profile] = [source_profile, minx, miny, mask_source_profile,
                                                                 xy_source]

                for co_active_cell, co_active_transients in co_active_frames_dict.items():
                    if (cell, co_active_cell) not in pair_of_cells_already_checked_dict:
                        # then if we loop starting with co_active_cell as cell, we will skip this loop
                        pair_of_cells_already_checked_dict[(co_active_cell, cell)] = True
                    else:
                        continue

                    # co_active_transients is a list of
                    # tuple of int, reprensenting the frame of the onset and the frame of the peak
                    for co_active_transient in co_active_transients:
                        # we want to use the all transient of the cell to measure correlation
                        # not just the part that is co-active
                        transient_ids = spike_nums_dur_numbers[cell,
                                        co_active_transient[0]:co_active_transient[1] + 1]
                        transient_ids = np.unique(transient_ids)
                        transient_id = transient_ids[transient_ids >= 0][0]
                        frames_to_remove = np.where(spike_nums_dur_numbers[cell] == transient_id)[0]
                        transient_cell = (frames_to_remove[0], frames_to_remove[-1])

                        transient_ids = spike_nums_dur_numbers[co_active_cell,
                                        co_active_transient[0]:co_active_transient[1] + 1]
                        transient_ids = np.unique(transient_ids)
                        transient_id = transient_ids[transient_ids >= 0][0]
                        frames_to_remove = np.where(spike_nums_dur_numbers[co_active_cell] == transient_id)[0]
                        transient_co_active_cell = (frames_to_remove[0], frames_to_remove[-1])

                        pearson_corr_cell = \
                            self.coord_obj.corr_between_source_and_transient(cell=cell,
                                                                             transient=transient_cell,
                                                                             source_profile_dict=source_profile_dict,
                                                                             tiff_movie=self.tiff_movie,
                                                                             traces=self.raw_traces,
                                                                             source_profile_corr_dict=
                                                                             source_profile_corr_dict,
                                                                             pixels_around=1)

                        pearson_corr_co_active_cell = \
                            self.coord_obj.corr_between_source_and_transient(cell=co_active_cell,
                                                                             transient=transient_co_active_cell,
                                                                             source_profile_dict=source_profile_dict,
                                                                             tiff_movie=self.tiff_movie,
                                                                             traces=self.raw_traces,
                                                                             source_profile_corr_dict=source_profile_corr_dict,
                                                                             pixels_around=1)
                        # pearson_corr = np.round(pearson_corr, 2)

                        # cell from which removing a transient
                        cell_to_use = None
                        transient_to_remove = None
                        if (pearson_corr_cell < 0.4) and (pearson_corr_co_active_cell > 0.6):
                            # then we conclude that the transient in cell is Fake
                            # we need to remove this transient for spike_nums_dur
                            cell_to_use = cell
                            transient_to_remove = transient_cell
                        elif (pearson_corr_cell > 0.6) and (pearson_corr_co_active_cell < 0.4):
                            # then we conclude that the transient in co_active_cell is Fake
                            cell_to_use = co_active_cell
                            transient_to_remove = transient_co_active_cell
                        if cell_to_use is None:
                            continue
                        # removing the transient
                        self.spike_struct.spike_nums_dur[cell_to_use,
                        transient_to_remove[0]:transient_to_remove[1] + 1] = 0
                        n_fake_transients_by_cell[cell_to_use] += 1
                        if cell_to_use not in fake_transients_periods_by_cell_dict:
                            fake_transients_periods_by_cell_dict[cell_to_use] = []
                        fake_transients_periods_by_cell_dict[cell_to_use].append(transient_to_remove)

            total_n_fake_transients = np.sum(n_fake_transients_by_cell)
            total_n_transients = np.sum(self.spike_struct.spike_nums)
            print(f"{self.description}, n_co_active_transients_detected: "
                  f"{n_co_active_transients_detected}, n_fake_transients {total_n_fake_transients}, "
                  f"total_n_transients {total_n_transients}")

            save_formats = "pdf"

            display_misc.plot_hist_distribution(distribution_data=n_fake_transients_by_cell,
                                                description=f"{self.description}_fake_transients_distribution_by_cell:",
                                                param=self.param,
                                                path_results=self.param.path_results,
                                                tight_x_range=True,
                                                twice_more_bins=True,
                                                xlabel="N fake transients by cell", save_formats=save_formats)

            file_name = f'{self.param.path_results}/{self.description}_stat_fake_transients_distribution_by_' \
                        f'cell_{self.param.time_str}.txt'

            with open(file_name, "w", encoding='UTF-8') as file:
                file.write(f"N fake transients by cell for {self.description}" + '\n')
                file.write("" + '\n')
                file.write(f"n_co_active_transients_detected: "
                           f"{n_co_active_transients_detected}, n_fake_transients {total_n_fake_transients}, "
                           f"total_n_transients {total_n_transients}")

                file.write("" + '\n')

                for cell in np.arange(n_cells):
                    file.write(f"{cell}: {n_fake_transients_by_cell[cell]}")
                    if cell in fake_transients_periods_by_cell_dict:
                        file.write(': ')
                        for period in fake_transients_periods_by_cell_dict[cell]:
                            file.write(f"{period}, ")
                    file.write('\n')

            file_name = f"{self.description}_filtered_predicted_raster_dur_{prediction_key}.npy"
            np.save(os.path.join(self.param.path_data, path_name, file_name), self.spike_struct.spike_nums_dur)

    def load_richard_data(self, path_data):
        print(f"{self.description} loading_data")
        # Wake_Frames, Active_Wake_Frames, Quiet_Wake_Frames, REMs_Frames, NREMs_Frames
        keys = ["Wake_Frames", "Active_Wake_Frames", "Quiet_Wake_Frames", "REMs_Frames", "NREMs_Frames"]
        # Tread_Position_cm.mat
        self.richard_dict = dict()
        for key in keys:
            self.richard_dict[key] = np.zeros(0, dtype="int16")
        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(self.param.path_data + path_data):
            file_names.extend(local_filenames)
            break

        if len(file_names) == 0:
            return
        n_frames = 0
        if "Tread_Position_cm.mat" in file_names:
            file_name = os.path.join(self.param.path_data, path_data, "Tread_Position_cm.mat")
            data = hdf5storage.loadmat(file_name)
            tread_position = data["Tread_Position_cm"][0]
            n_frames = len(tread_position)
            active_frames = []
            tread_position_diff = np.diff(tread_position)
            active_frames_diff = np.where(tread_position_diff != 0)[0]
            for frame in active_frames_diff:
                if frame < 5:
                    # first frames might be artefact
                    continue
                # if 2 frames are spaced of less than 5 frames, then we consider the mice was still running in between
                n_frames_to_fusion = 20
                if (len(active_frames) > 0) and (frame - active_frames[-1] < n_frames_to_fusion):
                    active_frames.extend(list(range(active_frames[-1] + 1, frame)))
                active_frames.extend(list(range(max(0, frame - 5), frame + 2)))
            print(f"len(active_frames) {len(active_frames)}")
            self.richard_dict["Active_Wake_Frames"] = np.unique(active_frames)
        else:
            print(f"No file Tread_Position_cm.mat for {self.description}")
            return

        Hypnogram_Frames = None
        if "Hypnogram_Frames.mat" in file_names:
            file_name = os.path.join(self.param.path_data, path_data, "Hypnogram_Frames.mat")
            data = hdf5storage.loadmat(file_name)
            Hypnogram_Frames = data["Hypnogram_Frames"]
            # 1=wake, 2=nrems, 3=rems
            wake_frames = np.where(Hypnogram_Frames == 1)[0]
            self.richard_dict["Wake_Frames"] = wake_frames
            self.richard_dict["Quiet_Wake_Frames"] = np.setdiff1d(wake_frames,
                                                                  self.richard_dict["Active_Wake_Frames"])
            self.richard_dict["REMs_Frames"] = np.where(Hypnogram_Frames == 2)[0]
            self.richard_dict["NREMs_Frames"] = np.where(Hypnogram_Frames == 3)[0]
        else:
            self.richard_dict["Wake_Frames"] = np.arange(n_frames)
            self.richard_dict["Quiet_Wake_Frames"] = np.setdiff1d(np.arange(n_frames),
                                                                  self.richard_dict["Active_Wake_Frames"])

    def load_raw_traces_from_npy(self, path):
        for (dirpath, dirnames, local_filenames) in os.walk(self.param.path_data + path):
            for file_name in local_filenames:
                if (self.description.lower() in file_name.lower()) and ("raw_traces" in file_name.lower()) \
                        and file_name.endswith(".npy"):
                    self.raw_traces = np.load(os.path.join(self.param.path_data, path, file_name))
                    print(f"{self.description} raw traces loaded from file npy")
                    return True
            break
        print(f"{self.description} raw traces not loaded")
        return False

    def save_raw_traces(self, path):
        if self.raw_traces is None:
            print(f"{self.description} raw traces None, not saved")
            return
        np.save(os.path.join(self.param.path_data, path, f"{self.description}_raw_traces.npy".lower()),
                self.raw_traces)

    def load_raw_motion_translation_shift_data(self, path_to_load):
        """
        Load data from xy_translation after motion correction
        :param path_to_load:
        :return:
        """
        if path_to_load is None:
            print(f"{self.description} load_raw_motion_translation_shift_data "
                  f"path_to_load is None")
            return

        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(self.param.path_data, path_to_load)):
            for file_name in local_filenames:
                if (("params" in file_name.lower()) and (self.description.lower() in file_name.lower())) \
                        and file_name.endswith(".mat"):
                    data = hdf5storage.loadmat(os.path.join(self.param.path_data, path_to_load, file_name))
                    variables_mapping = {"xshifts": "xshifts",
                                         "yshifts": "yshifts"}
                    self.x_shifts = data[variables_mapping["xshifts"]][0]
                    self.y_shifts = data[variables_mapping["yshifts"]][0]
                elif (("params" in file_name.lower()) and (self.description.lower() in file_name.lower())) \
                        and file_name.endswith(".npy"):

                    ops = np.load(os.path.join(self.param.path_data, path_to_load, file_name), allow_pickle=True)
                    data = ops.item()

                    variables_mapping = {"xshifts": "xoff",
                                         "yshifts": "yoff"}
                    self.x_shifts = data[variables_mapping["xshifts"]]
                    self.y_shifts = data[variables_mapping["yshifts"]]
            break

    def load_graph_data(self, path_to_load):
        """
                        Load the graph data in path_to_load for the ms
                        :param path_to_load:
                        :return:
                        """
        if self.spike_struct.graph_out is not None:
            return

        if (path_to_load is None):
            print(f"{self.description} load_graph_data "
                  f"path_to_load is None")
            return

        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(self.param.path_data, path_to_load)):
            for file_name in local_filenames:
                if (("graph" in file_name.lower()) and (self.description.lower() in file_name.lower())) \
                        and file_name.endswith(".graphml"):

                    graph = nx.read_graphml(path=(os.path.join(self.param.path_data, path_to_load, file_name)),
                                            node_type=int)
                    if "graph_out" in file_name.lower():
                        print(f"{self.description} graph_out loaded from file")
                        self.spike_struct.graph_out = graph
                    elif "graph_in" in file_name.lower():
                        print(f"{self.description} graph_in loaded from file")
                        self.spike_struct.graph_in = graph

            break
        if self.spike_struct.graph_out is None:
            print(f"{self.description} no graph data file found")

    def load_lfp_data(self, path_to_load):
        """
                Load the lfp data in path_to_load for the ms
                :param path_to_load:
                :return:
                """
        if self.lfp_signal is not None:
            return

        if (path_to_load is None):
            print(f"{self.description} load_lfp_data "
                  f"path_to_load is None")
            return

        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(self.param.path_data, path_to_load)):
            for file_name in local_filenames:
                if (("lfp" in file_name.lower()) and (self.description.lower() in file_name.lower())) \
                        and file_name.endswith(".npy") and ("hz" in file_name.lower()):
                    self.lfp_signal = np.load(os.path.join(self.param.path_data, path_to_load, file_name))
                    index_npy = file_name.index(".npy")
                    index_ = len(file_name) - file_name[::-1].index("_") - 1
                    # format exemple: P7_19_03_05_a000_abf_lfp_channel_3_1000hz.npy
                    self.lfp_sampling_rate = int(file_name[index_ + 1:index_npy - 2])
                    # print(f"{file_name} {self.lfp_sampling_rate}")
                    return
            break
        print(f"{self.description} no lfp data file found")

    def load_speed_from_file(self, path_to_load):
        """
        Load the speed file in path_to_load for the ms
        :param path_to_load:
        :return:
        """
        if self.speed_by_frame is not None:
            return

        if (path_to_load is None):
            print(f"{self.description} load_speed_from_file "
                  f"path_to_load is None")
            return

        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(self.param.path_data, path_to_load)):
            for file_name in local_filenames:
                if (("speed" in file_name.lower()) and (self.description.lower() in file_name.lower())) \
                        and file_name.endswith(".npy"):
                    self.speed_by_frame = np.load(os.path.join(self.param.path_data, path_to_load, file_name))
                    return
            break
        print(f"{self.description} no speed data file found")

    def load_data_from_period_selection_gui(self, variables_mapping, file_name_to_load=None, path_to_load=None):
        if self.shift_data_dict is not None:
            return

        if (file_name_to_load is None) and (path_to_load is None):
            print(f"{self.description} load_data_from_period_selection_gui "
                  f"file_name_to_load and path_to_load are None")
            return

        if file_name_to_load is not None:
            if not file_name_to_load.endswith(".npz"):
                print(f"load_data_from_period_selection_gui not a npz file {file_name_to_load}")
                return
            try:
                data = np.load(os.path.join(self.param.path_data, file_name_to_load))
                self.shift_data_dict = dict()
                for key, value in variables_mapping.items():
                    self.shift_data_dict[key] = data[value]
            except (FileNotFoundError, OSError) as e:
                print(f"File not found: {file_name_to_load} in load_data_from_period_selection_gui {self.description}")
                return
        else:
            shift_data_found = False
            for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(self.param.path_data, path_to_load)):
                for file_name in local_filenames:
                    if (("mvt_categories" in file_name.lower()) or ("mvts_categories" in file_name.lower())) \
                            and file_name.endswith(".npz"):
                        data = np.load(os.path.join(self.param.path_data, path_to_load, file_name))
                        self.shift_data_dict = dict()
                        for key, value in variables_mapping.items():
                            self.shift_data_dict[key] = data[value]
                        shift_data_found = True
                break
            if not shift_data_found:
                print(f"{self.description} no period_selection_gui data found")

    def load_data_from_file(self, file_name_to_load, variables_mapping, frames_filter=None,
                            from_gui=False, from_fiji=False, save_caiman_apart=False):
        """

        :param file_name_to_load:
        :param variables_mapping:
        :param frames_filter: if not None, will keep only the frames in frames_filter
        :param save_caiman_apart: means we save caiman raster dur in a special variable caiman_spike_nums_dur
        :return:
        """
        matlab_format = True
        try:
            if file_name_to_load.endswith(".npy") and ("params" in file_name_to_load.lower()):
                matlab_format = False
                # valid only for shifts data so far
                ops = np.load(os.path.join(self.param.path_data, file_name_to_load))
                data = ops.item()
            elif file_name_to_load.endswith(".npz"):
                matlab_format = False
                data = np.load(os.path.join(self.param.path_data, file_name_to_load))
            else:
                data = hdf5storage.loadmat(os.path.join(self.param.path_data, file_name_to_load))
        except (FileNotFoundError, OSError) as e:
            print(f"File not found: {os.path.join(self.param.path_data, file_name_to_load)}")
            return
        # print(f'load_data_from_file: {list(data.keys())}')
        if "shift_periods_bool" in variables_mapping:
            if matlab_format is False:
                self.shift_periods_bool = data[variables_mapping["shift_periods_bool"]]
        if "spike_nums" in variables_mapping:
            self.spike_struct.spike_nums = data[variables_mapping["spike_nums"]].astype(int)
            if frames_filter is not None:
                self.spike_struct.spike_nums = self.spike_struct.spike_nums[:, frames_filter]
            if self.spike_struct.labels is None:
                self.spike_struct.labels = np.arange(len(self.spike_struct.spike_nums))
            if self.spike_struct.n_cells is None:
                self.spike_struct.n_cells = len(self.spike_struct.spike_nums)
            if self.spike_struct.n_in_matrix is None:
                self.spike_struct.n_in_matrix = np.zeros((self.spike_struct.n_cells, self.spike_struct.n_cells))
                self.spike_struct.n_out_matrix = np.zeros((self.spike_struct.n_cells, self.spike_struct.n_cells))
        if "global_roi" in variables_mapping:
            self.global_roi = data[variables_mapping["global_roi"]][0]
        if "doubtful_frames_nums" in variables_mapping:
            if variables_mapping["doubtful_frames_nums"] in data:
                self.doubtful_frames_nums = data[variables_mapping["doubtful_frames_nums"]].astype(int)
        if "xshifts" in variables_mapping:
            if matlab_format:
                self.x_shifts = data[variables_mapping["xshifts"]][0]
            else:
                self.x_shifts = data[variables_mapping["xshifts"]]
        if "yshifts" in variables_mapping:
            if matlab_format:
                self.y_shifts = data[variables_mapping["yshifts"]][0]
            else:
                self.y_shifts = data[variables_mapping["yshifts"]]
        if "spike_nums_dur" in variables_mapping:
            if save_caiman_apart:
                self.caiman_spike_nums_dur = data[variables_mapping["spike_nums_dur"]].astype(int)
            else:
                self.spike_struct.spike_nums_dur = data[variables_mapping["spike_nums_dur"]].astype(int)
                if frames_filter is not None:
                    self.spike_struct.spike_nums_dur = self.spike_struct.spike_nums_dur[:, frames_filter]
                if self.spike_struct.labels is None:
                    self.spike_struct.labels = np.arange(len(self.spike_struct.spike_nums_dur))
                if self.spike_struct.n_cells is None:
                    self.spike_struct.n_cells = len(self.spike_struct.spike_nums_dur)
                if self.spike_struct.n_in_matrix is None:
                    self.spike_struct.n_in_matrix = np.zeros((self.spike_struct.n_cells, self.spike_struct.n_cells))
                    self.spike_struct.n_out_matrix = np.zeros((self.spike_struct.n_cells, self.spike_struct.n_cells))
        if "traces" in variables_mapping:
            self.traces = data[variables_mapping["traces"]].astype(float)
            if frames_filter is not None:
                self.traces = self.traces[:, frames_filter]
        if "raw_traces" in variables_mapping:
            self.raw_traces = data[variables_mapping["raw_traces"]].astype(float)
            if frames_filter is not None:
                self.raw_traces = self.raw_traces[:, frames_filter]
            self.smooth_traces = np.copy(self.raw_traces)
            # smoothing the trace
            windows = ['hanning', 'hamming', 'bartlett', 'blackman']
            i_w = 1
            window_length = 11
            for i in np.arange(self.raw_traces.shape[0]):
                smooth_signal = smooth_convolve(x=self.smooth_traces[i], window_len=window_length,
                                                window=windows[i_w])
                beg = (window_length - 1) // 2
                self.smooth_traces[i] = smooth_signal[beg:-beg]
        if "coord" in variables_mapping:
            # coming from matlab
            self.coord = data[variables_mapping["coord"]][0]
            self.coord_obj = CoordClass(coord=self.coord, nb_col=self.movie_len_x,
                                        nb_lines=self.movie_len_y, from_fiji=from_fiji)
        if "spike_durations" in variables_mapping:
            self.spike_struct.set_spike_durations(data[variables_mapping["spike_durations"]])
        elif self.spike_struct.spike_nums_dur is not None:
            self.spike_struct.set_spike_durations()
        if "spike_amplitudes" in variables_mapping:
            self.spike_struct.set_spike_amplitudes(data[variables_mapping["spike_amplitudes"]])

        if "cells_to_remove" in variables_mapping:
            if len(data[variables_mapping["cells_to_remove"]]) == 0:
                self.cells_to_remove = np.zeros(0, dtype="int16")
            else:
                self.cells_to_remove = data[variables_mapping["cells_to_remove"]].astype(int)[0]

        if "peak_nums" in variables_mapping:
            self.spike_struct.peak_nums = data[variables_mapping["peak_nums"]].astype(int)
        self.spike_struct.set_spike_trains_from_spike_nums()

        # if (self.spike_struct.spike_nums_dur is not None) or (self.spike_struct.spike_nums is not None):
        #     self.detect_n_in_n_out()

    def load_cells_to_remove_from_txt(self, file_name):
        cells_to_remove = []
        with open(self.param.path_data + file_name, "r", encoding='UTF-8') as file:
            for nb_line, line in enumerate(file):
                line_list = line.split()
                cells_list = [int(i) for i in line_list]
                cells_to_remove.extend(cells_list)
        # print(f"cells_to_remove {cells_to_remove}")
        self.cells_to_remove = cells_to_remove

    def clean_data_using_cells_to_remove(self):
        # temporarly deactivated for converting GT to cinac format
        # return

        if (self.cells_to_remove is None) or len(self.cells_to_remove) == 0:
            return
        n_cells = self.spike_struct.n_cells
        self.removed_cells_mapping = np.ones(n_cells, dtype="int16")
        self.removed_cells_mapping *= -1
        new_coord = []
        new_cell_index = 0
        for cell in np.arange(n_cells):
            if cell in self.cells_to_remove:
                continue
            new_coord.append(self.coord[cell])
            self.removed_cells_mapping[cell] = new_cell_index
            new_cell_index += 1

        self.coord_obj = CoordClass(coord=new_coord, nb_col=self.movie_len_x,
                                    nb_lines=self.movie_len_y)

        cells_to_remove = np.array(self.cells_to_remove)
        mask = np.ones(n_cells, dtype="bool")
        mask[cells_to_remove] = False
        if self.doubtful_frames_nums is not None:
            self.doubtful_frames_nums = self.doubtful_frames_nums[mask]

        self.spike_struct.clean_data_using_cells_to_remove(cells_to_remove=self.cells_to_remove)
        # raise Exception("titi")

    def get_new_cell_indices_if_cells_removed(self, cell_indices_array):
        """
        Take an array of int, and return another one with new index in case some cells would have been removed
        using clean_data_using_cells_to_remove()
        :param cell_indices_array:
        :return: new_cell_indices_array: indices or existing cell, with new indexing
        original_cell_indices_mapping = for each new cell index, contains the corresponding original index
        """
        if self.removed_cells_mapping is None:
            return np.copy(cell_indices_array), np.copy(cell_indices_array)

        new_cell_indices_array = self.removed_cells_mapping[cell_indices_array]
        # removing cell indices of cell that has been removed
        copy_new_cell_indices_array = np.copy(new_cell_indices_array)
        new_cell_indices_array = new_cell_indices_array[new_cell_indices_array >= 0]
        original_cell_indices_mapping = np.copy(cell_indices_array[copy_new_cell_indices_array >= 0])

        return new_cell_indices_array, original_cell_indices_mapping

    def clean_raster_at_concatenation(self):
        self.spike_struct.clean_raster_at_concatenation()

    def detect_n_in_n_out(self):
        self.spike_struct.detect_n_in_n_out()

    def build_default_doubtful_frames(self):
        """
        Build an empty doubtful_frames matrix (none of the frames are doubtful then)
        self.spike_struct.spike_nums_dur should not be none
        Returns:

        """
        if self.spike_struct.spike_nums_dur is not None:
            self.doubtful_frames_nums = np.zeros(self.spike_struct.spike_nums_dur.shape, dtype="int8")

    def build_spike_nums_dur(self):
        # build spike_nums_dur based on peak_nums and spike_nums
        self.spike_struct.build_spike_nums_dur()
