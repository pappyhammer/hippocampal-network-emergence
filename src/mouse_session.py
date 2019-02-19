import pandas as pd
# from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from bisect import bisect
from scipy import signal
# important to avoid a bug when using virtualenv
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
import time
# import copy
from datetime import datetime
# import keras
import os
import pyabf
import matplotlib.image as mpimg
import random
import networkx as nx
from pattern_discovery.graph.misc import welsh_powell
# to add homemade package, go to preferences, then project interpreter, then click on the wheel symbol
# then show all, then select the interpreter and lick on the more right icon to display a list of folder and
# add the one containing the folder pattern_discovery
from pattern_discovery.seq_solver.markov_way import MarkovParameters
from pattern_discovery.seq_solver.markov_way import find_significant_patterns
from pattern_discovery.seq_solver.markov_way import find_sequences_in_ordered_spike_nums
from pattern_discovery.seq_solver.markov_way import save_on_file_seq_detection_results
import pattern_discovery.tools.misc as tools_misc
from pattern_discovery.tools.misc import get_time_correlation_data
from pattern_discovery.tools.misc import get_continous_time_periods
from pattern_discovery.tools.misc import find_continuous_frames_period
from pattern_discovery.display.raster import plot_spikes_raster
from pattern_discovery.display.misc import time_correlation_graph
from pattern_discovery.display.cells_map_module import CoordClass
from pattern_discovery.tools.sce_detection import get_sce_detection_threshold, detect_sce_with_sliding_window, \
    get_low_activity_events_detection_threshold
from sortedcontainers import SortedList, SortedDict
from pattern_discovery.clustering.kmean_version.k_mean_clustering import compute_and_plot_clusters_raster_kmean_version
from pattern_discovery.clustering.kmean_version.k_mean_clustering import give_stat_one_sce
from pattern_discovery.clustering.fca.fca import compute_and_plot_clusters_raster_fca_version
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
from hne_spike_structure import HNESpikeStructure
from mvt_selection_gui import MvtSelectionGui
from pattern_discovery.tools.signal import smooth_convolve
import PIL


class MouseSession:
    def __init__(self, age, session_id, param, nb_ms_by_frame, weight=None, spike_nums=None, spike_nums_dur=None,
                 percentile_for_low_activity_threshold=1):
        # should be a list of int
        self.param = param
        self.age = age
        self.session_id = str(session_id)
        self.nb_ms_by_frame = nb_ms_by_frame
        self.description = f"P{self.age}_{self.session_id}"
        self.spike_struct = HNESpikeStructure(mouse_session=self, spike_nums=spike_nums, spike_nums_dur=spike_nums_dur)
        # spike_nums represents the onsets of the neuron spikes
        # bin from 25000 frames caiman "onsets"
        self.caiman_spike_nums = None
        # raster_dur built upon the 25000 frames caiman "onsets"
        self.caiman_spike_nums_dur = None
        self.caiman_active_periods = None
        self.traces = None
        self.raw_traces = None
        self.smooth_traces = None
        self.z_score_traces = None
        self.z_score_raw_traces = None
        self.z_score_smooth_traces = None
        self.coord = None
        # comes from the gui
        self.cells_to_remove = None
        # array of float, each index corresponds to a cell and the value is the prediction made by the cell classifier
        self.cell_cnn_predictions = None
        self.load_cnn_cell_classifier_results()
        self.activity_threshold = None
        self.low_activity_threshold_by_percentile = dict()
        self.percentile_for_low_activity_threshold = percentile_for_low_activity_threshold
        self.low_activity_threshold = None
        self.avg_cell_map_img = None
        self.avg_cell_map_img_file_name = None
        self.tif_movie_file_name = None
        # will be use by the cell classifier
        self.tiff_movie = None
        # will be use by the cell and transient classifier, normalize between 0 and 1
        self.tiff_movie_norm_0_1 = None
        self.tiff_movie_normalized = None
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

        if (self.param is not None) and (self.param.cell_assemblies_data_path is not None):
            self.load_cell_assemblies_data()

        # for seq
        self.best_order_loaded = None
        if (self.param is not None) and (self.param.best_order_data_path is not None):
            self.load_best_order_data()
        self.weight = weight
        self.coord_obj = None

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
        start_time = time.time()
        if self.traces is None:
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
        caiman_spike_nums = data_file["spikenums"].astype(int)

        spike_nums_bin = np.zeros((caiman_spike_nums.shape[0], caiman_spike_nums.shape[1] // 2),
                                  dtype="int8")
        for cell in np.arange(spike_nums_bin.shape[0]):
            binned_cell = caiman_spike_nums[cell].reshape(-1, 2).mean(axis=1)
            binned_cell[binned_cell > 0] = 1
            spike_nums_bin[cell] = binned_cell.astype("int")

        self.caiman_spike_nums = spike_nums_bin

        n_cells = self.traces.shape[0]
        n_times = self.traces.shape[1]

        # copying traces
        traces = self.traces[:]

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
                print(f"{self.description} max tiff_movie {str(np.round(max_value, 3))}, "
                      f"mean tiff_movie {str(np.round(np.mean(self.tiff_movie), 3))}, "
                      f"median tiff_movie {str(np.round(np.median(self.tiff_movie), 3))}")
                # self.tiff_movie_normalized = (self.tiff_movie - min_value) / (max_value - min_value)
                self.tiff_movie_normalized = self.tiff_movie / max_value

        if do_z_score_normalization:
            # z-score standardization
            if (self.tiff_movie is not None) and (self.tiff_movie_normalized is None):
                max_value = np.max(self.tiff_movie)
                print(f"{self.description} max tiff_movie {str(np.round(max_value, 3))}, "
                      f"mean tiff_movie {str(np.round(np.mean(self.tiff_movie), 3))}, "
                      f"std tiff_movie {str(np.round(np.std(self.tiff_movie), 3))}")
                self.tiff_movie_normalized = (self.tiff_movie - np.mean(self.tiff_movie)) / np.std(self.tiff_movie)

    def normalize_traces(self):
        n_cells = self.traces.shape[0]
        self.z_score_traces = np.zeros(self.traces.shape)
        self.z_score_raw_traces = np.zeros(self.raw_traces.shape)
        self.z_score_smooth_traces = np.zeros(self.smooth_traces.shape)
        # z_score traces
        for i in np.arange(n_cells):
            self.z_score_traces[i, :] = (self.traces[i, :] - np.mean(self.traces[i, :])) / np.std(self.traces[i, :])
            if self.raw_traces is not None:
                self.z_score_raw_traces[i, :] = (self.raw_traces[i, :] - np.mean(self.raw_traces[i, :])) \
                                        / np.std(self.raw_traces[i, :])
                self.z_score_smooth_traces[i, :] = (self.smooth_traces[i, :] - np.mean(self.smooth_traces[i, :])) \
                                        / np.std(self.smooth_traces[i, :])

    def plot_psth_over_twitches_time_correlation_graph_style(self):
        """
        Same shape as a time-correlation graph but the zeo will correpsond the twitches time, the celle will be
        correlated with itself
        :return:
        """
        if self.twitches_frames_periods is None:
            return

        # results = get_time_correlation_data(spike_nums=self.spike_struct.spike_nums,
        #                                     events_times=self.twitches_frames_periods, time_around_events=10)
        # self.time_lags_list, self.correlation_list, \
        # self.time_lags_dict, self.correlation_dict, self.time_lags_window, cells_list = results
        spike_nums = self.spike_struct.spike_nums
        events_times = self.twitches_frames_periods
        n_twitches = len(events_times)
        print(f"{self.description}: number of twitches {n_twitches}")
        time_around_events = 10

        nb_neurons = len(spike_nums)
        n_times = len(spike_nums[0, :])
        # values for each cell
        time_lags_dict = dict()
        correlation_dict = dict()
        # for ploting
        time_lags_list = []
        correlation_list = []
        cells_list = []

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
                event_time = (event_times[0] + event_times[1]) // 2
                min_limit = max(0, (event_time - time_around_events))
                max_limit = min(event_time + 1 + time_around_events, n_times)
                # min((peak_time + time_window), (n_times - 1))
                if np.sum(spike_nums[neuron, min_limit:max_limit]) == 0:
                    continue
                # see to consider the case in which the cell spikes 2 times around a peak during the tim_window
                neuron_spike_time = spike_nums[neuron, min_limit:max_limit]
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
            adding_this_neuron = True

            if np.sum(distribution_array[time_around_events - 2:]) < 2:
                adding_this_neuron = False

            if np.sum(distribution_array[time_around_events - 2:]) <= 2 * np.sum(
                    distribution_array[:time_around_events - 2]):
                adding_this_neuron = False

            if np.sum(distribution_array[time_around_events - 2:]) > 1:
                nb_spikes = np.sum(distribution_array[time_around_events - 2:])
                print(f"cell {neuron}: distribution_array {distribution_array}")
                # TODO: plot distribution of % ratio
                # TODO: plot piezzo for top cells in distribution
                ratio_cells.append(neuron)
                ratio_spike_twitch_total_twitches.append((nb_spikes / n_twitches) * 100)
                ratio_spike_twitch_total_spikes.append((nb_spikes / np.sum(spike_nums[neuron, :])) * 100)
                print(f"ratio spikes after twitch / n_twitches: "
                      f"{np.round(nb_spikes / n_twitches, 3)}")
                print(f"ratio spikes after twitch / n_spikes: "
                      f"{np.round(nb_spikes / np.sum(spike_nums[neuron, :]), 3)}")

            # adding the cell only if it has at least a spike around peak times
            if adding_this_neuron:
                correlation_value = np.max(distribution_array) / total_spikes
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

        inter_neurons = np.array(self.spike_struct.inter_neurons)
        # keeping only interneurons that are among the selected cells
        inter_neurons = np.intersect1d(ratio_cells, inter_neurons)
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
        if len(ratio_interneurons) > 0:
            values_to_scatter.append(np.mean(ratio_interneurons))
            values_to_scatter.extend(ratio_interneurons)
            labels.extend(["mean", f"interneuron (x{len(inter_neurons)})"])
            scatter_shapes.extend(["o"])
            scatter_shapes.extend(["*"] * len(inter_neurons))
            colors.extend(["red"])
            colors.extend(["red"] * len(inter_neurons))

        plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spike_twitch_total_spikes,
                                      description=f"{self.description}_hist_spike_twitches_ratio_over_total_spikes",
                                      values_to_scatter=np.array(values_to_scatter),
                                      labels=labels,
                                      scatter_shapes=scatter_shapes,
                                      colors=colors, twice_more_bins=True,
                                      xlabel="spikes in twitch vs total spikes (%)",
                                      param=self.param)

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
        if len(ratio_interneurons) > 0:
            values_to_scatter.append(np.mean(ratio_interneurons))
            values_to_scatter.extend(ratio_interneurons)
            labels.extend(["mean", f"interneuron (x{len(inter_neurons)})"])
            scatter_shapes.extend(["o"])
            scatter_shapes.extend(["*"] * len(inter_neurons))
            colors.extend(["red"])
            colors.extend(["red"] * len(inter_neurons))

        plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spike_twitch_total_twitches,
                                      description=f"{self.description}_hist_spike_twitches_ratio_over_total_twitches",
                                      values_to_scatter=np.array(values_to_scatter),
                                      labels=labels,
                                      scatter_shapes=scatter_shapes,
                                      colors=colors, twice_more_bins=True,
                                      xlabel="spikes in twitch vs total twitches (%)",
                                      param=self.param)
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
                               cells_groups=cells_groups,
                               groups_colors=groups_colors,
                               data_id=self.description + "_around_twitches_psth_style",
                               param=self.param,
                               set_x_limit_to_max=True,
                               time_stamps_by_ms=0.01,
                               ms_scale=200,
                               size_cells_in_groups=150,
                               show_percentiles=show_percentiles)

        # now we take the cells that are the most correlated to twitches

    def plot_time_correlation_graph_over_twitches(self):
        if self.twitches_frames_periods is None:
            return
        results = get_time_correlation_data(spike_nums=self.spike_struct.spike_nums,
                                            events_times=self.twitches_frames_periods, time_around_events=10)
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
                               plot_cell_numbers=True,
                               cells_groups=cells_groups,
                               groups_colors=groups_colors,
                               data_id=self.description + "around_twitches",
                               param=self.param,
                               set_x_limit_to_max=True,
                               time_stamps_by_ms=0.01,
                               ms_scale=200,
                               size_cells_in_groups=150,
                               show_percentiles=show_percentiles)

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

        if twitches_group == 7:
            sce_times_related_to_twitch = np.zeros(len(sce_bool), dtype="bool")
            for twitch_period in self.twitches_frames_periods:
                is_in_sce = np.any(sce_bool[twitch_period[0]: twitch_period[1] + 1])
                if is_in_sce:
                    indices = np.where(sce_bool[twitch_period[0]: twitch_period[1] + 1])[0] + twitch_period[0]
                    sce_times_related_to_twitch[indices] = True
                # looking if there is a sce less than a second after
                end_time = np.min((twitch_period[1] + 1 + 10, len(sce_bool)))
                sce_after = np.any(sce_bool[twitch_period[1]:end_time])

                if sce_after:
                    indices = np.where(sce_bool[twitch_period[1]:end_time])[0] + twitch_period[1]
                    sce_times_related_to_twitch[indices] = True

            for sce_period in sce_periods:
                if not np.any(sce_times_related_to_twitch[sce_period[0]:sce_period[1] + 1]):
                    twitches_times.append((sce_period[0] + sce_period[1]) // 2)
                    twitches_periods.append((sce_period[0], sce_period[1]))

        for twitch_period in self.twitches_frames_periods:
            if (sce_bool is None) or (twitches_group == 0):
                twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
                twitches_periods.append((twitch_period[0], twitch_period[1]))
                continue
            is_in_sce = np.any(sce_bool[twitch_period[0]: twitch_period[1] + 1])
            if twitches_group == 1:
                if is_in_sce:
                    twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
                    twitches_periods.append((twitch_period[0], twitch_period[1]))
                continue

            if twitches_group == 2:
                if not is_in_sce:
                    twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
                    twitches_periods.append((twitch_period[0], twitch_period[1]))
                continue

            if is_in_sce:
                continue
            # looking if there is a sce less than a second after
            end_time = np.min((twitch_period[1] + 1 + 10, len(sce_bool)))
            sce_after = np.any(sce_bool[twitch_period[1]:end_time])
            if twitches_group == 3:
                if sce_after:
                    twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
                    twitches_periods.append((twitch_period[0], twitch_period[1]))
                    continue
            if twitches_group == 4:
                if not sce_after:
                    twitches_times.append((twitch_period[0] + twitch_period[1]) // 2)
                    twitches_periods.append((twitch_period[0], twitch_period[1]))
                    continue
        return twitches_times, twitches_periods

    def get_spikes_values_around_twitches(self, sce_bool=None, time_around=100,
                                          twitches_group=0, low_percentile=25, high_percentile=75):
        spike_nums_dur = self.spike_struct.spike_nums_dur
        # spike_nums = self.spike_struct.spike_nums
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
        return n_twitches, time_x_values, np.array(mean_values), \
               np.array(median_values), np.array(low_values), np.array(high_values), np.array(std_values)

    def plot_psth_twitches(self, time_around=100,
                           twitches_group=0, line_mode=False,
                           with_other_ms=None,
                           save_formats="pdf"):
        """

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

        if self.twitches_frames_periods is None:
            return

        sce_bool = self.sce_bool

        if with_other_ms is not None:
            line_mode = True

        results = \
            self.get_spikes_values_around_twitches(sce_bool=sce_bool, time_around=time_around,
                                                   twitches_group=twitches_group)

        if results is None:
            return
        n_twitches, time_x_values, mean_values, median_values, low_values, high_values, std_values = results

        n_cells = len(self.spike_struct.spike_nums_dur)
        activity_threshold_percentage = (self.activity_threshold / n_cells) * 100

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

        for mean_version in [True, False]:
            max_value = 0
            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    gridspec_kw={'height_ratios': [1]},
                                    figsize=(15, 10))
            ax1.set_facecolor("black")
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
                        results = \
                            ms.get_spikes_values_around_twitches(sce_bool=ms.sce_bool, time_around=time_around,
                                                                 twitches_group=twitches_group)

                        if results is None:
                            continue
                        ms_n_twitches, ms_time_x_values, ms_mean_values, ms_median_values, \
                        ms_low_values, ms_high_values, ms_std_values = results

                    if with_other_ms is None:
                        color = hist_color
                    else:
                        color = cm.nipy_spectral(float(index_ms + 1) / (len(with_other_ms) + 2))
                    if mean_version:
                        plt.plot(time_x_values,
                                 ms_mean_values, color=color, lw=2, label=f"{ms.description}")
                        if with_other_ms is None:
                            ax1.fill_between(time_x_values, ms_mean_values - ms_std_values,
                                             ms_mean_values + ms_std_values,
                                             alpha=0.5, facecolor=color)
                        max_value = np.max((max_value, np.max(ms_mean_values + ms_std_values)))
                    else:
                        plt.plot(time_x_values,
                                 ms_median_values, color=color, lw=2, label=f"{ms.description}")
                        if with_other_ms is None:
                            ax1.fill_between(time_x_values, ms_low_values, ms_high_values,
                                             alpha=0.5, facecolor=color)
                        max_value = np.max((max_value, np.max(ms_high_values)))
            else:
                plt.bar(time_x_values,
                        mean_values, color=hist_color, edgecolor=edge_color)
                max_value = np.max((max_value, np.max(mean_values)))
            ax1.vlines(0, 0,
                       np.max(mean_values), color="white", linewidth=2,
                       linestyles="dashed")
            ax1.hlines(activity_threshold_percentage, -1 * time_around, time_around,
                       color="white", linewidth=1,
                       linestyles="dashed")

            if with_other_ms is not None:
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

            plt.title(f"{descr} {n_twitches} twitches bar chart {title_option} {extra_info}")
            ax1.set_ylabel(f"Spikes (%)")
            ax1.set_xlabel("time (frames)")
            ax1.set_ylim(0, np.max((activity_threshold_percentage, max_value)) + 1)
            # xticks = np.arange(0, len(data_dict))
            # ax1.set_xticks(xticks)
            # # sce clusters labels
            # ax1.set_xticklabels(labels)
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            for save_format in save_formats:
                fig.savefig(f'{self.param.path_results}/{descr}_bar_chart_'
                            f'{n_twitches}_twitches_{title_option}'
                            f'_{extra_info}{self.param.time_str}.{save_format}',
                            format=f"{save_format}")

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
            self.cell_assemblies = []
            self.sce_times_in_single_cell_assemblies = dict()
            self.sce_times_in_multiple_cell_assemblies = []
            self.sce_times_in_cell_assemblies = []
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
                        if line.startswith("single_sce_in_ca"):
                            line_list = line.split(':')
                            ca_index = int(line_list[1])
                            self.sce_times_in_single_cell_assemblies[ca_index] = []
                            couples_of_times = line_list[2].split("#")
                            for couple_of_time in couples_of_times:
                                times = couple_of_time.split(" ")
                                self.sce_times_in_single_cell_assemblies[ca_index].append([int(t) for t in times])
                                self.sce_times_in_cell_assemblies.append([int(t) for t in times])
                        if line.startswith("multiple_sce_in_ca"):
                            line_list = line.split(':')
                            sces_times = line_list[1].split("#")
                            for sce_time in sces_times:
                                times = sce_time.split(" ")
                                self.sce_times_in_multiple_cell_assemblies.append([int(t) for t in times])
                                self.sce_times_in_cell_assemblies.append([int(t) for t in times])
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
                            lw=2, label=f"n° {ax_index+first_cell}{extra_info}")
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
                            grouped_mean_values[ax_index], color="blue", lw=2, label=f"cell {ax_index+first_cell}")
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
        span_areas_coords.append(self.mvt_frames_periods)
        span_area_colors.append('red')
        span_areas_coords.append(self.sce_times_in_cell_assemblies)
        span_area_colors.append('green')
        span_areas_coords.append(self.twitches_frames_periods)
        span_area_colors.append('blue')

        ## ratio
        for cell_assembly_index in np.arange(-1, len(self.cell_assemblies)):

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

    def plot_each_inter_neuron_connect_map(self):
        # plot n_in and n_out map of the interneurons
        inter_neurons = self.spike_struct.inter_neurons
        n_inter_neurons = len(inter_neurons)
        if n_inter_neurons == 0:
            return

        for inter_neuron in inter_neurons:
            self.plot_connectivity_maps_of_a_cell(cell_to_map=inter_neuron, cell_descr="inter_neuron")

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

    def plot_all_cells_on_map(self, save_plot=True, return_fig=False):
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
        fig = self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      fill_polygons=False,
                                      title_option="all cells", connections_dict=None,
                                      cells_groups=cells_groups,
                                      cells_groups_colors=cells_groups_colors,
                                      cells_groups_edge_colors=cells_groups_edge_colors,
                                      with_edge=True, cells_groups_alpha=cells_groups_alpha,
                                      dont_fill_cells_not_in_groups=False,
                                      with_cell_numbers=True, save_formats=["png"],
                                      save_plot=save_plot, return_fig=return_fig)
        if return_fig:
            return fig

    def plot_cell_assemblies_on_map(self):
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

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      fill_polygons=False,
                                      title_option="cell_assemblies", connections_dict=None,
                                      cells_groups=self.cell_assemblies,
                                      cells_groups_colors=cells_groups_colors,
                                      dont_fill_cells_not_in_groups=True,
                                      with_cell_numbers=True, save_formats=["png"])

    def set_low_activity_threshold(self, threshold, percentile_value):
        self.low_activity_threshold_by_percentile[percentile_value] = threshold
        if self.percentile_for_low_activity_threshold in self.low_activity_threshold_by_percentile:
            self.low_activity_threshold = \
                self.low_activity_threshold_by_percentile[self.percentile_for_low_activity_threshold]

    def set_inter_neurons(self, inter_neurons):
        self.spike_struct.inter_neurons = np.array(inter_neurons).astype(int)

    def load_abf_file(self, abf_file_name, threshold_piezo=None, with_run=False,
                      frames_channel=0, piezo_channel=1, run_channel=2, threshold_ratio=2,
                      sampling_rate=50000, offset=None, just_load_npz_file=True):

        print(f"abf: ms {self.description}")

        self.abf_sampling_rate = sampling_rate
        self.threshold_piezo = threshold_piezo

        if with_run:
            self.with_run = True
        else:
            self.with_piezo = True

        # first checking if the data has been saved in a file before
        index_reverse = abf_file_name[::-1].find("/")
        path_abf_data = abf_file_name[:len(abf_file_name) - index_reverse]
        file_names = []

        npz_loaded = False
        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
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
                                tools_misc.find_continuous_frames_period(self.intermediate_behavourial_events_frames)
                        if "noise_mvt_frames" in npzfile:
                            self.noise_mvt_frames = npzfile['noise_mvt_frames']
                            self.noise_mvt_frames_periods = \
                                tools_misc.find_continuous_frames_period(self.noise_mvt_frames)
                        # if (not with_run) and do_detect_twitches:
                        #     self.detect_twitches()
            # np.savez(self.param.path_data + path_abf_data + self.description + "_mvts_from_abf_new.npz",
            #          twitches_frames=self.twitches_frames,
            #          short_lasting_mvt_frames=self.short_lasting_mvt_frames,
            #          complex_mvt_frames=self.complex_mvt_frames,
            #          intermediate_behavourial_events_frames=self.intermediate_behavourial_events_frames,
            #          noise_mvt_frames=self.noise_mvt_frames)
        if just_load_npz_file:
            return
        # 50000 Hz
        abf = pyabf.ABF(self.param.path_data + abf_file_name)

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
        if with_run:
            abf.setSweep(sweepNumber=0, channel=run_channel)
        else:
            abf.setSweep(sweepNumber=0, channel=piezo_channel)
        mvt_data = abf.sweepY
        if offset is not None:
            mvt_data = mvt_data + offset

        # first frame
        first_frame_index = np.where(frames_data < 0.01)[0][0]
        # removing the part before the recording
        # print(f"first_frame_index {first_frame_index}")
        times_in_sec = times_in_sec[:-first_frame_index]
        frames_data = frames_data[first_frame_index:]
        mvt_data = mvt_data[first_frame_index:]

        if (self.abf_sampling_rate < 50000):
            mask_frames_data = np.ones(len(frames_data), dtype="bool")
            # we need to detect the frames manually, but first removing data between movies
            selection = np.where(frames_data >= 0.05)[0]
            # frames_period = find_continuous_frames_period(selection)
            # # for period in frames_period:
            # #     mask_frames_data[period[0]:period[1]+1] = False

            # for channel in np.arange(0, 1):
            #     print("showing frames")
            #     fig, ax = plt.subplots(nrows=1, ncols=1,
            #                            gridspec_kw={'height_ratios': [1]},
            #                            figsize=(20, 8))
            #     plt.plot(times_in_sec, frames_data, lw=.5)
            #     for mvt_period in frames_period:
            #         color = "red"
            #         ax.axvspan(mvt_period[0] / self.abf_sampling_rate, mvt_period[1] / self.abf_sampling_rate,
            #                    alpha=0.5, facecolor=color, zorder=1)
            #     plt.title(f"channel {channel} {self.description} after correction")
            #     plt.show()
            #     plt.close()
            mask_selection = np.zeros(len(selection), dtype="bool")
            pos = np.diff(selection)
            # looking for continuous data between movies
            to_keep_for_removing = np.where(pos == 1)[0] + 1
            mask_selection[to_keep_for_removing] = True
            selection = selection[mask_selection]
            # print(f"len(selection) {len(selection)}")
            mask_frames_data[selection] = False
            frames_data = frames_data[mask_frames_data]
            len_frames_data_in_s = np.round(len(frames_data) / self.abf_sampling_rate, 3)
            # print(f"frames_data in sec {len_frames_data_in_s}")
            # print(f"frames_data in 100 ms {np.round(len_frames_data_in_s/0.1, 2)}")
            mvt_data = mvt_data[mask_frames_data]
            times_in_sec = times_in_sec[:-len(np.where(mask_frames_data == 0)[0])]
            active_frames = np.linspace(0, len(frames_data), 12500).astype(int)
            mean_diff_active_frames = np.mean(np.diff(active_frames)) / self.abf_sampling_rate
            print(f"mean diff active_frames {np.round(mean_diff_active_frames, 3)}")
            if mean_diff_active_frames < 0.09:
                raise Exception("mean_diff_active_frames < 0.09")
            # for channel in np.arange(0, 1):
            #     fig, ax = plt.subplots(nrows=1, ncols=1,
            #                            gridspec_kw={'height_ratios': [1]},
            #                            figsize=(20, 8))
            #     plt.plot(times_in_sec, frames_data, lw=.5)
            #     plt.title(f"channel {channel} {self.description} after correction")
            #     plt.show()
            #     plt.close()
        else:
            binary_frames_data = np.zeros(len(frames_data), dtype="int8")
            binary_frames_data[frames_data >= 0.05] = 1
            binary_frames_data[frames_data < 0.05] = 0
            # +1 due to the shift of diff
            active_frames = np.where(np.diff(binary_frames_data) == 1)[0] + 1

        mvt_data_without_abs = mvt_data
        mvt_data = np.abs(mvt_data)
        if not with_run:
            self.raw_piezo = mvt_data
            self.raw_piezo_without_abs = mvt_data_without_abs

        self.abf_times_in_sec = times_in_sec
        # active_frames = np.concatenate(([0], active_frames))
        # print(f"active_frames {active_frames}")
        nb_frames = len(active_frames)
        self.abf_frames = active_frames
        print(f"nb_frames {nb_frames}")

        # manual selection deactivated
        do_manual_selection = not npz_loaded
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
            mvt_periods, speed_during_mvt_periods = self.detect_run_periods(mvt_data=mvt_data, min_speed=0.5)
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
        # continuous_frames_periods = tools_misc.find_continuous_frames_period(self.mvt_frames)
        # print(f"continuous_frames_periods {continuous_frames_periods}")
        # print(f"self.mvt_frames_periods {self.mvt_frames_periods}")
        # print(f"len(mvt_frames) {len(self.mvt_frames)}")

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
        return mvt_periods, speed_during_mvt_periods

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

    def load_tif_movie(self, path):
        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(self.param.path_data + path):
            file_names.extend(local_filenames)
            break
        if len(file_names) == 0:
            return

        for file_name in file_names:
            file_name_original = file_name
            file_name = file_name.lower()
            descr = self.description.lower() + ".tif"
            if descr != file_name:
                continue
            self.tif_movie_file_name = self.param.path_data + path + file_name_original
            # print(f"self.tif_movie_file_name {self.tif_movie_file_name}")

    def load_data_from_file(self, file_name_to_load, variables_mapping, frames_filter=None,
                            from_gui=False):
        """

        :param file_name_to_load:
        :param variables_mapping:
        :param frames_filter: if not None, will keep only the frames in frames_filter
        :return:
        """
        data = hdf5storage.loadmat(self.param.path_data + file_name_to_load)
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
        if "spike_nums_dur" in variables_mapping:
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
            self.coord_obj = CoordClass(coord=self.coord, nb_col=200,
                                        nb_lines=200)
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
        if (self.cells_to_remove is None) or len(self.cells_to_remove) == 0:
            return
        n_cells = self.spike_struct.n_cells
        # print(f"self.coord[1].shape {self.coord[1].shape}")
        # raise Exception("titi")
        new_coord = []
        for cell in np.arange(n_cells):
            if cell in self.cells_to_remove:
                continue
            new_coord.append(self.coord[cell])

        self.coord_obj = CoordClass(coord=new_coord, nb_col=200,
                                    nb_lines=200)
        self.spike_struct.clean_data_using_cells_to_remove(cells_to_remove=self.cells_to_remove)
        # raise Exception("titi")

    def detect_n_in_n_out(self):
        self.spike_struct.detect_n_in_n_out()

    def build_spike_nums_dur(self):
        # build spike_nums_dur based on peak_nums and spike_nums
        self.spike_struct.build_spike_nums_dur()