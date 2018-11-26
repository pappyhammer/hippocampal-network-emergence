import pandas as pd
# from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from bisect import bisect

# important to avoid a bug when using virtualenv
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
import math
# import copy
from datetime import datetime
# import keras
import os
import pyabf
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


class HNEParameters(MarkovParameters):
    def __init__(self, path_results, time_str, time_inter_seq, min_duration_intra_seq, min_len_seq, min_rep_nb,
                 path_data,
                 max_branches, stop_if_twin, no_reverse_seq, error_rate, spike_rate_weight,
                 bin_size=1, cell_assemblies_data_path=None, best_order_data_path=None):
        super().__init__(time_inter_seq=time_inter_seq, min_duration_intra_seq=min_duration_intra_seq,
                         min_len_seq=min_len_seq, min_rep_nb=min_rep_nb, no_reverse_seq=no_reverse_seq,
                         max_branches=max_branches, stop_if_twin=stop_if_twin, error_rate=error_rate,
                         spike_rate_weight=spike_rate_weight,
                         bin_size=bin_size, path_results=path_results, time_str=time_str)
        self.path_data = path_data
        self.cell_assemblies_data_path = cell_assemblies_data_path
        self.best_order_data_path = best_order_data_path
        # for plotting ages
        self.markers = ['o', '*', 's', 'v', '<', '>', '^', 'x', '+', "."]  # d losange
        self.colors = ["darkmagenta", "white", "saddlebrown", "blue", "red", "darkgrey", "chartreuse", "cornflowerblue",
                       "pink", "darkgreen", "gold"]


class MouseSession:
    def __init__(self, age, session_id, param, nb_ms_by_frame, weight=None, spike_nums=None, spike_nums_dur=None,
                 percentile_for_low_activity_threshold=1):
        # should be a list of int
        self.age = age
        self.session_id = str(session_id)
        self.nb_ms_by_frame = nb_ms_by_frame
        self.description = f"P{self.age}_{self.session_id}"
        self.spike_struct = HNESpikeStructure(mouse_session=self, spike_nums=spike_nums, spike_nums_dur=spike_nums_dur)
        # spike_nums represents the onsets of the neuron spikes
        self.traces = None
        self.coord = None
        self.activity_threshold = None
        self.low_activity_threshold_by_percentile = dict()
        self.percentile_for_low_activity_threshold = percentile_for_low_activity_threshold
        self.low_activity_threshold = None
        self.param = param
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

        if self.param.cell_assemblies_data_path is not None:
            self.load_cell_assemblies_data()
        # for seq
        self.best_order_loaded = None
        if self.param.best_order_data_path is not None:
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
        # represents the frames such as defined in the abf file, used to match the frames index to the raw_piezo data.
        # for frames f with index 10, x = abf_frames[f] will give us the index of f such that self.raw_piezo[x] represent
        # the piezzo value at frame x
        self.abf_frames = None
        self.abf_times_in_sec = None
        self.threshold_piezo = None
        self.twitches_frames_periods = None
        self.twitches_frames = None
        self.sce_bool = None
        self.sce_times_numbers = None
        self.SCE_times = None

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

        # inter_neurons = np.array(self.spike_struct.inter_neurons)
        values_to_scatter = []
        # non_inter_neurons = np.setdiff1d(np.arange(len(ratio_spike_twitch_total_spikes)), inter_neurons)
        # ratio_interneurons = list(ratio_spike_twitch_total_spikes[inter_neurons])
        # ratio_non_interneurons = list(ratio_spike_twitch_total_spikes[non_inter_neurons])
        labels = []
        scatter_shapes = []
        colors = []
        values_to_scatter.append(np.mean(ratio_spike_twitch_total_spikes))
        labels.extend(["mean"])
        scatter_shapes.extend(["o"])
        colors.extend(["white"])
        # if len(ratio_non_interneurons) > 0:
        #     values_to_scatter.append(np.mean(ratio_non_interneurons))
        #     values_to_scatter.append(np.median(ratio_non_interneurons))
        #     labels.extend(["mean", "median"])
        #     scatter_shapes.extend(["o", "s"])
        #     colors.extend(["white", "white"])
        # if len(ratio_interneurons) > 0:
        #     values_to_scatter.append(np.mean(ratio_interneurons))
        #     values_to_scatter.append(np.median(ratio_interneurons))
        #     values_to_scatter.extend(ratio_interneurons)
        #     labels.extend(["mean", "median", f"interneuron (x{len(inter_neurons)})"])
        #     scatter_shapes.extend(["o", "s"])
        #     scatter_shapes.extend(["*"] * len(inter_neurons))
        #     colors.extend(["red", "red"])
        #     colors.extend(["red"] * len(inter_neurons))

        plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spike_twitch_total_spikes,
                                      description=f"{self.description}_hist_spike_twitches_ratio_over_total_spikes",
                                      values_to_scatter=np.array(values_to_scatter),
                                      labels=labels,
                                      scatter_shapes=scatter_shapes,
                                      colors=colors,
                                      xlabel="spikes in twitch vs total spikes (%)",
                                      param=self.param)

        values_to_scatter = []
        # non_inter_neurons = np.setdiff1d(np.arange(len(ratio_spike_twitch_total_twitches)), inter_neurons)
        # ratio_interneurons = list(ratio_spike_twitch_total_twitches[inter_neurons])
        # ratio_non_interneurons = list(ratio_spike_twitch_total_twitches[non_inter_neurons])
        labels = []
        scatter_shapes = []
        colors = []
        values_to_scatter.append(np.mean(ratio_spike_twitch_total_twitches))
        labels.extend(["mean"])
        scatter_shapes.extend(["o"])
        colors.extend(["white"])
        # if len(ratio_non_interneurons) > 0:
        #     values_to_scatter.append(np.mean(ratio_non_interneurons))
        #     values_to_scatter.append(np.median(ratio_non_interneurons))
        #     labels.extend(["mean", "median"])
        #     scatter_shapes.extend(["o", "s"])
        #     colors.extend(["white", "white"])
        # if len(ratio_interneurons) > 0:
        #     values_to_scatter.append(np.mean(ratio_interneurons))
        #     values_to_scatter.append(np.median(ratio_interneurons))
        #     values_to_scatter.extend(ratio_interneurons)
        #     labels.extend(["mean", "median", f"interneuron (x{len(inter_neurons)})"])
        #     scatter_shapes.extend(["o", "s"])
        #     scatter_shapes.extend(["*"] * len(inter_neurons))
        #     colors.extend(["red", "red"])
        #     colors.extend(["red"] * len(inter_neurons))

        plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spike_twitch_total_twitches,
                                      description=f"{self.description}_hist_spike_twitches_ratio_over_total_twitches",
                                      values_to_scatter=np.array(values_to_scatter),
                                      labels=labels,
                                      scatter_shapes=scatter_shapes,
                                      colors=colors,
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
        if self.twitches_frames is None:
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

    def plot_connectivity_maps_of_a_cell(self, cell_to_map, cell_descr,
                                         cell_color="red", links_cell_color="cornflowerblue"):
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

        self.coord_obj.compute_center_coord(cells_groups=cells_groups,
                                            cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      title_option=f"n_in_{cell_descr}_{cell_to_map}",
                                      connections_dict=connections_dict_in,
                                      with_cell_numbers=True)

        cells_groups_colors = [cell_color]
        cells_groups = [[cell_to_map]]
        if at_least_on_out_link and color_each_cells_link_to_cell:
            links_cells = list(connections_dict_out[cell_to_map].keys())
            # removing fellow inter_neurons
            links_cells = np.setdiff1d(np.array(links_cells), np.array(cell_to_map))
            if len(links_cells) > 0:
                cells_groups.append(list(connections_dict_out[cell_to_map].keys()))
                cells_groups_colors.append(links_cell_color)

        self.coord_obj.compute_center_coord(cells_groups=cells_groups,
                                            cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      title_option=f"n_out_{cell_descr}_{cell_to_map}",
                                      connections_dict=connections_dict_out,
                                      with_cell_numbers=True)

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

        self.coord_obj.compute_center_coord(cells_groups=cells_groups,
                                            cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      title_option=f"n_in_interneurons_x_{n_inter_neurons}",
                                      connections_dict=connections_dict_in,
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

        self.coord_obj.compute_center_coord(cells_groups=cells_groups,
                                            cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=False,
                                      title_option=f"n_out_interneurons_x_{n_inter_neurons}",
                                      connections_dict=connections_dict_out,
                                      with_cell_numbers=True)

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
        self.coord_obj.compute_center_coord(cells_groups=self.cell_assemblies,
                                            cells_groups_colors=cells_groups_colors)

        self.coord_obj.plot_cells_map(param=self.param,
                                      data_id=self.description, show_polygons=True,
                                      fill_polygons=True,
                                      title_option="cell_assemblies", connections_dict=None,
                                      with_cell_numbers=True)

    def set_low_activity_threshold(self, threshold, percentile_value):
        self.low_activity_threshold_by_percentile[percentile_value] = threshold
        if self.percentile_for_low_activity_threshold in self.low_activity_threshold_by_percentile:
            self.low_activity_threshold = \
                self.low_activity_threshold_by_percentile[self.percentile_for_low_activity_threshold]

    def set_inter_neurons(self, inter_neurons):
        self.spike_struct.inter_neurons = np.array(inter_neurons).astype(int)

    def load_abf_file(self, abf_file_name, threshold_piezo=None, with_run=False,
                      frames_channel=0, piezo_channel=1, run_channel=2,
                      sampling_rate=50000, offset=None):
        # return
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

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(self.param.path_data + path_abf_data):
            file_names.extend(local_filenames)
            break
        if len(file_names) > 0:
            for file_name in file_names:
                if file_name.endswith(".npz"):
                    if file_name.find("abf") > -1:
                        # loading data
                        npzfile = np.load(self.param.path_data + path_abf_data + file_name)
                        self.mvt_frames = npzfile['mvt_frames']
                        self.mvt_frames_periods = tools_misc.find_continuous_frames_period(self.mvt_frames)
                        if "speed_by_mvt_frame" in npzfile:
                            self.speed_by_mvt_frame = npzfile['speed_by_mvt_frame']
                        if "raw_piezo" in npzfile:
                            self.raw_piezo = npzfile['raw_piezo']
                        if "abf_frames" in npzfile:
                            self.abf_frames = npzfile['abf_frames']
                        if "abf_times_in_sec" in npzfile:
                            self.abf_times_in_sec = npzfile['abf_times_in_sec']
                        if not with_run:
                            self.detect_twitches()
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
        mvt_data = np.abs(mvt_data[first_frame_index:])

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
        if not with_run:
            self.raw_piezo = mvt_data
        # active_frames = np.concatenate(([0], active_frames))
        # print(f"active_frames {active_frames}")
        nb_frames = len(active_frames)
        print(f"nb_frames {nb_frames}")

        if (not with_run) and (threshold_piezo is None):
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   gridspec_kw={'height_ratios': [1]},
                                   figsize=(20, 8))
            plt.plot(times_in_sec, mvt_data, lw=.5)
            plt.title(f"piezo {self.description}")
            plt.show()
            plt.close()
            return

        if with_run:
            mvt_periods, speed_during_mvt_periods = self.detect_run_periods(mvt_data=mvt_data, min_speed=0.5)
        else:
            mvt_periods = self.detect_mvt_periods_with_piezo_and_diff(piezo_data=mvt_data,
                                                                      piezo_threshold=threshold_piezo,
                                                                      min_time_between_periods=2 * sampling_rate)
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
        if not with_run:
            self.detect_twitches()

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
            if threshold_piezo is not None:
                # print(f"len(self.raw_piezo): {len(self.raw_piezo)}")
                self.abf_frames = active_frames
                self.abf_times_in_sec = times_in_sec
                np.savez(self.param.path_data + path_abf_data + self.description + "_mvts_from_abf.npz",
                         mvt_frames=self.mvt_frames, raw_piezo=self.raw_piezo, abf_frames=active_frames,
                         abf_times_in_sec=times_in_sec)
        # continuous_frames_periods = tools_misc.find_continuous_frames_period(self.mvt_frames)
        # print(f"continuous_frames_periods {continuous_frames_periods}")
        # print(f"self.mvt_frames_periods {self.mvt_frames_periods}")
        # print(f"len(mvt_frames) {len(self.mvt_frames)}")

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
                                         min_time_between_periods=min_time_between_periods)

    def merging_time_periods(self, time_periods, min_time_between_periods):
        n_periods = len(time_periods)
        # print(f"n_periods {n_periods}")
        # for i, time_period in enumerate(time_periods):
        #     print(f"time_period {i}: {np.round(time_period[0]/50000, 2)} - {np.round(time_period[1]/50000, 2)}")
        merged_time_periods = []
        index = 0
        while index < n_periods:
            time_period = time_periods[index]
            if len(merged_time_periods) == 0:
                merged_time_periods.append([time_period[0], time_period[1]])
                index += 1
                continue
            # we check if the time between both is superior at min_time_between_periods
            last_time_period = merged_time_periods[-1]
            if (time_period[0] - last_time_period[1]) < min_time_between_periods:
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

    def load_data_from_file(self, file_name_to_load, variables_mapping, frames_filter=None):
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
        if "coord" in variables_mapping:
            self.coord = data[variables_mapping["coord"]][0]
            self.coord_obj = CoordClass(coord=self.coord, nb_col=200,
                                        nb_lines=200)
        if "spike_durations" in variables_mapping:
            self.spike_struct.set_spike_durations(data[variables_mapping["spike_durations"]])
        elif self.spike_struct.spike_nums_dur is not None:
            self.spike_struct.set_spike_durations()
        if "spike_amplitudes" in variables_mapping:
            self.spike_struct.set_spike_amplitudes(data[variables_mapping["spike_amplitudes"]])

        self.spike_struct.set_spike_trains_from_spike_nums()

        # if (self.spike_struct.spike_nums_dur is not None) or (self.spike_struct.spike_nums is not None):
        #     self.detect_n_in_n_out()

    def detect_n_in_n_out(self):
        self.spike_struct.detect_n_in_n_out()


class HNESpikeStructure:

    def __init__(self, mouse_session, labels=None, spike_nums=None, spike_trains=None,
                 spike_nums_dur=None, activity_threshold=None,
                 title=None, ordered_indices=None, ordered_spike_data=None):
        self.mouse_session = mouse_session
        self.spike_nums = spike_nums
        self.spike_nums_dur = spike_nums_dur
        if (self.spike_nums is not None) or (self.spike_nums_dur is not None):
            if self.spike_nums is not None:
                self.n_cells = len(self.spike_nums)
            else:
                self.n_cells = len(self.spike_nums_dur)
        else:
            self.n_cells = None
        self.spike_trains = spike_trains
        self.ordered_spike_data = ordered_spike_data
        self.activity_threshold = activity_threshold
        self.title = title
        self.labels = labels
        self.ordered_indices = ordered_indices
        self.ordered_labels = None
        if self.ordered_indices is not None:
            if self.spike_nums is not None:
                self.ordered_spike_nums = np.copy(self.spike_nums[ordered_indices, :])
            else:
                self.ordered_spike_nums = None
            if self.spike_nums_dur is not None:
                self.ordered_spike_nums_dur = np.copy(self.spike_nums_dur[ordered_indices, :])
            else:
                self.ordered_spike_nums_dur = None
            if self.spike_trains is not None:
                self.ordered_spike_trains = []
                for index in ordered_indices:
                    self.ordered_spike_trains.append(self.spike_trains[index])
            else:
                self.ordered_spike_trains = None
            self.ordered_labels = []
            if self.labels is not None:
                for old_cell_index in self.ordered_indices:
                    self.ordered_labels.append(self.labels[old_cell_index])

        # list of size n_cells, each list is array representing the duration (in frames) of each spike of the cell
        self.spike_durations = None
        self.inter_neurons = None
        # list of size n_cells, each list is array representing the amplitude of each spike of the cell
        self.spike_amplitudes = None

        # nb frames (1 frame == 100 ms) to look for connection near a neuron that spike
        self.nb_frames_for_func_connect = 5
        # contain the list of neurons connected to the EB as keys, and the number of connection as values
        # first key is a dict of neuron, the second key is other neurons to which the first connect,
        # then the number of times is connected to it
        self.n_in_dict = dict()
        self.n_out_dict = dict()
        self.n_in_matrix = None
        self.n_out_matrix = None

    def detect_n_in_n_out(self):
        # look neuron by neuron, at each spike and make a pair wise for each other neurons according to the spike
        # distribution around 500ms before and after. If the distribution is not uniform then we look where is the max
        # and we add it to n_out or n_in if before or after. If it is at the same time, then we don't add it.
        nb_neurons = len(self.spike_nums)
        n_times = len(self.spike_nums[0, :])
        for neuron in np.arange(nb_neurons):
            self.n_in_dict[neuron] = dict()
            self.n_out_dict[neuron] = dict()
            neurons_to_consider = np.arange(len(self.spike_nums))
            mask = np.ones(len(self.spike_nums), dtype="bool")
            mask[neuron] = False
            neurons_to_consider = neurons_to_consider[mask]
            # look at onsets
            neuron_spikes, = np.where(self.spike_nums[neuron, :])
            # is_early_born = (neuron == ms.early_born_cell)

            if len(neuron_spikes) == 0:
                continue

            spike_nums_to_use = self.spike_nums

            distribution_array_2_d = np.zeros((nb_neurons, ((self.nb_frames_for_func_connect * 2) + 1)),
                                              dtype="int16")

            event_index = self.nb_frames_for_func_connect
            # looping on each spike of the main neuron
            for n, event in enumerate(neuron_spikes):
                # only taking in consideration events that are not too close from bottom range or upper range
                min_limit = max(event - self.nb_frames_for_func_connect, 0)
                max_limit = min((event + self.nb_frames_for_func_connect), (n_times - 1))
                mask = np.zeros((nb_neurons, ((self.nb_frames_for_func_connect * 2) + 1)),
                                dtype="bool")
                mask_start = 0
                if (event - self.nb_frames_for_func_connect) < 0:
                    mask_start = -1 * (event - self.nb_frames_for_func_connect)
                mask_end = mask_start + (max_limit - min_limit) + 1
                mask[:, mask_start:mask_end] = spike_nums_to_use[:, min_limit:(max_limit + 1)] > 0
                distribution_array_2_d[mask] += 1

            # going neuron by neuron
            for neuron_to_consider in neurons_to_consider:
                distribution_array = distribution_array_2_d[neuron_to_consider, :]
                distribution_for_test = np.zeros(np.sum(distribution_array))
                frames_time = np.arange(-self.nb_frames_for_func_connect, self.nb_frames_for_func_connect + 1)
                i_n = 0
                for i_time, sum_spike in enumerate(distribution_array):
                    if sum_spike > 0:
                        distribution_for_test[i_n:i_n + sum_spike] = frames_time[i_time]
                        i_n += sum_spike
                if len(distribution_for_test) >= 20:
                    stat_n, p_value = stats.normaltest(distribution_for_test)
                    ks, p_ks = stats.kstest(distribution_for_test, stats.randint.cdf,
                                            args=(np.min(distribution_for_test),
                                                  np.max(distribution_for_test)))
                    is_normal_distribution = p_value >= 0.05
                    is_uniform_distribution = p_ks >= 0.05
                    # if the distribution is normal or uniform, we skip it
                    if is_normal_distribution or is_uniform_distribution:
                        continue

                n_in_sum = np.sum(distribution_array[:event_index])
                n_out_sum = np.sum(distribution_array[(event_index + 1):])
                # means we have the same number of spikes before and after
                if n_in_sum == n_out_sum:
                    continue
                max_value = max(n_in_sum, n_out_sum)
                min_value = min(n_in_sum, n_out_sum)
                # we should have at least twice more spikes on one side
                if max_value < (min_value * 2):
                    continue
                # and twice as more as the spikes at time 0
                if max_value < (distribution_array[event_index] * 2):
                    continue

                if n_in_sum > n_out_sum:
                    self.n_in_dict[neuron][neuron_to_consider] = 1
                    self.n_in_matrix[neuron][neuron_to_consider] = 1
                else:
                    self.n_out_dict[neuron][neuron_to_consider] = 1
                    self.n_out_matrix[neuron][neuron_to_consider] = 1

                # show_bar_chart = False
                # if show_bar_chart and len(distribution_for_test) >= 20:
                #     if (np.max(distribution_array) > 3):
                #         # if (neuron == self.early_born_cell):
                #         print(f'neuron {neuron}, neuron_to_consider {neuron_to_consider}')
                #
                #         print(f"### distribution_for_normal_test {distribution_for_test}")
                #
                #         print(f"stat_n {stat_n}, p_value {p_value} "
                #               f"{'Non normal distribution' if (p_value < 0.05) else 'Normal distribution'}")
                #
                #         print(f"ks {ks}, p_ks {p_ks} "
                #               f"{'Non uniform distribution' if (p_ks < 0.05) else 'Uniform distribution'}")
                #
                #         print("")
                #         plt.bar(np.arange(-1 * self.nb_frames_for_func_connect, self.nb_frames_for_func_connect + 1),
                #                 distribution_array, color="blue")
                #         plt.title(f"neuron {neuron} vs neuron {neuron_to_consider} "
                #                   f"{'Non normal distribution' if (p_value < 0.05) else 'Normal distribution'}, "
                #                   f"{'Non uniform distribution' if (p_ks < 0.05) else 'Uniform distribution'}")
                #         plt.xlabel("Cell")
                #         plt.ylabel("Nb of spikes")
                #         plt.show()
                #         plt.close()

    def set_spike_durations(self, spike_durations_array=None):
        if self.spike_durations is not None:
            return

        self.spike_durations = []
        n_cells = 0
        if self.spike_nums is not None:
            n_cells = len(self.spike_nums)
        elif self.spike_nums_dur is not None:
            n_cells = len(self.spike_nums_dur)
        if n_cells == 0:
            print("set_spike_durations no cell")
            return

        avg_spike_duration_by_cell = np.zeros(n_cells)

        if self.spike_nums_dur is None:
            for cell_id, spikes_d in enumerate(spike_durations_array):
                self.spike_durations.append(spikes_d[spikes_d > 0])
                if len(self.spike_durations[-1]) > 0:
                    avg_spike_duration_by_cell[cell_id] = np.mean(self.spike_durations[-1])
                else:
                    avg_spike_duration_by_cell[cell_id] = 0
        else:
            self.spike_durations = tools_misc.get_spikes_duration_from_raster_dur(spike_nums_dur=self.spike_nums_dur)
            # test
            # if spike_durations_array is not None:
            #     # test_spike_duration = []
            #     for cell_id, spikes_d in enumerate(spike_durations_array):
            #         # test_spike_duration.append(spikes_d[spikes_d > 0])
            #         print(f"cell {cell_id}")
            #         print(f"diff: {np.array(self.spike_durations[cell_id])}")
            #         print(f"Robin: {spikes_d[spikes_d > 0]}")
            #         print(f"raster: {np.where(self.spike_nums_dur[cell_id, :])[0]}")
            #         print("")
            #     raise Exception()

            for cell_id, spike_duration in enumerate(self.spike_durations):
                if len(spike_duration) > 0:
                    avg_spike_duration_by_cell[cell_id] = np.mean(spike_duration)
                else:
                    avg_spike_duration_by_cell[cell_id] = 0

        # inter_neurons can be manually added
        if self.inter_neurons is not None:
            return
        # otherwise, we select them throught a spike duration treshold

        # cells_ordered_by_duration = np.argsort(avg_spike_duration_by_cell)
        # avg_spike_duration_by_cell_ordered = np.sort(avg_spike_duration_by_cell)
        # n_cells = len(avg_spike_duration_by_cell)
        # interneurons_threshold_99 = avg_spike_duration_by_cell_ordered[int(n_cells * (1 - 0.01)]
        # interneurons_threshold_99_indices = cells_ordered_by_duration[int(n_cells * (1 - 0.01)]
        #

        # if self.mouse_session.session_id == "18_10_23_a001":
        #     for cell_id, avg in enumerate(avg_spike_duration_by_cell):
        #         print(f"cell_id {cell_id}, avg {avg}")

        interneurons_threshold_95 = np.percentile(avg_spike_duration_by_cell, 95)
        interneurons_threshold_99 = np.percentile(avg_spike_duration_by_cell, 99)

        self.inter_neurons = np.where(avg_spike_duration_by_cell >= interneurons_threshold_99)[0]

        print(f"{self.mouse_session.description}")
        inter_neurons_95 = np.where(avg_spike_duration_by_cell >= interneurons_threshold_95)[0]
        print(f"interneurons 95: {inter_neurons_95}")
        print(f"durations: {np.round(avg_spike_duration_by_cell[inter_neurons_95], 2)}")
        print("")

        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(12, 12))
        ax1.set_facecolor("black")
        distribution = avg_spike_duration_by_cell
        bins = int(np.sqrt(len(distribution)))
        weights = (np.ones_like(distribution) / (len(distribution))) * 100
        hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins,
                                                    facecolor="blue",
                                                    edgecolor="white",
                                                    weights=weights, log=False)

        plt.scatter(x=interneurons_threshold_99, y=20, marker="*",
                    color=["white"], s=150, zorder=20)

        plt.title(f"{self.mouse_session.description}")
        # plt.legend()
        plt.show()
        plt.close()

    def set_spike_amplitudes(self, spike_amplitudes_array):
        self.spike_amplitudes = []
        for spikes_d in spike_amplitudes_array:
            self.spike_amplitudes.append(spikes_d[spikes_d > 0])

    def set_order(self, ordered_indices):
        if ordered_indices is None:
            self.ordered_spike_nums = np.copy(self.spike_nums)
        else:
            if self.spike_nums is not None:
                self.ordered_spike_nums = np.copy(self.spike_nums[ordered_indices, :])
            # else:
            #     self.ordered_spike_nums = None
            if self.spike_trains is not None:
                self.ordered_spike_trains = []
                for index in ordered_indices:
                    self.ordered_spike_trains.append(self.spike_trains[index])
            # else:
            #     self.ordered_spike_trains = None
            self.ordered_indices = ordered_indices
            self.ordered_labels = []
            for old_cell_index in self.ordered_indices:
                self.ordered_labels.append(self.labels[old_cell_index])

    def set_spike_trains_from_spike_nums(self):
        # n_cells = len(self.spike_nums)
        # n_times = len(self.spike_nums[0, :])
        if self.spike_nums is None:
            return
        self.spike_trains = []
        for cell_spikes in self.spike_nums:
            self.spike_trains.append(np.where(cell_spikes)[0].astype(float))


def connec_func_stat(mouse_sessions, data_descr, param, show_interneurons=True, cells_to_highlights=None,
                     cells_to_highlights_shape=None, cells_to_highlights_colors=None, cells_to_highlights_legend=None):
    # print(f"connec_func_stat {mouse_session.session_numbers[0]}")
    interneurons_pos = np.zeros(0, dtype="uint16")
    total_nb_neurons = 0
    for ms in mouse_sessions:
        total_nb_neurons += ms.spike_struct.n_cells
    n_outs_total = np.zeros(total_nb_neurons)
    n_ins_total = np.zeros(total_nb_neurons)
    neurons_count_so_far = 0
    for ms_nb, ms in enumerate(mouse_sessions):
        nb_neurons = ms.spike_struct.n_cells
        n_ins = np.sum(ms.spike_struct.n_in_matrix, axis=1) / nb_neurons
        n_ins = np.round(n_ins * 100, 2)
        n_outs = np.sum(ms.spike_struct.n_out_matrix, axis=1) / nb_neurons
        n_outs = np.round(n_outs * 100, 2)
        if len(ms.spike_struct.inter_neurons) > 0:
            interneurons_pos = np.concatenate((interneurons_pos,
                                               np.array(ms.spike_struct.inter_neurons) + neurons_count_so_far))
        n_ins_total[neurons_count_so_far:(neurons_count_so_far + nb_neurons)] = n_ins
        n_outs_total[neurons_count_so_far:(neurons_count_so_far + nb_neurons)] = n_outs
        neurons_count_so_far += nb_neurons

    values_to_scatter = []
    labels = []
    scatter_shapes = []
    colors = []
    values_to_scatter.append(np.mean(n_outs_total))
    values_to_scatter.append(np.median(n_outs_total))
    labels.extend(["mean", "median"])
    scatter_shapes.extend(["o", "s"])
    colors.extend(["white", "white"])
    if show_interneurons and len(interneurons_pos) > 0:
        values_to_scatter.extend(list(n_outs_total[interneurons_pos]))
        labels.extend([f"interneuron (x{len(interneurons_pos)})"])
        scatter_shapes.extend(["*"] * len(n_outs_total[interneurons_pos]))
        colors.extend(["red"] * len(n_outs_total[interneurons_pos]))
    if cells_to_highlights is not None:
        for index, cells in enumerate(cells_to_highlights):
            values_to_scatter.extend(list(n_outs_total[np.array(cells)]))
            labels.append(cells_to_highlights_legend[index])
            scatter_shapes.extend([cells_to_highlights_shape[index]] * len(cells))
            colors.extend([cells_to_highlights_colors[index]] * len(cells))

    plot_hist_ratio_spikes_events(ratio_spikes_events=n_outs_total,
                                  description=f"{data_descr}_distribution_n_out",
                                  values_to_scatter=np.array(values_to_scatter),
                                  labels=labels,
                                  scatter_shapes=scatter_shapes,
                                  colors=colors,
                                  tight_x_range=True,
                                  xlabel="Active cells (%)",
                                  ylabel="Probability distribution (%)",
                                  param=param)

    values_to_scatter = []
    scatter_shapes = []
    labels = []
    colors = []
    values_to_scatter.append(np.mean(n_ins_total))
    values_to_scatter.append(np.median(n_ins_total))
    labels.extend(["mean", "median"])
    scatter_shapes.extend(["o", "s"])
    colors.extend(["white", "white"])
    if show_interneurons and len(interneurons_pos) > 0:
        values_to_scatter.extend(list(n_ins_total[interneurons_pos]))
        labels.extend([f"interneuron (x{len(interneurons_pos)})"])
        scatter_shapes.extend(["*"] * len(n_ins_total[interneurons_pos]))
        colors.extend(["red"] * len(n_ins_total[interneurons_pos]))
    if cells_to_highlights is not None:
        for index, cells in enumerate(cells_to_highlights):
            values_to_scatter.extend(list(n_ins_total[np.array(cells)]))
            labels.append(cells_to_highlights_legend)
            scatter_shapes.extend([cells_to_highlights_shape[index]] * len(cells))
            colors.extend([cells_to_highlights_colors[index]] * len(cells))

    plot_hist_ratio_spikes_events(ratio_spikes_events=n_ins_total,
                                  description=f"{data_descr}_distribution_n_in",
                                  values_to_scatter=np.array(values_to_scatter),
                                  labels=labels,
                                  scatter_shapes=scatter_shapes,
                                  colors=colors,
                                  tight_x_range=True,
                                  xlabel="Active cells (%)",
                                  ylabel="Probability distribution (%)",
                                  param=param)
    return n_ins_total, n_outs_total


def save_stat_by_age(ratio_spikes_events_by_age, ratio_spikes_total_events_by_age, mouse_sessions,
                     interneurons_indices_by_age, param):
    file_name = f'{param.path_results}/stat_for_all_mice_{param.time_str}.txt'
    round_factor = 1

    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"Stat by age" + '\n')
        file.write("" + '\n')

        for age in ratio_spikes_events_by_age.keys():
            ratio_spikes_events = np.array(ratio_spikes_events_by_age[age])
            inter_neurons = np.array(interneurons_indices_by_age[age])
            non_inter_neurons = np.setdiff1d(np.arange(len(ratio_spikes_events)), inter_neurons)
            file.write('\n')
            file.write('\n')
            file.write("#" * 50 + '\n')
            file.write('\n')
            file.write(f"p{age}")
            for ms in mouse_sessions:
                if ms.age == age:
                    file.write(f"{ms.description} + '\n'")
            file.write('\n')
            file.write("Ratio spikes in events vs all spikes for each non-interneuron cells:" + '\n')
            file.write(f"mean: {np.round(np.mean(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                       f"std: {np.round(np.std(ratio_spikes_events[non_inter_neurons]), round_factor)}" + '\n')
            file.write(f"median: {np.round(np.median(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                       f"5th percentile: {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 5), round_factor)}, "
                       f"25th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 25), round_factor)}, "
                       f"75th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 75), round_factor)}, "
                       f"95th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 95), round_factor)}" + '\n')
            file.write('\n')
            file.write('\n')

            if len(inter_neurons) == 0:
                file.write("No interneurons" + '\n')
            else:
                file.write("Ratio spikes in events vs all spikes for each interneuron cells:" + '\n')
                file.write(f"mean: {np.round(np.mean(ratio_spikes_events[inter_neurons]), round_factor)}, "
                           f"std: {np.round(np.std(ratio_spikes_events[inter_neurons]), round_factor)}" + '\n')
                file.write(f"median: {np.round(np.median(ratio_spikes_events[inter_neurons]), round_factor)}, "
                           f"5th percentile: {np.round(np.percentile(ratio_spikes_events[inter_neurons], 5), round_factor)}, "
                           f"25th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 25), round_factor)}, "
                           f"75th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 75), round_factor)}, "
                           f"95th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 95), round_factor)}" + '\n')
            file.write('\n')
            file.write('\n')

            file.write('\n')
            file.write('\n')
            ratio_spikes_events = np.array(ratio_spikes_total_events_by_age[age])
            file.write("Ratio spikes in events vs all events for non-interneuron cells:" + '\n')
            file.write(f"mean: {np.round(np.mean(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                       f"std: {np.round(np.std(ratio_spikes_events[non_inter_neurons]), round_factor)}" + '\n')
            file.write(f"median: {np.round(np.median(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                       f"5th percentile: {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 5), round_factor)}, "
                       f"25th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 25), round_factor)}, "
                       f"75th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 75), round_factor)}, "
                       f"95th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 95), round_factor)}" + '\n')
            file.write('\n')
            file.write('\n')
            if len(inter_neurons) == 0:
                file.write("No interneurons" + '\n')
            else:
                file.write("Ratio spikes in events vs all events for interneuron cells:" + '\n')
                file.write(f"mean: {np.round(np.mean(ratio_spikes_events[inter_neurons]), round_factor)}, "
                           f"std: {np.round(np.std(ratio_spikes_events[inter_neurons]), round_factor)}" + '\n')
                file.write(f"median: {np.round(np.median(ratio_spikes_events[inter_neurons]), round_factor)}, "
                           f"5th percentile: {np.round(np.percentile(ratio_spikes_events[inter_neurons], 5), round_factor)}, "
                           f"25th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 25), round_factor)}, "
                           f"75th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 75), round_factor)}, "
                           f"95th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 95), round_factor)}" + '\n')


def save_stat_sce_detection_methods(spike_nums_to_use, activity_threshold, ms,
                                    SCE_times, param, sliding_window_duration,
                                    perc_threshold, use_raster_dur,
                                    keep_max_each_surrogate, ratio_spikes_events,
                                    ratio_spikes_total_events,
                                    n_surrogate_activity_threshold):
    round_factor = 1
    raster_option = "raster_dur" if use_raster_dur else "onsets"
    technique_details = " max of each surrogate " if keep_max_each_surrogate else ""
    technique_details_file = "max_each_surrogate" if keep_max_each_surrogate else ""
    file_name = f'{param.path_results}/{ms.description}_stat_{raster_option}_{perc_threshold}_perc_' \
                f'{technique_details_file}_{param.time_str}.txt'

    n_sce = len(SCE_times)
    n_cells = len(spike_nums_to_use)
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(
            f"Stat {ms.description} using {raster_option}{technique_details} {perc_threshold} percentile " + '\n')
        file.write("" + '\n')
        file.write(f"{n_cells} cells, {n_sce} events" + '\n')
        file.write(f"Event participation threshold {activity_threshold} / "
                   f"{np.round((activity_threshold*100)/n_cells, 2)}%, "
                   f"{perc_threshold} percentile, "
                   f"{n_surrogate_activity_threshold} surrogates" + '\n')
        file.write(f"Sliding window duration {sliding_window_duration}" + '\n')
        file.write("" + '\n')
        file.write("" + '\n')

        inter_neurons = ms.spike_struct.inter_neurons
        non_inter_neurons = np.setdiff1d(np.arange(n_cells), inter_neurons)
        # ratio_spikes_events
        file.write("Ratio spikes in events vs all spikes for non interneuron cells:" + '\n')
        file.write(f"mean: {np.round(np.mean(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                   f"std: {np.round(np.std(ratio_spikes_events[non_inter_neurons]), round_factor)}" + '\n')
        file.write(f"median: {np.round(np.median(ratio_spikes_events[non_inter_neurons]), round_factor)}, "
                   f"5th percentile: {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 5), round_factor)}, "
                   f"25th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 25), round_factor)}, "
                   f"75th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 75), round_factor)}, "
                   f"95th percentile {np.round(np.percentile(ratio_spikes_events[non_inter_neurons], 95), round_factor)}" + '\n')
        file.write('\n')
        file.write('\n')

        if len(inter_neurons) == 0:
            file.write("No interneurons" + '\n')
        else:
            file.write("Ratio spikes in events vs all spikes for interneurons:" + '\n')
            file.write(f"mean: {np.round(np.mean(ratio_spikes_events[inter_neurons]), round_factor)}, "
                       f"std: {np.round(np.std(ratio_spikes_events[inter_neurons]), round_factor)}" + '\n')
            file.write(f"median: {np.round(np.median(ratio_spikes_events[inter_neurons]), round_factor)}, "
                       f"5th percentile: {np.round(np.percentile(ratio_spikes_events[inter_neurons], 5), round_factor)}, "
                       f"25th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 25), round_factor)}, "
                       f"75th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 75), round_factor)}, "
                       f"95th percentile {np.round(np.percentile(ratio_spikes_events[inter_neurons], 95), round_factor)}"
                       + '\n')
        file.write('\n')
        file.write('\n')
        file.write("#" * 50 + '\n')
        file.write('\n')
        file.write('\n')

        # ratio_spikes_events on total events
        file.write("Ratio spikes in events vs all events for non interneurons cell:" + '\n')
        file.write(f"mean: {np.round(np.mean(ratio_spikes_total_events[non_inter_neurons]), round_factor)}, "
                   f"std: {np.round(np.std(ratio_spikes_total_events[non_inter_neurons]), round_factor)}" + '\n')
        file.write(f"median: {np.round(np.median(ratio_spikes_total_events[non_inter_neurons]), round_factor)}, "
                   f"5th percentile: {np.round(np.percentile(ratio_spikes_total_events[non_inter_neurons], 5), round_factor)}, "
                   f"25th percentile {np.round(np.percentile(ratio_spikes_total_events[non_inter_neurons], 25), round_factor)}, "
                   f"75th percentile {np.round(np.percentile(ratio_spikes_total_events[non_inter_neurons], 75), round_factor)}, "
                   f"95th percentile {np.round(np.percentile(ratio_spikes_total_events[non_inter_neurons], 95), round_factor)}" + '\n')
        file.write('\n')
        file.write('\n')
        inter_neurons = ms.spike_struct.inter_neurons
        if len(inter_neurons) == 0:
            file.write("No interneurons" + '\n')
        else:
            file.write("Ratio spikes in events vs all events for interneurons:" + '\n')
            file.write(f"mean: {np.round(np.mean(ratio_spikes_total_events[inter_neurons]), round_factor)}, "
                       f"std: {np.round(np.std(ratio_spikes_total_events[inter_neurons]), round_factor)}" + '\n')
            file.write(f"median: {np.round(np.median(ratio_spikes_total_events[inter_neurons]), round_factor)}, "
                       f"5th percentile: {np.round(np.percentile(ratio_spikes_total_events[inter_neurons], 5), round_factor)}, "
                       f"25th percentile {np.round(np.percentile(ratio_spikes_total_events[inter_neurons], 25), round_factor)}, "
                       f"75th percentile {np.round(np.percentile(ratio_spikes_total_events[inter_neurons], 75), round_factor)}, "
                       f"95th percentile {np.round(np.percentile(ratio_spikes_total_events[inter_neurons], 95), round_factor)}"
                       + '\n')
        file.write('\n')
        file.write('\n')
        file.write("#" * 50 + '\n')
        file.write('\n')
        file.write('\n')

        if n_sce > 0:
            file.write("All events (SCEs) stat:" + '\n')
            file.write('\n')
            # duration in frames
            duration_values = np.zeros(n_sce, dtype="uint16")
            max_activity_values = np.zeros(n_sce)
            mean_activity_values = np.zeros(n_sce)
            overall_activity_values = np.zeros(n_sce)
            # Collecting data from each SCE
            for sce_id in np.arange(len(SCE_times)):
                result = give_stat_one_sce(sce_id=sce_id,
                                           spike_nums_to_use=spike_nums_to_use,
                                           SCE_times=SCE_times,
                                           sliding_window_duration=sliding_window_duration)
                duration_values[sce_id], max_activity_values[sce_id], mean_activity_values[sce_id], \
                overall_activity_values[sce_id] = result

            # percentage as the total number of cells
            max_activity_values = (max_activity_values / n_cells) * 100
            mean_activity_values = (mean_activity_values / n_cells) * 100
            overall_activity_values = (overall_activity_values / n_cells) * 100

            # # normalizing by duration
            # max_activity_values = max_activity_values / duration_values
            # mean_activity_values = mean_activity_values / duration_values
            # overall_activity_values = overall_activity_values / duration_values
            file.write(f"Duration: mean (std) / median {np.round(np.mean(duration_values), round_factor)} "
                       f"({np.round(np.std(duration_values), round_factor)}) / "
                       f"{np.round(np.median(duration_values), round_factor)}\n")
            file.write(f"Overall participation: mean (std) / median "
                       f"{np.round(np.mean(overall_activity_values), round_factor)} "
                       f"({np.round(np.std(overall_activity_values), round_factor)}) "
                       f"/ {np.round(np.median(overall_activity_values), round_factor)}\n")
            file.write(f"Max participation: mean (std) / median "
                       f"{np.round(np.mean(max_activity_values), round_factor)} "
                       f" ({np.round(np.std(max_activity_values), round_factor)}) / "
                       f"{np.round(np.median(max_activity_values), round_factor)}\n")
            file.write(f"Mean participation: mean (std) / median "
                       f"{np.round(np.mean(mean_activity_values), round_factor)} "
                       f"({np.round(np.std(mean_activity_values), round_factor)}) "
                       f"/ {np.round(np.median(mean_activity_values), round_factor)}\n")

            doing_each_sce_stat = False

            if doing_each_sce_stat:
                file.write('\n')
                file.write('\n')
                file.write("#" * 50 + '\n')
                file.write('\n')
                file.write('\n')
                # for each SCE
                for sce_id in np.arange(len(SCE_times)):
                    file.write(f"SCE {sce_id}" + '\n')
                    file.write(f"Duration_in_frames {duration_values[sce_id]}" + '\n')
                    file.write(
                        f"Overall participation {np.round(overall_activity_values[sce_id], round_factor)}" + '\n')
                    file.write(f"Max participation {np.round(max_activity_values[sce_id], round_factor)}" + '\n')
                    file.write(f"Mean participation {np.round(mean_activity_values[sce_id], round_factor)}" + '\n')

                    file.write('\n')
                    file.write('\n')

                file.write('\n')
                file.write('\n')
                file.write("#" * 50 + '\n')
                file.write('\n')

        file.write("ISI stats" '\n')
        file.write('\n')
        file.write('\n')

        cells_isi = tools_misc.get_isi(spike_data=ms.spike_struct.spike_nums, spike_trains_format=False)

        non_interneurons_isi = []
        interneurons_isi = []
        for cell_index in np.arange(n_cells):
            if cell_index in inter_neurons:
                interneurons_isi.extend(list(cells_isi[cell_index]))
            else:
                non_interneurons_isi.extend(list(cells_isi[cell_index]))

        file.write("ISI non interneurons:" + '\n')
        file.write(f"mean: {np.round(np.mean(non_interneurons_isi), round_factor)}, "
                   f"std: {np.round(np.std(non_interneurons_isi), round_factor)}" + '\n')
        file.write(f"median: {np.round(np.median(non_interneurons_isi), round_factor)}, "
                   f"5th percentile: {np.round(np.percentile(non_interneurons_isi, 5), round_factor)}, "
                   f"25th percentile {np.round(np.percentile(non_interneurons_isi, 25), round_factor)}, "
                   f"75th percentile {np.round(np.percentile(non_interneurons_isi, 75), round_factor)}, "
                   f"95th percentile {np.round(np.percentile(non_interneurons_isi, 95), round_factor)}" + '\n')
        if len(inter_neurons) == 0:
            file.write("No interneurons:" + '\n')
        else:
            file.write("ISI interneurons:" + '\n')
            file.write(f"mean: {np.round(np.mean(interneurons_isi), round_factor)}, "
                       f"std: {np.round(np.std(interneurons_isi), round_factor)}" + '\n')
            file.write(f"median: {np.round(np.median(interneurons_isi), round_factor)}, "
                       f"5th percentile: {np.round(np.percentile(interneurons_isi, 5), round_factor)}, "
                       f"25th percentile {np.round(np.percentile(interneurons_isi, 25), round_factor)}, "
                       f"75th percentile {np.round(np.percentile(interneurons_isi, 75), round_factor)}, "
                       f"95th percentile {np.round(np.percentile(interneurons_isi, 95), round_factor)}" + '\n')

        doing_each_isi_stat = False
        if doing_each_isi_stat:
            for cell_index in np.arange(len(ms.spike_struct.spike_nums)):
                file.write(f"cell {cell_index}" + '\n')
                file.write(f"median isi: {np.round(np.median(cells_isi[cell_index]), round_factor)}, " + '\n')
                file.write(f"mean isi {np.round(np.mean(cells_isi[cell_index]), round_factor)}" + '\n')
                file.write(f"std isi {np.round(np.std(cells_isi[cell_index]), round_factor)}" + '\n')

                file.write('\n')
                file.write('\n')

    return duration_values, max_activity_values, mean_activity_values, overall_activity_values


def plot_activity_duration_vs_age(mouse_sessions, ms_ages, duration_values_list,
                                  max_activity_values_list,
                                  mean_activity_values_list,
                                  overall_activity_values_list,
                                  param,
                                  save_formats="pdf"):
    duration_values_list_backup = duration_values_list
    overall_activity_values_list_backup = overall_activity_values_list
    # grouping mouses from the same age
    duration_dict = dict()
    max_activity_dict = dict()
    mean_activity_dict = dict()
    overall_activity_dict = dict()
    for i, age in enumerate(ms_ages):
        if age not in duration_dict:
            duration_dict[age] = duration_values_list[i]
            max_activity_dict[age] = max_activity_values_list[i]
            mean_activity_dict[age] = mean_activity_values_list[i]
            overall_activity_dict[age] = overall_activity_values_list[i]
        else:
            duration_dict[age] = np.concatenate((duration_dict[age], duration_values_list[i]))
            max_activity_dict[age] = np.concatenate((max_activity_dict[age], max_activity_values_list[i]))
            mean_activity_dict[age] = np.concatenate((mean_activity_dict[age], mean_activity_values_list[i]))
            overall_activity_dict[age] = np.concatenate((overall_activity_dict[age], overall_activity_values_list[i]))

    ms_ages = np.unique(ms_ages)
    duration_values_list = []
    max_activity_values_list = []
    mean_activity_values_list = []
    overall_activity_values_list = []
    for age in ms_ages:
        duration_values_list.append(duration_dict[age])
        max_activity_values_list.append(max_activity_dict[age])
        mean_activity_values_list.append(mean_activity_dict[age])
        overall_activity_values_list.append(overall_activity_dict[age])

    # normalizing by duration
    overall_activity_values_list_normalized = []
    for i, overall_activity_values in enumerate(overall_activity_values_list):
        overall_activity_values_list_normalized.append(overall_activity_values / duration_values_list[i])

    raw_y_datas = [duration_values_list, max_activity_values_list, mean_activity_values_list,
                   overall_activity_values_list,
                   overall_activity_values_list_normalized]
    y_data_labels = ["Duration", "Max participation", "Average participation", "Overall participation",
                     "Overall participation normalized"]

    # TODO: merge data from the same age
    # TODO: Show data with stds
    for index_raw, raw_y_data in enumerate(raw_y_datas):
        fcts_to_apply = [np.median]
        fcts_descr = ["median"]
        for index_fct, fct_to_apply in enumerate(fcts_to_apply):
            # raw_y_data is a list of np.array, we need to apply a fct to each array so we keep only one value
            y_data = np.zeros(len(ms_ages))
            stds = np.zeros(len(ms_ages))
            high_percentile_participation = np.zeros(len(ms_ages))
            low_percentile_participation = np.zeros(len(ms_ages))
            for index, data in enumerate(raw_y_data):
                y_data[index] = fct_to_apply(data)
                stds[index] = np.std(data)
                high_percentile_participation[index] = np.percentile(data, 95)
                low_percentile_participation[index] = np.percentile(data, 5)

            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    gridspec_kw={'height_ratios': [1]},
                                    figsize=(12, 12))
            ax1.set_facecolor("black")
            # ax1.scatter(ms_ages, y_data, color="blue", marker="o",
            #             s=12, zorder=20)
            # ax1.errorbar(ms_ages, y_data, yerr=stds, fmt='o', color="blue")
            # ax1.scatter(ms_ages, y_data, color="blue", marker="o",
            #             s=12, zorder=20)
            ax1.plot(ms_ages, y_data, color="blue")
            # stds = np.array(std_power_spectrum[freq_min_index:freq_max_index])
            ax1.fill_between(ms_ages, low_percentile_participation, high_percentile_participation,
                             alpha=0.5, facecolor="blue")

            ax1.set_ylabel(y_data_labels[index_raw] + f" ({fcts_descr[index_fct]})")
            ax1.set_xlabel("age")

            if isinstance(save_formats, str):
                save_formats = [save_formats]
            for save_format in save_formats:
                fig.savefig(f'{param.path_results}/{y_data_labels[index_raw]}_{fcts_descr[index_fct]}_vs_age'
                            f'_{param.time_str}.{save_format}',
                            format=f"{save_format}")

        plt.close()

    # then keeping only overall participation, non normalized
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(20, 20))
    ax1.set_facecolor("black")

    # markers = ['o', '*', 's', 'v', '<', '>', '^', 'x', '+', "."]  # d losange
    # colors = ["darkmagenta", "white", "saddlebrown", "blue", "red", "darkgrey", "chartreuse", "cornflowerblue",
    #           "pink", "darkgreen", "gold"]
    max_x = 0
    for index, age in enumerate(ms_ages):
        # color = cm.nipy_spectral(float(age-5) / (len(ms_ages)))
        # indices_rand used for jittering
        jitter_range_x = 0.5
        indices_rand = np.linspace(-jitter_range_x, jitter_range_x, len(duration_values_list[index]))
        jitter_range_y = 0.3
        indices_rand_y = np.linspace(-jitter_range_y, jitter_range_y, len(duration_values_list[index]))
        np.random.shuffle(indices_rand)
        np.random.shuffle(indices_rand_y)
        max_x = np.max((max_x, np.max(duration_values_list[index])))
        ax1.scatter(duration_values_list[index] + indices_rand,
                    overall_activity_values_list[index] + indices_rand_y,
                    color=param.colors[index % (len(param.colors))],
                    marker=param.markers[index % (len(param.markers))],
                    s=80, alpha=0.8, label=f"P{age}", edgecolors='none')

    ax1.set_xscale("log")
    ax1.set_xlim(0.4, max_x + 5)
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_ylabel("Participation (%)")
    ax1.set_xlabel("Duration (frames)")

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/participation_vs_duration'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")
    plt.close()

    ###################
    ###################
    # then the same with band of colors
    ###################

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(20, 20))
    ax1.set_facecolor("black")

    # colors = ["darkmagenta", "white", "saddlebrown", "blue", "red", "darkgrey", "chartreuse", "cornflowerblue",
    #           "pink", "darkgreen", "gold"]
    max_x = 0
    # from 1 to 79, 15 bins
    ranges = np.logspace(0, 1.9, 15)
    for index, age in enumerate(ms_ages):
        # first we want to bin the duration values
        # gathering participation for each common duration
        duration_d = SortedDict()
        overall_activity_values = overall_activity_values_list[index]
        for i_duration, duration_value in enumerate(duration_values_list[index]):
            range_pos = bisect(ranges, duration_value) - 1
            range_value = ranges[range_pos]
            if range_value not in duration_d:
                duration_d[range_value] = [overall_activity_values[i_duration]]
            else:
                duration_d[range_value].append(overall_activity_values[i_duration])

        average_participation = np.zeros(len(duration_d))
        stds_participation = np.zeros(len(duration_d))
        high_percentile_participation = np.zeros(len(duration_d))
        low_percentile_participation = np.zeros(len(duration_d))
        durations = np.zeros(len(duration_d))

        for i, duration in enumerate(duration_d.keys()):
            durations[i] = duration
            participations = duration_d[duration]
            average_participation[i] = np.median(participations)
            stds_participation[i] = np.std(participations)
            high_percentile_participation[i] = np.percentile(participations, 75)
            low_percentile_participation[i] = np.percentile(participations, 25)

        max_x = np.max((max_x, np.max(duration_values_list[index])))

        ax1.plot(durations, average_participation, color=param.colors[index % (len(param.colors))], label=f"P{age}")
        # ax1.fill_between(durations, average_participation - stds_participation, average_participation + stds_participation,
        #                  alpha=0.5, facecolor=colors[index % (len(colors))])

        ax1.fill_between(durations, low_percentile_participation, high_percentile_participation,
                         alpha=0.5, facecolor=param.colors[index % (len(param.colors))])

    ax1.set_xscale("log")
    ax1.set_xlim(0.4, max_x + 5)
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_ylabel("Participation (%)")
    ax1.set_xlabel("Duration (frames)")

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/participation_vs_duration_bands'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")
    plt.close()

    # ##################################################################################
    # ########################## WEIGHT vs normalized participation ##################
    # ################################################################################

    # grouping mouses from the same weight
    duration_dict = dict()
    overall_activity_dict = dict()
    duration_values_list = duration_values_list_backup
    overall_activity_values_list = overall_activity_values_list_backup
    ms_weights = []
    for ms in mouse_sessions:
        ms_weights.append(ms.weight)

    for i, weight in enumerate(ms_weights):
        if weight is None:
            continue
        if weight not in duration_dict:
            duration_dict[weight] = duration_values_list[i]
            overall_activity_dict[weight] = overall_activity_values_list[i]
        else:
            duration_dict[weight] = np.concatenate((duration_dict[weight], duration_values_list[i]))
            overall_activity_dict[weight] = np.concatenate((overall_activity_dict[weight],
                                                            overall_activity_values_list[i]))
    filter_ms_weights = []
    for weight in ms_weights:
        if weight is None:
            continue
        filter_ms_weights.append(weight)
    ms_weights = filter_ms_weights
    ms_weights = np.unique(ms_weights)
    duration_values_list = []
    overall_activity_values_list = []
    for weight in ms_weights:
        duration_values_list.append(duration_dict[weight])
        overall_activity_values_list.append(overall_activity_dict[weight])

    # normalizing by duration
    overall_activity_values_list_normalized = []
    for i, overall_activity_values in enumerate(overall_activity_values_list):
        overall_activity_values_list_normalized.append(overall_activity_values / duration_values_list[i])

    y_data = np.zeros(len(ms_weights))
    stds = np.zeros(len(ms_weights))
    high_percentile_participation = np.zeros(len(ms_weights))
    low_percentile_participation = np.zeros(len(ms_weights))
    for index, data in enumerate(overall_activity_values_list_normalized):
        y_data[index] = np.median(data)
        stds[index] = np.std(data)
        high_percentile_participation[index] = np.percentile(data, 95)
        low_percentile_participation[index] = np.percentile(data, 5)

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    ax1.set_facecolor("black")
    ax1.plot(ms_weights, y_data, color="blue")
    # stds = np.array(std_power_spectrum[freq_min_index:freq_max_index])
    ax1.fill_between(ms_weights, low_percentile_participation, high_percentile_participation,
                     alpha=0.5, facecolor="blue")

    ax1.set_ylabel("Overall participation normalized")
    ax1.set_xlabel("weight (grams)")

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/Overall participation normalized_vs_weight'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")

    plt.close()


def plot_duration_spikes_by_age(mouse_sessions, ms_ages,
                                duration_spikes_by_age, param, save_formats="pdf"):
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(20, 20))
    ax1.set_facecolor("black")

    y_data = np.zeros(len(ms_ages))
    # stds = np.zeros(len(ms_ages))
    high_percentile_participation = np.zeros(len(ms_ages))
    low_percentile_participation = np.zeros(len(ms_ages))
    for index, age in enumerate(ms_ages):
        if age not in duration_spikes_by_age:
            y_data[index] = 1
            high_percentile_participation[index] = 1
            low_percentile_participation[index] = 1
            continue

        y_data[index] = np.median(duration_spikes_by_age[age])
        # stds[index] = np.std(data)
        high_percentile_participation[index] = np.percentile(duration_spikes_by_age[age], 95)
        low_percentile_participation[index] = np.percentile(duration_spikes_by_age[age], 5)

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    ax1.set_facecolor("black")
    # ax1.scatter(ms_ages, y_data, color="blue", marker="o",
    #             s=12, zorder=20)
    # ax1.errorbar(ms_ages, y_data, yerr=stds, fmt='o', color="blue")
    # ax1.scatter(ms_ages, y_data, color="blue", marker="o",
    #             s=12, zorder=20)
    ax1.plot(ms_ages, y_data, color="blue")
    # stds = np.array(std_power_spectrum[freq_min_index:freq_max_index])
    ax1.fill_between(ms_ages, low_percentile_participation, high_percentile_participation,
                     alpha=0.5, facecolor="blue")

    ax1.set_ylabel("Spikes duration (frames)")
    ax1.set_xlabel("age")

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/duration_spikes_by_age'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")
    plt.close()


def plot_hist_ratio_spikes_events(ratio_spikes_events, description, values_to_scatter,
                                  labels, scatter_shapes, colors, param, tight_x_range=False,
                                  xlabel="", ylabel=None, save_formats="pdf"):
    distribution = np.array(ratio_spikes_events)
    hist_color = "blue"
    edge_color = "white"
    if tight_x_range:
        max_range = np.max(distribution)
        min_range = np.min(distribution)
    else:
        max_range = 100
        min_range = 0
    weights = (np.ones_like(distribution) / (len(distribution))) * 100

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    ax1.set_facecolor("black")
    bins = int(np.sqrt(len(distribution)))
    hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins, range=(min_range, max_range),
                                                facecolor=hist_color,
                                                edgecolor=edge_color,
                                                weights=weights, log=False)

    scatter_bins = np.ones(len(values_to_scatter), dtype="int16")
    scatter_bins *= -1

    for i, edge in enumerate(edges_plt):
        # print(f"i {i}, edge {edge}")
        if i >= len(hist_plt):
            # means that scatter left are on the edge of the last bin
            scatter_bins[scatter_bins == -1] = i - 1
            break

        if len(values_to_scatter[values_to_scatter <= edge]) > 0:
            if (i + 1) < len(edges_plt):
                bool_list = values_to_scatter < edge  # edges_plt[i + 1]
                for i_bool, bool_value in enumerate(bool_list):
                    if bool_value:
                        if scatter_bins[i_bool] == -1:
                            new_i = max(0, i - 1)
                            scatter_bins[i_bool] = new_i
            else:
                bool_list = values_to_scatter < edge
                for i_bool, bool_value in enumerate(bool_list):
                    if bool_value:
                        if scatter_bins[i_bool] == -1:
                            scatter_bins[i_bool] = i

    decay = np.linspace(1.1, 1.15, len(values_to_scatter))
    for i, value_to_scatter in enumerate(values_to_scatter):
        if i < len(labels):
            plt.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                        color=colors[i], s=60, zorder=20, label=labels[i])
        else:
            plt.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                        color=colors[i], s=60, zorder=20)
    # if tight_x_range:
    #     plt.xlim(min_range, max_range)
    plt.xlim(0, 100)
    xticks = np.arange(0, 110, 10)

    ax1.set_xticks(xticks)
    # sce clusters labels
    ax1.set_xticklabels(xticks)

    if ylabel is None:
        ax1.set_ylabel("Distribution (%)")
    else:
        ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)

    ax1.legend()

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/{description}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")

    plt.close()


def compute_stat_about_significant_seq(files_path, param, save_formats="pdf"):
    file_names = []
    n_categories = 4
    marker_cat = ["*", "d", "o", "s"]
    # categories that should be displayed
    banned_categories = []
    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(files_path):
        file_names.extend(local_filenames)
        break

    # dict1: key age (int) value dict2
    # dict2: key category (int, from 1 to 4), value dict3
    # dict3: key length seq (int), value dict4
    # dict4: key repetitions (int), value nb of seq with this length and this repetition
    data_dict = SortedDict()
    # key is an int representing age, and value is an int representing the number of sessions for a given age
    nb_ms_by_age = dict()

    for file_name in file_names:
        file_name_original = file_name
        if not file_name.startswith("p"):
            if not file_name.startswith("significant_sorting_results"):
                continue
            # p_index = len("significant_sorting_results")+1
            file_name = file_name[len("significant_sorting_results"):]
        index_ = file_name.find("_")
        if index_ < 1:
            continue
        age = int(file_name[1:index_])
        nb_ms_by_age[age] = nb_ms_by_age.get(age, 0) + 1
        # print(f"age {age}")
        if age not in data_dict:
            data_dict[age] = dict()
            for cat in np.arange(1, 1 + n_categories):
                data_dict[age][cat] = dict()

        with open(files_path + file_name_original, "r", encoding='UTF-8') as file:
            for nb_line, line in enumerate(file):
                line_list = line.split(':')
                seq_n_cells = int(line_list[0])
                line_list = line_list[1].split("]")
                # we remove the '[' on the first position
                repetitions_str = line_list[0][1:].split(",")
                repetitions = []
                for rep in repetitions_str:
                    repetitions.append(int(rep))
                # we remove the ' [' on the first position
                categories_str = line_list[1][2:].split(",")
                categories = []
                for cat in categories_str:
                    categories.append(int(cat))

                for index, cat in enumerate(categories):
                    if seq_n_cells not in data_dict[age][cat]:
                        data_dict[age][cat][seq_n_cells] = dict()

                    rep = repetitions[index]
                    if rep not in data_dict[age][cat][seq_n_cells]:
                        data_dict[age][cat][seq_n_cells][rep] = 0
                    data_dict[age][cat][seq_n_cells][rep] += 1
                # print(f"{seq_n_cells} cells: {repetitions} {categories}")

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(15, 15))
    ax1.set_facecolor("black")

    n_jitter = 10
    jitter_range_x = 0.4
    indices_rand = np.linspace(-jitter_range_x, jitter_range_x, n_jitter)
    jitter_range_y = 0.4
    indices_rand_y = np.linspace(-jitter_range_y, jitter_range_y, n_jitter)
    np.random.shuffle(indices_rand)
    np.random.shuffle(indices_rand_y)

    age_index = 0
    i_jitter = 0
    max_rep = 0
    min_rep = 100
    min_len = 100
    max_len = 0
    len_grid = tuple((3, 5))
    # key is a len, value dict2
    #  dict 2: key is rep, and value is np.array of n*n dimension, bool
    grid_dict = dict()
    for age, cat_dict in data_dict.items():
        for cat, len_dict in cat_dict.items():
            if cat in banned_categories:
                continue
            for len_seq, rep_dict in len_dict.items():
                if len_seq not in grid_dict:
                    grid_dict[len_seq] = dict()
                min_len = np.min((len_seq, min_len))
                max_len = np.max((len_seq, max_len))
                for rep, n_seq in rep_dict.items():
                    if rep not in grid_dict[len_seq]:
                        grid_dict[len_seq][rep] = np.ones(len_grid[0] * len_grid[1], dtype="bool")
                    max_rep = np.max((max_rep, rep))
                    min_rep = np.min((min_rep, rep))
                    grid = grid_dict[len_seq][rep]
                    free_spots = np.where(grid)[0]
                    if len(free_spots) == 0:
                        print("error no more free spots")
                        return
                    np.random.shuffle(free_spots)
                    spot = free_spots[0]
                    grid[spot] = False
                    grid_pos_y = int(spot / len_grid[1])
                    grid_pos_x = spot % len_grid[1]
                    x_pos = len_seq - 0.4 + (grid_pos_x * (0.8 / (len_grid[1] - 1)))
                    y_pos = rep - 0.35 + (grid_pos_y * (0.7 / (len_grid[0] - 1)))
                    n_seq_normalized = np.round((n_seq / nb_ms_by_age[age]), 1)
                    if n_seq_normalized % 1 == 0:
                        n_seq_normalized = int(n_seq_normalized)
                    ax1.scatter(x_pos,
                                y_pos,
                                color=param.colors[age_index % (len(param.colors))],
                                marker=param.markers[cat - 1],
                                s=15 + 5 * np.sqrt(n_seq_normalized), alpha=1, edgecolors='none')
                    ax1.text(x=x_pos, y=y_pos,
                             s=f"{n_seq_normalized}", color="black", zorder=22,
                             ha='center', va="center", fontsize=0.9, fontweight='bold')
                    i_jitter += 1
        age_index += 1
    with_grid = False
    if with_grid:
        for len_seq in np.arange(min_len, max_len + 2):
            ax1.vlines(len_seq - 0.5, 0,
                       max_rep + 1, color="white", linewidth=0.5,
                       linestyles="dashed", zorder=1)
        for rep in np.arange(min_rep, max_rep + 2):
            ax1.hlines(rep - 0.5, min_len - 1,
                       max_len + 1, color="white", linewidth=0.5,
                       linestyles="dashed", zorder=1)
    legend_elements = []
    # [Line2D([0], [0], color='b', lw=4, label='Line')
    age_index = 0
    for age, cat_dict in data_dict.items():
        legend_elements.append(Patch(facecolor=param.colors[age_index % (len(param.colors))],
                                     edgecolor='black', label=f'p{age}'))
        age_index += 1

    for cat in np.arange(1, n_categories + 1):
        if cat in banned_categories:
            continue
        legend_elements.append(Line2D([0], [0], marker=param.markers[cat - 1], color="w", lw=0, label="*" * cat,
                                      markerfacecolor='black', markersize=15))

    ax1.legend(handles=legend_elements)

    # plt.title(title)
    ax1.set_ylabel(f"Repetition (#)")
    ax1.set_xlabel("length")
    ax1.set_ylim(min_rep - 0.5, max_rep + 0.5)
    ax1.set_xlim(min_len - 0.5, max_len + 0.5)
    # xticks = np.arange(0, len(data_dict))
    # ax1.set_xticks(xticks)
    # # sce clusters labels
    # ax1.set_xticklabels(labels)

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/scatter_significant_seq'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")

    plt.close()


def box_plot_data_by_age(data_dict, title, filename, y_label, param, save_formats="pdf"):
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    ax1.set_facecolor("black")

    colorfull = True
    labels = []
    data_list = []
    for age, data in data_dict.items():
        data_list.append(data)
        labels.append(age)
    bplot = plt.boxplot(data_list, patch_artist=colorfull,
                        labels=labels, sym='', zorder=1)  # whis=[5, 95], sym='+'
    # color=["b", "cornflowerblue"],
    # fill with colors

    # edge_color="silver"

    for element in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(bplot[element], color="white")

    for element in ['means', 'medians']:
        plt.setp(bplot[element], color="silver")

    if colorfull:
        colors = param.colors[:len(data_dict)]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # plt.xlim(0, 100)
    plt.title(title)
    ax1.set_ylabel(f"{y_label}")
    ax1.set_xlabel("age")
    xticks = np.arange(1, len(data_dict) + 1)
    ax1.set_xticks(xticks)
    # sce clusters labels
    ax1.set_xticklabels(labels)

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/{filename}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")

    plt.close()


def plot_psth_interneurons_events(ms, spike_nums_dur, spike_nums, SCE_times, sliding_window_duration,
                                  param, save_formats="pdf"):
    if spike_nums_dur is None:
        return
    inter_neurons = ms.spike_struct.inter_neurons
    n_times = len(spike_nums[0, :])
    for inter_neuron in inter_neurons:
        # key is an int which reprensent the sum of spikes at a certain distance (in frames) of the event,
        # negative or positive
        spike_at_time_dict = dict()

        for sce_id, time_tuple in enumerate(SCE_times):
            time_tuple = SCE_times[sce_id]
            duration_in_frames = (time_tuple[1] - time_tuple[0]) + 1
            n_slidings = (duration_in_frames - sliding_window_duration) + 1

            sum_activity_for_each_frame = np.zeros(n_slidings)
            for n in np.arange(n_slidings):
                # see to use window_duration to find the amount of participation
                time_beg = time_tuple[0] + n
                sum_activity_for_each_frame[n] = len(np.where(np.sum(spike_nums_dur[:,
                                                                     time_beg:(time_beg + sliding_window_duration)],
                                                                     axis=1))[0])
            max_activity_index = np.argmax(sum_activity_for_each_frame)
            # time_max represent the time at which the event is at its top, we will center histogram on it
            time_max = time_tuple[0] + max_activity_index + int(sliding_window_duration / 2)
            # beg_time = 0 if sce_id == 0 else SCE_times[sce_id - 1][1]
            # end_time = n_times if sce_id == (len(SCE_times) - 1) else SCE_times[sce_id + 1][0]
            range_in_frames = 50
            min_SCE = 0 if sce_id == 0 else SCE_times[sce_id - 1][1]
            max_SCE = n_times if sce_id == (len(SCE_times) - 1) else SCE_times[sce_id + 1][0]
            beg_time = np.max((0, time_max - range_in_frames, min_SCE))
            end_time = np.min((n_times, time_max + range_in_frames, max_SCE))
            # print(f"time_max {time_max}, beg_time {beg_time}, end_time {end_time}")
            # before the event
            time_spikes = np.where(spike_nums[inter_neuron, beg_time:time_max])[0]
            # print(f"before time_spikes {time_spikes}")
            if len(time_spikes) > 0:
                time_spike = np.max(time_spikes)
                time_spike = time_spike - ((time_max + 1) - beg_time)
                # print(f"before time_spike {time_spike}")
                # for time_spike in time_spikes:
                spike_at_time_dict[time_spike] = spike_at_time_dict.get(time_spike, 0) + 1
            # after the event
            time_spikes = np.where(spike_nums[inter_neuron, time_max:end_time])[0]
            if len(time_spikes) > 0:
                time_spike = np.min(time_spikes)
                # for time_spike in time_spikes:
                spike_at_time_dict[time_spike] = spike_at_time_dict.get(time_spike, 0) + 1

        distribution = []
        for time, nb_spikes_at_time in spike_at_time_dict.items():
            # print(f"time {time}")
            distribution.extend([time] * nb_spikes_at_time)
        if len(distribution) == 0:
            continue
        distribution = np.array(distribution)
        hist_color = "blue"
        edge_color = "white"
        max_range = np.max((np.max(distribution), range_in_frames))
        min_range = np.min((np.min(distribution), -range_in_frames))
        weights = (np.ones_like(distribution) / (len(distribution))) * 100

        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(15, 10))
        ax1.set_facecolor("black")
        # as many bins as time
        bins = (max_range - min_range) // 4
        # bins = int(np.sqrt(len(distribution)))
        hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins, range=(min_range, max_range),
                                                    facecolor=hist_color,
                                                    edgecolor=edge_color,
                                                    weights=weights, log=False)
        ax1.vlines(0, 0,
                   np.max(hist_plt), color="white", linewidth=2,
                   linestyles="dashed")
        plt.title(f"{ms.description} interneuron {inter_neuron}")
        ax1.set_ylabel(f"Spikes (%)")
        ax1.set_xlabel("time (frames)")
        ax1.set_ylim(0, 100)
        # xticks = np.arange(0, len(data_dict))
        # ax1.set_xticks(xticks)
        # # sce clusters labels
        # ax1.set_xticklabels(labels)

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/psth_{ms.description}_interneuron_{inter_neuron}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}")

        plt.close()


def get_ratio_spikes_on_events_vs_total_events_by_cell(spike_nums,
                                                       spike_nums_dur,
                                                       sce_times_numbers):
    n_cells = len(spike_nums)
    result = np.zeros(n_cells)

    for cell in np.arange(n_cells):
        n_sces = np.max(sce_times_numbers) + 1
        if spike_nums_dur is not None:
            spikes_index = np.where(spike_nums_dur[cell, :])[0]
        else:
            spikes_index = np.where(spike_nums[cell, :])[0]
        sce_numbers = sce_times_numbers[spikes_index]
        # will give us all sce in which the cell spikes
        sce_numbers = np.unique(sce_numbers)
        # removing the -1, which is when it spikes not in a SCE
        if len(np.where(sce_numbers == - 1)[0]) > 0:
            sce_numbers = sce_numbers[1:]
        # print(f"len sce_numbers {len(sce_numbers)}, sce_numbers {sce_numbers}, n_spikes {n_spikes}")
        # print(f"len sce_times_numbers {len(np.unique(sce_times_numbers))}, sce_numbers {np.unique(sce_times_numbers)}")
        result[cell] = np.min((((len(sce_numbers) / n_sces) * 100), 100))
    # result = result[result >= 0]
    return result


def test_seq_detect(ms):
    # print(f"test_seq_detect {ms.description} {ms.best_order_loaded}")
    if ms.best_order_loaded is None:
        return

    spike_nums_dur = ms.spike_struct.spike_nums_dur
    spike_nums_dur_ordered = spike_nums_dur[ms.best_order_loaded, :]
    seq_dict = find_sequences_in_ordered_spike_nums(spike_nums_dur_ordered, param=ms.param)
    # save_on_file_seq_detection_results(best_cells_order=ms.best_order_loaded,
    #                                    seq_dict=seq_dict,
    #                                    file_name=f"sorting_results_with_timestamps{ms.description}.txt",
    #                                    param=ms.param,
    #                                    significant_category_dict=None)

    colors_for_seq_list = ["blue", "red", "limegreen", "grey", "orange", "cornflowerblue", "yellow", "seagreen",
                           "magenta"]
    ordered_labels_real_data = []
    labels = np.arange(len(spike_nums_dur_ordered))
    for old_cell_index in ms.best_order_loaded:
        ordered_labels_real_data.append(labels[old_cell_index])
    plot_spikes_raster(spike_nums=spike_nums_dur_ordered, param=ms.param,
                       title=f"{ms.description}_spike_nums_ordered_seq_test",
                       spike_train_format=False,
                       file_name=f"{ms.description}_spike_nums_ordered_seq_test",
                       y_ticks_labels=ordered_labels_real_data,
                       save_raster=True,
                       show_raster=False,
                       sliding_window_duration=1,
                       show_sum_spikes_as_percentage=True,
                       plot_with_amplitude=False,
                       activity_threshold=ms.activity_threshold,
                       save_formats="pdf",
                       seq_times_to_color_dict=seq_dict,
                       link_seq_color=colors_for_seq_list,
                       link_seq_line_width=1,
                       link_seq_alpha=0.9,
                       jitter_links_range=5,
                       min_len_links_seq=3,
                       spike_shape="|",
                       spike_shape_size=10)

    print(f"n_cells: {len(spike_nums_dur_ordered)}")

    if ms.cell_assemblies is not None:
        total_cells_in_ca = 0
        for cell_assembly_index, cell_assembly in enumerate(ms.cell_assemblies):
            total_cells_in_ca += len(cell_assembly)
        #     print(f"CA {cell_assembly_index}: {cell_assembly}")
        # print(f"n_cells in cell assemblies: {total_cells_in_ca}")
        sequences_with_ca_numbers = []
        cells_seq_with_correct_indices = []
        # we need to find the indices from the organized seq
        for seq in seq_dict.keys():
            cells_seq_with_correct_indices.append(ms.best_order_loaded[np.array(seq)])
        for seq in cells_seq_with_correct_indices:
            new_seq = np.ones(len(seq), dtype="int16")
            new_seq *= - 1
            for cell_assembly_index, cell_assembly in enumerate(ms.cell_assemblies):
                for index_cell, cell in enumerate(seq):
                    if cell in cell_assembly:
                        new_seq[index_cell] = cell_assembly_index
            sequences_with_ca_numbers.append(new_seq)

    # print("")
    # print("Seq with cell assemblies index")
    choose_manually = True
    if choose_manually:
        max_index_seq = 0
        max_rep = 0
        seq_elu = None
        seq_dict_for_best_seq = dict()
        for seq in seq_dict.keys():
            # if (seq[0] > 10):
            #     break
            if len(seq_dict[seq]) > max_rep:  # len(seq_dict[seq]): max_rep < len(seq)
                max_rep = len(seq_dict[seq])  # len(seq)  #
                max_index_seq = np.max(seq)
                seq_elu = seq
                # max_index_seq = np.max((max_index_seq, np.max(seq)))
        for seq in seq_dict.keys():
            if seq[-1] <= seq_elu[-1]:
                seq_dict_for_best_seq[seq] = seq_dict[seq]

        # for index, seq in enumerate(sequences_with_ca_numbers):
        #     print(f"Original: {cells_seq_with_correct_indices[index]}")
        #     print(f"Cell assemblies {seq}")
    else:
        max_index_seq = 15

    cells_to_highlight = []
    cells_to_highlight_colors = []

    n_cell_assemblies = len(ms.cell_assemblies)

    for cell_assembly_index, cell_assembly in enumerate(ms.cell_assemblies):
        color = cm.nipy_spectral(float(cell_assembly_index + 1) / (n_cell_assemblies + 1))
        cell_indices_to_color = []
        for cell in cell_assembly:
            cell_index = np.where(ms.best_order_loaded == cell)[0][0]
            if cell_index <= max_index_seq:
                cell_indices_to_color.append(cell_index)
        cells_to_highlight.extend(cell_indices_to_color)
        cells_to_highlight_colors.extend([color] * len(cell_indices_to_color))

        span_areas_coords = []
        span_area_colors = []
        span_areas_coords.append(ms.mvt_frames_periods)
        span_area_colors.append('red')
        span_areas_coords.append(ms.sce_times_in_cell_assemblies)
        span_area_colors.append('green')
        span_areas_coords.append(ms.twitches_frames_periods)
        span_area_colors.append('blue')

    colors_for_seq_list = ["white"]
    # span_area_coords = [ms.SCE_times]
    # span_area_colors = ['lightgrey']
    plot_spikes_raster(spike_nums=spike_nums_dur_ordered[:max_index_seq + 1, :], param=ms.param,
                       title=f"{ms.description}_spike_nums_ordered_cell_assemblies_colored",
                       spike_train_format=False,
                       file_name=f"{ms.description}_spike_nums_ordered_cell_assemblies_colored",
                       y_ticks_labels=ordered_labels_real_data[:max_index_seq + 1],
                       save_raster=True,
                       show_raster=False,
                       sliding_window_duration=1,
                       show_sum_spikes_as_percentage=True,
                       plot_with_amplitude=False,
                       save_formats="pdf",
                       cells_to_highlight=cells_to_highlight,
                       cells_to_highlight_colors=cells_to_highlight_colors,
                       # seq_times_to_color_dict=seq_dict_for_best_seq,
                       # link_seq_color=colors_for_seq_list,
                       # link_seq_line_width=0.8,
                       # link_seq_alpha=0.9,
                       jitter_links_range=0,
                       min_len_links_seq=3,
                       spike_shape="o",
                       spike_shape_size=1,
                       # span_area_coords=span_areas_coords,
                       # span_area_colors=span_area_colors,
                       # span_area_coords=span_area_coords,
                       # span_area_colors=span_area_colors,
                       # span_area_only_on_raster=False,
                       without_activity_sum=True,
                       size_fig=(15, 6))
    # with amplitude, using traces
    # print(f"ms.traces.shape {ms.traces.shape}")
    amplitude_spike_nums = ms.traces
    n_times = len(amplitude_spike_nums[0, :])
    # normalizing it
    for cell, amplitudes in enumerate(amplitude_spike_nums):
        # print(f"cell {cell}, min: {np.mean(amplitudes)}, max {np.max(amplitudes)}, mean {np.mean(amplitudes)}")
        # amplitude_spike_nums[cell, :] = amplitudes / np.median(amplitudes)
        amplitude_spike_nums[cell, :] = amplitude_spike_nums[cell, :] / np.median(amplitude_spike_nums[cell, :])
        amplitude_spike_nums[cell, :] = norm01(gaussblur1D(amplitude_spike_nums[cell, :], n_times / 2, 0))
        amplitude_spike_nums[cell, :] = norm01(amplitude_spike_nums[cell, :])
        amplitude_spike_nums[cell, :] = amplitude_spike_nums[cell, :] - np.median(amplitude_spike_nums[cell, :])
    #     print(f"cell {cell}, min: {np.mean(amplitudes)}, max {np.max(amplitudes)}, mean {np.mean(amplitudes)}")
    amplitude_spike_nums_ordered = amplitude_spike_nums[ms.best_order_loaded, :]

    n_cells = len(amplitude_spike_nums_ordered)
    cells_range_to_display = np.arange(max_index_seq + 1)
    cells_range_to_display = np.arange(n_cells)
    use_heatmap = True
    if use_heatmap:
        with_one_ax = True
        if with_one_ax:
            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    gridspec_kw={'height_ratios': [1],
                                                 'width_ratios': [1]},
                                    figsize=(15, 6))
            ax1.imshow(amplitude_spike_nums_ordered[cells_range_to_display, :],
                       cmap=plt.get_cmap("hot"), extent=[0, 10, 0, 10], aspect='auto', vmin=0, vmax=0.5)
            ax1.axis('image')
            ax1.axis('off')

            # ax1.imshow(,  cmap=plt.get_cmap("hot")) # extent=[0, 1, 0, 1],
            # sns.heatmap(amplitude_spike_nums_ordered[:max_index_seq + 1, :],
            #             cbar=False, ax=ax1, cmap=plt.get_cmap("hot"), rasterized=True) #

            fig.savefig(f'{ms.param.path_results}/{ms.description}spike_nums_ordered_heatmap_.pdf',
                        format=f"pdf")
            show_fig = True
            if show_fig:
                plt.show()
            plt.close()
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1,
                                                     gridspec_kw={'height_ratios': [0.25, 0.25, 0.25, 0.25],
                                                                  'width_ratios': [1]},
                                                     figsize=(15, 6))
            vmax = 0.9
            ax1.imshow(amplitude_spike_nums_ordered[cells_range_to_display, :n_times // 4],
                       cmap=plt.get_cmap("hot"), extent=[0, 10, 0, 1], aspect='auto', vmin=0, vmax=0.5)
            ax1.axis('image')
            ax1.axis('off')
            ax2.imshow(amplitude_spike_nums_ordered[cells_range_to_display, (n_times // 4):(2 * (n_times // 4))],
                       cmap=plt.get_cmap("hot"), extent=[0, 10, 0, 1], aspect='auto', vmin=0, vmax=0.5)
            # matshow
            ax2.axis('image')
            ax2.axis('off')
            ax3.imshow(amplitude_spike_nums_ordered[cells_range_to_display, (2 * (n_times // 4)):(3 * (n_times // 4))],
                       cmap=plt.get_cmap("hot"), extent=[0, 10, 0, 1], aspect='auto', vmin=0,
                       vmax=0.5)  # vmin=0, vmax=vmax,

            ax3.axis('image')
            ax3.axis('off')
            ax4.imshow(amplitude_spike_nums_ordered[cells_range_to_display, (3 * (n_times // 4)):(4 * (n_times // 4))],
                       cmap=plt.get_cmap("hot"), extent=[0, 10, 0, 1], aspect='auto', vmin=0, vmax=0.5)
            ax4.axis('image')
            ax4.axis('off')

            # ax1.imshow(,  cmap=plt.get_cmap("hot")) # extent=[0, 1, 0, 1],
            # sns.heatmap(amplitude_spike_nums_ordered[:max_index_seq + 1, :],
            #             cbar=False, ax=ax1, cmap=plt.get_cmap("hot"), rasterized=True) #

            fig.savefig(f'{ms.param.path_results}/{ms.description}spike_nums_ordered_heatmap_.pdf',
                        format=f"pdf")
            show_fig = True
            if show_fig:
                plt.show()
            plt.close()
    else:
        # cells_range_to_display = np.arange(n_cells)
        labels_to_display = []
        for i in cells_range_to_display:
            labels_to_display.append(ordered_labels_real_data[i])
        # putting all values > 0.8 to 1
        amplitude_spike_nums_ordered[amplitude_spike_nums_ordered > 0.8] = 1
        plot_spikes_raster(spike_nums=amplitude_spike_nums_ordered[cells_range_to_display, :], param=ms.param,
                           title=f"{ms.description}_spike_nums_ordered_cell_assemblies_colored",
                           spike_train_format=False,
                           file_name=f"{ms.description}_spike_nums_ordered_with_amplitude",
                           y_ticks_labels=labels_to_display,
                           save_raster=True,
                           show_raster=False,
                           sliding_window_duration=1,
                           show_sum_spikes_as_percentage=True,
                           plot_with_amplitude=True,
                           save_formats="pdf",
                           # cells_to_highlight=cells_to_highlight,
                           # cells_to_highlight_colors=cells_to_highlight_colors,
                           # seq_times_to_color_dict=seq_dict_for_best_seq,
                           # link_seq_color=colors_for_seq_list,
                           # link_seq_line_width=0.8,
                           # link_seq_alpha=0.9,
                           # jitter_links_range=0,
                           # min_len_links_seq=3,
                           spike_shape="|",
                           spike_shape_size=0.2,
                           # span_area_coords=span_areas_coords,
                           # span_area_colors=span_area_colors,
                           # span_area_coords=span_area_coords,
                           # span_area_colors=span_area_colors,
                           # span_area_only_on_raster=False,
                           without_activity_sum=False,
                           spike_nums_for_activity_sum=spike_nums_dur_ordered[cells_range_to_display, :],
                           size_fig=(15, 6),
                           cmap_name="hot")

    # cells_to_highlight = None,
    # cells_to_highlight_colors = None,


def norm01(data):
    min_value = np.min(data)
    max_value = np.max(data)

    difference = max_value - min_value

    data -= min_value

    if difference > 0:
        data = data / difference

    return data


def gaussblur1D(data, dw, dim):
    n_times = data.shape[dim]

    if n_times % 2 == 0:
        kt = np.arange((-n_times / 2) + 0.5, (n_times / 2) - 0.5 + 1)
    else:
        kt = np.arange(-(n_times - 1) / 2, ((n_times - 1) / 2 + 1))

    if dim == 0:
        fil = np.exp(-np.square(kt) / dw ** 2) * np.ones(data.shape[0])
    elif dim == 1:
        fil = np.ones(data.shape[1]) * np.exp(-np.square(kt) / dw ** 2)
    elif dim == 2:
        fil = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        for i in np.arange(n_times):
            fil[:, :, i] = np.ones((data.shape[0], data.shape[1])) * np.exp(-np.square(kt[i]) / dw ** 2)

    tfA = np.fft.fftshift(np.fft.fft(data, axis=dim))
    b = np.real(np.fft.ifft(np.fft.ifftshift(tfA * fil), axis=dim))

    return b


"""
function B=GaussBlur1d(A,dw,dim)


    nt=size(A,dim);
    if rem(nt,2)==0
        kt=-nt/2+0.5:nt/2-0.5;
    else
        kt=-(nt-1)/2:(nt-1)/2;
    end
    if dim==1
        Fil=exp(-kt'.^2/dw^2)*ones(1,size(A,2));
    end
    if dim==2
        Fil=ones(size(A,1),1)*exp(-kt.^2/dw^2);
    end
    if dim==3
        for i=1:nt
            Fil(:,:,i)=ones(size(A,1),size(A,2))*exp(-kt(i).^2/dw^2);
        end
    end
    tfA=fftshift(fft(A,[],dim));
    B=real(ifft(ifftshift(tfA.*Fil),[],dim));
"""


def get_ratio_spikes_on_events_vs_total_spikes_by_cell(spike_nums,
                                                       spike_nums_dur,
                                                       sce_times_numbers):
    n_cells = len(spike_nums)
    result = np.zeros(n_cells)

    for cell in np.arange(n_cells):
        n_spikes = np.sum(spike_nums[cell, :])
        if spike_nums_dur is not None:
            spikes_index = np.where(spike_nums_dur[cell, :])[0]
        else:
            spikes_index = np.where(spike_nums[cell, :])[0]

        sce_numbers = sce_times_numbers[spikes_index]
        # will give us all sce in which the cell spikes
        sce_numbers = np.unique(sce_numbers)
        # removing the -1, which is when it spikes not in a SCE
        if len(np.where(sce_numbers == - 1)[0]) > 0:
            sce_numbers = sce_numbers[1:]
        # print(f"len sce_numbers {len(sce_numbers)}, sce_numbers {sce_numbers}, n_spikes {n_spikes}")
        # print(f"len sce_times_numbers {len(np.unique(sce_times_numbers))}, sce_numbers {np.unique(sce_times_numbers)}")
        if n_spikes == 0:
            result[cell] = 0
        else:
            result[cell] = np.min((((len(sce_numbers) / n_spikes) * 100), 100))
    # removing cells without spikes
    # result = result[result >= 0]
    return result


class SurpriseMichou:
    def __init__(self, n_lines, deactivated=False):
        self.actual_line = 0
        self.n_lines = 0
        self.bottom_width = n_lines
        self.top_width = n_lines * 3
        self.deactivated = deactivated

    def get_one_part_str(self):

        return ""

    def print_next_line(self):
        if self.deactivated:
            return
        width = self.top_width - (self.actual_line * 2)
        result = ""
        if actual_line < (n_lines - 1):
            pass

        self.actual_line += 1


def print_surprise_for_michou(n_lines, actual_line):
    bottom_width = n_lines
    top_width = n_lines * 3

    width = top_width - (actual_line * 2)

    result = ""
    if actual_line < (n_lines - 1):
        result += f"{' ' * 5}"
        result += f"{' '* actual_line}"
        if actual_line > (n_lines / 2):
            result += "\\"
        else:
            result += "|"
        if actual_line == (n_lines // 2):
            result += f"{' ' * (width//2 - 1)}"
            result += " O "
            result += f"{' ' * (width//2 - 1)}"
        else:
            result += f"{' ' * width}"

        if actual_line > (n_lines / 2):
            result += "/"
        else:
            result += "|"

        result += f"{' ' * (top_width // 4)}"

        # result += f"{' ' * actual_line}"
        result += "|"
        if actual_line == (n_lines // 2):
            result += f"{' ' * (width//2 - 1)}"
            result += " O "
            result += f"{' ' * (width//2 - 1)}"
        else:
            result += f"{' ' * width}"
        result += "|"
    else:
        result += f"{' ' * 5}"
        result += f"{' ' * actual_line}"
        result += f"{'|' * bottom_width}"

    print(f"{result}")


def load_mouse_sessions(ms_str_to_load, param, load_traces):
    ms_str_to_ms_dict = dict()

    if "p6_18_02_07_a001_ms" in ms_str_to_load:
        p6_18_02_07_a001_ms = MouseSession(age=6, session_id="18_02_07_a001", nb_ms_by_frame=100, param=param,
                                           weight=4.35)
        # calculated with 99th percentile on raster dur
        p6_18_02_07_a001_ms.activity_threshold = 15
        # p6_18_02_07_a001_ms.set_low_activity_threshold(threshold=3, percentile_value=1)
        # p6_18_02_07_a001_ms.set_low_activity_threshold(threshold=5, percentile_value=5)
        p6_18_02_07_a001_ms.set_inter_neurons([28, 36, 54, 75])
        # duration of those interneurons: [ 18.58 17.78   19.  17.67]
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p6_18_02_07_a001_ms.load_data_from_file(file_name_to_load=
                                                "p6/p6_18_02_07_a001/p6_18_02_07_001_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p6_18_02_07_a001_ms.load_data_from_file(file_name_to_load="p6/p6_18_02_07_a001/p6_18_02_07_a001_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p6_18_02_07_a001_ms.load_data_from_file(file_name_to_load="p6/p6_18_02_07_a001/p6_18_02_07_a001_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p6_18_02_07_a001_ms.load_abf_file(abf_file_name="p6/p6_18_02_07_a001/p6_18_02_07_001.abf",
                                          threshold_piezo=25)  # 7
        ms_str_to_ms_dict["p6_18_02_07_a001_ms"] = p6_18_02_07_a001_ms
        # p6_18_02_07_a001_ms.plot_cell_assemblies_on_map()

    if "p6_18_02_07_a002_ms" in ms_str_to_load:
        p6_18_02_07_a002_ms = MouseSession(age=6, session_id="18_02_07_a002", nb_ms_by_frame=100, param=param,
                                           weight=4.35)
        # calculated with 99th percentile on raster dur
        p6_18_02_07_a002_ms.activity_threshold = 8
        # p6_18_02_07_a002_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        # p6_18_02_07_a002_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
        p6_18_02_07_a002_ms.set_inter_neurons([40, 90])
        # duration of those interneurons: 16.27  23.33
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p6_18_02_07_a002_ms.load_data_from_file(file_name_to_load=
                                                "p6/p6_18_02_07_a002/p6_18_02_07_002_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p6_18_02_07_a002_ms.load_data_from_file(file_name_to_load="p6/p6_18_02_07_a002/p6_18_02_07_a002_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p6_18_02_07_a002_ms.load_data_from_file(file_name_to_load="p6/p6_18_02_07_a002/p6_18_02_07_a002_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p6_18_02_07_a002_ms.load_abf_file(abf_file_name="p6/p6_18_02_07_a002/p6_18_02_07_002.abf",
                                          threshold_piezo=25)
        ms_str_to_ms_dict["p6_18_02_07_a002_ms"] = p6_18_02_07_a002_ms

    if "p7_171012_a000_ms" in ms_str_to_load:
        p7_171012_a000_ms = MouseSession(age=7, session_id="17_10_12_a000", nb_ms_by_frame=100, param=param,
                                         weight=None)
        # calculated with 99th percentile on raster dur
        p7_171012_a000_ms.activity_threshold = 19
        # p7_171012_a000_ms.set_low_activity_threshold(threshold=6, percentile_value=1)
        # p7_171012_a000_ms.set_low_activity_threshold(threshold=7, percentile_value=5)
        p7_171012_a000_ms.set_inter_neurons([305, 360, 398, 412])
        # duration of those interneurons: 13.23  12.48  10.8   11.88
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p7_171012_a000_ms.load_data_from_file(
            file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_Corrected_RasterDur.mat",
            variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p7_171012_a000_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_Traces.mat",
                                                  variables_mapping=variables_mapping)
        # variables_mapping = {"coord": "ContoursAll"} ContoursSoma ContoursIntNeur
        # p7_171012_a000_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_12_a000/p7_17_10_12_a000_CellDetect.mat",
        #                                          variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p7_171012_a000_ms"] = p7_171012_a000_ms

    if "p7_17_10_18_a002_ms" in ms_str_to_load:
        p7_17_10_18_a002_ms = MouseSession(age=7, session_id="17_10_18_a002", nb_ms_by_frame=100, param=param,
                                           weight=None)
        # calculated with 99th percentile on raster dur
        p7_17_10_18_a002_ms.activity_threshold = 14
        # p7_17_10_18_a002_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        # p7_17_10_18_a002_ms.set_low_activity_threshold(threshold=4, percentile_value=5)
        p7_17_10_18_a002_ms.set_inter_neurons([51])
        # duration of those interneurons: 14.13
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p7_17_10_18_a002_ms.load_data_from_file(file_name_to_load=
                                                "p7/p7_17_10_18_a002/p7_17_10_18_a002_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p7_17_10_18_a002_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a002/p7_17_10_18_a002_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p7_17_10_18_a002_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a002/p7_17_10_18_a002_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p7_17_10_18_a002_ms"] = p7_17_10_18_a002_ms

    if "p7_17_10_18_a004_ms" in ms_str_to_load:
        p7_17_10_18_a004_ms = MouseSession(age=7, session_id="17_10_18_a004", nb_ms_by_frame=100, param=param,
                                           weight=None)
        # calculated with 99th percentile on raster dur
        p7_17_10_18_a004_ms.activity_threshold = 13
        # p7_17_10_18_a004_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        # p7_17_10_18_a004_ms.set_low_activity_threshold(threshold=3, percentile_value=5)
        p7_17_10_18_a004_ms.set_inter_neurons([298])
        # duration of those interneurons: 15.35
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p7_17_10_18_a004_ms.load_data_from_file(file_name_to_load=
                                                "p7/p7_17_10_18_a004/p7_17_10_18_a004_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p7_17_10_18_a004_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a004/p7_17_10_18_a004_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p7_17_10_18_a004_ms.load_data_from_file(file_name_to_load="p7/p7_17_10_18_a004/p7_17_10_18_a004_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p7_17_10_18_a004_ms"] = p7_17_10_18_a004_ms

    if "p7_18_02_08_a000_ms" in ms_str_to_load:
        p7_18_02_08_a000_ms = MouseSession(age=7, session_id="18_02_08_a000", nb_ms_by_frame=100, param=param,
                                           weight=3.85)
        # calculated with 99th percentile on raster dur
        p7_18_02_08_a000_ms.activity_threshold = 10
        # p7_18_02_08_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        # p7_18_02_08_a000_ms.set_low_activity_threshold(threshold=2, percentile_value=5)
        p7_18_02_08_a000_ms.set_inter_neurons([56, 95, 178])
        # duration of those interneurons: 12.88  13.94  13.04
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p7_18_02_08_a000_ms.load_data_from_file(file_name_to_load=
                                                "p7/p7_18_02_08_a000/p7_18_02_18_a000_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p7_18_02_08_a000_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a000/p7_18_02_08_a000_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p7_18_02_08_a000_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a000/p7_18_02_08_a000_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p7_18_02_08_a000_ms.load_abf_file(abf_file_name="p7/p7_18_02_08_a000/p7_18_02_08_a000.abf",
                                          threshold_piezo=4)
        ms_str_to_ms_dict["p7_18_02_08_a000_ms"] = p7_18_02_08_a000_ms

    if "p7_18_02_08_a001_ms" in ms_str_to_load:
        p7_18_02_08_a001_ms = MouseSession(age=7, session_id="18_02_08_a001", nb_ms_by_frame=100, param=param,
                                           weight=3.85)
        # calculated with 99th percentile on raster dur
        p7_18_02_08_a001_ms.activity_threshold = 12
        # p7_18_02_08_a001_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        # p7_18_02_08_a001_ms.set_low_activity_threshold(threshold=3, percentile_value=5)
        p7_18_02_08_a001_ms.set_inter_neurons([151])
        # duration of those interneurons: 22.11
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p7_18_02_08_a001_ms.load_data_from_file(file_name_to_load=
                                                "p7/p7_18_02_08_a001/p7_18_02_18_a001_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p7_18_02_08_a001_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a001/p7_18_02_08_a001_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p7_18_02_08_a001_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a001/p7_18_02_08_a001_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p7_18_02_08_a001_ms.load_abf_file(abf_file_name="p7/p7_18_02_08_a001/p7_18_02_08_a001.abf",
                                          threshold_piezo=4)
        ms_str_to_ms_dict["p7_18_02_08_a001_ms"] = p7_18_02_08_a001_ms

    if "p7_18_02_08_a002_ms" in ms_str_to_load:
        p7_18_02_08_a002_ms = MouseSession(age=7, session_id="18_02_08_a002", nb_ms_by_frame=100, param=param,
                                           weight=3.85)
        # calculated with 99th percentile on raster dur
        p7_18_02_08_a002_ms.activity_threshold = 9
        # p7_18_02_08_a002_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        # p7_18_02_08_a002_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
        p7_18_02_08_a002_ms.set_inter_neurons([207])
        # duration of those interneurons: 22.3
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p7_18_02_08_a002_ms.load_data_from_file(file_name_to_load=
                                                "p7/p7_18_02_08_a002/p7_18_02_08_a002_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p7_18_02_08_a002_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a002/p7_18_02_08_a002_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p7_18_02_08_a002_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a002/p7_18_02_08_a002_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p7_18_02_08_a002_ms.load_abf_file(abf_file_name="p7/p7_18_02_08_a002/p7_18_02_08_a002.abf",
                                          threshold_piezo=2.5)
        ms_str_to_ms_dict["p7_18_02_08_a002_ms"] = p7_18_02_08_a002_ms

    if "p7_18_02_08_a003_ms" in ms_str_to_load:
        p7_18_02_08_a003_ms = MouseSession(age=7, session_id="18_02_08_a003", nb_ms_by_frame=100, param=param,
                                           weight=3.85)
        # calculated with 99th percentile on raster dur
        p7_18_02_08_a003_ms.activity_threshold = 7
        # p7_18_02_08_a003_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        # p7_18_02_08_a003_ms.set_low_activity_threshold(threshold=0, percentile_value=5)
        p7_18_02_08_a003_ms.set_inter_neurons([171])
        # duration of those interneurons: 14.92
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p7_18_02_08_a003_ms.load_data_from_file(file_name_to_load=
                                                "p7/p7_18_02_08_a003/p7_18_02_08_a003_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p7_18_02_08_a003_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a003/p7_18_02_08_a003_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p7_18_02_08_a003_ms.load_data_from_file(file_name_to_load="p7/p7_18_02_08_a003/p7_18_02_08_a003_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p7_18_02_08_a003_ms.load_abf_file(abf_file_name="p7/p7_18_02_08_a003/p7_18_02_08_a003.abf",
                                          threshold_piezo=9)  # used to be 2.5
        ms_str_to_ms_dict["p7_18_02_08_a003_ms"] = p7_18_02_08_a003_ms

    if "p8_18_02_09_a000_ms" in ms_str_to_load:
        p8_18_02_09_a000_ms = MouseSession(age=8, session_id="18_02_09_a000", nb_ms_by_frame=100, param=param,
                                           weight=None)
        # calculated with 99th percentile on raster dur
        p8_18_02_09_a000_ms.activity_threshold = 8
        # p8_18_02_09_a000_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        # p8_18_02_09_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
        p8_18_02_09_a000_ms.set_inter_neurons([64, 91])
        # duration of those interneurons: 12.48  11.47
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p8_18_02_09_a000_ms.load_data_from_file(file_name_to_load=
                                                "p8/p8_18_02_09_a000/p8_18_02_09_a000_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p8_18_02_09_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a000/p8_18_02_09_a000_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p8_18_02_09_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a000/p8_18_02_09_a000_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p8_18_02_09_a000_ms.load_abf_file(abf_file_name="p8/p8_18_02_09_a000/p8_18_02_09_a000.abf",
                                          threshold_piezo=2)  # used to be 1.5
        ms_str_to_ms_dict["p8_18_02_09_a000_ms"] = p8_18_02_09_a000_ms

    if "p8_18_02_09_a001_ms" in ms_str_to_load:
        p8_18_02_09_a001_ms = MouseSession(age=8, session_id="18_02_09_a001", nb_ms_by_frame=100, param=param,
                                           weight=None)
        # calculated with 99th percentile on raster dur
        p8_18_02_09_a001_ms.activity_threshold = 10
        # p8_18_02_09_a001_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        # p8_18_02_09_a001_ms.set_low_activity_threshold(threshold=2, percentile_value=5)
        p8_18_02_09_a001_ms.set_inter_neurons([])
        # duration of those interneurons:
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p8_18_02_09_a001_ms.load_data_from_file(file_name_to_load=
                                                "p8/p8_18_02_09_a001/p8_18_02_09_a001_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p8_18_02_09_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a001/p8_18_02_09_a001_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p8_18_02_09_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_02_09_a001/p8_18_02_09_a001_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p8_18_02_09_a001_ms.load_abf_file(abf_file_name="p8/p8_18_02_09_a001/p8_18_02_09_a001.abf",
                                          threshold_piezo=3)  # 1.5 before then 2
        ms_str_to_ms_dict["p8_18_02_09_a001_ms"] = p8_18_02_09_a001_ms

    if "p8_18_10_17_a000_ms" in ms_str_to_load:
        p8_18_10_17_a000_ms = MouseSession(age=8, session_id="18_10_17_a000", nb_ms_by_frame=100, param=param,
                                           weight=6)
        # calculated with 99th percentile on raster dur
        p8_18_10_17_a000_ms.activity_threshold = 11
        # p8_18_10_17_a000_ms.set_low_activity_threshold(threshold=, percentile_value=1)
        # p8_18_10_17_a000_ms.set_low_activity_threshold(threshold=, percentile_value=5)
        p8_18_10_17_a000_ms.set_inter_neurons([27, 70])
        # duration of those interneurons: 23.8, 43
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p8_18_10_17_a000_ms.load_data_from_file(
            file_name_to_load="p8/p8_18_10_17_a000/P8_18_10_17_a000_Corrected_RasterDur.mat",
            variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p8_18_10_17_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_17_a000/p8_18_10_17_a000_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p8_18_10_17_a000_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_17_a000/p8_18_10_17_a000_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p8_18_10_17_a000_ms"] = p8_18_10_17_a000_ms

    if "p8_18_10_17_a001_ms" in ms_str_to_load:
        p8_18_10_17_a001_ms = MouseSession(age=8, session_id="18_10_17_a001", nb_ms_by_frame=100, param=param,
                                           weight=6)
        # calculated with 99th percentile on raster dur
        p8_18_10_17_a001_ms.activity_threshold = 9
        # p8_18_10_17_a001_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        # p8_18_10_17_a001_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
        p8_18_10_17_a001_ms.set_inter_neurons([117, 135, 217, 271])
        # duration of those interneurons: 32.33, 171, 144.5, 48.8
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p8_18_10_17_a001_ms.load_data_from_file(file_name_to_load=
                                                "p8/p8_18_10_17_a001/p8_18_10_17_a001_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p8_18_10_17_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_17_a001/p8_18_10_17_a001_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p8_18_10_17_a001_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_17_a001/p8_18_10_17_a001_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        # CORRUPTED ABF ??
        p8_18_10_17_a001_ms.load_abf_file(abf_file_name="p8/p8_18_10_17_a001/p8_18_10_17_a001.abf",
                                          threshold_piezo=0.4, piezo_channel=2, sampling_rate=10000)
        ms_str_to_ms_dict["p8_18_10_17_a001_ms"] = p8_18_10_17_a001_ms

    if "p8_18_10_24_a005_ms" in ms_str_to_load:
        # 6.4
        p8_18_10_24_a005_ms = MouseSession(age=8, session_id="18_10_24_a005", nb_ms_by_frame=100, param=param,
                                           weight=6.4)
        # calculated with 99th percentile on raster dur
        p8_18_10_24_a005_ms.activity_threshold = 9
        # p8_18_10_24_a005_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        # p8_18_10_24_a005_ms.set_low_activity_threshold(threshold=1, percentile_value=5)
        p8_18_10_24_a005_ms.set_inter_neurons([33, 112, 206])
        # duration of those interneurons: 18.92, 27.33, 20.55
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p8_18_10_24_a005_ms.load_data_from_file(file_name_to_load=
                                                "p8/p8_18_10_24_a005/p8_18_10_24_a005_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p8_18_10_24_a005_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_24_a005/p8_18_10_24_a005_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p8_18_10_24_a005_ms.load_data_from_file(file_name_to_load="p8/p8_18_10_24_a005/p8_18_10_24_a005_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p8_18_10_24_a005_ms.load_abf_file(abf_file_name="p8/p8_18_10_24_a005/p8_18_10_24_a005.abf",
                                          threshold_piezo=0.5)  # used to be 0.4
        ms_str_to_ms_dict["p8_18_10_24_a005_ms"] = p8_18_10_24_a005_ms

    # p9_17_11_29_a002 low participation comparing to other, dead shortly after the recording
    # p9_17_11_29_a002_ms = MouseSession(age=9, session_id="17_11_29_a002", nb_ms_by_frame=100, param=param,
    #                                    weight=5.7)
    # # calculated with 99th percentile on raster dur
    # p9_17_11_29_a002_ms.activity_threshold = 10
    # p9_17_11_29_a002_ms.set_inter_neurons([170])
    # # limit ??
    # # duration of those interneurons: 21
    # variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
    #                      "spike_nums": "filt_Bin100ms_spikedigital",
    #                      "spike_durations": "LOC3"}
    # p9_17_11_29_a002_ms.load_data_from_file(file_name_to_load="p9/p9_17_11_29_a002/p9_17_11_29_a002_RasterDur.mat",
    #                                         variables_mapping=variables_mapping)

    # p9_17_11_29_a003_ms = MouseSession(age=9, session_id="17_11_29_a003", nb_ms_by_frame=100, param=param,
    #                                    weight=5.7)
    # # calculated with 99th percentile on raster dur
    # p9_17_11_29_a003_ms.activity_threshold = 7
    # p9_17_11_29_a003_ms.set_inter_neurons([1, 13, 54])
    # # duration of those interneurons: 21.1 22.75  23
    # variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
    #                      "spike_nums": "filt_Bin100ms_spikedigital",
    #                      "spike_durations": "LOC3"}
    # p9_17_11_29_a003_ms.load_data_from_file(file_name_to_load="p9/p9_17_11_29_a003/p9_17_11_29_a003_RasterDur.mat",
    #                                         variables_mapping=variables_mapping)
    if "p9_17_12_06_a001_ms" in ms_str_to_load:
        p9_17_12_06_a001_ms = MouseSession(age=9, session_id="17_12_06_a001", nb_ms_by_frame=100, param=param,
                                           weight=5.6)
        # calculated with 99th percentile on raster dur
        p9_17_12_06_a001_ms.activity_threshold = 8
        # p9_17_12_06_a001_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        p9_17_12_06_a001_ms.set_inter_neurons([72])
        # duration of those interneurons:15.88
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p9_17_12_06_a001_ms.load_data_from_file(file_name_to_load=
                                                "p9/p9_17_12_06_a001/p9_17_12_06_a001_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p9_17_12_06_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_06_a001/p9_17_12_06_a001_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p9_17_12_06_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_06_a001/p9_17_12_06_a001_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p9_17_12_06_a001_ms.load_abf_file(abf_file_name="p9/p9_17_12_06_a001/p9_17_12_06_a001.abf",
                                          threshold_piezo=1.5)
        ms_str_to_ms_dict["p9_17_12_06_a001_ms"] = p9_17_12_06_a001_ms

    if "p9_17_12_20_a001_ms" in ms_str_to_load:
        p9_17_12_20_a001_ms = MouseSession(age=9, session_id="17_12_20_a001", nb_ms_by_frame=100, param=param,
                                           weight=5.05)
        # calculated with 99th percentile on raster dur
        p9_17_12_20_a001_ms.activity_threshold = 8
        # p9_17_12_20_a001_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        p9_17_12_20_a001_ms.set_inter_neurons([32])
        # duration of those interneurons: 10.35
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p9_17_12_20_a001_ms.load_data_from_file(file_name_to_load=
                                                "p9/p9_17_12_20_a001/p9_17_12_20_a001_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p9_17_12_20_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_20_a001/p9_17_12_20_a001_Traces.mat",
                                                    variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p9_17_12_20_a001_ms.load_data_from_file(file_name_to_load="p9/p9_17_12_20_a001/p9_17_12_20_a001_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p9_17_12_20_a001_ms.load_abf_file(abf_file_name="p9/p9_17_12_20_a001/p9_17_12_20_a001.abf",
                                          threshold_piezo=3)  # used to be 2
        ms_str_to_ms_dict["p9_17_12_20_a001_ms"] = p9_17_12_20_a001_ms

    if "p9_18_09_27_a003_ms" in ms_str_to_load:
        p9_18_09_27_a003_ms = MouseSession(age=9, session_id="18_09_27_a003", nb_ms_by_frame=100, param=param,
                                           weight=6.65)
        # calculated with 99th percentile on raster dur
        # p9_18_09_27_a003_ms.activity_threshold = 9
        # p9_18_09_27_a003_ms.set_low_activity_threshold(threshold=, percentile_value=1)
        p9_18_09_27_a003_ms.set_inter_neurons([2, 9, 67, 206])
        # duration of those interneurons: 59.1, 32, 28, 35.15
        variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p9_18_09_27_a003_ms.load_data_from_file(file_name_to_load=
                                                "p9/p9_18_09_27_a003/p9_18_09_27_a003_Corrected_RasterDur.mat",
                                                variables_mapping=variables_mapping)

        variables_mapping = {"coord": "ContoursAll"}
        p9_18_09_27_a003_ms.load_data_from_file(file_name_to_load="p9/p9_18_09_27_a003/p9_18_09_27_a003_CellDetect.mat",
                                                variables_mapping=variables_mapping)
        p9_18_09_27_a003_ms.load_abf_file(abf_file_name="p9/p9_18_09_27_a003/p9_18_09_27_a003.abf",
                                          threshold_piezo=0.06, piezo_channel=2, sampling_rate=10000,
                                          offset=0.1)
        ms_str_to_ms_dict["p9_18_09_27_a003_ms"] = p9_18_09_27_a003_ms

    if "p10_17_11_16_a003_ms" in ms_str_to_load:
        p10_17_11_16_a003_ms = MouseSession(age=10, session_id="17_11_16_a003", nb_ms_by_frame=100, param=param,
                                            weight=6.1)
        # calculated with 99th percentile on raster dur
        p10_17_11_16_a003_ms.activity_threshold = 6
        # p10_17_11_16_a003_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        p10_17_11_16_a003_ms.set_inter_neurons([8])
        # duration of those interneurons: 28
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p10_17_11_16_a003_ms.load_data_from_file(file_name_to_load=
                                                 "p10/p10_17_11_16_a003/p10_17_11_16_a003_Corrected_RasterDur.mat",
                                                 variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p10_17_11_16_a003_ms.load_data_from_file(
                file_name_to_load="p10/p10_17_11_16_a003/p10_17_11_16_a003_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p10_17_11_16_a003_ms.load_data_from_file(
            file_name_to_load="p10/p10_17_11_16_a003/p10_17_11_16_a003_CellDetect.mat",
            variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p10_17_11_16_a003_ms"] = p10_17_11_16_a003_ms

    if "p11_17_11_24_a000_ms" in ms_str_to_load:
        p11_17_11_24_a000_ms = MouseSession(age=11, session_id="17_11_24_a000", nb_ms_by_frame=100, param=param,
                                            weight=6.7)
        # calculated with 99th percentile on raster dur
        p11_17_11_24_a000_ms.activity_threshold = 11
        # p11_17_11_24_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        p11_17_11_24_a000_ms.set_inter_neurons([193])
        # duration of those interneurons: 19.09
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p11_17_11_24_a000_ms.load_data_from_file(file_name_to_load=
                                                 "p11/p11_17_11_24_a000/p11_17_11_24_a000_Corrected_RasterDur.mat",
                                                 variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p11_17_11_24_a000_ms.load_data_from_file(
                file_name_to_load="p11/p11_17_11_24_a000/p11_17_11_24_a000_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p11_17_11_24_a000_ms.load_data_from_file(
            file_name_to_load="p11/p11_17_11_24_a000/p11_17_11_24_a000_CellDetect.mat",
            variables_mapping=variables_mapping)
        # p11_17_11_24_a000_ms.plot_cell_assemblies_on_map()
        ms_str_to_ms_dict["p11_17_11_24_a000_ms"] = p11_17_11_24_a000_ms

    if "p11_17_11_24_a001_ms" in ms_str_to_load:
        p11_17_11_24_a001_ms = MouseSession(age=11, session_id="17_11_24_a001", nb_ms_by_frame=100, param=param,
                                            weight=6.7)
        # calculated with 99th percentile on raster dur
        p11_17_11_24_a001_ms.activity_threshold = 10
        # p11_17_11_24_a001_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        p11_17_11_24_a001_ms.set_inter_neurons([])
        # duration of those interneurons:
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p11_17_11_24_a001_ms.load_data_from_file(file_name_to_load=
                                                 "p11/p11_17_11_24_a001/p11_17_11_24_a001_Corrected_RasterDur.mat",
                                                 variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p11_17_11_24_a001_ms.load_data_from_file(
                file_name_to_load="p11/p11_17_11_24_a001/p11_17_11_24_a001_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p11_17_11_24_a001_ms.load_data_from_file(
            file_name_to_load="p11/p11_17_11_24_a001/p11_17_11_24_a001_CellDetect.mat",
            variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p11_17_11_24_a001_ms"] = p11_17_11_24_a001_ms

    if "p12_171110_a000_ms" in ms_str_to_load:
        p12_171110_a000_ms = MouseSession(age=12, session_id="171110_a000", nb_ms_by_frame=100, param=param,
                                          weight=7)
        # calculated with 99th percentile on raster dur
        p12_171110_a000_ms.activity_threshold = 9
        # p12_171110_a000_ms.set_low_activity_threshold(threshold=1, percentile_value=1)
        p12_171110_a000_ms.set_inter_neurons([106, 144])
        # duration of those interneurons: 18.29  14.4
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p12_171110_a000_ms.load_data_from_file(file_name_to_load=
                                               "p12/p12_17_11_10_a000/p12_17_11_10_a000_Corrected_RasterDur.mat",
                                               variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p12_171110_a000_ms.load_data_from_file(
                file_name_to_load="p12/p12_17_11_10_a000/p12_17_11_10_a000_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p12_171110_a000_ms.load_data_from_file(
            file_name_to_load="p12/p12_17_11_10_a000/p12_17_11_10_a000_CellDetect.mat",
            variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p12_171110_a000_ms"] = p12_171110_a000_ms

    if "p12_17_11_10_a002_ms" in ms_str_to_load:
        p12_17_11_10_a002_ms = MouseSession(age=12, session_id="17_11_10_a002", nb_ms_by_frame=100, param=param,
                                            weight=7)
        # calculated with 99th percentile on raster dur
        p12_17_11_10_a002_ms.activity_threshold = 11
        # p12_17_11_10_a002_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        p12_17_11_10_a002_ms.set_inter_neurons([150, 252])
        # duration of those interneurons: 16.17, 24.8
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p12_17_11_10_a002_ms.load_data_from_file(file_name_to_load=
                                                 "p12/p12_17_11_10_a002/p12_17_11_10_a002_Corrected_RasterDur.mat",
                                                 variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p12_17_11_10_a002_ms.load_data_from_file(
                file_name_to_load="p12/p12_17_11_10_a002/p12_17_11_10_a002_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p12_17_11_10_a002_ms.load_data_from_file(
            file_name_to_load="p12/p12_17_11_10_a002/p12_17_11_10_a002_CellDetect.mat",
            variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p12_17_11_10_a002_ms"] = p12_17_11_10_a002_ms

    if "p13_18_10_29_a000_ms" in ms_str_to_load:
        p13_18_10_29_a000_ms = MouseSession(age=13, session_id="18_10_29_a000", nb_ms_by_frame=100, param=param,
                                            weight=9.4)
        # calculated with 99th percentile on raster dur
        p13_18_10_29_a000_ms.activity_threshold = 13
        # p13_18_10_29_a000_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        p13_18_10_29_a000_ms.set_inter_neurons([5, 26, 27, 35, 38])
        # duration of those interneurons: 13.57, 16.8, 22.4, 12, 14.19
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p13_18_10_29_a000_ms.load_data_from_file(file_name_to_load=
                                                 "p13/p13_18_10_29_a000/p13_18_10_29_a000_Corrected_RasterDur.mat",
                                                 variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p13_18_10_29_a000_ms.load_data_from_file(
                file_name_to_load="p13/p13_18_10_29_a000/p13_18_10_29_a000_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p13_18_10_29_a000_ms.load_data_from_file(file_name_to_load=
                                                 "p13/p13_18_10_29_a000/p13_18_10_29_a000_CellDetect.mat",
                                                 variables_mapping=variables_mapping)
        p13_18_10_29_a000_ms.load_abf_file(abf_file_name="p13/p13_18_10_29_a000/p13_18_10_29_a000.abf",
                                           threshold_piezo=None, with_run=True, sampling_rate=10000)
        ms_str_to_ms_dict["p13_18_10_29_a000_ms"] = p13_18_10_29_a000_ms

    if "p13_18_10_29_a001_ms" in ms_str_to_load:
        p13_18_10_29_a001_ms = MouseSession(age=13, session_id="18_10_29_a001", nb_ms_by_frame=100, param=param,
                                            weight=9.4)
        # calculated with 99th percentile on raster dur
        p13_18_10_29_a001_ms.activity_threshold = 11
        # p13_18_10_29_a001_ms.set_low_activity_threshold(threshold=2, percentile_value=1)
        p13_18_10_29_a001_ms.set_inter_neurons([68])
        # duration of those interneurons: 13.31
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p13_18_10_29_a001_ms.load_data_from_file(file_name_to_load=
                                                 "p13/p13_18_10_29_a001/p13_18_10_29_a001_Corrected_RasterDur.mat",
                                                 variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p13_18_10_29_a001_ms.load_data_from_file(
                file_name_to_load="p13/p13_18_10_29_a001/p13_18_10_29_a001_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p13_18_10_29_a001_ms.load_data_from_file(file_name_to_load=
                                                 "p13/p13_18_10_29_a001/p13_18_10_29_a001_CellDetect.mat",
                                                 variables_mapping=variables_mapping)
        p13_18_10_29_a001_ms.load_abf_file(abf_file_name="p13/p13_18_10_29_a001/p13_18_10_29_a001.abf",
                                           threshold_piezo=None, with_run=True, sampling_rate=10000)
        ms_str_to_ms_dict["p13_18_10_29_a001_ms"] = p13_18_10_29_a001_ms

    if "p14_18_10_23_a000_ms" in ms_str_to_load:
        p14_18_10_23_a000_ms = MouseSession(age=14, session_id="18_10_23_a000", nb_ms_by_frame=100, param=param,
                                            weight=10.35)
        # calculated with 99th percentile on raster dur
        p14_18_10_23_a000_ms.activity_threshold = 8
        # p14_18_10_23_a000_ms.set_low_activity_threshold(threshold=0, percentile_value=1)
        p14_18_10_23_a000_ms.set_inter_neurons([0])
        # duration of those interneurons: 24.33
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p14_18_10_23_a000_ms.load_data_from_file(file_name_to_load=
                                                 "p14/p14_18_10_23_a000/p14_18_10_23_a000_Corrected_RasterDur.mat",
                                                 variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p14_18_10_23_a000_ms.load_data_from_file(
                file_name_to_load="p14/p14_18_10_23_a000/p14_18_10_23_a000_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p14_18_10_23_a000_ms.load_data_from_file(
            file_name_to_load="p14/p14_18_10_23_a000/p14_18_10_23_a000_CellDetect.mat",
            variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p14_18_10_23_a000_ms"] = p14_18_10_23_a000_ms

    if "p14_18_10_23_a001_ms" in ms_str_to_load:
        # only interneurons in p14_18_10_23_a001_ms
        p14_18_10_23_a001_ms = MouseSession(age=14, session_id="18_10_23_a001", nb_ms_by_frame=100, param=param,
                                            weight=10.35)
        # calculated with 99th percentile on raster dur
        p14_18_10_23_a001_ms.activity_threshold = 8
        # p14_18_10_23_a001_ms.set_inter_neurons(np.arange(31))
        p14_18_10_23_a001_ms.set_inter_neurons([])
        # duration of those interneurons: 24.33
        variables_mapping = {"spike_nums_dur": "corrected_rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p14_18_10_23_a001_ms.load_data_from_file(file_name_to_load=
                                                 "p14/p14_18_10_23_a001/p14_18_10_23_a001_Corrected_RasterDur.mat",
                                                 variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p14_18_10_23_a001_ms.load_data_from_file(
                file_name_to_load="p14/p14_18_10_23_a001/p14_18_10_23_a001_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p14_18_10_23_a001_ms.load_data_from_file(
            file_name_to_load="p14/p14_18_10_23_a001/p14_18_10_23_a001_CellDetect.mat",
            variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p14_18_10_23_a001_ms"] = p14_18_10_23_a001_ms

    if "p14_18_10_30_a001_ms" in ms_str_to_load:
        p14_18_10_30_a001_ms = MouseSession(age=14, session_id="18_10_30_a001", nb_ms_by_frame=100, param=param,
                                            weight=8.9)
        # calculated with 99th percentile on raster dur
        p14_18_10_30_a001_ms.activity_threshold = 11
        # p14_18_10_30_a001_ms.set_low_activity_threshold(threshold=, percentile_value=1)
        p14_18_10_30_a001_ms.set_inter_neurons([0])
        # duration of those interneurons: 24.33
        variables_mapping = {"spike_nums_dur": "rasterdur",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p14_18_10_30_a001_ms.load_data_from_file(file_name_to_load=
                                                 "p14/p14_18_10_30_a001/p14_18_10_30_a001_RasterDur.mat",
                                                 variables_mapping=variables_mapping)
        if load_traces:
            variables_mapping = {"traces": "C_df"}
            p14_18_10_30_a001_ms.load_data_from_file(
                file_name_to_load="p14/p14_18_10_30_a001/p14_18_10_30_a001_Traces.mat",
                variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p14_18_10_30_a001_ms.load_data_from_file(
            file_name_to_load="p14/p14_18_10_30_a001/p14_18_10_30_a001_CellDetect.mat",
            variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p14_18_10_30_a001_ms"] = p14_18_10_30_a001_ms

    # arnaud_ms = MouseSession(age=24, session_id="arnaud", nb_ms_by_frame=50, param=param)
    # arnaud_ms.activity_threshold = 13
    # arnaud_ms.set_inter_neurons([])
    # variables_mapping = {"spike_nums": "spikenums"}
    # arnaud_ms.load_data_from_file(file_name_to_load="spikenumsarnaud.mat", variables_mapping=variables_mapping)

    if "p60_arnaud_ms" in ms_str_to_load:
        p60_arnaud_ms = MouseSession(age=60, session_id="arnaud_a_529", nb_ms_by_frame=100, param=param)
        p60_arnaud_ms.activity_threshold = 9
        p60_arnaud_ms.set_inter_neurons([])
        # duration of those interneurons:
        variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p60_arnaud_ms.load_data_from_file(file_name_to_load=
                                          "p60/a529/Arnaud_RasterDur.mat",
                                          variables_mapping=variables_mapping)

        # variables_mapping = {"traces": "C_df"}
        # p60_arnaud_ms.load_data_from_file(file_name_to_load="p60/a529/Arnaud_a_529_corr_Traces.mat",
        #                                          variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p60_arnaud_ms.load_data_from_file(file_name_to_load="p60/a529/Arnaud_a_529_corr_CellDetect.mat",
                                          variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p60_arnaud_ms"] = p60_arnaud_ms

    if "p60_a529_2015_02_25_ms" in ms_str_to_load:
        p60_a529_2015_02_25_ms = MouseSession(age=60, session_id="a529_2015_02_25", nb_ms_by_frame=100, param=param)
        p60_a529_2015_02_25_ms.activity_threshold = 10
        p60_a529_2015_02_25_ms.set_inter_neurons([])
        # duration of those interneurons:
        variables_mapping = {"spike_nums_dur": "rasterdur", "traces": "C_df",
                             "spike_nums": "filt_Bin100ms_spikedigital",
                             "spike_durations": "LOC3"}
        p60_a529_2015_02_25_ms.load_data_from_file(file_name_to_load=
                                                   "p60/a529_2015_02_25/a529_2015_02_25_RasterDur.mat",
                                                   variables_mapping=variables_mapping)
        variables_mapping = {"coord": "ContoursAll"}
        p60_a529_2015_02_25_ms.load_data_from_file(
            file_name_to_load="p60/a529_2015_02_25/MotCorre_529_15_02_25_CellDetect.mat",
            variables_mapping=variables_mapping)
        ms_str_to_ms_dict["p60_a529_2015_02_25_ms"] = p60_a529_2015_02_25_ms

    if "p60_a529_2015_02_25_v_arnaud_ms" in ms_str_to_load:
        p60_a529_2015_02_25_v_arnaud_ms = MouseSession(age=60, session_id="a529_2015_02_25_v_arnaud",
                                                       nb_ms_by_frame=100, param=param)
        # p60_a529_2015_02_25_v_arnaud_ms.activity_threshold = 5
        p60_a529_2015_02_25_v_arnaud_ms.set_inter_neurons([])
        # duration of those interneurons:
        variables_mapping = {"traces": "Tr1b",
                             "spike_nums": "Raster"}
        p60_a529_2015_02_25_v_arnaud_ms.load_data_from_file(file_name_to_load=
                                                            "p60/a529_2015_02_25_v_arnaud/a529-20150225_Raster_all_cells.mat.mat",
                                                            variables_mapping=variables_mapping)

        ms_str_to_ms_dict["p60_a529_2015_02_25_v_arnaud_ms"] = p60_a529_2015_02_25_v_arnaud_ms

    return ms_str_to_ms_dict


def main():
    # for line in np.arange(15):
    #     print_surprise_for_michou(n_lines=15, actual_line=line)
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    path_data = root_path + "data/"
    path_results_raw = root_path + "results_hne/"
    cell_assemblies_data_path = path_data + "cell_assemblies/v3/"
    best_order_data_path = path_data + "best_order_data/v2/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results_raw + f"{time_str}"
    os.mkdir(path_results)

    # --------------------------------------------------------------------------------
    # ------------------------------ param section ------------------------------
    # --------------------------------------------------------------------------------

    # param will be set later when the spike_nums will have been constructed
    param = HNEParameters(time_str=time_str, path_results=path_results, error_rate=2,
                          cell_assemblies_data_path=cell_assemblies_data_path,
                          best_order_data_path=best_order_data_path,
                          time_inter_seq=50, min_duration_intra_seq=-3, min_len_seq=10, min_rep_nb=4,
                          max_branches=20, stop_if_twin=False,
                          no_reverse_seq=False, spike_rate_weight=False, path_data=path_data)

    just_compute_significant_seq_stat = False
    if just_compute_significant_seq_stat:
        compute_stat_about_significant_seq(files_path=f"{path_data}significant_seq/v5/", param=param)
        return

    load_traces = True

    available_ms_str = ["p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms",
                        "p7_171012_a000_ms", "p7_18_02_08_a000_ms",
                        "p7_17_10_18_a002_ms", "p7_17_10_18_a004_ms",
                        "p7_18_02_08_a001_ms", "p7_18_02_08_a002_ms",
                        "p7_18_02_08_a003_ms",
                        "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms",
                        "p8_18_10_24_a005_ms", "p8_18_10_17_a001_ms",
                        "p8_18_10_17_a000_ms",  # new
                        "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms",
                        "p9_18_09_27_a003_ms",  # new
                        "p10_17_11_16_a003_ms",
                        "p11_17_11_24_a001_ms", "p11_17_11_24_a000_ms",
                        "p12_17_11_10_a002_ms", "p12_171110_a000_ms",
                        "p13_18_10_29_a000_ms",  # new
                        "p13_18_10_29_a001_ms",
                        "p14_18_10_23_a000_ms",
                        "p14_18_10_30_a001_ms",
                        "p60_arnaud_ms", "p60_a529_2015_02_25_ms"]
    abf_corrupted = ["p8_18_10_17_a001_ms", "p9_18_09_27_a003_ms"]

    ms_with_piezo = ["p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms", "p7_18_02_08_a000_ms",
                     "p7_18_02_08_a001_ms", "p7_18_02_08_a002_ms", "p7_18_02_08_a003_ms", "p8_18_02_09_a000_ms",
                     "p8_18_02_09_a001_ms", "p8_18_10_17_a001_ms",
                     "p8_18_10_24_a005_ms", "p9_18_09_27_a003_ms", "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms"]
    ms_with_run = ["p13_18_10_29_a001_ms", "p13_18_10_29_a000_ms"]
    run_ms_str = ["p12_17_11_10_a000_ms", "p12_17_11_10_a002_ms", "p13_18_10_29_a000_ms",
                  "p13_18_10_29_a001_ms"]
    ms_10000_sampling = ["p8_18_10_17_a001_ms", "p9_18_09_27_a003_ms"]

    oriens_ms_str = ["p14_18_10_23_a001_ms"]

    ms_str_to_load = ["p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms",
                      "p8_18_10_24_a005_ms", "p8_18_10_17_a001_ms",
                      "p8_18_10_17_a000_ms",  # new
                      "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms",
                      "p9_18_09_27_a003_ms",  # new
                      "p10_17_11_16_a003_ms",
                      "p11_17_11_24_a001_ms", "p11_17_11_24_a000_ms",
                      "p12_17_11_10_a002_ms", "p12_171110_a000_ms",
                      "p13_18_10_29_a000_ms",  # new
                      "p13_18_10_29_a001_ms",
                      "p14_18_10_23_a000_ms",
                      "p14_18_10_30_a001_ms",
                      "p60_arnaud_ms"]
    ms_with_cell_assemblies = ["p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms",
                               "p9_18_09_27_a003_ms", "p10_17_11_16_a003_ms",
                               "p11_17_11_24_a000_ms"]
    ms_str_to_load = available_ms_str
    ms_str_to_load = ms_with_run
    # ms_str_to_load = ["p60_a529_2015_02_25_v_arnaud_ms"]
    ms_str_to_load = ["p7_18_02_08_a001_ms"]
    ms_str_to_load = ["p10_17_11_16_a003_ms"]
    ms_str_to_load = available_ms_str
    ms_str_to_load = ["p9_18_09_27_a003_ms", "p10_17_11_16_a003_ms"]
    ms_str_to_load = ms_with_cell_assemblies
    ms_str_to_load = ["p6_18_02_07_a001_ms", ]
    ms_str_to_load = ["p7_18_02_08_a000_ms"]
    ms_str_to_load = ["p6_18_02_07_a001_ms", "p12_17_11_10_a002_ms"]
    ms_str_to_load = ["p60_arnaud_ms"]
    ms_str_to_load = available_ms_str
    ms_str_to_load = ["p60_a529_2015_02_25_ms"]
    ms_str_to_load = ["p6_18_02_07_a002_ms"]
    ms_str_to_load = ["p7_18_02_08_a000_ms"]
    ms_str_to_load = ms_with_piezo
    ms_str_to_load = ["p6_18_02_07_a002_ms"]
    ms_str_to_load = ["p9_18_09_27_a003_ms"]
    ms_str_to_load = ms_with_piezo

    # loading data
    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_str_to_load, param=param,
                                            load_traces=load_traces)

    available_ms = []
    for ms_str in ms_str_to_load:
        available_ms.append(ms_str_to_ms_dict[ms_str])
    # for ms in available_ms:
    #     ms.plot_each_inter_neuron_connect_map()
    #     return
    ms_to_analyse = available_ms

    just_do_stat_on_event_detection_parameters = True
    do_plot_psth_twitches = False
    just_plot_raster = False

    # for events (sce) detection
    perc_threshold = 80
    use_max_of_each_surrogate = False
    n_surrogate_activity_threshold = 1000
    use_raster_dur = True
    no_redundancy = False
    determine_low_activity_by_variation = False

    do_plot_interneurons_connect_maps = False
    do_plot_connect_hist = False
    do_plot_connect_hist_for_all_ages = False
    do_time_graph_correlation = False
    do_time_graph_correlation_and_connect_best = False

    # ##########################################################################################
    # #################################### CLUSTERING ###########################################
    # ##########################################################################################
    do_clustering = False
    # if False, clustering will be done using kmean
    do_fca_clustering = False
    do_clustering_with_twitches_events = False
    with_cells_in_cluster_seq_sorted = False

    # ##### for fca #####
    n_surrogate_fca = 20

    # #### for kmean  #####
    with_shuffling = False
    print(f"use_raster_dur {use_raster_dur}")
    range_n_clusters_k_mean = np.arange(4, 10)
    # range_n_clusters_k_mean = np.array([6])
    n_surrogate_k_mean = 10
    keep_only_the_best_kmean_cluster = False

    # ##########################################################################################
    # ################################ PATTERNS SEARCH #########################################
    # ##########################################################################################
    do_pattern_search = False
    keep_the_longest_seq = False
    split_pattern_search = False
    use_only_uniformity_method = True
    use_loss_score_to_keep_the_best_from_tree = False
    use_sce_times_for_pattern_search = False
    use_ordered_spike_nums_for_surrogate = True
    n_surrogate_for_pattern_search = 100
    # seq params:
    # TODO: error_rate that change with the number of element in the sequence
    param.error_rate = 0.25  # 0.25
    param.max_branches = 10
    param.time_inter_seq = 30  # 50
    param.min_duration_intra_seq = 0
    param.min_len_seq = 5  # 5
    param.min_rep_nb = 3

    debug_mode = False

    # ------------------------------ end param section ------------------------------

    ms_by_age = dict()
    for ms_index, ms in enumerate(ms_to_analyse):
        # ms.plot_time_correlation_graph_over_twitches()
        # if ms_index == len(ms_to_analyse) - 1:
        #     raise Exception("loko")
        # continue
        # ms.plot_raster_with_cells_assemblies_events_and_mvts()
        # if ms_index == len(ms_to_analyse) - 1:
        #     raise Exception("koko")
        # continue

        spike_nums_to_use = ms.spike_struct.spike_nums_dur
        sliding_window_duration = 1
        if ms.activity_threshold is None:
            activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums_to_use,
                                                             window_duration=sliding_window_duration,
                                                             spike_train_mode=False,
                                                             use_max_of_each_surrogate=use_max_of_each_surrogate,
                                                             n_surrogate=n_surrogate_activity_threshold,
                                                             perc_threshold=perc_threshold,
                                                             debug_mode=False)

            ms.spike_struct.activity_threshold = activity_threshold
            # param.activity_threshold = activity_threshold
        else:
            activity_threshold = ms.activity_threshold
            ms.spike_struct.activity_threshold = ms.activity_threshold

        sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_nums_to_use,
                                                              window_duration=sliding_window_duration,
                                                              perc_threshold=perc_threshold,
                                                              activity_threshold=activity_threshold,
                                                              debug_mode=False,
                                                              no_redundancy=no_redundancy)

        print(f"sce_with_sliding_window detected")
        # tuple of times
        SCE_times = sce_detection_result[1]

        # print(f"SCE_times {SCE_times}")
        sce_times_numbers = sce_detection_result[3]
        sce_times_bool = sce_detection_result[0]
        # useful for plotting twitches
        ms.sce_bool = sce_times_bool
        ms.sce_times_numbers = sce_times_numbers
        ms.SCE_times = SCE_times

        # ms.plot_piezo_with_extra_info(show_plot=False, save_formats="pdf")
        # ms.plot_piezo_around_event(range_in_sec=5, save_formats="png")
        # ms.plot_raw_traces_around_twitches()
        ms.plot_psth_over_twitches_time_correlation_graph_style()
        # ms.plot_piezo_with_extra_info(show_plot=True, with_cell_assemblies_sce=False, save_formats="pdf")
        if ms_index == len(ms_to_analyse) - 1:
            raise Exception("lala")
        continue
        #
        # test_seq_detect(ms)
        # raise Exception("toto")

        # ms.plot_cell_assemblies_on_map()
        # raise Exception("toto")
        if ms.age not in ms_by_age:
            ms_by_age[ms.age] = []

        ms_by_age[ms.age].append(ms)

        if do_plot_interneurons_connect_maps or do_plot_connect_hist:
            ms.detect_n_in_n_out()
        elif do_time_graph_correlation_and_connect_best and do_time_graph_correlation:
            ms.detect_n_in_n_out()

        if do_time_graph_correlation:
            spike_struct = ms.spike_struct
            n_cells = ms.spike_struct.n_cells
            sliding_window_duration = 1
            if ms.activity_threshold is None:
                activity_threshold = get_sce_detection_threshold(spike_nums=spike_struct.spike_nums_dur,
                                                                 window_duration=sliding_window_duration,
                                                                 spike_train_mode=False,
                                                                 use_max_of_each_surrogate=use_max_of_each_surrogate,
                                                                 n_surrogate=n_surrogate_activity_threshold,
                                                                 perc_threshold=perc_threshold,
                                                                 debug_mode=False)

                spike_struct.activity_threshold = activity_threshold
                # param.activity_threshold = activity_threshold
            else:
                activity_threshold = ms.activity_threshold
                spike_struct.activity_threshold = ms.activity_threshold

            sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_struct.spike_nums_dur,
                                                                  window_duration=sliding_window_duration,
                                                                  perc_threshold=perc_threshold,
                                                                  activity_threshold=activity_threshold,
                                                                  debug_mode=False,
                                                                  no_redundancy=no_redundancy)

            # tuple of times
            SCE_times = sce_detection_result[1]

            events_peak_times = []
            for sce_time in SCE_times:
                events_peak_times.append(sce_time[0] +
                                         np.argmax(spike_struct.spike_nums_dur[:, sce_time[0]:sce_time[1] + 1]))

            results = get_time_correlation_data(spike_nums=spike_struct.spike_nums,
                                                events_times=SCE_times, time_around_events=5)
            ms.time_lags_list, ms.correlation_list, \
            ms.time_lags_dict, ms.correlation_dict, ms.time_lags_window, cells_list = results

            if do_time_graph_correlation_and_connect_best and ms.coord_obj is not None:
                # keep value with correlation > 95th percentile
                correlation_threshold = np.percentile(ms.correlation_list, 99)
                indices = np.where(np.array(ms.correlation_list) > correlation_threshold)[0]
                hub_cells = np.array(cells_list)[indices]

                # then show their connectivity in and out
                connec_func_stat(mouse_sessions=[ms], data_descr=f"{ms.description} with hub cells",
                                 param=param, show_interneurons=False, cells_to_highlights=[hub_cells],
                                 cells_to_highlights_shape=["o"], cells_to_highlights_colors=["red"],
                                 cells_to_highlights_legend=["hub cells"])

                # and show their connectivty map
                for cell_to_map in hub_cells:
                    ms.plot_connectivity_maps_of_a_cell(cell_to_map=cell_to_map, cell_descr="hub_cell",
                                                        cell_color="red", links_cell_color="cornflowerblue")

        if do_plot_connect_hist:
            connec_func_stat([ms], data_descr=ms.description, param=param)

        if do_plot_interneurons_connect_maps:
            if ms.coord is None:
                continue
            ms.plot_each_inter_neuron_connect_map()

    if do_time_graph_correlation:
        max_value = 0
        for ms_time_graph in ms_to_analyse:
            max_value_ms = np.max((np.abs(np.min(ms_time_graph.time_lags_list)),
                                   np.abs(np.max(ms_time_graph.time_lags_list))))
            max_value = np.max((max_value, max_value_ms))

        time_window_to_include_them_all = (max_value * 1.1) / 2

        for ms in ms_to_analyse:
            cells_groups = []
            groups_colors = []
            spike_struct = ms.spike_struct
            if (spike_struct.inter_neurons is not None) and (len(spike_struct.inter_neurons) > 0):
                cells_groups.append(spike_struct.inter_neurons)
                groups_colors.append("red")

            common_time_window = True
            if common_time_window:
                time_window = time_window_to_include_them_all
            else:
                time_window = ms.time_lags_window
            if do_time_graph_correlation_and_connect_best:
                show_percentiles = [99]
            else:
                show_percentiles = None
            # show_percentiles = [99]
            # first plotting each individual time-correlation graph with the same x-limits
            time_correlation_graph(time_lags_list=ms.time_lags_list,
                                   correlation_list=ms.correlation_list,
                                   time_lags_dict=ms.time_lags_dict,
                                   correlation_dict=ms.correlation_dict,
                                   n_cells=ms.spike_struct.n_cells,
                                   time_window=time_window,
                                   plot_cell_numbers=True,
                                   cells_groups=cells_groups,
                                   groups_colors=groups_colors,
                                   data_id=ms.description,
                                   param=param,
                                   set_x_limit_to_max=True,
                                   time_stamps_by_ms=0.01,
                                   ms_scale=200,
                                   show_percentiles=show_percentiles)

            # normalized version
            # time_lags_list_z_score = (np.array(ms.time_lags_list) - np.mean(ms.time_lags_list)) / \
            #                          np.std(ms.time_lags_list)
            # time_lags_dict_z_score = dict()
            # for cell, time_lag in ms.time_lags_dict.items():
            #     time_lags_dict_z_score[cell] = (time_lag - np.mean(ms.time_lags_list))) / np.std(ms.time_lags_list)

        time_lags_list_by_age = dict()
        correlation_list_by_age = dict()
        time_lags_dict_by_age = dict()
        correlation_dict_by_age = dict()
        time_lags_window_by_age = dict()

        for age, ms_this_age in ms_by_age.items():
            cells_so_far = 0
            time_lags_list_by_age[age] = []
            correlation_list_by_age[age] = []
            time_lags_dict_by_age[age] = dict()
            correlation_dict_by_age[age] = dict()
            cells_groups = [[]]
            groups_colors = ["red"]
            for ms in ms_this_age:
                time_lags_list_by_age[age].extend(ms.time_lags_list)
                correlation_list_by_age[age].extend(ms.correlation_list)
                spike_struct = ms.spike_struct
                if (spike_struct.inter_neurons is not None) and (len(spike_struct.inter_neurons) > 0):
                    cells_groups[0].extend(list(np.array(spike_struct.inter_neurons) + cells_so_far))

                for cell in ms.time_lags_dict.keys():
                    time_lags_dict_by_age[age][cell + cells_so_far] = ms.time_lags_dict[cell]
                    correlation_dict_by_age[age][cell + cells_so_far] = ms.correlation_dict[cell]
                cells_so_far += len(ms.time_lags_dict)

            time_correlation_graph(time_lags_list=time_lags_list_by_age[age],
                                   correlation_list=correlation_list_by_age[age],
                                   time_lags_dict=time_lags_dict_by_age[age],
                                   correlation_dict=correlation_dict_by_age[age],
                                   n_cells=ms.spike_struct.n_cells,
                                   time_window=time_window_to_include_them_all,
                                   plot_cell_numbers=True,
                                   cells_groups=cells_groups,
                                   groups_colors=groups_colors,
                                   data_id=f"p{age}",
                                   param=param,
                                   set_x_limit_to_max=True,
                                   time_stamps_by_ms=0.01,
                                   ms_scale=200)

    if do_plot_connect_hist_for_all_ages:
        n_ins_by_age = dict()
        n_outs_by_age = dict()
        for age, ms_list in ms_by_age.items():
            n_ins_by_age[age], n_outs_by_age[age] = connec_func_stat(ms_list, data_descr=f"p{age}", param=param)
        box_plot_data_by_age(data_dict=n_ins_by_age,
                             title="Connectivity in by age",
                             filename="box_plots_connectivity_in_by_age",
                             y_label="Active cells (%)",
                             param=param)
        box_plot_data_by_age(data_dict=n_outs_by_age,
                             title="Connectivity out by age",
                             filename="box_plots_connectivity_out_by_age",
                             y_label="Active cells (%)",
                             param=param)

    if just_do_stat_on_event_detection_parameters:
        # keep the value for each ms
        duration_values_list = []
        max_activity_values_list = []
        mean_activity_values_list = []
        overall_activity_values_list = []
        ms_ages = []
        ratio_spikes_events_by_age = dict()
        interneurons_indices_by_age = dict()
        duration_spikes_by_age = dict()
        ratio_spikes_total_events_by_age = dict()

        for ms in available_ms:
            ms_ages.append(ms.age)
            t = 1
            # for each session, we want to detect SCE using 6 methods:
            # 1) Keep the max of each surrogate and take the 95th percentile
            # 2) Keep the 95th percentile of n_times * n°surrogates events activity
            # 3) Keep the 99th percentile of n_times * n°surrogates events activity
            #  and for each with raster_dur and with_onsets for n surrogates

            spike_struct = ms.spike_struct
            n_cells = len(spike_struct.spike_nums)
            if ms.spike_struct.spike_nums_dur is not None:
                use_raster_durs = [True]
            else:
                use_raster_durs = [False]
            # 0 : 99th percentile, 1 & 2: 95th percentile
            selection_options = [0]
            for use_raster_dur in use_raster_durs:
                for selection_option in selection_options:
                    if use_raster_dur:
                        sliding_window_duration = 1
                        spike_nums_to_use = spike_struct.spike_nums_dur
                    else:
                        sliding_window_duration = 5
                        spike_nums_to_use = spike_struct.spike_nums

                    # ######  parameters setting #########
                    # data_descr = f"{ms.description}"
                    n_surrogate_activity_threshold = 1000

                    if selection_option > 0:
                        perc_threshold = 95
                    else:
                        perc_threshold = 99

                    if selection_option == 2:
                        use_max_of_each_surrogate = True
                    else:
                        use_max_of_each_surrogate = False

                    ###############  TEST    ##################
                    # if not determine_low_activity_by_variation:
                    #     perc_low_activity_threshold = 5
                    #     if perc_low_activity_threshold not in ms.low_activity_threshold_by_percentile:
                    #         low_activity_events_thsld = get_low_activity_events_detection_threshold(
                    #             spike_nums=spike_nums_to_use,
                    #             window_duration=sliding_window_duration,
                    #             spike_train_mode=False,
                    #             use_min_of_each_surrogate=False,
                    #             n_surrogate=n_surrogate_activity_threshold,
                    #             perc_threshold=perc_low_activity_threshold,
                    #             debug_mode=False)
                    #         print(f"ms {ms.description}")
                    #         print(f"low_activity_events_thsld {low_activity_events_thsld}, "
                    #               f"{np.round((low_activity_events_thsld/n_cells), 3)}%")
                    #         continue
                    # else:
                    #     pass
                    # ########### END TEST ###########

                    if ms.activity_threshold is None:
                        activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums_to_use,
                                                                         window_duration=sliding_window_duration,
                                                                         spike_train_mode=False,
                                                                         use_max_of_each_surrogate=use_max_of_each_surrogate,
                                                                         n_surrogate=n_surrogate_activity_threshold,
                                                                         perc_threshold=perc_threshold,
                                                                         debug_mode=False)

                        spike_struct.activity_threshold = activity_threshold
                        # param.activity_threshold = activity_threshold
                    else:
                        activity_threshold = ms.activity_threshold
                        spike_struct.activity_threshold = ms.activity_threshold

                    sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_nums_to_use,
                                                                          window_duration=sliding_window_duration,
                                                                          perc_threshold=perc_threshold,
                                                                          activity_threshold=activity_threshold,
                                                                          debug_mode=False,
                                                                          no_redundancy=no_redundancy)

                    print(f"sce_with_sliding_window detected")
                    # tuple of times
                    SCE_times = sce_detection_result[1]

                    # print(f"SCE_times {SCE_times}")
                    sce_times_numbers = sce_detection_result[3]
                    sce_times_bool = sce_detection_result[0]
                    # useful for plotting twitches
                    ms.sce_bool = sce_times_bool
                    ms.sce_times_numbers = sce_times_numbers
                    ms.SCE_times = SCE_times

                    print(f"ms {ms.description}, {len(SCE_times)} sce "
                          f"activity threshold {activity_threshold}, "
                          f"use_raster_dur {use_raster_dur},  "
                          f"{perc_threshold} percentile, use_max_of_each_surrogate {use_max_of_each_surrogate}"
                          f", np.shape(spike_nums_to_use) {np.shape(spike_nums_to_use)}")

                    raster_option = "raster_dur" if use_raster_dur else "onsets"
                    technique_details_file = "max_each_surrogate" if use_max_of_each_surrogate else ""
                    file_name = f'{ms.description}_stat_{raster_option}_{perc_threshold}_perc_' \
                                f'{technique_details_file}'

                    spike_shape = '|' if use_raster_dur else 'o'
                    inter_neurons = ms.spike_struct.inter_neurons
                    span_area_coords = [SCE_times]
                    span_area_colors = ['lightgrey']
                    if ms.mvt_frames_periods is not None:
                        if (not ms.with_run) and (ms.twitches_frames_periods is not None):
                            span_area_coords.append(ms.twitches_frames_periods)
                            # span_area_coords.append(ms.mvt_frames_periods)
                            span_area_colors.append("red")
                        elif ms.with_run:
                            span_area_coords.append(ms.mvt_frames_periods)
                            # span_area_coords.append(ms.mvt_frames_periods)
                            span_area_colors.append("red")
                    if do_plot_psth_twitches:
                        line_mode = True
                        ms.plot_psth_twitches(line_mode=line_mode)
                        ms.plot_psth_twitches(twitches_group=1, line_mode=line_mode)
                        ms.plot_psth_twitches(twitches_group=2, line_mode=line_mode)
                        ms.plot_psth_twitches(twitches_group=3, line_mode=line_mode)
                        ms.plot_psth_twitches(twitches_group=4, line_mode=line_mode)
                    plot_spikes_raster(spike_nums=spike_nums_to_use, param=ms.param,
                                       span_cells_to_highlight=inter_neurons,
                                       span_cells_to_highlight_colors=["red"] * len(inter_neurons),
                                       spike_train_format=False,
                                       title=file_name,
                                       file_name=file_name,
                                       y_ticks_labels=np.arange(n_cells),
                                       y_ticks_labels_size=2,
                                       save_raster=True,
                                       show_raster=False,
                                       plot_with_amplitude=False,
                                       activity_threshold=spike_struct.activity_threshold,
                                       # 500 ms window
                                       sliding_window_duration=sliding_window_duration,
                                       show_sum_spikes_as_percentage=True,
                                       # vertical_lines=SCE_times,
                                       # vertical_lines_colors=['white'] * len(SCE_times),
                                       # vertical_lines_sytle="solid",
                                       # vertical_lines_linewidth=[0.2] * len(SCE_times),
                                       span_area_coords=span_area_coords,
                                       span_area_colors=span_area_colors,
                                       span_area_only_on_raster=False,
                                       spike_shape=spike_shape,
                                       spike_shape_size=0.5,
                                       save_formats="pdf")
                    if just_plot_raster:
                        continue
                    plot_psth_interneurons_events(ms=ms, spike_nums_dur=ms.spike_struct.spike_nums_dur,
                                                  spike_nums=ms.spike_struct.spike_nums,
                                                  SCE_times=SCE_times, sliding_window_duration=sliding_window_duration,
                                                  param=param)

                    # return an array of size n_cells, with each value the ratio as a float (percentage)
                    ratio_spikes_events = get_ratio_spikes_on_events_vs_total_spikes_by_cell(
                        spike_nums=spike_struct.spike_nums,
                        spike_nums_dur=spike_struct.spike_nums_dur,
                        sce_times_numbers=sce_times_numbers)

                    ratio_spikes_total_events = get_ratio_spikes_on_events_vs_total_events_by_cell(
                        spike_nums=spike_struct.spike_nums,
                        spike_nums_dur=spike_struct.spike_nums_dur,
                        sce_times_numbers=sce_times_numbers)
                    if ms.age not in ratio_spikes_events_by_age:
                        ratio_spikes_events_by_age[ms.age] = list(ratio_spikes_events)
                        interneurons_indices_by_age[ms.age] = list(ms.spike_struct.inter_neurons)
                        all_spike_duration = []
                        if ms.spike_struct.spike_durations is not None:
                            for spike_duration in ms.spike_struct.spike_durations:
                                all_spike_duration.extend(spike_duration)
                            duration_spikes_by_age[ms.age] = all_spike_duration
                    else:
                        nb_elem = len(ratio_spikes_events_by_age[ms.age])
                        ratio_spikes_events_by_age[ms.age].extend(list(ratio_spikes_events))
                        interneurons_indices_by_age[ms.age].extend(list(ms.spike_struct.inter_neurons +
                                                                        nb_elem))
                        all_spike_duration = []
                        if ms.spike_struct.spike_durations is not None:
                            for spike_duration in ms.spike_struct.spike_durations:
                                all_spike_duration.extend(spike_duration)
                            duration_spikes_by_age[ms.age].extend(all_spike_duration)

                    if ms.age not in ratio_spikes_total_events_by_age:
                        ratio_spikes_total_events_by_age[ms.age] = list(ratio_spikes_total_events)
                    else:
                        ratio_spikes_total_events_by_age[ms.age].extend(list(ratio_spikes_total_events))

                    res = save_stat_sce_detection_methods(spike_nums_to_use=spike_nums_to_use,
                                                          activity_threshold=activity_threshold,
                                                          ms=ms,
                                                          ratio_spikes_events=ratio_spikes_events,
                                                          ratio_spikes_total_events=ratio_spikes_total_events,
                                                          SCE_times=SCE_times, param=param,
                                                          sliding_window_duration=sliding_window_duration,
                                                          perc_threshold=perc_threshold,
                                                          use_raster_dur=use_raster_dur,
                                                          keep_max_each_surrogate=use_max_of_each_surrogate,
                                                          n_surrogate_activity_threshold=n_surrogate_activity_threshold)

                    duration_values_list.append(res[0])
                    max_activity_values_list.append(res[1])
                    mean_activity_values_list.append(res[2])
                    overall_activity_values_list.append(res[3])

                    values_to_scatter = []
                    non_inter_neurons = np.setdiff1d(np.arange(len(ratio_spikes_events)), inter_neurons)
                    ratio_interneurons = list(ratio_spikes_events[inter_neurons])
                    ratio_non_interneurons = list(ratio_spikes_events[non_inter_neurons])
                    labels = []
                    scatter_shapes = []
                    colors = []
                    if len(ratio_non_interneurons) > 0:
                        values_to_scatter.append(np.mean(ratio_non_interneurons))
                        values_to_scatter.append(np.median(ratio_non_interneurons))
                        labels.extend(["mean", "median"])
                        scatter_shapes.extend(["o", "s"])
                        colors.extend(["white", "white"])
                    if len(ratio_interneurons) > 0:
                        values_to_scatter.append(np.mean(ratio_interneurons))
                        values_to_scatter.append(np.median(ratio_interneurons))
                        values_to_scatter.extend(ratio_interneurons)
                        labels.extend(["mean", "median", f"interneuron (x{len(inter_neurons)})"])
                        scatter_shapes.extend(["o", "s"])
                        scatter_shapes.extend(["*"] * len(inter_neurons))
                        colors.extend(["red", "red"])
                        colors.extend(["red"] * len(inter_neurons))

                    plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spikes_events,
                                                  description=f"{ms.description}_hist_spike_events_ratio",
                                                  values_to_scatter=np.array(values_to_scatter),
                                                  labels=labels,
                                                  scatter_shapes=scatter_shapes,
                                                  colors=colors,
                                                  xlabel="spikes in event vs total spikes (%)",
                                                  param=param)

                    values_to_scatter = []
                    ratio_interneurons = list(ratio_spikes_total_events[inter_neurons])
                    ratio_non_interneurons = list(ratio_spikes_total_events[non_inter_neurons])
                    if len(ratio_non_interneurons) > 0:
                        values_to_scatter.append(np.mean(ratio_non_interneurons))
                        values_to_scatter.append(np.median(ratio_non_interneurons))
                    if len(ratio_interneurons) > 0:
                        values_to_scatter.append(np.mean(ratio_interneurons))
                        values_to_scatter.append(np.median(ratio_interneurons))
                        values_to_scatter.extend(ratio_interneurons)
                    plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spikes_total_events,
                                                  description=f"{ms.description}_hist_spike_total_events_ratio",
                                                  values_to_scatter=np.array(values_to_scatter),
                                                  labels=labels,
                                                  scatter_shapes=scatter_shapes,
                                                  colors=colors,
                                                  xlabel="spikes in event vs total events (%)",
                                                  param=param)
        if do_plot_psth_twitches:
            for age, ms_of_this_age in ms_by_age.items():
                ms_of_this_age[0].plot_psth_twitches(line_mode=line_mode, with_other_ms=ms_of_this_age[1:])
                ms_of_this_age[0].plot_psth_twitches(twitches_group=1,
                                                     line_mode=line_mode, with_other_ms=ms_of_this_age[1:])
                ms_of_this_age[0].plot_psth_twitches(twitches_group=2,
                                                     line_mode=line_mode, with_other_ms=ms_of_this_age[1:])
                ms_of_this_age[0].plot_psth_twitches(twitches_group=3,
                                                     line_mode=line_mode, with_other_ms=ms_of_this_age[1:])
                ms_of_this_age[0].plot_psth_twitches(twitches_group=4,
                                                     line_mode=line_mode, with_other_ms=ms_of_this_age[1:])

        ratio_spikes_events_non_interneurons_by_age = dict()
        ratio_spikes_events_interneurons_by_age = dict()
        for age, ratio_spikes in ratio_spikes_events_by_age.items():
            inter_neurons = np.array(interneurons_indices_by_age[age]).astype(int)
            non_inter_neurons = np.setdiff1d(np.arange(len(ratio_spikes)), inter_neurons)
            ratio_spikes = np.array(ratio_spikes)
            values_to_scatter = []
            ratio_interneurons = list(ratio_spikes[inter_neurons])
            ratio_spikes_events_interneurons_by_age[age] = ratio_interneurons
            ratio_non_interneurons = list(ratio_spikes[non_inter_neurons])
            ratio_spikes_events_non_interneurons_by_age[age] = ratio_non_interneurons
            labels = []
            scatter_shapes = []
            colors = []
            if len(ratio_non_interneurons) > 0:
                values_to_scatter.append(np.mean(ratio_non_interneurons))
                values_to_scatter.append(np.median(ratio_non_interneurons))
                labels.extend(["mean", "median"])
                scatter_shapes.extend(["o", "s"])
                colors.extend(["white", "white"])
            if len(ratio_interneurons) > 0:
                values_to_scatter.append(np.mean(ratio_interneurons))
                values_to_scatter.append(np.median(ratio_interneurons))
                values_to_scatter.extend(ratio_interneurons)
                labels.extend(["mean", "median", f"interneuron (x{len(inter_neurons)})"])
                scatter_shapes.extend(["o", "s"])
                scatter_shapes.extend(["*"] * len(inter_neurons))
                colors.extend(["red", "red"])
                colors.extend(["red"] * len(inter_neurons))
            plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spikes,
                                          description=f"p{age}_hist_spike_events_ratio",
                                          values_to_scatter=np.array(values_to_scatter),
                                          labels=labels,
                                          scatter_shapes=scatter_shapes,
                                          colors=colors,
                                          xlabel="spikes in event vs total spikes (%)",
                                          param=param)

        ratio_spikes_total_events_non_interneurons_by_age = dict()
        ratio_spikes_total_events_interneurons_by_age = dict()
        for age, ratio_spikes in ratio_spikes_total_events_by_age.items():
            inter_neurons = np.array(interneurons_indices_by_age[age]).astype(int)
            non_inter_neurons = np.setdiff1d(np.arange(len(ratio_spikes)), inter_neurons)
            ratio_spikes = np.array(ratio_spikes)
            values_to_scatter = []
            ratio_interneurons = list(ratio_spikes[inter_neurons])
            ratio_spikes_total_events_interneurons_by_age[age] = ratio_interneurons
            ratio_non_interneurons = list(ratio_spikes[non_inter_neurons])
            ratio_spikes_total_events_non_interneurons_by_age[age] = ratio_non_interneurons
            labels = []
            scatter_shapes = []
            colors = []
            if len(ratio_non_interneurons) > 0:
                values_to_scatter.append(np.mean(ratio_non_interneurons))
                values_to_scatter.append(np.median(ratio_non_interneurons))
                labels.extend(["mean", "median"])
                scatter_shapes.extend(["o", "s"])
                colors.extend(["white", "white"])
            if len(ratio_interneurons) > 0:
                values_to_scatter.append(np.mean(ratio_interneurons))
                values_to_scatter.append(np.median(ratio_interneurons))
                values_to_scatter.extend(ratio_interneurons)
                labels.extend(["mean", "median", f"interneuron (x{len(inter_neurons)})"])
                scatter_shapes.extend(["o", "s"])
                scatter_shapes.extend(["*"] * len(inter_neurons))
                colors.extend(["red", "red"])
                colors.extend(["red"] * len(inter_neurons))
            plot_hist_ratio_spikes_events(ratio_spikes_events=ratio_spikes,
                                          description=f"p{age}_hist_spike_total_events_ratio",
                                          values_to_scatter=np.array(values_to_scatter),
                                          labels=labels,
                                          scatter_shapes=scatter_shapes,
                                          colors=colors,
                                          xlabel="spikes in event vs total events (%)",
                                          param=param)

        # plotting boxplots
        box_plot_data_by_age(data_dict=ratio_spikes_events_by_age,
                             title="Ratio spikes on events / total spikes for all cells",
                             filename="box_plots_ratio_spikes_events_total_spikes_by_age_all_cells",
                             y_label="",
                             param=param)
        box_plot_data_by_age(data_dict=ratio_spikes_events_non_interneurons_by_age,
                             title="Ratio spikes on events / total spikes for pyramidal cells",
                             filename="box_plots_ratio_spikes_events_total_spikes_by_age_pyramidal_cells",
                             y_label="",
                             param=param)
        box_plot_data_by_age(data_dict=ratio_spikes_events_interneurons_by_age,
                             title="Ratio spikes on events / total spikes for interneurons",
                             filename="box_plots_ratio_spikes_events_total_spikes_by_age_interneurons",
                             y_label="",
                             param=param)

        box_plot_data_by_age(data_dict=ratio_spikes_total_events_by_age,
                             title="Ratio spikes on events / total events for all cells",
                             filename="box_plots_ratio_spikes_events_total_events_by_age_all_cells",
                             y_label="",
                             param=param)
        box_plot_data_by_age(data_dict=ratio_spikes_total_events_non_interneurons_by_age,
                             title="Ratio spikes on events / total events for pyramidal cells",
                             filename="box_plots_ratio_spikes_events_total_events_by_age_pyramidal_cells",
                             y_label="",
                             param=param)
        box_plot_data_by_age(data_dict=ratio_spikes_total_events_interneurons_by_age,
                             title="Ratio spikes on events / total events for interneurons",
                             filename="box_plots_ratio_spikes_events_total_events_by_age_interneurons",
                             y_label="",
                             param=param)

        save_stat_by_age(ratio_spikes_events_by_age=ratio_spikes_events_by_age,
                         ratio_spikes_total_events_by_age=ratio_spikes_total_events_by_age,
                         interneurons_indices_by_age=interneurons_indices_by_age,
                         mouse_sessions=available_ms,
                         param=param)

        plot_activity_duration_vs_age(mouse_sessions=available_ms, ms_ages=ms_ages,
                                      duration_values_list=duration_values_list,
                                      max_activity_values_list=max_activity_values_list,
                                      mean_activity_values_list=mean_activity_values_list,
                                      overall_activity_values_list=overall_activity_values_list,
                                      param=param)

        plot_duration_spikes_by_age(mouse_sessions=available_ms, ms_ages=ms_ages,
                                    duration_spikes_by_age=duration_spikes_by_age, param=param)
        return

    if (not do_pattern_search) and (not do_clustering):
        return

    for ms in ms_to_analyse:
        ###################################################################
        ###################################################################
        # ###########    SCE detection and clustering        ##############
        ###################################################################
        ###################################################################

        spike_struct = ms.spike_struct
        n_cells = len(spike_struct.spike_nums)
        # spike_struct.build_spike_trains()

        # ######  parameters setting #########
        data_descr = f"{ms.description}"
        print(f"ms: {data_descr}")

        if do_fca_clustering:
            sliding_window_duration = 5
            spike_nums_to_use = spike_struct.spike_nums
            # sigma is the std of the random distribution used to jitter the data
            sigma = 20
        else:
            if use_raster_dur:
                sliding_window_duration = 1
                spike_nums_to_use = spike_struct.spike_nums_dur
                data_descr += "_raster_dur"
            else:
                sliding_window_duration = 5
                spike_nums_to_use = spike_struct.spike_nums

        # ######  end parameters setting #########
        if ms.activity_threshold is None:
            activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums_to_use,
                                                             window_duration=sliding_window_duration,
                                                             spike_train_mode=False,
                                                             n_surrogate=n_surrogate_activity_threshold,
                                                             perc_threshold=perc_threshold,
                                                             use_max_of_each_surrogate=use_max_of_each_surrogate,
                                                             debug_mode=True)
        else:
            activity_threshold = ms.activity_threshold

        print(f"perc_threshold {perc_threshold}, "
              f"activity_threshold {activity_threshold}, {np.round((activity_threshold/n_cells)*100, 2)}%")
        print(f"sliding_window_duration {sliding_window_duration}")
        spike_struct.activity_threshold = activity_threshold
        # param.activity_threshold = activity_threshold

        print("plot_spikes_raster")

        plot_spikes_raster(spike_nums=spike_nums_to_use, param=ms.param,
                           spike_train_format=False,
                           title=f"raster plot {data_descr}",
                           file_name=f"spike_nums_{data_descr}",
                           y_ticks_labels=spike_struct.labels,
                           y_ticks_labels_size=4,
                           save_raster=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           activity_threshold=spike_struct.activity_threshold,
                           # 500 ms window
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           spike_shape="|",
                           spike_shape_size=1,
                           save_formats="pdf")

        # TODO: detect_sce_with_sliding_window with spike_trains
        sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_nums_to_use,
                                                              window_duration=sliding_window_duration,
                                                              perc_threshold=perc_threshold,
                                                              activity_threshold=activity_threshold,
                                                              debug_mode=False,
                                                              no_redundancy=no_redundancy,
                                                              keep_only_the_peak=True)
        print(f"sce_with_sliding_window detected")
        cellsinpeak = sce_detection_result[2]
        SCE_times = sce_detection_result[1]
        sce_times_bool = sce_detection_result[0]
        sce_times_numbers = sce_detection_result[3]
        # useful for plotting twitches
        ms.sce_bool = sce_times_bool
        ms.sce_times_numbers = sce_times_numbers
        ms.SCE_times = SCE_times

        print(f"Nb SCE: {cellsinpeak.shape}")
        # print(f"Nb spikes by SCE: {np.sum(cellsinpeak, axis=0)}")
        display_isi_info = False
        if display_isi_info:
            cells_isi = tools_misc.get_isi(spike_data=spike_struct.spike_nums, spike_trains_format=False)
            for cell_index in np.arange(len(spike_struct.spike_nums)):
                print(f"{spike_struct.labels[cell_index]} median isi: {np.round(np.median(cells_isi[cell_index]), 2)}, "
                      f"mean isi {np.round(np.mean(cells_isi[cell_index]), 2)}")

        # return a dict of list of list of neurons, representing the best clusters
        # (as many as nth_best_clusters).
        # the key is the K from the k-mean
        if do_clustering:
            if do_fca_clustering:
                compute_and_plot_clusters_raster_fca_version(spike_trains=spike_struct.spike_trains,
                                                             spike_nums=spike_struct.spike_nums,
                                                             data_descr=data_descr, param=param,
                                                             sliding_window_duration=sliding_window_duration,
                                                             SCE_times=SCE_times, sce_times_numbers=sce_times_numbers,
                                                             perc_threshold=perc_threshold,
                                                             n_surrogate_activity_threshold=
                                                             n_surrogate_activity_threshold,
                                                             sigma=sigma, n_surrogate_fca=n_surrogate_fca,
                                                             labels=spike_struct.labels,
                                                             activity_threshold=activity_threshold,
                                                             fca_early_stop=True,
                                                             with_cells_in_cluster_seq_sorted=
                                                             with_cells_in_cluster_seq_sorted,
                                                             use_uniform_jittering=True)
            else:
                if do_clustering_with_twitches_events:
                    n_times = len(sce_times_numbers)
                    ms.define_twitches_events()
                    for twitch_group in [9]:  # 1, 3, 4, 5, 6, 7, 8
                        twitches_times = ms.events_by_twitches_group[twitch_group]
                        cellsinpeak = np.zeros((n_cells, len(twitches_times)), dtype="int16")
                        for twitch_index, twitch_period in enumerate(twitches_times):
                            cellsinpeak[:, twitch_index] = np.sum(
                                spike_nums_to_use[:, twitch_period[0]:twitch_period[1] + 1], axis=1)
                            cellsinpeak[cellsinpeak[:, twitch_index] > 0, twitch_index] = 1

                        twitches_times_numbers = np.ones(n_times, dtype="int16")
                        twitches_times_numbers *= -1
                        for twitch_index, twitch_period in enumerate(twitches_times):
                            twitches_times_numbers[twitch_period[0]:twitch_period[1] + 1] = twitch_index

                        twitches_times_bool = np.zeros(n_times, dtype="bool")
                        for twitch_index, twitch_period in enumerate(twitches_times):
                            twitches_times_bool[twitch_period[0]:twitch_period[1] + 1] = True
                        descr_twitch = ""
                        descr_twitch += data_descr
                        descr_twitch += "_" + ms.twitches_group_title[twitch_group]

                        print(f"twitch_group {twitch_group}: {len(twitches_times)}")
                        if len(twitches_times) < 10:
                            continue
                        print(f"")
                        range_n_clusters_k_mean = np.arange(2, np.min((len(twitches_times) // 2, 10)))

                        compute_and_plot_clusters_raster_kmean_version(labels=ms.spike_struct.labels,
                                                                       activity_threshold=
                                                                       ms.spike_struct.activity_threshold,
                                                                       range_n_clusters_k_mean=range_n_clusters_k_mean,
                                                                       n_surrogate_k_mean=n_surrogate_k_mean,
                                                                       with_shuffling=with_shuffling,
                                                                       spike_nums_to_use=spike_nums_to_use,
                                                                       cellsinpeak=cellsinpeak,
                                                                       data_descr=descr_twitch,
                                                                       param=ms.param,
                                                                       sliding_window_duration=sliding_window_duration,
                                                                       SCE_times=twitches_times,
                                                                       sce_times_numbers=twitches_times_numbers,
                                                                       sce_times_bool=twitches_times_bool,
                                                                       perc_threshold=perc_threshold,
                                                                       n_surrogate_activity_threshold=
                                                                       n_surrogate_activity_threshold,
                                                                       debug_mode=debug_mode,
                                                                       fct_to_keep_best_silhouettes=np.median,
                                                                       with_cells_in_cluster_seq_sorted=
                                                                       with_cells_in_cluster_seq_sorted,
                                                                       keep_only_the_best=
                                                                       keep_only_the_best_kmean_cluster)
                else:
                    compute_and_plot_clusters_raster_kmean_version(labels=ms.spike_struct.labels,
                                                                   activity_threshold=ms.spike_struct.activity_threshold,
                                                                   range_n_clusters_k_mean=range_n_clusters_k_mean,
                                                                   n_surrogate_k_mean=n_surrogate_k_mean,
                                                                   with_shuffling=with_shuffling,
                                                                   spike_nums_to_use=spike_nums_to_use,
                                                                   cellsinpeak=cellsinpeak,
                                                                   data_descr=data_descr,
                                                                   param=ms.param,
                                                                   sliding_window_duration=sliding_window_duration,
                                                                   SCE_times=SCE_times,
                                                                   sce_times_numbers=sce_times_numbers,
                                                                   sce_times_bool=sce_times_bool,
                                                                   perc_threshold=perc_threshold,
                                                                   n_surrogate_activity_threshold=
                                                                   n_surrogate_activity_threshold,
                                                                   debug_mode=debug_mode,
                                                                   fct_to_keep_best_silhouettes=np.median,
                                                                   with_cells_in_cluster_seq_sorted=
                                                                   with_cells_in_cluster_seq_sorted,
                                                                   keep_only_the_best=keep_only_the_best_kmean_cluster)

        ###################################################################
        ###################################################################
        # ##############    Sequences detection        ###################
        ###################################################################
        ###################################################################

        if do_pattern_search:
            sce_times_bool_to_use = sce_times_bool if use_sce_times_for_pattern_search else None
            if split_pattern_search:
                n_splits = 5
                splits_indices = np.linspace(0, len(spike_nums_to_use[0, :]), n_splits + 1).astype(int)
                for split_id in np.arange(n_splits):
                    find_significant_patterns(spike_nums=
                                              spike_nums_to_use[:,
                                              splits_indices[split_id]:splits_indices[split_id + 1]],
                                              param=param,
                                              activity_threshold=activity_threshold,
                                              sliding_window_duration=sliding_window_duration,
                                              n_surrogate=n_surrogate_for_pattern_search,
                                              use_ordered_spike_nums_for_surrogate=use_ordered_spike_nums_for_surrogate,
                                              data_id=ms.description, debug_mode=False,
                                              extra_file_name=f"part_{split_id+1}",
                                              sce_times_bool=sce_times_bool_to_use,
                                              use_only_uniformity_method=use_only_uniformity_method,
                                              use_loss_score_to_keep_the_best_from_tree=
                                              use_loss_score_to_keep_the_best_from_tree,
                                              spike_shape="|",
                                              spike_shape_size=10
                                              )

            else:
                print("Start of use_new_pattern_package")
                find_significant_patterns(spike_nums=spike_nums_to_use, param=param,
                                          activity_threshold=activity_threshold,
                                          sliding_window_duration=sliding_window_duration,
                                          n_surrogate=n_surrogate_for_pattern_search,
                                          data_id=ms.description, debug_mode=False,
                                          use_ordered_spike_nums_for_surrogate=use_ordered_spike_nums_for_surrogate,
                                          extra_file_name="",
                                          sce_times_bool=sce_times_bool_to_use,
                                          use_only_uniformity_method=use_only_uniformity_method,
                                          use_loss_score_to_keep_the_best_from_tree=
                                          use_loss_score_to_keep_the_best_from_tree,
                                          spike_shape="|",
                                          spike_shape_size=5,
                                          keep_the_longest_seq=keep_the_longest_seq)

    return


main()
