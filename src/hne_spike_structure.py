import pandas as pd
# from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from bisect import bisect
from scipy.signal import find_peaks
from scipy import signal
# important to avoid a bug when using virtualenv
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
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
