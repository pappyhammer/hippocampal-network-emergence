import pandas as pd
# from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.cm as cm
import scipy.io as sio
import scipy.stats as scipy_stats
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
# import seqnmf
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
    get_low_activity_events_detection_threshold, detect_sce_potatoes_style
from sortedcontainers import SortedList, SortedDict
from pattern_discovery.clustering.kmean_version.k_mean_clustering import compute_and_plot_clusters_raster_kmean_version
from pattern_discovery.clustering.kmean_version.k_mean_clustering import give_stat_one_sce
from pattern_discovery.clustering.fca.fca import compute_and_plot_clusters_raster_fca_version
from pattern_discovery.graph.force_directed_graphs import plot_graph_using_fa2
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import time
from scipy import stats
from mouse_session import MouseSession
from hne_parameters import HNEParameters
from hne_spike_structure import HNESpikeStructure
from mouse_session_loader import load_mouse_sessions
import networkx as nxfrom
from articifical_movie_patch_generator import produce_movie
import articifical_movie_patch_generator as art_movie_gen
from ScanImageTiffReader import ScanImageTiffReader
from pattern_discovery.tools.signal import smooth_convolve
import PIL
from PIL import ImageSequence


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
                                  colors=colors, twice_more_bins=True,
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
                                  colors=colors, twice_more_bins=True,
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
                                  twice_more_bins=False,
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
    if twice_more_bins:
        bins *= 2
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
    if tight_x_range:
        plt.xlim(min_range, max_range)
    else:
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


def correlate_global_roi_and_shift(path_data, param):
    if not os.path.isdir(path_data):
        return

    # each key corresponds to the name of directory containing the files
    data_dict = {}
    i = 0
    for (dirpath, dirnames, local_filenames) in os.walk(path_data):
        # if dirpath[-1] == "/":
        #     dirpath = dirpath[:-1]
        # print(f"dirpath {dirpath}")
        # print(f"dirnames {dirnames}")
        dirnames_to_walk = []
        for dirname in dirnames:
            if len(dirname) < 4 and (dirname[0].lower() == "p"):
                dirnames_to_walk.append(dirname)
        # print(f"dirnames_to_walk {dirnames_to_walk}")
        for dirname_to_walk in dirnames_to_walk:
            for (dir_path, dir_names, local_filenames) in os.walk(os.path.join(dirpath, dirname_to_walk)):
                parent_dir = os.path.split(dir_path)[1]
                if parent_dir[0].lower() != "p" or (len(parent_dir) < 4):
                    continue
                # looking for files only in dir starting by p and with name length superior to 3
                # print(f"parent_dir {parent_dir}")
                try:
                    index_ = parent_dir.index("_")
                    if (index_ < 2) or (index_ > 3):
                        continue
                        # print(f"directory name not in the right format: {dir_name}")
                    age = int(parent_dir[1:index_])
                except ValueError:
                    continue

                data_dict[parent_dir] = dict()
                data_dict[parent_dir]["id"] = parent_dir
                data_dict[parent_dir]["age"] = age
                data_dict[parent_dir]["dirpath"] = dir_path
                for file_name in local_filenames:
                    if file_name.startswith("."):
                        continue
                    if file_name.endswith(".mat"):
                        if "global_roi" in file_name.lower():
                            roi_array = hdf5storage.loadmat(os.path.join(dir_path, file_name))
                            data_dict[parent_dir]["global_roi"] = roi_array["global_roi"][0]
                        elif "params" in file_name.lower():
                            mvt_x_y = hdf5storage.loadmat(os.path.join(dir_path, file_name))
                            data_dict[parent_dir]["xshifts"] = mvt_x_y['xshifts'][0]
                            data_dict[parent_dir]["yshifts"] = mvt_x_y['yshifts'][0]
                    if (file_name.endswith(".tif") or file_name.endswith(".tiff")) and ("avg" not in file_name.lower()):
                        data_dict[parent_dir]["tiff_file"] = file_name

                if ("xshifts" not in data_dict[parent_dir]) or ("tiff_file" not in data_dict[parent_dir]):
                    # if the shift info or the movie is missing, we remove this session
                    del data_dict[parent_dir]

    if len(data_dict) == 0:
        print("No directory found")
        return

    # loading movie and calculating the global roi if not done previously
    for key, value in data_dict.items():
        if "global_roi" not in value:
            # then we load the movie, measure to global roi, put it in the data_dict[key] and save it for future
            # loading
            use_scan_tiff_reader = False
            if use_scan_tiff_reader:
                start_time = time.time()
                tiff_movie = ScanImageTiffReader(os.path.join(value["dirpath"], value["tiff_file"])).data()
                stop_time = time.time()
                print(f"Time for loading movie {value['tiff_file']} with scan_image_tiff: "
                      f"{np.round(stop_time - start_time, 3)} s")
            else:
                start_time = time.time()
                im = PIL.Image.open(os.path.join(value["dirpath"], value["tiff_file"]))
                n_frames = len(list(ImageSequence.Iterator(im)))
                dim_x, dim_y = np.array(im).shape
                print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")
                tiff_movie = np.zeros((n_frames, dim_x, dim_y), dtype="uint16")
                for frame, page in enumerate(ImageSequence.Iterator(im)):
                    tiff_movie[frame] = np.array(page)
                stop_time = time.time()
                print(f"Time for loading movie: "
                      f"{np.round(stop_time - start_time, 3)} s")
            global_roi = np.mean(tiff_movie, axis=(1, 2))
            # print(f"global_roi {global_roi.shape}")
            value["global_roi"] = global_roi
            sio.savemat(os.path.join(value["dirpath"], f"{value['id']}_global_roi.mat"), {"global_roi": global_roi})
            del tiff_movie

    corr_by_age = dict()
    corr_by_age_bin = dict()
    # now we produce 2 subplots to plot the mvt and roi value of each session
    for value in data_dict.values():
        roi = value["global_roi"]
        # normalization
        roi = (roi - np.mean(roi)) / np.std(roi)

        # print(f'value["xshifts"] {np.abs(value["xshifts"])}')
        mvt = np.abs(value["xshifts"]) + np.abs(value["yshifts"])
        # normalization
        mvt = (mvt - np.mean(mvt)) / np.std(mvt)
        mvt = mvt - np.abs(np.min(roi)) - np.max(mvt)

        non_lag_roi = np.copy(roi)
        non_lag_mvt = np.copy(mvt)

        add_lag = False
        if add_lag:
            lag = -1
            if lag < 0:
                mvt = mvt[np.abs(lag):]
                roi = roi[:lag]
            else:
                if lag == 0:
                    print("lag can't be set to 0")
                    return
                mvt = mvt[:-lag]
                roi = roi[lag:]

        rho, p = scipy_stats.pearsonr(roi, mvt)

        if np.isnan(rho):
            continue

        if value["age"] not in corr_by_age:
            corr_by_age[value["age"]] = []

        if value["age"] not in corr_by_age_bin:
            corr_by_age_bin[value["age"]] = []

        print(f"{value['id']} rho {str(np.round(rho, 3))}, p {p}")
        corr_by_age[value["age"]].append(rho)

        bin_size = 100
        roi_bin = non_lag_roi.reshape((len(non_lag_roi)//bin_size, bin_size)).sum(axis=1)
        mvt_bin = non_lag_mvt.reshape((len(non_lag_mvt)//bin_size, bin_size)).sum(axis=1)
        rho_bin, p_bin = scipy_stats.pearsonr(roi_bin, mvt_bin)
        print(f"{value['id']} rho_bin {str(np.round(rho_bin, 3))}, p_bin {p_bin}")
        corr_by_age_bin[value["age"]].append(rho_bin)

        rho_bin, p_bin = scipy_stats.spearmanr(roi_bin, mvt_bin)
        print(f"{value['id']} rho_bin {str(np.round(rho_bin, 3))}, p_bin {p_bin}")

        windows = ['hanning', 'hamming', 'bartlett', 'blackman']
        i_w = 1
        window_length = 11
        beg = (window_length - 1) // 2

        # smooth_roi = smooth_convolve(x=roi, window_len=window_length,
        #                                     window=windows[i_w])
        # smooth_roi = smooth_roi[beg:-beg]
        #
        # smooth_mvt = smooth_convolve(x=mvt, window_len=window_length,
        #                              window=windows[i_w])
        # smooth_mvt = smooth_mvt[beg:-beg]

        # rho_smooth, p_smooth = scipy_stats.pearsonr(smooth_roi, smooth_mvt)
        # print(f"{value['id']} rho_smooth {str(np.round(rho_smooth, 3))}, p_smooth {p_smooth}")

        # rho_diff, p_diff = scipy_stats.pearsonr(np.diff(roi), np.diff(mvt))
        # print(f"{value['id']} rho_diff {str(np.round(rho_diff, 3))}, p_diff {p_diff}")


        # roi_filter = np.copy(roi)
        # roi_filter[roi_filter < np.mean(roi)] = np.min(roi)
        # roi_filter[roi_filter >= np.mean(roi)] = np.max(roi)
        # mvt_filter = np.copy(mvt)
        # mvt_filter[mvt_filter < np.mean(mvt)] = np.min(mvt)
        # mvt_filter[mvt_filter >= np.mean(mvt)] = np.max(mvt)
        #
        # rho_filter, p_filter = scipy_stats.pearsonr(roi_filter, mvt_filter)
        # print(f"{value['id']} rho_filter {str(np.round(rho_filter, 3))}, p_filter {p_filter}")

        n_frames = len(roi)

        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex='col',
                                gridspec_kw={'height_ratios': [1], 'width_ratios': [1]},
                                figsize=(15, 5))

        ax1.set_facecolor("black")
        
        ax1.plot(np.arange(n_frames), roi, color="cornflowerblue", lw=1, label=f"ROI", zorder=10)
        ax1.plot(np.arange(n_frames), mvt, color="red", lw=1, label=f"SHIFT", zorder=10)
        min_value = np.min(mvt)
        max_value = np.max(roi)
        interval = 200

        ax1.vlines(np.arange(interval, n_frames, interval), min_value, max_value,
                   color="white", linewidth=0.2,
                   linestyles="dashed", zorder=5)

        ax1.set_ylim(min_value, max_value)
        ax1.set_xlim(0, n_frames)
        # ax1.text(x=n_frames-500, y=np.max(roi)-1, s=f"{value['id']}", color="cornflowerblue", zorder=20,
        #          ha='center', va="center", fontsize=5, fontweight='bold')
        plt.title(value['id'], color="blue")

        ax1.legend()
        axes_to_clean = [ax1]
        for ax in axes_to_clean:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax1.margins(0)
        fig.tight_layout()

        save_formats = ["pdf"]
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/roi_vs_shift_{value["id"]}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}")
        plt.close()

        # then we plot the correlation graph
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex='col',
                                gridspec_kw={'height_ratios': [1], 'width_ratios': [1]},
                                figsize=(10, 10))

        ax1.set_facecolor("black")

        ax1.scatter(roi, mvt,
                    marker='o', c="white",
                    edgecolors="cornflowerblue", s=10,
                    zorder=10)

        # ax1.set_xticks(ages)
        # # sce clusters labels
        # ax1.set_xticklabels(ages)

        ax1.set_xlabel("roi")
        ax1.set_ylabel("shift")

        save_formats = ["pdf"]
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/roi_vs_shift_plot_{value["id"]}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}")
        plt.close()

    # scatter plot with correlation non binarised
    ages = np.array(list(corr_by_age.keys()))
    ages.sort()
    rhos = np.zeros(len(ages))
    stds = np.zeros(len(ages))
    for age_index, age in enumerate(ages):
        rhos[age_index] = np.mean(corr_by_age[age])
        if len(corr_by_age[age]) > 1:
            stds[age_index] = np.std(corr_by_age[age])
    # then we plot the correlation graph
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex='col',
                            gridspec_kw={'height_ratios': [1], 'width_ratios': [1]},
                            figsize=(10, 10))

    ax1.set_facecolor("black")

    ax1.errorbar(ages, rhos, stds, marker='o', markeredgecolor="blue",
                 markerfacecolor="white", markersize=10, ecolor="cornflowerblue", linestyle="-",
                 color="white",
                 linewidth="1",
                 zorder=10)

    ax1.set_xticks(ages)
    # sce clusters labels
    ax1.set_xticklabels(ages)
    # ax1.scatter(ages, rhos, marker='o', c="white",
    #             edgecolors="blue", s=30,
    #             zorder=10)

    ax1.set_xlabel("age")
    ax1.set_ylabel("rho")

    save_formats = ["pdf"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/roi_vs_shift_corr_by_age'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")
    plt.close()

    # scatter plot with correlation non binarised
    ages = np.array(list(corr_by_age.keys()))
    ages.sort()
    rhos = np.zeros(len(ages))
    stds = np.zeros(len(ages))
    for age_index, age in enumerate(ages):
        rhos[age_index] = np.mean(corr_by_age[age])
        if len(corr_by_age[age]) > 1:
            stds[age_index] = np.std(corr_by_age[age])
    # then we plot the correlation graph
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex='col',
                            gridspec_kw={'height_ratios': [1], 'width_ratios': [1]},
                            figsize=(10, 10))

    ax1.set_facecolor("black")

    ax1.errorbar(ages, rhos, stds, marker='o', markeredgecolor="blue",
                 markerfacecolor="white", markersize=10, ecolor="cornflowerblue", linestyle="-",
                 color="white",
                 linewidth="1",
                 zorder=10)

    ax1.set_xticks(ages)
    # sce clusters labels
    ax1.set_xticklabels(ages)
    # ax1.scatter(ages, rhos, marker='o', c="white",
    #             edgecolors="blue", s=30,
    #             zorder=10)

    ax1.set_xlabel("age")
    ax1.set_ylabel("rho")

    save_formats = ["pdf"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/roi_vs_shift_corr_by_age'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")
    plt.close()

    # scatter plot with correlation binarised
    ages = np.array(list(corr_by_age_bin.keys()))
    ages.sort()
    rhos = np.zeros(len(ages))
    stds = np.zeros(len(ages))
    for age_index, age in enumerate(ages):
        rhos[age_index] = np.mean(corr_by_age_bin[age])
        if len(corr_by_age_bin[age]) > 1:
            stds[age_index] = np.std(corr_by_age_bin[age])
    # then we plot the correlation graph
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex='col',
                            gridspec_kw={'height_ratios': [1], 'width_ratios': [1]},
                            figsize=(10, 10))

    ax1.set_facecolor("black")

    ax1.errorbar(ages, rhos, stds, marker='o', markeredgecolor="blue",
                 markerfacecolor="white", markersize=10, ecolor="cornflowerblue", linestyle="-",
                 color="white",
                 linewidth="1",
                 zorder=10)

    ax1.set_xticks(ages)
    # sce clusters labels
    ax1.set_xticklabels(ages)
    # ax1.scatter(ages, rhos, marker='o', c="white",
    #             edgecolors="blue", s=30,
    #             zorder=10)

    ax1.set_xlabel("age")
    ax1.set_ylabel("rho")

    save_formats = ["pdf"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/roi_vs_shift_corr_by_age_bin'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")
    plt.close()

def compute_stat_about_significant_seq(files_path, param, color_option="use_cmap_gradient", cmap_name="Reds",
                                       scale_scatter=False, use_different_shapes_for_stat=False,
                                       save_formats="pdf"):
    """

    :param files_path:
    :param param:
    :param scale_scatter: if True will scale the scatter to the number of sequences of a given length and repetition
    :param use_different_shapes_for_stat: if True, will put a different shape for each type of significant stat
    :param color_option: "use_cmap_random", "use_cmap_gradien", "use_param_colors"
    :param cmap: "Reds", "nipy_spectral"
    :param save_formats:
    :return:
    """
    n_categories = 4
    marker_cat = ["*", "d", "o", "s"]
    # categories that should be displayed
    banned_categories = []
    file_names = []
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

    ages_groups = dict()
    # will allow to group age by groups, for each age, we give a string value which is going to be a key for another
    # dict
    age_tuples = [(6, 7), (8, 10), (11, 14), (60,)]
    manual_colors = dict()
    manual_colors["6-7"] = "white"
    # manual_colors["8-10"] = "navajowhite"
    # manual_colors["11-14"] = "coral"
    manual_colors["8-10"] = "lawngreen"
    manual_colors["11-14"] = "cornflowerblue"
    manual_colors["60"] = "red"
    # used for display
    ages_key_order = []
    for age_tuple in age_tuples:
        if len(age_tuple) == 1:
            age_str = f"{age_tuple[0]}"
        else:
            age_str = f"{age_tuple[0]}-{age_tuple[1]}"
        if len(age_tuple) == 1:
            ages_groups[age_tuple[0]] = age_str
        else:
            for age in np.arange(age_tuple[0], age_tuple[1] + 1):
                ages_groups[age] = age_str
        ages_key_order.append(age_str)

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
        age_file = int(file_name[1:index_])
        age = ages_groups[age_file]
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
    for age in ages_key_order:
        cat_dict = data_dict[age]
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
                    if color_option == "use_cmap_random":
                        color = plt.get_cmap(cmap_name)(float(age_index + 1) / (len(ages_key_order) + 1))
                    elif color_option == "use_cmap_gradient":
                        color = plt.get_cmap(cmap_name)(float(age_index + 1) / (len(ages_key_order) + 1))
                    elif color_option == "manual":
                        color = manual_colors[age]
                    else:
                        color = param.colors[age_index % (len(param.colors))]
                    scatter_size = 50 + 1.2 * x_pos + 1.2 * y_pos
                    if scale_scatter:
                        scatter_size = 15 + 5 * np.sqrt(n_seq_normalized)
                    marker_to_use = "o"
                    if use_different_shapes_for_stat:
                        marker_to_use = param.markers[cat - 1]
                    ax1.scatter(x_pos,
                                y_pos,
                                color=color,
                                marker=marker_to_use,
                                s=scatter_size, alpha=1, edgecolors='none')
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
    for age in ages_key_order:
        if color_option == "use_cmap_random":
            color = plt.get_cmap(cmap_name)(float(age_index + 1) / (len(ages_key_order) + 1))
        elif color_option == "use_cmap_gradient":
            values = np.linspace(0, 1, len(ages_key_order))
            color = plt.get_cmap(cmap_name)(values[age_index])
        elif color_option == "manual":
            color = manual_colors[age]
        else:
            color = param.colors[age_index % (len(param.colors))]
        legend_elements.append(Patch(facecolor=color,
                                     edgecolor='black', label=f'p{age}'))
        age_index += 1
    if use_different_shapes_for_stat:
        for cat in np.arange(1, n_categories + 1):
            if cat in banned_categories:
                continue
            legend_elements.append(Line2D([0], [0], marker=param.markers[cat - 1], color="w", lw=0, label="*" * cat,
                                          markerfacecolor='black', markersize=15))

    ax1.legend(handles=legend_elements)

    # plt.title(title)
    ax1.set_ylabel(f"Repetition (#)", fontsize=20)
    ax1.set_xlabel("Cells (#)", fontsize=20)
    ax1.set_ylim(min_rep - 0.5, max_rep + 1)
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


def test_seq_detect(ms, span_area_coords=None, span_area_colors=None):
    # print(f"test_seq_detect {ms.description} {ms.best_order_loaded}")
    if ms.best_order_loaded is None:
        return

    spike_nums_dur = ms.spike_struct.spike_nums_dur
    spike_nums_dur_ordered = spike_nums_dur[ms.best_order_loaded, :]
    # seq_dict = find_sequences_in_ordered_spike_nums(spike_nums_dur_ordered, param=ms.param)
    # save_on_file_seq_detection_results(best_cells_order=ms.best_order_loaded,
    #                                    seq_dict=seq_dict,
    #                                    file_name=f"sorting_results_with_timestamps{ms.description}.txt",
    #                                    param=ms.param,
    #                                    significant_category_dict=None)

    # colors_for_seq_list = ["blue", "red", "limegreen", "grey", "orange", "cornflowerblue", "yellow", "seagreen",
    #                        "magenta"]
    ordered_labels_real_data = []
    labels = np.arange(len(spike_nums_dur_ordered))
    for old_cell_index in ms.best_order_loaded:
        ordered_labels_real_data.append(labels[old_cell_index])
    # plot_spikes_raster(spike_nums=spike_nums_dur_ordered, param=ms.param,
    #                    title=f"{ms.description}_spike_nums_ordered_seq_test",
    #                    spike_train_format=False,
    #                    file_name=f"{ms.description}_spike_nums_ordered_seq_test",
    #                    y_ticks_labels=ordered_labels_real_data,
    #                    save_raster=True,
    #                    show_raster=False,
    #                    sliding_window_duration=1,
    #                    show_sum_spikes_as_percentage=True,
    #                    plot_with_amplitude=False,
    #                    activity_threshold=ms.activity_threshold,
    #                    save_formats="pdf",
    #                    seq_times_to_color_dict=seq_dict,
    #                    link_seq_color=colors_for_seq_list,
    #                    link_seq_line_width=1,
    #                    link_seq_alpha=0.9,
    #                    jitter_links_range=5,
    #                    min_len_links_seq=3,
    #                    spike_shape="|",
    #                    spike_shape_size=10)

    plot_spikes_raster(spike_nums=spike_nums_dur_ordered, param=ms.param,
                       title=f"{ms.description}_spike_nums_ordered",
                       spike_train_format=False,
                       file_name=f"{ms.description}_spike_nums_ordered",
                       y_ticks_labels=ordered_labels_real_data,
                       save_raster=True,
                       show_raster=False,
                       sliding_window_duration=1,
                       show_sum_spikes_as_percentage=True,
                       plot_with_amplitude=False,
                       activity_threshold=ms.activity_threshold,
                       save_formats="pdf",
                       link_seq_line_width=1,
                       link_seq_alpha=0.9,
                       jitter_links_range=5,
                       min_len_links_seq=3,
                       spike_shape="|",
                       span_area_coords=span_area_coords,
                       span_area_colors=span_area_colors,
                       spike_shape_size=10)

    print(f"n_cells: {len(spike_nums_dur_ordered)}")

    # if ms.cell_assemblies is not None:
    #     total_cells_in_ca = 0
    #     for cell_assembly_index, cell_assembly in enumerate(ms.cell_assemblies):
    #         total_cells_in_ca += len(cell_assembly)
    #     #     print(f"CA {cell_assembly_index}: {cell_assembly}")
    #     # print(f"n_cells in cell assemblies: {total_cells_in_ca}")
    #     sequences_with_ca_numbers = []
    #     cells_seq_with_correct_indices = []
    #     # we need to find the indices from the organized seq
    #     for seq in seq_dict.keys():
    #         cells_seq_with_correct_indices.append(ms.best_order_loaded[np.array(seq)])
    #     for seq in cells_seq_with_correct_indices:
    #         new_seq = np.ones(len(seq), dtype="int16")
    #         new_seq *= - 1
    #         for cell_assembly_index, cell_assembly in enumerate(ms.cell_assemblies):
    #             for index_cell, cell in enumerate(seq):
    #                 if cell in cell_assembly:
    #                     new_seq[index_cell] = cell_assembly_index
    #         sequences_with_ca_numbers.append(new_seq)

    # print("")
    # print("Seq with cell assemblies index")
    choose_manually = False
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
        max_index_seq = len(spike_nums_dur_ordered)  # 50

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

        # span_areas_coords = []
        # span_area_colors = []
        # span_areas_coords.append(ms.mvt_frames_periods)
        # span_area_colors.append('red')
        # span_areas_coords.append(ms.sce_times_in_cell_assemblies)
        # span_area_colors.append('green')
        # span_areas_coords.append(ms.twitches_frames_periods)
        # span_area_colors.append('blue')

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
                       # spike_shape="o",
                       # spike_shape_size=1,
                       spike_shape="|",
                       spike_shape_size=10,
                       # span_area_coords=span_area_coords,
                       # span_area_colors=span_area_colors,
                       # span_area_coords=span_area_coords,
                       # span_area_colors=span_area_colors,
                       # span_area_only_on_raster=False,
                       without_activity_sum=True,
                       size_fig=(15, 6))
    # with amplitude, using traces
    # print(f"ms.traces.shape {ms.traces.shape}")
    if ms.traces is None:
        return

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


def find_hubs(graph, ms):
    n_cells = ms.spike_struct.n_cells
    # first selecting cells conencted to more than 5% cells
    cells_connectivity_perc_threshold = 5
    # step 1
    cells_selected_s1 = []
    cells_connectivity = []
    for cell in np.arange(n_cells):
        cells_connectivity.append(len(graph[cell]))
        if ((len(graph[cell]) / n_cells) * 100) >= cells_connectivity_perc_threshold:
            cells_selected_s1.append(cell)
    if len(cells_selected_s1) == 0:
        return cells_selected_s1
    cells_selected_s2 = []
    connec_treshold = np.percentile(cells_connectivity, 80)
    for cell in cells_selected_s1:
        if cells_connectivity[cell] >= connec_treshold:
            cells_selected_s2.append(cell)

    if len(cells_selected_s2) == 0:
        return cells_selected_s2

    cells_selected_s3 = []
    bc_dict = nx.betweenness_centrality(graph)  # , np.arange(n_cells)
    bc_values = list(bc_dict.values())
    bc_perc_threshold = np.percentile(bc_values, 80)
    for cell in cells_selected_s2:
        if bc_dict[cell] >= bc_perc_threshold:
            cells_selected_s3.append(cell)

    return cells_selected_s3


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


def main():
    # for line in np.arange(15):
    #     print_surprise_for_michou(n_lines=15, actual_line=line)
    # loading the root_path
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")

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
    just_correlate_global_roi_and_shift = False
    if just_compute_significant_seq_stat:
        compute_stat_about_significant_seq(files_path=f"{path_data}significant_seq/v4/", param=param,
                                           save_formats=["pdf"],
                                           color_option="manual", cmap_name="Reds")
        return

    if just_correlate_global_roi_and_shift:
        correlate_global_roi_and_shift(path_data=os.path.join(path_data), param=param)
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
    ms_new_from_Robin_2nd_dec = ["p12_171110_a000_ms", "p6_18_02_07_a002_ms"]
    ms_str_to_load = available_ms_str
    ms_str_to_load = ms_with_run
    # ms_str_to_load = ["p60_a529_2015_02_25_v_arnaud_ms"]
    ms_str_to_load = ["p7_18_02_08_a001_ms"]
    ms_str_to_load = ["p10_17_11_16_a003_ms"]
    ms_str_to_load = available_ms_str
    ms_str_to_load = ["p9_18_09_27_a003_ms", "p10_17_11_16_a003_ms"]
    ms_str_to_load = ms_with_cell_assemblies
    ms_str_to_load = ["p6_18_02_07_a001_ms", "p12_17_11_10_a002_ms"]
    ms_str_to_load = ["p60_arnaud_ms"]
    ms_str_to_load = available_ms_str
    ms_str_to_load = ["p6_18_02_07_a002_ms"]
    ms_str_to_load = ms_with_piezo
    ms_str_to_load = ms_with_piezo
    ms_str_to_load = ["p7_18_02_08_a000_ms"]
    ms_str_to_load = ["p7_17_10_18_a002_ms"]
    # ms_str_to_load = ["p60_a529_2015_02_25_ms"]
    ms_str_to_load = ms_new_from_Robin_2nd_dec
    ms_str_to_load = ["p9_18_09_27_a003_ms"]
    ms_str_to_load = ["p6_18_02_07_a001_ms"]
    ms_str_to_load = ["p9_18_09_27_a003_ms"]
    ms_str_to_load = ["p60_a529_2015_02_25_ms"]
    ms_str_to_load = ["p6_18_02_07_a001_ms"]
    no_spike_nums = ["p6_18_02_07_a002_ms", "p12_171110_a000_ms"]
    ms_str_to_load = ["p13_18_10_29_a000_ms",  # new
                      "p13_18_10_29_a001_ms",
                      "p14_18_10_23_a000_ms",
                      "p14_18_10_30_a001_ms",
                      "p60_arnaud_ms", "p60_a529_2015_02_25_ms"]
    for_graph = ["p6_18_02_07_a001_ms",
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
                 "p12_17_11_10_a002_ms",
                 "p13_18_10_29_a000_ms",  # new
                 "p13_18_10_29_a001_ms",
                 "p14_18_10_23_a000_ms",
                 "p14_18_10_30_a001_ms",
                 "p60_arnaud_ms", "p60_a529_2015_02_25_ms"]
    ms_str_to_load = for_graph
    ms_str_to_load = ["p60_arnaud_ms", "p60_a529_2015_02_25_ms"]
    ms_str_to_load = ["p13_18_10_29_a001_ms"]
    ms_str_to_load = ["p6_18_02_07_a002_ms"]
    ms_str_to_load = ["p6_18_02_07_a001_ms"]
    ms_str_to_load = ["p7_18_02_08_a000_ms"]
    ms_str_to_load = ["p12_171110_a000_ms"]
    ms_str_to_load = ["p9_18_09_27_a003_ms"]
    ms_str_to_load = ["p12_171110_a000_ms"]
    ms_str_to_load = ["p60_a529_2015_02_25_ms"]
    ms_str_to_load = ["p7_171012_a000_ms"]
    ms_str_to_load = ["p7_171012_a000_ms"]
    # ms_str_to_load = ["richard_015_D74_P2_ms"]
    # ms_str_to_load = ["richard_015_D89_P2_ms"]
    # ms_str_to_load = ["richard_015_D66_P2_ms"]
    # ms_str_to_load = ["richard_015_D75_P2_ms"]
    # ms_str_to_load = ["richard_018_D32_P2_ms"]
    # ms_str_to_load = ["richard_018_D28_P2_ms"]
    # ms_str_to_load = ["richard_028_D1_P1_ms"]
    # ms_str_to_load = ["richard_028_D2_P1_ms"]
    ms_str_to_load = ["p12_171110_a000_ms"]
    # ms_str_to_load = ["p8_18_10_24_a005_ms"]

    # 256

    # loading data
    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_str_to_load, param=param,
                                            load_traces=load_traces, load_abf=True)

    available_ms = []
    for ms_str in ms_str_to_load:
        available_ms.append(ms_str_to_ms_dict[ms_str])
    # for ms in available_ms:
    #     ms.plot_each_inter_neuron_connect_map()
    #     return
    ms_to_analyse = available_ms

    just_do_stat_on_event_detection_parameters = False
    just_plot_raster = False
    just_plot_time_correlation_graph_over_twitches = False
    just_plot_raster_with_cells_assemblies_events_and_mvts = False
    just_plot_traces_raster = False
    just_plot_piezo_with_extra_info = False
    just_plot_raw_traces_around_each_sce_for_each_cell = False
    just_plot_cell_assemblies_on_map = False
    just_plot_all_cells_on_map = False
    do_plot_psth_twitches = False
    just_do_seqnmf = False
    just_generate_artificial_movie_from_rasterdur = False
    just_do_pca_on_raster = False
    just_display_seq_with_cell_assembly = False
    just_produce_animation = True

    # for events (sce) detection
    perc_threshold = 99
    use_max_of_each_surrogate = False
    n_surrogate_activity_threshold = 1000
    use_raster_dur = True
    no_redundancy = False
    determine_low_activity_by_variation = False

    do_plot_interneurons_connect_maps = False
    do_plot_connect_hist = False
    do_plot_graph = False
    do_find_hubs = False
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
    use_richard_option = False
    # wake, sleep, quiet_wake, sleep_quiet_wake, active_wake
    richard_option = "wake"

    # ##### for fca #####
    n_surrogate_fca = 20

    # #### for kmean  #####
    with_shuffling = False
    print(f"use_raster_dur {use_raster_dur}")
    range_n_clusters_k_mean = np.arange(2, 10)
    # range_n_clusters_k_mean = np.array([5])
    n_surrogate_k_mean = 20
    keep_only_the_best_kmean_cluster = False

    # ##########################################################################################
    # ################################ PATTERNS SEARCH #########################################
    # ##########################################################################################
    do_pattern_search = False
    keep_the_longest_seq = False
    split_pattern_search = False
    use_only_uniformity_method = False
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
    param.min_rep_nb = 5

    debug_mode = False

    # ------------------------------ end param section ------------------------------

    ms_by_age = dict()
    for ms_index, ms in enumerate(ms_to_analyse):
        if do_pattern_search or do_clustering:
            break
        print(f"ms {ms.description}")
        # np.savez(ms.param.path_data + ms.description + "_rasters_reduced.npz",
        #          spike_nums=ms.spike_struct.spike_nums[:50, :5000],
        #          spike_nums_dur=ms.spike_struct.spike_nums_dur[:50, :5000])
        # raise Exception("ambre")
        if just_plot_time_correlation_graph_over_twitches:
            ms.plot_time_correlation_graph_over_twitches()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("loko")
            continue

        if just_plot_all_cells_on_map:
            ms.plot_all_cells_on_map()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_all_cells_on_map exception")
            continue

        if just_produce_animation:
            ms.produce_animation()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_produce_animation exception")
            continue

        if just_display_seq_with_cell_assembly:
            print("test_seq_detect")
            span_area_coords = None
            span_area_colors = None
            show_richard_active_frames = True
            if show_richard_active_frames:
                active_frames = ms.richard_dict["Active_Wake_Frames"]
                bin_array = np.zeros(ms.spike_struct.spike_nums_dur.shape[1], dtype="int8")
                bin_array[np.unique(active_frames)] = 1
                periods = get_continous_time_periods(bin_array)
                span_area_coords = [periods]
                span_area_colors = ["red"]
            test_seq_detect(ms, span_area_coords=span_area_coords, span_area_colors=span_area_colors)
            raise Exception("toto")

        if just_do_pca_on_raster:
            spike_nums_to_use = ms.spike_struct.spike_nums_dur
            sce_detection_result = detect_sce_potatoes_style(spike_nums=spike_nums_to_use, perc_threshold=95,
                                                             debug_mode=True)

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

            span_area_coords = [SCE_times]
            span_area_colors = ['lightgrey']
            ms.pca_on_raster(span_area_coords=span_area_coords, span_area_colors=span_area_colors)
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("pca_done")
            continue

        if just_plot_raster_with_cells_assemblies_events_and_mvts:
            ms.plot_raster_with_cells_assemblies_events_and_mvts()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("koko")
            continue

        if just_generate_artificial_movie_from_rasterdur:
            param_movie = art_movie_gen.DataForMs(path_data=param.path_data, path_results=param.path_results,
                                                  time_str=param.time_str,
                                                  with_mvt=False, use_fake_cells=False, dimensions=(180, 175),
                                                  n_vessels=0, same_baseline_from_cell_than_background=True)
            # dimension is height x width
            produce_movie(map_coords=ms.coord, raster_dur=ms.spike_struct.spike_nums_dur, param=param_movie,
                          cells_with_overlap=None,
                          overlapping_cells=None, padding=0,
                          vessels=[], use_traces_for_amplitude=ms.raw_traces,
                          file_name=f"{ms.description}_artificial_reverse")
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("momo")
            continue

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
            print(f"{ms.description} activity_threshold: {activity_threshold}")
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

        print(f"n_cells {ms.spike_struct.n_cells}, n_sces {len(ms.SCE_times)}")

        if just_plot_traces_raster:
            print("just_plot_traces_raster")
            raw_traces = ms.raw_traces
            for i in np.arange(len(raw_traces)):
                raw_traces[i] = (raw_traces[i] - np.mean(raw_traces[i]) / np.std(raw_traces[i]))
                raw_traces[i] = norm01(raw_traces[i]) * 5

            span_area_coords = []
            span_area_colors = []
            if (not ms.with_run) and (ms.twitches_frames_periods is not None):
                span_area_coords.append(ms.twitches_frames_periods)
                span_area_colors.append("blue")
                span_area_coords.append(ms.complex_mvt_frames_periods)
                span_area_colors.append("red")
                span_area_coords.append(ms.intermediate_behavourial_events_frames_periods)
                span_area_colors.append("red")
                span_area_coords.append(ms.short_lasting_mvt_frames_periods)
                span_area_colors.append("black")
            plot_spikes_raster(spike_nums=spike_nums_to_use, param=ms.param,
                               traces=raw_traces,
                               display_traces=True,
                               spike_train_format=False,
                               title="traces raster",
                               file_name="traces raster",
                               y_ticks_labels=np.arange(len(raw_traces)),
                               y_ticks_labels_size=2,
                               save_raster=True,
                               show_raster=True,
                               span_area_coords=span_area_coords,
                               span_area_colors=span_area_colors,
                               plot_with_amplitude=False,
                               activity_threshold=ms.spike_struct.activity_threshold,
                               raster_face_color="white",
                               # 500 ms window
                               sliding_window_duration=sliding_window_duration,
                               show_sum_spikes_as_percentage=True,
                               # vertical_lines=SCE_times,
                               # vertical_lines_colors=['white'] * len(SCE_times),
                               # vertical_lines_sytle="solid",
                               # vertical_lines_linewidth=[0.2] * len(SCE_times),
                               span_area_only_on_raster=False,
                               spike_shape_size=0.5,
                               save_formats="png")
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_traces_raster exception")
            continue

        if just_do_seqnmf:
            k = 2
            l = 30
            lambda_value = 0.001
            W, H, cost, loadings, power = seqnmf.seqnmf(ms.spike_struct.spike_nums, K=k, L=l,
                                                        Lambda=lambda_value)
            fig_seqnmf = seqnmf.plot(W, H)
            save_formats = "pdf"
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            for save_format in save_formats:
                fig_seqnmf.savefig(f'{ms.param.path_results}/{ms.description}_seqnmf_k_{k}_l_{l}_lambda_{lambda_value}'
                                   f'_{ms.param.time_str}.{save_format}',
                                   format=f"{save_format}")

            plt.show()
            plt.close()
            # (160, 20, 100) and (20, 14000)
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_piezo_with_extra_info exception")
            continue

        if just_plot_piezo_with_extra_info:
            ms.plot_piezo_with_extra_info(show_plot=False, save_formats="pdf")
            # # ms.plot_piezo_around_event(range_in_sec=5, save_formats="png")
            # # ms.plot_raw_traces_around_twitches()
            # # ms.plot_psth_over_twitches_time_correlation_graph_style()
            # # ms.plot_piezo_with_extra_info(show_plot=True, with_cell_assemblies_sce=False, save_formats="pdf")
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_piezo_with_extra_info exception")
            continue
        if just_plot_raw_traces_around_each_sce_for_each_cell:
            ms.plot_raw_traces_around_each_sce_for_each_cell()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("plot_raw_traces_around_each_sce_for_each_cell exception")
            continue

        if just_plot_cell_assemblies_on_map:
            ms.plot_cell_assemblies_on_map()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_cell_assemblies_on_map exception")
            continue

        if ms.age not in ms_by_age:
            ms_by_age[ms.age] = []

        ms_by_age[ms.age].append(ms)

        if do_plot_interneurons_connect_maps or do_plot_connect_hist or do_plot_graph or do_find_hubs:
            ms.detect_n_in_n_out()
            # For p9_a003 good out connec: cell 8, 235, 201,  151, 17
            # for cell_to_map in [8, 235, 201, 151, 17]:
            #     ms.plot_connectivity_maps_of_a_cell(cell_to_map=cell_to_map, cell_descr="",
            #                                         cell_color="red", links_cell_color="cornflowerblue")
            # raise Exception("it's over")
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

            # for Robin
            # for sce_index, sce_time in enumerate(SCE_times):
            #     print(f"sce n° {sce_index}: {sce_time[0]}-{sce_time[1]}: "
            #           f"{np.where(np.sum(spike_struct.spike_nums_dur[:, sce_time[0]:sce_time[1]+1], axis=1))[0]}")
            # return
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

        if do_find_hubs:
            for cell_to_map in [61, 73, 130, 138, 142]:
                ms.plot_connectivity_maps_of_a_cell(cell_to_map=cell_to_map, cell_descr="", not_in=False,
                                                    cell_color="red", links_cell_color="cornflowerblue")
            # hubs = find_hubs(graph=ms.spike_struct.graph_out, ms=ms)
            # print(f"{ms.description} hubs: {hubs}")
            # P13_18_10_29_a001 hubs: [61, 73, 130, 138, 142]
            # P60_arnaud_a_529 hubs: [65, 102]
            # P60_a529_2015_02_25 hubs: [2, 8, 88, 97, 109, 123, 127, 142]
        if do_plot_graph:
            plot_graph_using_fa2(graph=ms.spike_struct.graph_out, file_name=f"{ms.description} graph out",
                                 title=f"{ms.description}",
                                 param=param, iterations=15000, save_raster=True, with_labels=False,
                                 save_formats="pdf", show_plot=False)
            # ms.spike_struct.graph_out.add_edges_from(ms.spike_struct.graph_in.edges())
            # plot_graph_using_fa2(graph=ms.spike_struct.graph_out, file_name=f"{ms.description} graph in-out",
            #                      title=f"{ms.description} in-out",
            #                      param=param, iterations=5000, save_raster=True, with_labels=False,
            #                      save_formats="pdf", show_plot=False)
        if do_plot_connect_hist:
            connec_func_stat([ms], data_descr=ms.description, param=param)
            # best_cell = -1
            # best_score = 0
            # for cell in np.arange(ms.spike_struct.n_cells):
            #     score = np.sum(ms.spike_struct.n_out_matrix[cell])
            #     if best_score < score:
            #         best_cell = cell
            #         best_score = score
            # for cell_to_map in [best_cell]:
            #     ms.plot_connectivity_maps_of_a_cell(cell_to_map=cell_to_map, cell_descr="", not_in=False,
            #                                         cell_color="red", links_cell_color="cornflowerblue")

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
            n_cells = spike_struct.n_cells
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

                    # sce_detection_result = detect_sce_potatoes_style(spike_nums=spike_nums_to_use, perc_threshold=95,
                    #                                                  debug_mode=True)

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
                    if inter_neurons is None:
                        inter_neurons = []
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
        n_cells = len(spike_struct.spike_nums_dur)
        # spike_struct.build_spike_trains()

        # ######  parameters setting #########
        data_descr = f"{ms.description}"
        print(f"ms: {data_descr} {param.time_str}")

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
        if use_richard_option:
            # wake, sleep, quiet_wake, quiet + sleep
            print(f"richard_option {richard_option}")
            print(f"spike_nums_to_use n_frames before: {spike_nums_to_use.shape[1]}")
            data_descr += "_" + richard_option

            if richard_option == "wake":
                frames_selected = np.concatenate((ms.richard_dict["Active_Wake_Frames"],
                                                  ms.richard_dict["Quiet_Wake_Frames"]))
                frames_selected = np.unique(frames_selected)
            elif richard_option == "sleep":
                frames_selected = np.concatenate((ms.richard_dict["REMs_Frames"],
                                                  ms.richard_dict["NREMs_Frames"]))
                frames_selected = np.unique(frames_selected)
            elif richard_option == "quiet_wake":
                frames_selected = np.unique(ms.richard_dict["Quiet_Wake_Frames"])
            elif richard_option == "active_wake":
                frames_selected = np.unique(ms.richard_dict["Active_Wake_Frames"])
            elif richard_option == "sleep_quiet_wake":
                frames_selected = np.concatenate((ms.richard_dict["REMs_Frames"],
                                                  ms.richard_dict["NREMs_Frames"]))
                frames_selected = np.concatenate((frames_selected,
                                                  ms.richard_dict["Quiet_Wake_Frames"]))
                frames_selected = np.unique(frames_selected)
            # removing frames over the number of frames in the raster dur
            frames_selected = frames_selected[frames_selected < spike_nums_to_use.shape[1]]
            spike_nums_to_use = spike_nums_to_use[:, frames_selected]
            print(f"spike_nums_to_use n_frames after: {spike_nums_to_use.shape[1]}")
            # raise Exception("test richard")

        if (ms.activity_threshold is None) or use_richard_option:
            activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums_to_use,
                                                             window_duration=sliding_window_duration,
                                                             spike_train_mode=False,
                                                             n_surrogate=n_surrogate_activity_threshold,
                                                             perc_threshold=perc_threshold,
                                                             use_max_of_each_surrogate=use_max_of_each_surrogate,
                                                             debug_mode=False)
        else:
            activity_threshold = ms.activity_threshold

        print(f"perc_threshold {perc_threshold}, "
              f"activity_threshold {activity_threshold}, {np.round((activity_threshold/n_cells)*100, 2)}%")
        print(f"sliding_window_duration {sliding_window_duration}")
        spike_struct.activity_threshold = activity_threshold
        # param.activity_threshold = activity_threshold

        print("// plot_spikes_raster")

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
                                                              keep_only_the_peak=False)
        # sce_detection_result = detect_sce_potatoes_style(spike_nums=spike_nums_to_use, perc_threshold=95,
        #                                                  debug_mode=True, keep_only_the_peak=False)

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

                    #
                    # print(f"perc_threshold {perc_threshold}, "
                    #       f"activity_threshold {activity_threshold}, {np.round((activity_threshold/n_cells)*100, 2)}%")
                    #
                    # spike_struct.activity_threshold = activity_threshold
                    #
                    # sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_nums_to_use,
                    #                                                       window_duration=sliding_window_duration,
                    #                                                       perc_threshold=perc_threshold,
                    #                                                       activity_threshold=activity_threshold,
                    #                                                       debug_mode=False,
                    #                                                       no_redundancy=no_redundancy,
                    #                                                       keep_only_the_peak=False)
                    # print(f"sce_with_sliding_window detected")
                    # cellsinpeak = sce_detection_result[2]
                    # SCE_times = sce_detection_result[1]
                    # sce_times_bool = sce_detection_result[0]
                    # sce_times_numbers = sce_detection_result[3]
                    # # useful for plotting twitches
                    # ms.sce_bool = sce_times_bool
                    # ms.sce_times_numbers = sce_times_numbers
                    # ms.SCE_times = SCE_times

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
                                          extra_file_name=data_descr,
                                          sce_times_bool=sce_times_bool_to_use,
                                          use_only_uniformity_method=use_only_uniformity_method,
                                          use_loss_score_to_keep_the_best_from_tree=
                                          use_loss_score_to_keep_the_best_from_tree,
                                          spike_shape="|",
                                          spike_shape_size=5,
                                          keep_the_longest_seq=keep_the_longest_seq)

    return


main()
