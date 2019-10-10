max_threads = '10'
import os

# Configure threading
os.environ["MKL_NUM_THREADS"] = max_threads
os.environ["NUMEXPR_NUM_THREADS"] = max_threads
os.environ["OMP_NUM_THREADS"] = max_threads

# Importing numpy and model must be performed *after* multithreading is configured. Useful for cilva

import scipy.spatial.distance as sci_sp_dist
import matplotlib.cm as cm
import scipy.io as sio
import scipy.stats as scipy_stats
import seaborn as sns
from bisect import bisect
from pattern_discovery.display.cells_map_module import CoordClass
# important to avoid a bug when using virtualenv
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
# import copy
from datetime import datetime
from scipy import signal
# import keras
import pyabf
from sklearn.decomposition import PCA
# import seqnmf
# to add homemade package, go to preferences, then project interpreter, then click on the wheel symbol
# then show all, then select the interpreter and lick on the more right icon to display a list of folder and
# add the one containing the folder pattern_discovery
from pattern_discovery.seq_solver.markov_way import MarkovParameters
from pattern_discovery.seq_solver.seq_with_pca import find_seq_with_pca
from pattern_discovery.seq_solver.markov_way import find_significant_patterns
from pattern_discovery.seq_solver.seq_finder_using_graph import find_sequences_using_graph_main
from pattern_discovery.seq_solver.markov_way import find_sequences_in_ordered_spike_nums
from pattern_discovery.seq_solver.markov_way import save_on_file_seq_detection_results
import pattern_discovery.tools.misc as tools_misc
from pattern_discovery.tools.misc import get_time_correlation_data
from pattern_discovery.tools.misc import get_continous_time_periods, give_unique_id_to_each_transient_of_raster_dur
from pattern_discovery.display.raster import plot_spikes_raster, plot_with_imshow
from pattern_discovery.display.misc import time_correlation_graph
import pattern_discovery.display.misc as display_misc
from pattern_discovery.tools.sce_detection import get_sce_detection_threshold, detect_sce_with_sliding_window, \
    detect_sce_on_traces
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
from mouse_session_loader import load_mouse_sessions
from lexi_mouse_session_loader import load_lexi_mouse_sessions
import networkx as nx
from articifical_movie_patch_generator import produce_movie
import articifical_movie_patch_generator as art_movie_gen
from ScanImageTiffReader import ScanImageTiffReader
from pattern_discovery.tools.signal import smooth_convolve
import PIL
from PIL import ImageSequence
# import joypy
import math
import hdbscan
from scipy.spatial.distance import jensenshannon
import neo
import elephant.conversion as elephant_conv
import quantities as pq
import elephant.cell_assembly_detection as cad
from spot_dist import spotdist_function
from twitches_analysis import twitch_analysis, covnorm
from rastermap import Rastermap


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

    plot_hist_distribution(distribution_data=n_outs_total,
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

    plot_hist_distribution(distribution_data=n_ins_total,
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
                   f"{np.round((activity_threshold * 100) / n_cells, 2)}%, "
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


def plot_movement_activity(ms_to_analyse, param, save_formats="pdf"):
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    path_results = os.path.join(param.path_results, "movement_activity")
    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    data_types = ["shift_twitch", "shift_long"]
    movement_stat_by_age = dict()
    for data_type in data_types:
        movement_stat_by_age[data_type] = dict()

    for ms in ms_to_analyse:
        # if not None, filter the frame keeping the kind of mouvements choosen, if available
        # if "no_shift" then select the frame that are not in any period
        # Other keys are: shift_twitch, shift_long, shift_unclassified
        # or a list of those 3 keys and then will take all frames except those

        if (ms.shift_data_dict is None) or (ms.spike_struct.spike_nums_dur is None):
            continue

        n_cells = ms.spike_struct.spike_nums_dur.shape[0]
        for data_type in data_types:
            ms_distribution = []
            array_bool = ms.shift_data_dict[data_type]
            periods = get_continous_time_periods(array_bool.astype("int8"))
            for period in periods:
                # see to associate a period to a period of activity
                sum_activity = np.sum(ms.spike_struct.spike_nums_dur[:, period[0]:period[1] + 1], axis=1)
                sum_activity = len(np.where(sum_activity > 0)[0])
                sum_activity = sum_activity / n_cells
                sum_activity = sum_activity / ((period[1] - period[0] + 1) / ms.sampling_rate)
                ms_distribution.append(sum_activity)
            if ("p" + str(ms.age)) not in movement_stat_by_age[data_type]:
                movement_stat_by_age[data_type][("p" + str(ms.age))] = []
            movement_stat_by_age[data_type][("p" + str(ms.age))].extend(ms_distribution)

            plot_hist_distribution(distribution_data=ms_distribution,
                                   description=f"{ms.description}_hist_{data_type}_activity",
                                   param=param,
                                   path_results=path_results,
                                   tight_x_range=True,
                                   twice_more_bins=True,
                                   xlabel=f"{data_type} activity", save_formats=save_formats)
    for data_type in data_types:
        box_plot_data_by_age(data_dict=movement_stat_by_age[data_type], title="",
                             filename=f"{data_type}_activity_box_plot",
                             path_results=path_results, with_scatters=False,
                             y_label=f"{data_type} activity", colors=colors, param=param,
                             save_formats=save_formats)


def plot_psth_over_event_time_correlation_graph_style(ms_to_analyse, event_str, param, time_around_events=10,
                                                      save_formats="pdf"):
    """
            Will plot in one plot with subplots all time_correlation_graph_over event periods
            Event could be "shift_twitch" or "shift_long'
            :param ms_to_analyse:
            :param param:
            :param save_formats:
            :return:
            """
    print("plot_psth_over_event_time_correlation_graph_style")
    # from: http://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=8
    colors = ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
    # orange ones: http://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=8#type=sequential&scheme=YlOrBr&n=9
    # colors = ['#ffffe5', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
    # diverging, 11 colors : http://colorbrewer2.org/?type=diverging&scheme=RdYlBu&n=11
    # colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
    #           '#74add1', '#4575b4', '#313695']
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    background_color = "black"
    labels_color = "white"
    max_sum = 0

    n_plots = len(ms_to_analyse)

    max_n_lines = 2
    n_lines = n_plots if n_plots <= max_n_lines else max_n_lines
    n_col = math.ceil(n_plots / n_lines)

    # for scatter, ratio all spikes vs all twitches
    fig_for_scatter, axes_for_scatter = plt.subplots(nrows=n_lines, ncols=n_col,
                                                     gridspec_kw={'width_ratios': [1] * n_col,
                                                                  'height_ratios': [1] * n_lines},
                                                     figsize=(30, 20))
    fig_for_scatter.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig_for_scatter.patch.set_facecolor(background_color)
    axes_for_scatter = axes_for_scatter.flatten()

    # for histogram all spikes
    fig_all_spikes, axes_all_spikes = plt.subplots(nrows=n_lines, ncols=n_col,
                                                   gridspec_kw={'width_ratios': [1] * n_col,
                                                                'height_ratios': [1] * n_lines},
                                                   figsize=(30, 20))
    fig_all_spikes.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig_all_spikes.patch.set_facecolor(background_color)
    axes_all_spikes = axes_all_spikes.flatten()

    # for histogram all events
    fig_all_events, axes_all_events = plt.subplots(nrows=n_lines, ncols=n_col,
                                                   gridspec_kw={'width_ratios': [1] * n_col,
                                                                'height_ratios': [1] * n_lines},
                                                   figsize=(30, 20))
    fig_all_events.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig_all_events.patch.set_facecolor(background_color)
    axes_all_events = axes_all_events.flatten()

    # figure for the psth
    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                             figsize=(30, 20))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)
    axes = axes.flatten()
    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        axes_all_spikes[ax_index].set_facecolor(background_color)
        axes_all_events[ax_index].set_facecolor(background_color)
        axes_for_scatter[ax_index].set_facecolor(background_color)
        if ax_index >= len(ms_to_analyse):
            continue
        ms = ms_to_analyse[ax_index]
        ms.plot_psth_over_event_time_correlation_graph_style(event_str=event_str, time_around_events=time_around_events,
                                                             ax_to_use=ax,
                                                             color_to_use=colors[ax_index % len(colors)],
                                                             ax_to_use_total_spikes=axes_all_spikes[ax_index],
                                                             color_to_use_total_spikes=colors[ax_index % len(colors)],
                                                             ax_to_use_total_events=axes_all_events[ax_index],
                                                             color_to_use_total_events=colors[ax_index % len(colors)],
                                                             ax_to_use_for_scatter=axes_for_scatter[ax_index],
                                                             color_to_use_for_scatter=colors[ax_index % len(colors)])

    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/time_correlation_graph_over_{event_str}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
        fig_all_spikes.savefig(f'{param.path_results}/hist_spike_{event_str}_{time_around_events}_by_session'
                               f'_{param.time_str}.{save_format}',
                               format=f"{save_format}",
                               facecolor=fig.get_facecolor())
        fig_all_events.savefig(f'{param.path_results}/hist_event_{event_str}_{time_around_events}_by_session'
                               f'_{param.time_str}.{save_format}',
                               format=f"{save_format}",
                               facecolor=fig.get_facecolor())
        fig_for_scatter.savefig(
            f'{param.path_results}/scatter_all_spikes_vs_all_twitches_{event_str}_{time_around_events}_by_session'
            f'_{param.time_str}.{save_format}',
            format=f"{save_format}",
            facecolor=fig.get_facecolor())

    plt.close()


def plot_all_time_correlation_graph_over_events(ms_to_analyse, event_str, param, time_around_events=1,
                                                save_formats="pdf"):
    """
        Will plot in one plot with subplots all time_correlation_graph_over event periods
        Event could be "shift_twitch" or "shift_long'
        :param ms_to_analyse:
        :param param:
        :param save_formats:
        :return:
        """
    # from: http://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=8
    colors = ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
    # orange ones: http://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=8#type=sequential&scheme=YlOrBr&n=9
    # colors = ['#ffffe5', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
    # diverging, 11 colors : http://colorbrewer2.org/?type=diverging&scheme=RdYlBu&n=11
    # colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
    #           '#74add1', '#4575b4', '#313695']
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    background_color = "black"
    labels_color = "white"
    max_sum = 0

    n_plots = len(ms_to_analyse)

    max_n_lines = 3
    n_lines = n_plots if n_plots <= max_n_lines else max_n_lines
    n_col = math.ceil(n_plots / n_lines)

    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                             figsize=(30, 20))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)

    axes = axes.flatten()
    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        if ax_index >= len(ms_to_analyse):
            continue
        ms = ms_to_analyse[ax_index]
        ms.plot_time_correlation_graph_over_events(event_str=event_str, time_around_events=time_around_events,
                                                   ax_to_use=ax,
                                                   color_to_use=colors[ax_index % len(colors)])

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/time_correlation_graph_over_{event_str}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()


def plot_all_sum_spikes_dur(ms_to_analyse, param, save_formats="pdf"):
    """
    Will plot in one plot with subplots all sum of activity by age
    :param ms_to_analyse:
    :param param:
    :param save_formats:
    :return:
    """
    # from: http://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=8
    colors = ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
    # orange ones: http://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=8#type=sequential&scheme=YlOrBr&n=9
    # colors = ['#ffffe5', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
    # diverging, 11 colors : http://colorbrewer2.org/?type=diverging&scheme=RdYlBu&n=11
    # colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
    #           '#74add1', '#4575b4', '#313695']
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    background_color = "black"
    labels_color = "white"
    max_sum = 0

    sum_activity_by_age_dict = SortedDict()
    ms_description_by_age_dict = SortedDict()

    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums_dur is None:
            print(f"plot_all_sum_spikes_dur: {ms.description}: spike_nums_dur is None")
            continue
        if ms.age not in sum_activity_by_age_dict:
            sum_activity_by_age_dict[ms.age] = []
            ms_description_by_age_dict[ms.age] = []
        sum_spikes = np.sum(ms.spike_struct.spike_nums_dur, axis=0)
        # normalizing by the number of cell
        sum_spikes = (sum_spikes / ms.spike_struct.spike_nums_dur.shape[0]) * 100
        max_sum = max(max_sum, np.max(sum_spikes))
        sum_activity_by_age_dict[ms.age].append(sum_spikes)
        ms_description_by_age_dict[ms.age].append(ms.description)

    sum_activity_list = []
    ms_description_list = []
    for age, sum_activities in sum_activity_by_age_dict.items():
        sum_activity_list.extend(sum_activities)
        ms_description_list.extend(ms_description_by_age_dict[age])
    n_plots = len(sum_activity_list)

    max_n_lines = 20
    n_lines = n_plots if n_plots <= max_n_lines else max_n_lines
    n_col = math.ceil(n_plots / n_lines)

    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                             figsize=(30, 20))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)
    axes = axes.flatten()
    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        if ax_index >= n_plots:
            continue

        sum_spikes = sum_activity_list[ax_index]
        label = ms_description_list[ax_index]
        face_color = colors[ax_index % len(colors)]
        x_value = np.arange(len(sum_spikes))
        ax.fill_between(x_value, 0, sum_spikes, facecolor=face_color, label=label)
        ax.set_ylim(0, max_sum)
        ax.legend()

        # ax.yaxis.set_tick_params(labelsize=20)
        # ax.xaxis.set_tick_params(labelsize=20)
        ax.tick_params(axis='y', colors=labels_color)
        ax.tick_params(axis='x', colors=labels_color)

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/plots_sum_activity_by_age_'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()


def plot_all_long_mvt_psth_in_one_figure(ms_to_analyse, param, line_mode=True,
                                         duration_option=False, save_formats="pdf"):
    """

    :param ms_to_analyse:
    :param param:
    :param line_mode:
    :param duration_option:
    :param save_formats:
    :return:
    """
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
              '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
              '#74add1', '#4575b4', '#313695']
    background_color = "black"
    labels_color = "white"
    max_sum = 0

    if line_mode:
        n_plots = len(ms_to_analyse) + 1
    else:
        n_plots = len(ms_to_analyse)

    max_n_lines = 2
    n_lines = n_plots if n_plots <= max_n_lines else max_n_lines
    n_col = math.ceil(n_plots / n_lines)

    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                             figsize=(30, 20))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)

    axes = axes.flatten()
    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        if ax_index >= len(ms_to_analyse):
            continue
        ms = ms_to_analyse[ax_index]
        print(f"{ms.description} plot_all_mvt_psth_in_one_figure")
        ms.plot_psth_long_mvt(time_around=100, line_mode=line_mode, ax_to_use=ax, put_mean_line_on_plt=line_mode,
                              color_to_use=colors[ax_index % len(colors)], duration_option=duration_option)
    bonus_file_name = ""
    if duration_option:
        bonus_file_name = "_duration"
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/mvt_psth{bonus_file_name}_'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()


def plot_psth_twitches_for_multiple_ms(ms_list, description, path_results, time_around=100,
                                       line_mode=False,
                                       ax_to_use=None, put_mean_line_on_plt=False,
                                       color_to_use=None,
                                       with_other_ms=None,
                                       duration_option=False,
                                       use_traces=False,
                                       mean_version=True,
                                       save_formats="pdf"):
    sce_bool = None
    time_x_values = np.arange(-1 * time_around, time_around + 1)
    total_n_cells = 0
    total_n_twitches = 0
    # line represent each twitches, and columns the time surrounding the twitch
    psth_matrix = np.zeros((0, len(time_x_values)))

    for ms in ms_list:
        # print(f"ms description {ms.description}")
        spike_nums_dur = ms.spike_struct.spike_nums_dur
        # spike_nums = self.spike_struct.spike_nums
        if use_traces:
            # print(f"get_spikes_values_around_twitches use_traces")
            spike_nums_to_use = ms.raw_traces
        else:
            cells_to_keep = np.sum(ms.spike_struct.spike_nums, axis=1) > 2
            spike_nums_to_use = spike_nums_dur[cells_to_keep]
            # spike_nums_to_use = spike_nums_dur

        n_cells = len(spike_nums_to_use)

        # frames on which to center the ptsth
        twitches_times, twitches_periods = ms.get_twitches_times_by_group(sce_bool=sce_bool,
                                                                          twitches_group=0)

        results = ms.get_spikes_by_time_around_a_time(twitches_times, spike_nums_to_use, time_around)
        if results is None:
            continue

        total_n_cells += n_cells
        spike_sum_of_sum_at_time_dict, spikes_sums_at_time_dict, \
        spikes_at_time_dict = results

        # tmp_psth_matrix = None
        # first we determine the max nb of value (can change depending where the twitches are)
        max_n_twitches = 0
        for time_index, time_value in enumerate(time_x_values):
            if time_value in spikes_sums_at_time_dict:
                if len(spikes_sums_at_time_dict[time_value]) > max_n_twitches:
                    max_n_twitches = len(spikes_sums_at_time_dict[time_value])
        tmp_psth_matrix = np.zeros((max_n_twitches, len(time_x_values)))
        for time_index, time_value in enumerate(time_x_values):
            if time_value in spikes_sums_at_time_dict:
                n_twitches = len(spikes_sums_at_time_dict[time_value])
                # print(f"_len(spikes_sums_at_time_dict[time_value]) {len(spikes_sums_at_time_dict[time_value])}")
                tmp_psth_matrix[:n_twitches, time_index] = spikes_sums_at_time_dict[time_value]
                if n_twitches < len(tmp_psth_matrix):
                    tmp_psth_matrix[n_twitches:, time_index] = np.nan
                # mean_values.append((np.mean(spikes_sums_at_time_dict[time_value]) / n_cells) * 100)
                # median_values.append((np.median(spikes_sums_at_time_dict[time_value]) / n_cells) * 100)
                # std_values.append((np.std(spikes_sums_at_time_dict[time_value]) / n_cells) * 100)
                # low_values.append((np.percentile(spikes_sums_at_time_dict[time_value], low_percentile) / n_cells) * 100)
                # high_values.append(
                #     (np.percentile(spikes_sums_at_time_dict[time_value], high_percentile) / n_cells) * 100)
            else:
                print(f"time {time_value} not there")
                # mean_values.append(0)
        if tmp_psth_matrix is not None:
            total_n_twitches += len(tmp_psth_matrix)
            median_values = np.nanmedian(tmp_psth_matrix, axis=0)
            median_values = median_values / n_cells
            median_values = median_values * 100
            median_values = np.reshape(median_values, (1, len(median_values)))
            psth_matrix = np.concatenate((psth_matrix, median_values))
            # psth_matrix = np.concatenate((psth_matrix, tmp_psth_matrix))

    if len(psth_matrix) == 0:
        return
    # print(f"psth_matrix.shape {psth_matrix.shape}")
    # print(f"{description}, total cells filtered: {total_n_cells}")
    # psth_matrix = psth_matrix / total_n_cells
    # psth_matrix = psth_matrix * 100

    mean_values = np.nanmean(psth_matrix, axis=0)
    std_values = np.nanstd(psth_matrix, axis=0)
    median_values = np.nanmedian(psth_matrix, axis=0)
    low_values = np.nanpercentile(psth_matrix, 25)
    high_values = np.nanpercentile(psth_matrix, 75)

    n_twitches = len(psth_matrix)

    hist_color = "blue"
    edge_color = "white"
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

        if color_to_use is not None:
            color = color_to_use
        else:
            color = hist_color
        if mean_version:
            ax1.plot(time_x_values,
                     mean_values, color=color, lw=2, label=f"{description}, N={len(ms_list)}, "
                                                           f"{total_n_twitches} twitches")
            if put_mean_line_on_plt:
                plt.plot(time_x_values,
                         mean_values, color=color, lw=2)
            if with_other_ms is None:
                ax1.fill_between(time_x_values, mean_values - std_values,
                                 mean_values + std_values,
                                 alpha=0.5, facecolor=color)
            max_value = np.max((max_value, np.max(mean_values + std_values)))
        else:
            ax1.plot(time_x_values,
                     median_values, color=color, lw=2, label=f"{description} {n_twitches} twitches")
            if with_other_ms is None:
                ax1.fill_between(time_x_values, low_values, high_values,
                                 alpha=0.5, facecolor=color)
            max_value = np.max((max_value, np.max(high_values)))
    else:
        if color_to_use is not None:
            hist_color = color_to_use
            edge_color = "white"
        ax1.bar(time_x_values,
                mean_values, color=hist_color, edgecolor=edge_color,
                label=f"{description} {n_twitches} twitches")
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

    descr = description

    if duration_option:
        ax1.set_ylabel(f"Duration (ms)")
    else:
        ax1.set_ylabel(f"Spikes (%)")
    ax1.set_xlabel("time (frames)")
    # ax1.set_ylim(0, max_value + 1)
    ax1.set_ylim(0, 5)
    # ax1.set_ylim(0, np.max((activity_threshold_percentage, max_value)) + 1)

    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")
    # xticks = np.arange(0, len(data_dict))
    # ax1.set_xticks(xticks)
    # # sce clusters labels
    # ax1.set_xticklabels(labels)

    if ax_to_use is None:
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{path_results}/{descr}_psth_'
                        f'{n_twitches}_twitches'
                        f'_{extra_info}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())

        plt.close()


def plot_twitches_psth_by_age(ms_to_analyse, param, line_mode=True, use_traces=False,
                              save_formats="pdf"):
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    background_color = "black"
    labels_color = "white"
    max_sum = 0
    # age_categories = {5: "5-6", 6: "5-6", 7: "7-8", 8: "7-8", 9: "9"}
    age_categories = {5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}

    ms_by_age = dict()
    ages = []
    for ms in ms_to_analyse:
        # print(f"len(ms.spike_struct.spike_nums_dur) {len(ms.spike_struct.spike_nums_dur)}")
        if ms.spike_struct.spike_nums_dur is not None and ms.shift_data_dict is not None:
            age_category = age_categories[ms.age]
            if age_category not in ms_by_age:
                ms_by_age[age_category] = []
                ages.append(age_category)
            ms_by_age[age_category].append(ms)
    ages.sort()

    if line_mode:
        n_plots = len(ms_by_age) + 1
    else:
        n_plots = len(ms_by_age)

    if n_plots > 6:
        max_n_lines = 4
    else:
        max_n_lines = 2
    n_lines = n_plots if n_plots <= max_n_lines else max_n_lines
    n_col = math.ceil(n_plots / n_lines)

    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                             figsize=(30, 20))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)

    axes = axes.flatten()
    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        if ax_index >= len(ms_by_age):
            continue
        ms_list = ms_by_age[ages[ax_index]]
        descrs = [m.description for m in ms_list]
        # print(f"{ms.description} plot_all_twitch_psth_in_one_figure")
        plot_psth_twitches_for_multiple_ms(ms_list=ms_list, line_mode=line_mode, ax_to_use=ax,
                                           time_around=100,
                                           description=f"p{ages[ax_index]}", path_results=param.path_results,
                                           put_mean_line_on_plt=line_mode,
                                           color_to_use=colors[ax_index % len(colors)],
                                           use_traces=use_traces)

        plot_psth_twitches_for_multiple_ms(ms_list=ms_list, line_mode=line_mode,
                                           time_around=100,
                                           description=f"p{ages[ax_index]}", path_results=param.path_results,
                                           put_mean_line_on_plt=line_mode,
                                           color_to_use=colors[ax_index % len(colors)],
                                           use_traces=use_traces)

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/twitches_psth_by_age_'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()


def plot_all_twitch_psth_in_one_figure(ms_to_analyse, param, line_mode=True, use_traces=False,
                                       duration_option=False, save_formats="pdf"):
    """
    Will plot in one plot with subplots all twitches PSTH
    :param ms_to_analyse:
    :param param:
    :param save_formats:
    :return:
    """
    # from: http://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=8
    colors = ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
    # orange ones: http://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=8#type=sequential&scheme=YlOrBr&n=9
    # colors = ['#ffffe5', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
    # diverging, 11 colors : http://colorbrewer2.org/?type=diverging&scheme=RdYlBu&n=11
    # colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
    #           '#74add1', '#4575b4', '#313695']
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    background_color = "black"
    labels_color = "white"
    max_sum = 0

    if line_mode:
        n_plots = len(ms_to_analyse) + 1
    else:
        n_plots = len(ms_to_analyse)

    if n_plots > 6:
        max_n_lines = 5
    else:
        max_n_lines = 2
    n_lines = n_plots if n_plots <= max_n_lines else max_n_lines
    n_col = math.ceil(n_plots / n_lines)

    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                             figsize=(30, 20))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)

    axes = axes.flatten()
    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        if ax_index >= len(ms_to_analyse):
            continue
        ms = ms_to_analyse[ax_index]
        # print(f"{ms.description} plot_all_twitch_psth_in_one_figure")
        ms.plot_psth_twitches(line_mode=line_mode, ax_to_use=ax, put_mean_line_on_plt=line_mode,
                              color_to_use=colors[ax_index % len(colors)], duration_option=duration_option,
                              use_traces=use_traces)
    bonus_file_name = ""
    if duration_option:
        bonus_file_name = "_duration"
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/twitches_psth{bonus_file_name}_'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()


def plot_all_basic_stats(ms_to_analyse, param, use_animal_weight=False, save_formats="pdf"):
    # from: http://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=8
    colors = ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
    # orange ones: http://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=8#type=sequential&scheme=YlOrBr&n=9
    colors = ['#ffffe5', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
    # diverging, 11 colors : http://colorbrewer2.org/?type=diverging&scheme=RdYlBu&n=11
    colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
              '#74add1', '#4575b4', '#313695']
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']

    # plot_transient_durations(ms_to_analyse, param, colors=colors, save_formats=save_formats)
    plot_transient_frequency(ms_to_analyse, param, colors=colors, save_formats=save_formats,
                             use_animal_weight=use_animal_weight)
    # plot_transient_amplitude(ms_to_analyse, param, colors=colors, save_formats=save_formats)
    # plot_transient_durations_normalized_by_amplitude(ms_to_analyse, param, colors=colors, save_formats=save_formats)


def plot_transient_amplitude(ms_to_analyse, param, colors=None, save_formats="pdf"):
    path_results = os.path.join(param.path_results, "transient_amplitude")
    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    transient_amplitude_by_age = dict()
    transient_amplitude_by_age_avg_by_cell = dict()
    n_ms = 0
    for ms_index, ms in enumerate(ms_to_analyse):
        if ms.spike_struct.spike_nums_dur is not None:
            n_ms += 1

    background_color = "black"
    max_n_lines = 5
    n_lines = n_ms if n_ms <= max_n_lines else max_n_lines
    n_col = math.ceil(n_ms / n_lines)
    # for histogram all events
    fig_tr_amplitude, axes_tr_amplitude = plt.subplots(nrows=n_lines, ncols=n_col,
                                                       gridspec_kw={'width_ratios': [1] * n_col,
                                                                    'height_ratios': [1] * n_lines},
                                                       figsize=(30, 20))
    fig_tr_amplitude.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig_tr_amplitude.patch.set_facecolor(background_color)
    if n_lines + n_col == 2:
        axes_tr_amplitude = axes_tr_amplitude
    else:
        axes_tr_amplitude = axes_tr_amplitude.flatten()

    # figure for the psth
    fig_tr_amplitude_avg, axes_tr_amplitude_avg = plt.subplots(nrows=n_lines, ncols=n_col,
                                                               gridspec_kw={'width_ratios': [1] * n_col,
                                                                            'height_ratios': [1] * n_lines},
                                                               figsize=(30, 20))
    fig_tr_amplitude_avg.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig_tr_amplitude_avg.patch.set_facecolor(background_color)
    axes_tr_amplitude_avg = axes_tr_amplitude_avg.flatten()

    for ax_index, ax in enumerate(axes_tr_amplitude):
        ax.set_facecolor(background_color)
        axes_tr_amplitude_avg[ax_index].set_facecolor(background_color)

    real_ms_index = 0
    for ms_index, ms in enumerate(ms_to_analyse):
        if ms.spike_struct.spike_nums_dur is None:
            continue

        age_str = "p" + str(ms.age)
        if ms.raw_traces is None:
            ms.load_tiff_movie_in_memory()
            ms.raw_traces = ms.build_raw_traces_from_movie()
        raw_traces = ms.raw_traces
        transient_amplitude = []
        transient_amplitude_avg_by_cell = []
        for cell, cell_raster in enumerate(ms.spike_struct.spike_nums_dur):
            # DF / F
            raw_trace = raw_traces[cell] / np.mean(raw_traces[cell])
            periods = get_continous_time_periods(cell_raster)
            transient_amplitude_for_this_cell = []
            for period in periods:
                max_value = np.max(raw_trace[period[0]:period[1] + 1])
                transient_amplitude.append(max_value)
                transient_amplitude_for_this_cell.append(max_value)
            if len(transient_amplitude_for_this_cell) > 0:
                transient_amplitude_avg_by_cell.append(np.mean(transient_amplitude_for_this_cell))
        if age_str not in transient_amplitude_by_age:
            transient_amplitude_by_age[age_str] = []
        transient_amplitude_by_age[age_str].extend(transient_amplitude)
        if age_str not in transient_amplitude_by_age_avg_by_cell:
            transient_amplitude_by_age_avg_by_cell[age_str] = []
        transient_amplitude_by_age_avg_by_cell[age_str].extend(transient_amplitude_avg_by_cell)

        plot_hist_distribution(distribution_data=transient_amplitude,
                               description=f"{ms.description}_hist_transient_amplitude",
                               param=param,
                               path_results=path_results,
                               tight_x_range=True,
                               twice_more_bins=True,
                               ax_to_use=axes_tr_amplitude[real_ms_index],
                               color_to_use=colors[real_ms_index % len(colors)],
                               legend_str=f"{ms.description}",
                               xlabel="Amplitude (DF/F) of transients", save_formats=save_formats)

        plot_hist_distribution(distribution_data=transient_amplitude_avg_by_cell,
                               description=f"{ms.description}_hist_transient_amplitude_by_cell",
                               param=param,
                               path_results=path_results,
                               tight_x_range=True,
                               twice_more_bins=True,
                               ax_to_use=axes_tr_amplitude_avg[real_ms_index],
                               color_to_use=colors[real_ms_index % len(colors)],
                               legend_str=f"{ms.description}",
                               xlabel="Avg amplitude (DF/F) of transients by cell", save_formats=save_formats)
        real_ms_index += 1

    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig_tr_amplitude.savefig(f'{path_results}/_hist_transient_amplitude'
                                 f'_{param.time_str}.{save_format}',
                                 format=f"{save_format}",
                                 facecolor=fig_tr_amplitude.get_facecolor())
        fig_tr_amplitude_avg.savefig(f'{path_results}/hist_transient_amplitude_by_cell'
                                     f'_{param.time_str}.{save_format}',
                                     format=f"{save_format}",
                                     facecolor=fig_tr_amplitude_avg.get_facecolor())

    box_plot_data_by_age(data_dict=transient_amplitude_by_age, title="", filename="transient_amplitude_by_age",
                         path_results=path_results, with_scatters=False,
                         y_label="Amplitude (DF/F) of transients", colors=colors, param=param,
                         save_formats=save_formats)

    box_plot_data_by_age(data_dict=transient_amplitude_by_age_avg_by_cell, title="",
                         path_results=path_results, with_scatters=True,
                         filename="transient_amplitude_by_age_avg_by_cell",
                         y_label="Avg amplitude (DF/F) of transients by cell", colors=colors,
                         param=param, save_formats=save_formats)


def merge_coords_map(ms, param):
    if ms.description.lower() not in ["p6_19_02_18_a000", "p11_19_04_30_a001"]:
        print(f"merge_coords_map not available for {ms.description}")
        return
    # specific function to match segmentation from Fiji to the one with Caiman on matlab
    caiman_mat_file_name = os.path.join(param.path_data, f"p{ms.age}", ms.description.lower(),
                                        "caiman_matlab", f"{ms.description.lower()}_CellDetect_new.mat")
    data = hdf5storage.loadmat(caiman_mat_file_name)
    coord = data["ContoursAll"][0]
    coord_obj = CoordClass(coord=coord, nb_col=200,
                           nb_lines=200, from_fiji=False)

    caiman_suite2p_mapping = ms.coord_obj.match_cells_indices(coord_obj, param=param,
                                                              plot_title_opt=f"{ms.description}_fiji_vs_caiman")
    np.save(os.path.join(param.path_results, f"{ms.description}_fiji_vs_caiman.npy"),
            caiman_suite2p_mapping)


def cluster_using_grid(ms, param):
    if ms.tiff_movie is None:
        ms.load_tiff_movie_in_memory()

    # in pixels
    square_size = 160
    if ms.tiff_movie.shape[1] < square_size or ms.tiff_movie.shape[2] < square_size:
        print(f"{ms.description} movie shape is too small {ms.tiff_movie.shape}")
        return
    tiff_movie = np.copy(ms.tiff_movie).astype("float")
    n_frames = tiff_movie.shape[0]
    n_lines = 16
    if square_size % n_lines != 0:
        print(f"{square_size} should be divisible by {n_lines}")
    box_size = square_size // n_lines
    n_grids = n_lines * n_lines
    grid_traces = np.zeros((n_lines, n_lines, n_frames))
    remove_cells = True
    # create option to remove cells from the movie
    # before building grid traces
    # putting to np.nan all pixels of a cell

    if remove_cells:
        mask_movie = np.zeros((tiff_movie.shape[1], tiff_movie.shape[2]), dtype="bool")
        for cell in np.arange(ms.coord_obj.n_cells):
            if cell % 50 == 0:
                print(f"Removing mask of cell {cell}")
            cell_mask = ms.coord_obj.get_cell_mask(cell, (tiff_movie.shape[1], tiff_movie.shape[2]))
            mask_movie[cell_mask] = True
            tiff_movie[:, mask_movie] = np.nan
    # traces = np.zeros((n_frames, n_lines*n_lines))
    for x_box in np.arange(n_lines):
        for y_box in np.arange(n_lines):
            mask = np.zeros((tiff_movie.shape[1], tiff_movie.shape[2]), dtype="bool")
            mask[y_box * box_size:(y_box + 1) * box_size, x_box * box_size:(x_box + 1) * box_size] = True
            grid_traces[y_box, x_box, :] = np.nanmean(tiff_movie[:, mask], axis=1)
            # print(f"{y_box} {x_box} np.sum(grid) {np.sum(grid_traces[y_box, x_box])}")
            # normalizing the trace
            # grid_traces[y_box, x_box] = (grid_traces[y_box, x_box] - np.mean(grid_traces[y_box, x_box])) / \
            #                             np.std(grid_traces[y_box, x_box])

    traces = np.reshape(grid_traces, (n_lines * n_lines, n_frames))
    print(f"traces.shape {traces.shape}")

    cellsinpeak, sce_loc = detect_sce_on_traces(traces, use_speed=False,
                                                speed_threshold=None, sce_n_cells_threshold=5,
                                                sce_min_distance=4, use_median_norm=True,
                                                use_bleaching_correction=False,
                                                use_savitzky_golay_filt=True)
    sce_times_bool = np.zeros(n_frames, dtype="bool")
    sce_times_bool[sce_loc] = True
    SCE_times = get_continous_time_periods(sce_times_bool)
    sce_times_numbers = np.ones(n_frames, dtype="int16")
    sce_times_numbers *= -1
    for period_index, period in enumerate(SCE_times):
        # if period[0] == period[1]:
        #     print("both periods are equals")
        sce_times_numbers[period[0]:period[1] + 1] = period_index
    ms.sce_bool = sce_times_bool
    ms.sce_times_numbers = sce_times_numbers
    ms.SCE_times = SCE_times

    try_hdbscan(cells_in_sce=cellsinpeak, param=param, use_co_var=False,
                data_descr=f"{ms.description}_covar_grid_on_traces")

    # compute_and_plot_clusters_raster_kmean_version(labels=np.arange(n_grids),
    #                                                activity_threshold=None,
    #                                                range_n_clusters_k_mean=np.arange(3, 5),
    #                                                n_surrogate_k_mean=20,
    #                                                with_shuffling=False,
    #                                                spike_nums_to_use=None,
    #                                                cellsinpeak=cellsinpeak,
    #                                                data_descr=f"{ms.description}_k_mean_on_traces_grid",
    #                                                param=param,
    #                                                sliding_window_duration=1,
    #                                                SCE_times=SCE_times,
    #                                                sce_times_numbers=sce_times_numbers,
    #                                                sce_times_bool=sce_times_bool,
    #                                                perc_threshold=95,
    #                                                n_surrogate_activity_threshold=
    #                                                100,
    #                                                debug_mode=True,
    #                                                fct_to_keep_best_silhouettes=np.median,
    #                                                with_cells_in_cluster_seq_sorted=
    #                                                False,
    #                                                keep_only_the_best=False)


def get_pair_wise_pearson_correlation_distribution(raster_dur):
    """
    Distribution of pair-wise pearson correlation of all cells
    :param raster_dur:
    :return:
    """
    n_cells = raster_dur.shape[0]

    corr_distribution = []
    # count_high_p_value = 0
    # total_count = 0
    for cell in np.arange(n_cells - 1):
        if np.sum(raster_dur[cell]) == 0:
            continue
        for cell_bis in np.arange(cell + 1, n_cells):
            if np.sum(raster_dur[cell_bis]) == 0:
                continue
            corr, p_value = scipy_stats.pearsonr(raster_dur[cell].astype(float),
                                                 raster_dur[cell_bis].astype(float))
            # if p_value > 0.05:
            #     count_high_p_value += 1
            # total_count += 1
            # if p_value < 0.05:
            corr_distribution.append(corr)
    # print(f"total_count {total_count}, count_high_p_value {count_high_p_value}")
    return corr_distribution


def get_pair_wise_wasserstein_distance_distribution(raster_dur):
    """
       Distribution of pair-wise Wasserstein distance of all cells
       :param raster_dur:
       :return:
       """
    n_cells = raster_dur.shape[0]

    corr_distribution = []
    # count_high_p_value = 0
    # total_count = 0
    for cell in np.arange(n_cells - 1):
        if np.sum(raster_dur[cell]) == 0:
            continue
        for cell_bis in np.arange(cell + 1, n_cells):
            if np.sum(raster_dur[cell_bis]) == 0:
                continue
            if np.sum(raster_dur[cell]) == 0 or np.sum(raster_dur[cell_bis]) == 0:
                continue
            first_cell_raster = raster_dur[cell] / np.sum(raster_dur[cell])
            first_cell_bis_raster = raster_dur[cell_bis] / np.sum(raster_dur[cell_bis])
            n_frames = raster_dur.shape[1]
            wass_dist = scipy_stats.wasserstein_distance(np.arange(n_frames), np.arange(n_frames),
                                                         u_weights=first_cell_raster, v_weights=first_cell_bis_raster)
            # if p_value > 0.05:
            #     count_high_p_value += 1
            # total_count += 1
            # if p_value < 0.05:
            corr_distribution.append(wass_dist)
    # print(f"total_count {total_count}, count_high_p_value {count_high_p_value}")
    return corr_distribution


def get_pair_wise_hamming_distance_distribution(raster_dur):
    """
          Distribution of pair-wise Hamming distance of all cells
          :param raster_dur:
          :return:
          """
    n_cells = raster_dur.shape[0]

    corr_distribution = []
    # count_high_p_value = 0
    # total_count = 0
    for cell in np.arange(n_cells - 1):
        if np.sum(raster_dur[cell]) == 0:
            continue
        for cell_bis in np.arange(cell + 1, n_cells):
            if np.sum(raster_dur[cell_bis]) == 0:
                continue
            if np.sum(raster_dur[cell]) == 0 or np.sum(raster_dur[cell_bis]) == 0:
                continue
            n_frames = raster_dur.shape[1]
            hamm_dist = sci_sp_dist.hamming(raster_dur[cell], raster_dur[cell_bis])
            # if p_value > 0.05:
            #     count_high_p_value += 1
            # total_count += 1
            # if p_value < 0.05:
            corr_distribution.append(hamm_dist)
    # print(f"total_count {total_count}, count_high_p_value {count_high_p_value}")
    return corr_distribution


def plot_jsd_correlation(ms_to_analyse, param, metric, n_surrogate=50, save_formats="pdf"):
    print(f"Starting to plot distribution of pair-wise {metric}")
    possible_metrics = ["Pearson_correlation", "Wasserstein_distance", "Hamming_distance"]
    if metric not in possible_metrics:
        metric = "Pearson_correlation"
        raise Exception("This metric is not avalaible, keep going using Pearson correlation")
    if metric is None:
        metric = "Pearson_correlation"
        raise Exception("Metric was not specified, keep going using Pearson correlation")
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    just_plot_pearson_correlation_distribution = False

    n_ms = 0
    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums_dur is None:
            continue
        n_ms += 1

    background_color = "black"
    max_n_lines = 5
    n_lines = n_ms if n_ms <= max_n_lines else max_n_lines
    n_col = math.ceil(n_ms / n_lines)
    # for histogram all events
    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col,
                                          'height_ratios': [1] * n_lines},
                             figsize=(30, 25))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)
    if n_lines + n_col == 2:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        axes[ax_index].set_facecolor(background_color)

    jsd_by_age = dict()
    distrib_by_ms = dict()
    n_sessions_dict = dict()
    max_value_distribution = 0
    min_value_distribution = None
    n_ms_so_far = 0
    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums_dur is None:
            continue
        # ms.spike_struct.spike_nums_dur = ms.spike_struct.spike_nums_dur[:50, :12500]
        n_times = ms.spike_struct.spike_nums.shape[1]
        age_str = "p" + str(ms.age)
        if age_str not in jsd_by_age:
            jsd_by_age[age_str] = []
            n_sessions_dict[age_str] = set()
        # if ms.description not in distrib_by_ms:
        #     distrib_by_ms[ms.description] = []

        start_time = time.time()
        if metric == "Pearson_correlation":
            corr_ms_distribution = get_pair_wise_pearson_correlation_distribution(ms.spike_struct.spike_nums)
        if metric == "Wasserstein_distance":
            corr_ms_distribution = get_pair_wise_wasserstein_distance_distribution(ms.spike_struct.spike_nums_dur)
        if metric == "Hamming_distance":
            corr_ms_distribution = get_pair_wise_hamming_distance_distribution(ms.spike_struct.spike_nums_dur)

        distrib_by_ms[ms.description] = corr_ms_distribution
        max_value_distribution = max(np.max(corr_ms_distribution), max_value_distribution)
        min_value_distribution = np.min(corr_ms_distribution) if min_value_distribution is None \
            else min(np.min(corr_ms_distribution), min_value_distribution)
        stop_time = time.time()
        print(f"Time to get pair-wise {metric} for {ms.description}: "
              f"{np.round(stop_time - start_time, 3)} s")
        if just_plot_pearson_correlation_distribution:
            jsd_by_age[age_str].extend(corr_ms_distribution)
        else:
            # we measure the pair-wise correlation for each surrogate (rolling the raster)
            # corr_surrogate_distribution = []
            # for i in np.arange(n_surrogate):
            #     copy_raster = np.copy(ms.spike_struct.spike_nums_dur)
            #     for n, neuron_spikes in enumerate(copy_raster):
            #         # roll the data to a random displace number
            #         copy_raster[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
            #     corr = get_pair_wise_pearson_correlation_distribution(ms.spike_struct.spike_nums_dur)
            #     corr_surrogate_distribution.extend(corr)

            # then we want a probability vector for each distribution
            # with 20 bins, meaning 0.05 by bin
            # n_bins = 2000
            # # hist_ms, bin_edges = np.histogram(corr_ms_distribution, bins=n_bins, range=(-1, 1), density=True)
            # plot_hist_distribution(distribution_data=corr_ms_distribution, n_bins=n_bins,
            #                        description=f"{ms.description}_pair_wise_correlation",
            #                        legend_str=f"{ms.description}", tight_x_range=True,
            #                        ax_to_use=axes[n_ms_so_far],
            #                        color_to_use=colors[n_ms_so_far % len(colors)],
            #                        xlabel=f"Pair-wise correlation (pearson)",
            #                        param=param, density=True, use_log=True)

            # probs_ms = hist_ms / float(hist_ms.sum())
            # hist_surrogate, bin_edges = np.histogram(corr_surrogate_distribution, bins=n_bins,
            #                                          range=(0, 1))
            # probs_surrogate = hist_surrogate / float(hist_surrogate.sum())

            # jsd = jensenshannon(probs_ms, probs_surrogate, base=2) #
            # print(f"{ms.description}: jsd: {jsd}")
            # jsd_by_age[age_str].append(jsd)
            n_ms_so_far += 1
        # we remove the a00* at the end of description, to know if it's the same animal
        n_sessions_dict[age_str].add(ms.description[:-4])

    n_ms_so_far = 0
    if metric == "Pearson_correlation":
        xlabel = "Pair-wise correlation (pearson)"
        file_name = "Pair-wise_correlation_pearson"
    elif metric == "Wasserstein_distance":
        xlabel = "Pair-wise distance (Wasserstein)"
        file_name = "Pair-wise_Wassertein_distance"
    elif metric == "Hamming_distance":
        xlabel = "Pair-wise distance (Hamming)"
        file_name = "Pair-wise_Hamming_distance"

    for ms_description, ms_distribution in distrib_by_ms.items():
        # n_bins = min(50, int(np.sqrt(len(ms_distribution))))
        mean_value = np.mean(ms_distribution)
        print(f"{mean_value}")
        # hist_ms, bin_edges = np.histogram(corr_ms_distribution, bins=n_bins, range=(-1, 1), density=True)
        plot_hist_distribution(distribution_data=ms_distribution, values_to_scatter=np.array([mean_value]),
                               labels=["mean"],
                               scatter_shapes=['o'], colors=['white'],
                               description=f"{ms_description}_pair_wise_correlation",
                               legend_str=f"{ms_description}", tight_x_range=True,
                               ax_to_use=axes[n_ms_so_far], twice_more_bins=True,
                               x_range=(min_value_distribution, max_value_distribution),
                               color_to_use=colors[n_ms_so_far % len(colors)], xlabel=xlabel,  # n_bins=n_bins,
                               param=param, density=True, use_log=True)
        n_ms_so_far += 1
    print(f"end loop plot_hist_distribution")
    # to know how many different animal by age
    # for i, n_sessions_set in n_sessions_dict.items():
    #     n_sessions_dict[i] = len(n_sessions_set)

    # if just_plot_pearson_correlation_distribution:
    #     filename = "pair_wise_pearson_correlation"
    # else:
    #     filename = "JSD_by_age"
    #
    # if just_plot_pearson_correlation_distribution:
    #     y_label = "pair-wise pearson correlation"
    # else:
    #     y_label = "average JSD"

    # box_plot_data_by_age(data_dict=jsd_by_age, title="",
    #                      path_results=param.path_results, with_scatters=False,
    #                      filename=filename, scatter_size = 40,
    #                      y_label=y_label, colors=colors,
    #                      param=param, save_formats=save_formats)
    # we plot one distribution for each session with the ratio

    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/{file_name}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())


def plot_transient_frequency(ms_to_analyse, param, colors=None, save_formats="pdf", use_animal_weight=False,
                             with_hist_for_each=False):
    path_results = os.path.join(param.path_results, "transient_frequency")
    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    transient_frequency_by_age = SortedDict()

    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums_dur is None:
            print(f"{ms.description} no spike_nums_dur")
            continue

        # age_str = "p" + str(ms.age)
        label_key = "p" + str(ms.age)
        if use_animal_weight:
            if ms.weight is None:
                continue
            label_key = str(ms.weight) + "-" + label_key
        transient_frequency = []
        n_times = ms.spike_struct.spike_nums_dur.shape[1]
        for cell_raster in ms.spike_struct.spike_nums_dur:
            n_transients = len(get_continous_time_periods(cell_raster))
            transient_frequency.append(n_transients / (n_times / ms.sampling_rate))
        if label_key not in transient_frequency_by_age:
            transient_frequency_by_age[label_key] = []
        transient_frequency_by_age[label_key].extend(transient_frequency)
        if with_hist_for_each:
            plot_hist_distribution(distribution_data=transient_frequency,
                                   description=f"{ms.description}_hist_transients_frequency",
                                   param=param,
                                   path_results=path_results,
                                   tight_x_range=True,
                                   twice_more_bins=True,
                                   xlabel="Frequency of transients (Hz)", save_formats=save_formats)
    box_plot_data_by_age(data_dict=transient_frequency_by_age, title="", filename="transients_frequency_by_age",
                         path_results=path_results, with_scatters=False,
                         y_label="Frequency of transients (Hz)", colors=colors, param=param, save_formats=save_formats)


def plot_transient_durations_normalized_by_amplitude(ms_to_analyse, param, colors=None, save_formats="pdf"):
    path_results = os.path.join(param.path_results, "transient_duration_normalized_by_amplitude")
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    spike_durations_by_age = dict()
    spike_durations_by_age_avg_by_cell = dict()

    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums_dur is None:
            continue

        if ms.raw_traces is None:
            ms.load_tiff_movie_in_memory()
            ms.raw_traces = ms.build_raw_traces_from_movie()
        raw_traces = ms.raw_traces

        age_str = "p" + str(ms.age)

        # list of length n_cells, each element being a list of int representing the duration of the transient
        # in frames
        spike_durations = tools_misc.get_spikes_duration_from_raster_dur(spike_nums_dur=ms.spike_struct.spike_nums_dur)
        distribution_avg_by_cell = []
        distribution_all = []
        if age_str not in spike_durations_by_age:
            spike_durations_by_age[age_str] = []
        if age_str not in spike_durations_by_age_avg_by_cell:
            spike_durations_by_age_avg_by_cell[age_str] = []
        raster_dur = ms.spike_struct.spike_nums_dur
        for cell in np.arange(len(raster_dur)):
            # DF / F
            raw_trace = raw_traces[cell] / np.mean(raw_traces[cell])
            cell_raster_dur = raster_dur[cell]
            periods = get_continous_time_periods(cell_raster_dur)
            durations_for_a_cell = []
            for period in periods:
                amplitude = np.max(raw_trace[period[0]:period[1] + 1])
                duration = period[1] - period[0]
                duration = (duration / ms.sampling_rate) / amplitude
                distribution_all.append(duration)
                durations_for_a_cell.append(duration)
            if len(durations_for_a_cell) > 0:
                distribution_avg_by_cell.append(np.mean(durations_for_a_cell))
        spike_durations_by_age[age_str].extend(distribution_all)
        spike_durations_by_age_avg_by_cell[age_str].extend(distribution_avg_by_cell)

        plot_hist_distribution(distribution_data=distribution_all,
                               description=f"{ms.description}_hist_rising_time_durations_normalized_by_amplitude",
                               param=param,
                               path_results=path_results,
                               tight_x_range=True,
                               twice_more_bins=False,
                               xlabel="Duration of rising time (s)", save_formats=save_formats)
        plot_hist_distribution(distribution_data=distribution_avg_by_cell,
                               description=f"{ms.description}_hist_rising_time_duration_avg_by_cell",
                               param=param,
                               path_results=path_results,
                               tight_x_range=True,
                               twice_more_bins=True,
                               xlabel="Average duration of rising time for each cell (s)",
                               save_formats=save_formats)

    box_plot_data_by_age(data_dict=spike_durations_by_age, title="",
                         filename="rising_time_duration_by_age_normalized_by_amplitude",
                         y_label="Duration of rising time (s)", colors=colors,
                         path_results=path_results, with_scatters=False,
                         param=param, save_formats=save_formats)
    box_plot_data_by_age(data_dict=spike_durations_by_age_avg_by_cell, title="",
                         filename="rising_time_durations_by_age_avg_by_cell_normalized_by_amplitude",
                         path_results=path_results,
                         y_label="Average duration of rising time for each cell (s)",
                         colors=colors,
                         param=param, save_formats=save_formats)


def plot_transient_durations(ms_to_analyse, param, colors=None, save_formats="pdf"):
    path_results = os.path.join(param.path_results, "transient_duration")
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    spike_durations_by_age = dict()
    spike_durations_by_age_avg_by_cell = dict()

    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums_dur is None:
            continue
        # print(f"plot_transient_durations: {ms.description}")
        age_str = "p" + str(ms.age)
        # list of length n_cells, each element being a list of int representing the duration of the transient
        # in frames
        spike_durations = tools_misc.get_spikes_duration_from_raster_dur(spike_nums_dur=ms.spike_struct.spike_nums_dur)
        distribution_avg_by_cell = []
        distribution_all = []
        if age_str not in spike_durations_by_age:
            spike_durations_by_age[age_str] = []
        if age_str not in spike_durations_by_age_avg_by_cell:
            spike_durations_by_age_avg_by_cell[age_str] = []
        for spike_durations_by_cell in spike_durations:
            if len(spike_durations_by_cell) == 0:
                continue
            spike_durations_by_cell = [d / ms.sampling_rate for d in spike_durations_by_cell]
            distribution_all.extend(spike_durations_by_cell)
            spike_durations_by_age[age_str].extend(spike_durations_by_cell)

            distribution_avg_by_cell.append(np.mean(spike_durations_by_cell))
            spike_durations_by_age_avg_by_cell[age_str].append(np.mean(spike_durations_by_cell))

        plot_hist_distribution(distribution_data=distribution_all,
                               description=f"{ms.description}_hist_rising_time_durations",
                               param=param,
                               path_results=path_results,
                               tight_x_range=True,
                               twice_more_bins=False,
                               xlabel="Duration of rising time (s)", save_formats=save_formats)
        plot_hist_distribution(distribution_data=distribution_avg_by_cell,
                               description=f"{ms.description}_hist_rising_time_duration_avg_by_cell",
                               param=param,
                               path_results=path_results,
                               tight_x_range=True,
                               twice_more_bins=True,
                               xlabel="Average duration of rising time for each cell (s)", save_formats=save_formats)

    box_plot_data_by_age(data_dict=spike_durations_by_age, title="", filename="rising_time_duration_by_age",
                         y_label="Duration of rising time (s)", colors=colors, with_scatters=False,
                         path_results=path_results,
                         param=param, save_formats=save_formats)
    box_plot_data_by_age(data_dict=spike_durations_by_age_avg_by_cell, title="",
                         filename="rising_time_durations_by_age_avg_by_cell",
                         path_results=path_results,
                         y_label="Average duration of rising time for each cell (s)", colors=colors,
                         param=param, save_formats=save_formats)


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


def plot_hist_distribution(distribution_data, description, param, values_to_scatter=None,
                           n_bins=None, use_log=False, x_range=None,
                           labels=None, scatter_shapes=None, colors=None, tight_x_range=False,
                           twice_more_bins=False, background_color="black", labels_color="white",
                           xlabel="", ylabel=None, path_results=None, save_formats="pdf",
                           ax_to_use=None, color_to_use=None, legend_str="", density=False):
    """
    Plot a distribution in the form of an histogram, with option for adding some scatter values
    :param distribution_data:
    :param description:
    :param param:
    :param values_to_scatter:
    :param labels:
    :param scatter_shapes:
    :param colors:
    :param tight_x_range:
    :param twice_more_bins:
    :param xlabel:
    :param ylabel:
    :param save_formats:
    :return:
    """
    distribution = np.array(distribution_data)
    if color_to_use is None:
        hist_color = "blue"
    else:
        hist_color = color_to_use

    if x_range is not None:
        min_range = x_range[0]
        max_range = x_range[1]
    elif tight_x_range:
        max_range = np.max(distribution)
        min_range = np.min(distribution)
    else:
        max_range = 100
        min_range = 0
    weights = (np.ones_like(distribution) / (len(distribution))) * 100
    # weights=None

    if ax_to_use is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(12, 12))
        ax1.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
    else:
        ax1 = ax_to_use
    if n_bins is not None:
        bins = n_bins
    else:
        bins = int(np.sqrt(len(distribution)))
        if twice_more_bins:
            bins *= 2

    if bins > 100:
        edge_color = hist_color
    else:
        edge_color = "white"

    hist_plt, edges_plt, patches_plt = ax1.hist(distribution, bins=bins, range=(min_range, max_range),
                                                facecolor=hist_color, log=use_log,
                                                edgecolor=edge_color, label=f"{legend_str}",
                                                weights=weights, density=density)
    if values_to_scatter is not None:
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
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20, label=labels[i])
            else:
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20)
    ax1.legend()

    if tight_x_range:
        ax1.set_xlim(min_range, max_range)
    else:
        ax1.set_xlim(0, 100)
        xticks = np.arange(0, 110, 10)

        ax1.set_xticks(xticks)
        # sce clusters labels
        ax1.set_xticklabels(xticks)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)
    # TO remove the ticks but not the labels
    # ax1.xaxis.set_ticks_position('none')

    if ylabel is None:
        ax1.set_ylabel("Distribution (%)", fontsize=20, labelpad=20)
    else:
        ax1.set_ylabel(ylabel, fontsize=20, labelpad=20)
    ax1.set_xlabel(xlabel, fontsize=20, labelpad=20)

    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)

    if ax_to_use is None:
        # padding between ticks label and  label axis
        # ax1.tick_params(axis='both', which='major', pad=15)
        fig.tight_layout()

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        if path_results is None:
            path_results = param.path_results
        for save_format in save_formats:
            fig.savefig(f'{path_results}/{description}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())

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
                    if file_name.endswith(".npy"):
                        if "params" in file_name.lower():
                            ops = np.load(os.path.join(dir_path, file_name), allow_pickle=True)
                            ops = ops.item()
                            data_dict[parent_dir]["xshifts"] = ops['xoff']
                            data_dict[parent_dir]["yshifts"] = ops['yoff']
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
            try:
                start_time = time.time()
                tiff_movie = ScanImageTiffReader(os.path.join(value["dirpath"], value["tiff_file"])).data()
                stop_time = time.time()
                print(f"Time for loading movie {value['tiff_file']} with scan_image_tiff: "
                      f"{np.round(stop_time - start_time, 3)} s")
            except Exception:
                start_time = time.time()
                im = PIL.Image.open(os.path.join(value["dirpath"], value["tiff_file"]))
                n_frames = len(list(ImageSequence.Iterator(im)))
                dim_x, dim_y = np.array(im).shape
                print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")
                tiff_movie = np.zeros((n_frames, dim_x, dim_y), dtype="uint16")
                for frame, page in enumerate(ImageSequence.Iterator(im)):
                    tiff_movie[frame] = np.array(page)
                stop_time = time.time()
                print(f"Time for loading movie {value['tiff_file']}: "
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
        n_frames = len(roi)

        # roi = signal.detrend(roi)
        # normalization
        if np.nanstd(roi) != 0:
            roi = (roi - np.mean(roi)) / np.std(roi)
        else:
            roi = (roi - np.mean(roi))

        # mvt = np.abs(value["xshifts"]) + np.abs(value["yshifts"])
        mvt = np.sqrt((value["xshifts"] ** 2) + (value["yshifts"] ** 2))
        np.save(os.path.join(value["dirpath"], f"{value['id']}_shift.npy"), mvt)
        mvt -= np.nanmedian(mvt)
        mvt = np.abs(mvt)
        # mvt = signal.detrend(mvt)
        # if value["id"] == "p7_19_03_05_a000":
        #     print(f"{value['id']}: ValueError mvt: {mvt}")
        #     print(f"{value['id']}: ValueError np.mean(mvt): {np.nanmean(mvt)}")

        # normalization
        if np.nanstd(mvt) != 0:
            mvt = (mvt - np.nanmean(mvt)) / np.nanstd(mvt)
        else:
            mvt = (mvt - np.nanmean(mvt))
        mvt = mvt - np.abs(np.min(roi)) - np.nanmax(mvt)

        # plotting mvt vs ROI
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex='col',
                                gridspec_kw={'height_ratios': [1], 'width_ratios': [1]},
                                figsize=(15, 5))

        ax1.set_facecolor("black")

        ax1.plot(np.arange(n_frames), roi, color="red", lw=1, label=f"ROI", zorder=10)
        ax1.plot(np.arange(n_frames), mvt, color="cornflowerblue", lw=1, label=f"SHIFT", zorder=10)
        min_value = np.nanmin(mvt)
        max_value = np.max(roi)
        interval = 200

        ax1.vlines(np.arange(interval, n_frames, interval), min_value, max_value,
                   color="white", linewidth=0.2,
                   linestyles="dashed", zorder=5)
        try:
            ax1.set_ylim(min_value, max_value)
        except ValueError:
            print(f"{value['id']}: ValueError roi: {roi}")
            print(f"{value['id']}: ValueError mvt: {mvt}")
            print(f"{value['id']}: ValueError mvt: {value['xshifts']}")
            print(f"{value['id']}: ValueError mvt: {value['yshifts']}")
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
        roi_bin = non_lag_roi.reshape((len(non_lag_roi) // bin_size, bin_size)).sum(axis=1)
        mvt_bin = non_lag_mvt.reshape((len(non_lag_mvt) // bin_size, bin_size)).sum(axis=1)
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


def compute_stat_about_seq_with_slope(files_path, param,
                                      save_formats=["pdf"],
                                      color_option="manual", cmap_name="Reds"):
    """

    :param files_path:
    :param param:
    :param save_formats:
    :param color_option:
    :param cmap_name:
    :return:
    """
    plot_slopes_by_ages = False
    plot_seq_contour_map = False
    plot_seq_with_rep_by_age = False
    plot_synchronies_on_raster = False
    plot_3_kinds_of_slopes = True

    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    brewer_colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                     '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                     '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                     '#74add1', '#4575b4', '#313695']

    file_names = []
    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(files_path):
        file_names.extend(local_filenames)
        break

    # dict1: key age (int) value dict2
    # dict2: key category (int, from 1 to 4), value dict3
    # dict3: key length seq (int), value dict4
    # dict4: key repetitions (int), value nb of seq with this length and this repetition
    # data_dict = SortedDict()

    # dict to keep the number of rep of each seq depending on its size
    # dict1: key age (int) value dict2
    # dict2: key ms_description (str) value dict3
    # dict3: key is a tuple of int representing the cell indices of the seq, value dict4
    # dict4: key is the slope, value is a list with first_cell_spike_time, last_cell_spike_time, range_around and
    # indices of cells that fire in the sequence

    seq_dict = SortedDict()

    for file_name in file_names:
        file_name_original = file_name
        file_name = file_name.lower()
        if not file_name.startswith("p"):
            if not file_name.startswith("significant_sorting_results_with_timestamps_with_slope_"):
                continue

            file_name = file_name[len("significant_sorting_results_with_timestamps_with_slope_"):]

        index_ = file_name.find("_")
        if index_ < 1:
            continue
        age_file = int(file_name[1:index_])
        index_txt = file_name.find(".txt")
        ms_description = file_name[:index_txt]
        age = age_file
        # age = ages_groups[age_file]
        # nb_ms_by_age[age] = nb_ms_by_age.get(age, 0) + 1
        # print(f"age {age}")
        if age not in seq_dict:
            seq_dict[age] = dict()

        if ms_description not in seq_dict[age]:
            seq_dict[age][ms_description] = dict()

        current_cell_seq = None
        with open(files_path + file_name_original, "r", encoding='UTF-8') as file:
            for nb_line, line in enumerate(file):
                if "best_order" in line:
                    line_list = line.split(':')
                    cells_best_order = np.array(list(map(int, (line_list[1].split()))))
                    seq_dict[age][ms_description]["cells_best_order"] = cells_best_order
                    continue
                elif "shortest_paths" in line:
                    continue
                else:
                    if '#' in line:
                        current_cell_seq = tuple(map(int, (line[1:].split())))
                        seq_dict[age][ms_description][current_cell_seq] = dict()
                    else:
                        values_list = list(map(int, (line.split())))
                        slope_value = values_list[2]
                        if slope_value not in seq_dict[age][ms_description][current_cell_seq]:
                            seq_dict[age][ms_description][current_cell_seq][slope_value] = []
                        seq_dict[age][ms_description][current_cell_seq][slope_value].append(values_list[:2] + \
                                                                                            values_list[3:])
    min_slope = None
    max_slope = None
    # keep the count of each slope by age
    slopes_by_age = dict()
    n_sessions_dict = dict()
    # firt we want to do some stat about slopes
    for age, dict_ms in seq_dict.items():
        slopes_by_age[str(age)] = []
        for ms_description, dict_cells_seq in dict_ms.items():
            n_sessions_dict[str(age)] = n_sessions_dict.get(str(age), 0) + 1
            for cells_seq, dict_slope in dict_cells_seq.items():
                if cells_seq == "cells_best_order":
                    continue
                for slope, values in dict_slope.items():
                    if (min_slope is None) or (slope < min_slope):
                        min_slope = slope
                    if (max_slope is None) or (slope > max_slope):
                        max_slope = slope
                    slopes_by_age[str(age)].extend([slope] * len(values))
                #     print(f"{ms_description} {len(cells_seq)} cells, slope {slope}: {len(values)}")
                # print("")

    if plot_slopes_by_ages:
        box_plot_data_by_age(data_dict=slopes_by_age, title="",
                             filename=f"seq_slopes_by_age",
                             y_label=f"Seq slopes",
                             colors=brewer_colors, with_scatters=True,
                             n_sessions_dict=n_sessions_dict,
                             path_results=param.path_results, scatter_size=50,
                             with_y_jitter=0.5,
                             param=param, save_formats=save_formats)

    if plot_synchronies_on_raster or plot_seq_contour_map or plot_3_kinds_of_slopes:
        ms_str_to_load = []
        for age, dict_ms in seq_dict.items():
            for ms_description in dict_ms.keys():
                # if "p41_19_04_30_a000" == ms_description.lower():
                ms_str_to_load.append(ms_description.lower() + "_ms")
            # break
        load_traces = plot_3_kinds_of_slopes
        ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_str_to_load, param=param,
                                                load_traces=load_traces, load_abf=False)

    if plot_synchronies_on_raster:
        for age, dict_ms in seq_dict.items():
            for ms_description, dict_cells_seq in dict_ms.items():
                if ms_description.lower() + "_ms" not in ms_str_to_ms_dict:
                    continue
                # dict that takes for a key a tuple of int representing 2 cells, and as value a list of tuple of 2 float
                # representing the 2 extremities of a line between those 2 cells
                lines_to_display = dict()
                cells_to_highlight = []
                cells_to_highlight_colors = []
                cells_added_for_span = []
                range_around_slope_in_frames = 0
                ms = ms_str_to_ms_dict[ms_description.lower() + "_ms"]
                raster_dur = ms.spike_struct.spike_nums_dur
                n_frames = raster_dur.shape[1]
                n_cells = raster_dur.shape[0]
                cells_best_order = dict_cells_seq["cells_best_order"]
                # print(f"n_cells {n_cells}, len(cells_best_order) "
                #       f"{len(seq_dict[age][ms_description]['cells_best_order'])}")
                frames_to_remove = np.ones(n_frames, dtype="bool")
                for cells_seq, dict_slope in dict_cells_seq.items():
                    if cells_seq == "cells_best_order":
                        continue
                    # print(f"cells_seq {cells_seq}")
                    # first_cell = np.where(cells_best_order == cells_seq[0])[0][0]
                    # last_cell = np.where(cells_best_order == cells_seq[-1])[0][0]
                    first_cell = cells_seq[0]
                    last_cell = cells_seq[-1]
                    cells_pair_tuple = (first_cell, last_cell)
                    for slope, values_list in dict_slope.items():
                        if slope == 0:
                            for values in values_list:
                                if cells_pair_tuple not in lines_to_display:
                                    lines_to_display[cells_pair_tuple] = []
                                first_cell_spike_time = values[0]
                                last_cell_spike_time = values[1]
                                range_around = values[2]
                                range_around_slope_in_frames = range_around
                                lines_to_display[cells_pair_tuple].append((first_cell_spike_time, last_cell_spike_time))
                                frames_to_remove[max(0, first_cell_spike_time - range_around):
                                                 min(n_frames, first_cell_spike_time + range_around + 1)] = False
                                if cells_pair_tuple not in cells_added_for_span:
                                    cells_to_highlight.extend(list(range(cells_pair_tuple[0],
                                                                         cells_pair_tuple[1])))
                                    cells_to_highlight_colors.extend([brewer_colors[len(cells_added_for_span)
                                                                                    % len(brewer_colors)]] *
                                                                     (cells_pair_tuple[1] - cells_pair_tuple[0]))
                                    cells_added_for_span.append(cells_pair_tuple)
                raster_dur[:, frames_to_remove] = 0
                plot_spikes_raster(spike_nums=raster_dur[np.array(cells_best_order)], param=param,
                                   spike_train_format=False,
                                   file_name=f"{ms.description}_raster_dur_ordered_with_synchronies",
                                   y_ticks_labels=cells_best_order,
                                   save_raster=True,
                                   show_raster=False,
                                   show_sum_spikes_as_percentage=True,
                                   plot_with_amplitude=False,
                                   # span_cells_to_highlight=span_cells_to_highlight,
                                   # span_cells_to_highlight_colors=span_cells_to_highlight_colors,
                                   # span_area_coords=span_area_coords,
                                   # span_area_colors=span_area_colors,
                                   span_area_only_on_raster=False,
                                   spike_shape='|',
                                   spike_shape_size=0.1,
                                   lines_to_display=lines_to_display,
                                   lines_color="white",
                                   lines_width=0.2,
                                   lines_band=range_around_slope_in_frames,
                                   lines_band_color="white",
                                   save_formats="pdf")

    if plot_3_kinds_of_slopes:
        for age, dict_ms in seq_dict.items():
            for ms_description, dict_cells_seq in dict_ms.items():
                if ms_description.lower() + "_ms" not in ms_str_to_ms_dict:
                    continue
                ms = ms_str_to_ms_dict[ms_description.lower() + "_ms"]
                # if ms.age < 13:
                #     continue
                # working on speed data
                if ms.speed_by_frame is not None:
                    binary_speed = np.zeros(len(ms.speed_by_frame), dtype="int8")
                    binary_speed[ms.speed_by_frame > 0] = 1
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
                    periods_dict = ms.shift_data_dict
                    if periods_dict is not None:
                        print(f"{ms.description}:")
                        for name_period, period in periods_dict.items():
                            span_area_coords.append(get_continous_time_periods(period.astype("int8")))
                            span_area_colors.append(colors[i % len(colors)])
                            print(f"  Period {name_period} -> {colors[i]}")
                            i += 1
                    elif ms.speed_by_frame is not None:
                        span_area_coords = []
                        span_area_colors = []
                        span_area_coords.append(speed_periods)
                        span_area_colors.append("cornflowerblue")
                    else:
                        print(f"no mvt info for {ms.description}")

                # cells_to_highlight = []
                # cells_to_highlight_colors = []
                # cells_added_for_span = []
                raster_dur = ms.spike_struct.spike_nums_dur
                n_frames = raster_dur.shape[1]
                n_cells = raster_dur.shape[0]
                cells_best_order = dict_cells_seq["cells_best_order"]
                for cells_seq, dict_slope in dict_cells_seq.items():
                    if cells_seq == "cells_best_order":
                        continue
                    range_around_slope_in_frames = 0
                    # dict that takes for a key a tuple of int representing 2 cells,
                    # and as value a list of tuple of 2 float
                    # representing the 2 extremities of a line between those 2 cells
                    lines_to_display = dict()
                    first_cell = cells_seq[0]
                    last_cell = cells_seq[-1]
                    cells_pair_tuple = (0, len(cells_seq) - 1)
                    # number of negatives, synchrone and positive slopes
                    slopes_count = np.zeros(3, dtype="uint16")
                    for slope, values_list in dict_slope.items():
                        if slope < -1:
                            slopes_count[0] = slopes_count[0] + len(values_list)
                        elif slope > 1:
                            slopes_count[2] = slopes_count[2] + len(values_list)
                        else:
                            slopes_count[1] = slopes_count[1] + len(values_list)

                        for values in values_list:
                            if cells_pair_tuple not in lines_to_display:
                                lines_to_display[cells_pair_tuple] = []
                            first_cell_spike_time = values[0]
                            last_cell_spike_time = values[1]
                            range_around = values[2]
                            range_around_slope_in_frames = range_around
                            lines_to_display[cells_pair_tuple].append((first_cell_spike_time, last_cell_spike_time))
                    # print(f"{ms.description} {first_cell}-{last_cell}: {slopes_count} {range_around_slope_in_frames}")
                    # if np.all([x >= 3 for x in slopes_count]):
                    if slopes_count[2] > 3:
                        # we want a subplots
                        # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1,
                        #                         gridspec_kw={'height_ratios': [0.3, 0.1, 0.1, 0.3, 0.2],
                        #                                      'width_ratios': [1]},
                        #                         figsize=(10, 6))
                        file_name = f"{ms.description}_raster_dur_{first_cell}-{last_cell}"
                        plot_figure_with_map_and_raster_for_sequences(ms=ms,
                                                                      cells_in_seq=np.array(cells_best_order)
                                                                      [first_cell:last_cell + 1],
                                                                      file_name=file_name,
                                                                      lines_to_display=lines_to_display,
                                                                      range_around_slope_in_frames=
                                                                      range_around_slope_in_frames,
                                                                      span_area_coords=span_area_coords,
                                                                      span_area_colors=span_area_colors,
                                                                      save_formats=save_formats)

    if plot_seq_contour_map:
        for age, dict_ms in seq_dict.items():
            for ms_description, dict_cells_seq in dict_ms.items():
                if ms_description.lower() + "_ms" not in ms_str_to_ms_dict:
                    continue
                ms = ms_str_to_ms_dict[ms_description.lower() + "_ms"]
                cells_best_order = dict_cells_seq["cells_best_order"]
                for cells_seq in dict_cells_seq.keys():
                    if cells_seq == "cells_best_order":
                        continue
                    # color = (100 / 255, 215 / 255, 247 / 255, 1)  # #64D7F7"
                    color = (213 / 255, 38 / 255, 215 / 255, 1)  # #D526D7
                    cells_groups_colors = [color]
                    first_cell = cells_seq[0]
                    last_cell = cells_seq[-1]
                    cells_seq_for_map = np.array(cells_best_order)[first_cell:last_cell + 1]
                    cells_to_color = [cells_seq_for_map]
                    # check if at least a pair of cells intersect
                    at_least_one_intersect = False
                    for cell_index, cell_1 in enumerate(cells_seq_for_map[:-2]):
                        for cell_2 in cells_seq_for_map[cell_index + 1:]:
                            if cell_2 in ms.coord_obj.intersect_cells[cell_1]:
                                at_least_one_intersect = True
                                break
                        if at_least_one_intersect:
                            break

                    data_id = ms.description + f" {'_'.join(map(str, cells_seq_for_map))}"
                    if at_least_one_intersect:
                        data_id = "intersect_" + data_id

                    ms.coord_obj.plot_cells_map(param=param,
                                                data_id=data_id,
                                                show_polygons=False,
                                                fill_polygons=False,
                                                title_option="seq",
                                                connections_dict=None,
                                                with_edge=True,
                                                cells_groups=cells_to_color,
                                                cells_groups_colors=cells_groups_colors,
                                                dont_fill_cells_not_in_groups=True,
                                                with_cell_numbers=False,
                                                save_formats=save_formats)

    if plot_seq_with_rep_by_age:
        # key is an int representing age, and value is an int representing the number of sessions for a given age
        nb_ms_by_age = SortedDict()

        ages_groups = dict()
        # will allow to group age by groups, for each age, we give a string value which is going to be a key for another
        # dict
        agCe_tuples = [(5,), (6, 7), (8, 10), (11, 12), (13, 14), (15, 16), (17, 21), (41,)]  # , (60, )
        age_tuples = [(5, 7), (8,), (9, 10), (11, 14), (15, 21), (41,)]  # , (60, )
        manual_colors = dict()
        # manual_colors["5"] = "white"
        # manual_colors["6-7"] = "navajowhite"
        # manual_colors["8-10"] = "lawngreen"
        # manual_colors["11-12"] = "cornflowerblue"
        # manual_colors["13-14"] = "orange"
        # manual_colors["15-16"] = "coral"
        # manual_colors["17-21"] = "pink"
        # manual_colors["41"] = "red"
        manual_colors["5-7"] = "white"
        manual_colors["8"] = "lawngreen"
        manual_colors["9-10"] = "yellow"
        manual_colors["11-14"] = "cornflowerblue"
        manual_colors["15-21"] = "blue"
        manual_colors["41"] = "red"

        # manual_colors["60"] = "red"
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

        # we make group of slopes using a sliding window
        size_of_slopes_group = 3
        for first_slope in np.arange(min_slope, max_slope + 2 - size_of_slopes_group):
            slopes_in_group = list(range(first_slope, first_slope + size_of_slopes_group))
            # dict1: key age (int) value dict2
            # dict2: key length seq (int), value dict3
            # dict3: key repetitions (int), value nb of seq with this length and this repetition
            data_dict = SortedDict()
            # filling data_dict
            for age, dict_ms in seq_dict.items():
                if age in ages_groups:
                    age_str = ages_groups[age]
                else:
                    age_str = str(age)
                nb_ms_by_age[age_str] = nb_ms_by_age.get(age_str, 0) + 1
                if age_str not in data_dict:
                    data_dict[age_str] = dict()
                for ms_description, dict_cells_seq in dict_ms.items():
                    for cells_seq, dict_slope in dict_cells_seq.items():
                        if cells_seq == "cells_best_order":
                            continue
                        for slope, values in dict_slope.items():
                            # values is a list of list, each list represents a rep
                            if slope in slopes_in_group:
                                if len(cells_seq) not in data_dict[age_str]:
                                    data_dict[age_str][len(cells_seq)] = dict()
                                n_rep = len(values)
                                data_dict[age_str][len(cells_seq)][n_rep] = \
                                    data_dict[age_str][len(cells_seq)].get(n_rep, 0) + 1
            file_name = "scatter_significant_seq_slopes_" + ' '.join(map(str, slopes_in_group))
            plot_fig_nb_cells_in_seq_vs_rep_by_age(data_dict=data_dict, ages_key_order=ages_key_order,
                                                   param=param, min_len_seq_to_display=3,
                                                   nb_ms_by_age=nb_ms_by_age,
                                                   color_option=color_option,
                                                   manual_colors=manual_colors,
                                                   cmap_name=None,
                                                   file_name=file_name,
                                                   save_formats="pdf")


def plot_figure_with_map_and_raster_for_sequences(ms, cells_in_seq, span_area_coords, span_area_colors,
                                                  file_name, lines_to_display=None, range_around_slope_in_frames=0,
                                                  without_sum_activity_traces=False,
                                                  frames_to_use=None, color_map_for_seq=cm.Reds,
                                                  save_formats="pdf", dpi=500):
    param = ms.param
    if frames_to_use is not None:
        raster_dur = ms.spike_struct.spike_nums_dur[:, frames_to_use]
    else:
        raster_dur = ms.spike_struct.spike_nums_dur
    speed_array = None
    if ms.speed_by_frame is not None:
        speed_array = norm01(ms.speed_by_frame) - 1.5
    n_cells = raster_dur.shape[0]
    n_frames = raster_dur.shape[1]
    fig = plt.figure(figsize=(10, 11), dpi=dpi)  # constrained_layout=True,
    gs = fig.add_gridspec(12, 3)
    ax_empty = fig.add_subplot(gs[:3, 0])
    ax_map = fig.add_subplot(gs[:3, 1])
    ax_empty_2 = fig.add_subplot(gs[:3, 2])

    if without_sum_activity_traces:
        ax_raw_traces = fig.add_subplot(gs[3:7, :])
        ax_sum_raw_traces, ax_sum_all_raw_traces = (None, None)
    else:
        ax_raw_traces = fig.add_subplot(gs[3:5, :])
        ax_sum_raw_traces = fig.add_subplot(gs[5, :])
        ax_sum_all_raw_traces = fig.add_subplot(gs[6, :])

    ax_im_show = fig.add_subplot(gs[7:9, :])

    ax_raster = fig.add_subplot(gs[9:11, :])
    ax_empty_3 = fig.add_subplot(gs[11, :])
    background_color = "black"
    labels_color = "white"

    for ax in [ax_empty, ax_map, ax_empty_2, ax_raw_traces, ax_sum_raw_traces,
               ax_sum_all_raw_traces,
               ax_im_show, ax_raster, ax_empty_3]:
        if ax is None:
            continue
        ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)

    # first we display the cells map
    # color = (213 / 255, 38 / 255, 215 / 255, 1)  # #D526D7
    if color_map_for_seq is not None:
        cells_groups_colors = []
        cells_to_color = []
        for cell_index, cell in enumerate(cells_in_seq):
            cells_to_color.append([cell])
            color = color_map_for_seq(cell_index / (len(cells_in_seq)))
            # print(f"color {color}")
            cells_groups_colors.append(color)
    else:
        color = (67 / 255, 162 / 255, 202 / 255, 1)  # blue
        color = (33 / 255, 113 / 255, 181 / 255, 1)
        cells_groups_colors = [color]
        cells_to_color = [cells_in_seq]
    # check if at least a pair of cells intersect
    at_least_one_intersect = False
    for cell_index, cell_1 in enumerate(cells_in_seq[:-2]):
        for cell_2 in cells_in_seq[cell_index + 1:]:
            if cell_2 in ms.coord_obj.intersect_cells[cell_1]:
                at_least_one_intersect = True
                break
        if at_least_one_intersect:
            break

    data_id = ms.description + f" {'_'.join(map(str, cells_in_seq))}"
    if at_least_one_intersect:
        data_id = "intersect_" + data_id
    ms.coord_obj.plot_cells_map(param=param,
                                ax_to_use=ax_map,
                                data_id=data_id,
                                show_polygons=False,
                                fill_polygons=False,
                                title_option="seq",
                                connections_dict=None,
                                with_edge=True,
                                edge_line_width=0.2,
                                default_edge_color="#c6dbef",
                                cells_groups=cells_to_color,
                                cells_groups_colors=cells_groups_colors,
                                dont_fill_cells_not_in_groups=True,
                                with_cell_numbers=True,
                                cell_numbers_color="white",
                                text_size=2,
                                save_formats=save_formats)

    # one subplot for raw traces
    raw_traces = np.copy(ms.raw_traces)
    if frames_to_use is not None:
        raw_traces = raw_traces[:, frames_to_use]
    for i in np.arange(len(raw_traces)):
        raw_traces[i] = (raw_traces[i] - np.mean(raw_traces[i]) / np.std(raw_traces[i]))
        raw_traces[i] = norm01(raw_traces[i])
        raw_traces[i] = norm01(raw_traces[i]) * 5

    plot_spikes_raster(spike_nums=raster_dur[cells_in_seq],
                       param=ms.param,
                       display_spike_nums=False,
                       axes_list=[ax_raw_traces],
                       traces=raw_traces[cells_in_seq],
                       display_traces=True,
                       use_brewer_colors_for_traces=True,
                       spike_train_format=False,
                       y_ticks_labels=cells_in_seq,
                       y_ticks_labels_size=2,
                       save_raster=False,
                       show_raster=False,
                       span_area_coords=span_area_coords,
                       span_area_colors=span_area_colors,
                       alpha_span_area=0.3,
                       plot_with_amplitude=False,
                       # raster_face_color="white",
                       hide_x_labels=True,
                       without_activity_sum=True,
                       show_sum_spikes_as_percentage=True,
                       span_area_only_on_raster=False,
                       spike_shape='*',
                       spike_shape_size=1,
                       # lines_to_display=lines_to_display,
                       # lines_color="white",
                       # lines_width=0.35,
                       # lines_band=range_around_slope_in_frames,
                       # lines_band_color="white",
                       save_formats="pdf")

    if not without_sum_activity_traces:
        # ploting sum of activity of the traces shown
        for traces_index, trace_to_use in enumerate([raw_traces_01[cells_in_seq], raw_traces_01]):
            sum_traces = np.sum(trace_to_use,
                                axis=0) \
                         / len(trace_to_use)
            sum_traces *= 100
            if traces_index == 0:
                ax_to_use = ax_sum_raw_traces
            else:
                ax_to_use = ax_sum_all_raw_traces
            ax_to_use.set_facecolor(background_color)
            ax_to_use.fill_between(np.arange(n_frames), 0, sum_traces, facecolor="#c6dbef", zorder=10)
            if span_area_coords is not None:
                for index, span_area_coord in enumerate(span_area_coords):
                    for coord in span_area_coord:
                        if span_area_colors is not None:
                            color = span_area_colors[index]
                        else:
                            color = "lightgrey"
                        ax_to_use.axvspan(coord[0], coord[1], alpha=0.6, facecolor=color, zorder=8)
            ax_to_use.set_xlim(0, n_frames - 1)
            ax_to_use.set_ylim(0, np.max(sum_traces))
            ax_to_use.get_xaxis().set_visible(False)
            # ax_to_use.get_xaxis().set_ticks([])
            ax_to_use.yaxis.label.set_color("white")
            # ax_to_use.xaxis.label.set_color("white")
            ax_to_use.tick_params(axis='y', colors="white")
            # ax_to_use.tick_params(axis='x', colors="white")
            ax_to_use.tick_params(axis='both', which='both', length=0)
            ax_to_use.yaxis.set_tick_params(labelsize=2)

    traces_for_imshow = np.copy(ms.raw_traces[cells_in_seq])
    if frames_to_use is not None:
        traces_for_imshow = traces_for_imshow[:, frames_to_use]
    with_arnaud_normalization = True
    if with_arnaud_normalization:
        for i in np.arange(len(traces_for_imshow)):
            # traces_for_imshow[i] = traces_for_imshow[i] / np.median(traces_for_imshow[i])
            traces_for_imshow[i] = gaussblur1D(traces_for_imshow[i],
                                               traces_for_imshow.shape[1] / 2, 0)
            traces_for_imshow[i, :] = norm01(traces_for_imshow[i, :])
            traces_for_imshow[i, :] = traces_for_imshow[i, :] - np.median(traces_for_imshow[i, :])

    plot_with_imshow(raster=traces_for_imshow,
                     n_subplots=1, axes_list=[ax_im_show],
                     hide_x_labels=True,
                     y_ticks_labels_size=2,
                     y_ticks_labels=cells_in_seq,
                     fig=fig,
                     show_color_bar=False,
                     values_to_plot=None, cmap="hot",
                     without_ticks=True,
                     vmin=0, vmax=0.5,
                     reverse_order=True,
                     # lines_to_display=lines_to_display,
                     # lines_color="white",
                     # lines_width=0.2,
                     # lines_band=range_around_slope_in_frames,
                     # lines_band_color="white",
                     # lines_band_alpha=0.8,
                     speed_array=speed_array
                     )
    if frames_to_use is not None:
        x_ticks_labels = [x for x in frames_to_use if x % 100 == 0]
        x_ticks = [x for x in np.arange(0, 100 * len(x_ticks_labels), 100)]
    else:
        x_ticks_labels = [x for x in np.arange(n_frames) if x % 100 == 0]
        x_ticks = x_ticks_labels
    x_ticks_labels_size = 2

    cells_to_highlight = None
    cells_to_highlight_colors = None
    if color_map_for_seq is not None:
        cells_to_highlight_colors = []
        cells_to_highlight = []
        for cell_index, cell in enumerate(cells_in_seq):
            cells_to_highlight.append(cell_index)
            color = color_map_for_seq(cell_index / (len(cells_in_seq)))
            # print(f"color {color}")
            cells_to_highlight_colors.append(color)

    do_bin_raster = False
    bin_size = 12
    if do_bin_raster:
        raster_dur = tools_misc.bin_raster(raster=raster_dur, bin_size=bin_size, keep_same_dimension=True)

    plot_spikes_raster(spike_nums=raster_dur[cells_in_seq],
                       param=param,
                       x_ticks_labels=x_ticks_labels,
                       x_ticks_labels_size=x_ticks_labels_size,
                       x_ticks=x_ticks,
                       size_fig=(10, 2),
                       axes_list=[ax_raster],
                       spike_train_format=False,
                       y_ticks_labels=cells_in_seq,
                       save_raster=False,
                       show_raster=False,
                       show_sum_spikes_as_percentage=True,
                       without_activity_sum=True,
                       plot_with_amplitude=False,
                       span_area_coords=span_area_coords,
                       span_area_colors=span_area_colors,
                       cells_to_highlight=cells_to_highlight,
                       cells_to_highlight_colors=cells_to_highlight_colors,
                       alpha_span_area=0.5,
                       # cells_to_highlight=cells_to_highlight,
                       # cells_to_highlight_colors=cells_to_highlight_colors,
                       # span_cells_to_highlight=span_cells_to_highlight,
                       # span_cells_to_highlight_colors=span_cells_to_highlight_colors,
                       span_area_only_on_raster=False,
                       y_ticks_labels_size=1,
                       spike_shape='o',
                       spike_shape_size=0.1,
                       cell_spikes_color="red",
                       lines_to_display=lines_to_display,
                       lines_color="white",
                       lines_width=0.2,
                       lines_band=range_around_slope_in_frames,
                       lines_band_color="white",
                       save_formats="pdf")
    # fig.tight_layout()
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 0.1, 'h_pad': 0})
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/{file_name}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())

    plt.close()


def compute_stat_about_significant_seq(files_path, param, color_option="use_cmap_gradient", cmap_name="Reds",
                                       scale_scatter=False, use_different_shapes_for_stat=False,
                                       min_len_seq_to_display=4,
                                       save_formats="pdf", slope_version=True):
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
    # no more categories
    # n_categories = 4
    # marker_cat = ["*", "d", "o", "s"]
    # categories that should be displayed
    banned_categories = []
    file_names = []
    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(files_path):
        file_names.extend(local_filenames)
        break

    # dict1: key age (int) value dict3
    # not anymore dict2: key category (int, from 1 to 4), value dict3
    # dict3: key length seq (int), value dict4
    # dict4: key repetitions (int), value nb of seq with this length and this repetition
    data_dict = SortedDict()
    # key is an int representing age, and value is an int representing the number of sessions for a given age
    nb_ms_by_age = dict()

    ages_groups = dict()
    # will allow to group age by groups, for each age, we give a string value which is going to be a key for another
    # dict
    age_tuples = [(5,), (6, 7), (8, 10), (11, 12), (13, 14), (15, 21), (41,)]  # , (60, )
    manual_colors = dict()
    manual_colors["5"] = "white"
    manual_colors["6-7"] = "navajowhite"
    manual_colors["8-10"] = "lawngreen"
    manual_colors["11-12"] = "cornflowerblue"
    manual_colors["13-14"] = "orange"
    manual_colors["15-21"] = "coral"
    manual_colors["41"] = "red"

    # manual_colors["60"] = "red"
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
        file_name = file_name.lower()
        if not file_name.startswith("p"):
            if not file_name.startswith("significant_sorting_results_"):
                continue
            # p_index = len("significant_sorting_results")+1
            if "slope" in file_name:
                file_name = file_name[len("significant_sorting_results_with_slope_"):]
            else:
                file_name = file_name[len("significant_sorting_results_"):]
        index_ = file_name.find("_")
        if index_ < 1:
            continue
        age_file = int(file_name[1:index_])
        age = ages_groups[age_file]
        nb_ms_by_age[age] = nb_ms_by_age.get(age, 0) + 1
        # print(f"age {age}")
        if age not in data_dict:
            data_dict[age] = dict()
            # for cat in np.arange(1, 1 + n_categories):
            #     data_dict[age][cat] = dict()

        with open(files_path + file_name_original, "r", encoding='UTF-8') as file:
            for nb_line, line in enumerate(file):
                line_list = line.split(':')
                seq_n_cells = int(line_list[0])
                if not slope_version:
                    line_list = line_list[1].split("]")
                    # we remove the '[' on the first position
                    repetitions_str = line_list[0][1:].split(",")
                    repetitions = []
                    for rep in repetitions_str:
                        repetitions.append(int(rep))
                    n_total_rep = np.sum(repetitions)
                    if seq_n_cells not in data_dict[age]:
                        data_dict[age][seq_n_cells] = dict()
                    data_dict[age][seq_n_cells][n_total_rep] = 1
                else:
                    n_total_rep = int(line_list[1])
                    if seq_n_cells not in data_dict[age]:
                        data_dict[age][seq_n_cells] = dict()
                    data_dict[age][seq_n_cells][n_total_rep] = data_dict[age][seq_n_cells].get(n_total_rep, 0) + 1
                # we remove the ' [' on the first position
                # no more categories
                # categories_str = line_list[1][2:].split(",")
                # categories = []
                # for cat in categories_str:
                #     categories.append(int(cat))
                #
                # for index, cat in enumerate(categories):
                #     if seq_n_cells not in data_dict[age][cat]:
                #         data_dict[age][cat][seq_n_cells] = dict()
                #
                #     rep = repetitions[index]
                #     if rep not in data_dict[age][cat][seq_n_cells]:
                #         data_dict[age][cat][seq_n_cells][rep] = 0
                #     data_dict[age][cat][seq_n_cells][rep] += 1
                # print(f"{seq_n_cells} cells: {repetitions} {categories}")
    plot_fig_nb_cells_in_seq_vs_rep_by_age(data_dict, ages_key_order, param, min_len_seq_to_display,
                                           nb_ms_by_age, color_option, manual_colors, save_formats)


def plot_fig_nb_cells_in_seq_vs_rep_by_age(data_dict, ages_key_order, param, min_len_seq_to_display,
                                           nb_ms_by_age, color_option, manual_colors, scale_scatter=False,
                                           cmap_name=None, file_name=None,
                                           save_formats="pdf"):
    """

    :param data_dict: # dict1: key age (int) value dict2
            # dict2: key length seq (int), value dict3
            # dict3: key repetitions (int), value nb of seq with this length and this repetition
    :param ages_key_order:
    :param param:
    :param min_len_seq_to_display:
    :param nb_ms_by_age:
    :param color_option:
    :param manual_colors:
    :param cmap_name:
    :param save_formats:
    :return:
    """

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(15, 15))
    background_color = "black"
    labels_color = "white"
    ax1.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)

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
        len_dict = data_dict[age]
        for len_seq, rep_dict in len_dict.items():
            if len_seq < min_len_seq_to_display:
                continue
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
                scatter_size = 80 + 1.2 * x_pos + 1.2 * y_pos  # 50
                # scatter_size = 100
                if scale_scatter:
                    scatter_size = 15 + 5 * np.sqrt(n_seq_normalized)
                marker_to_use = "o"
                # if use_different_shapes_for_stat:
                #     marker_to_use = param.markers[cat - 1]
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
    # if use_different_shapes_for_stat:
    #     for cat in np.arange(1, n_categories + 1):
    #         if cat in banned_categories:
    #             continue
    #         legend_elements.append(Line2D([0], [0], marker=param.markers[cat - 1], color="w", lw=0, label="*" * cat,
    #                                       markerfacecolor='black', markersize=15))

    ax1.legend(handles=legend_elements)

    # plt.title(title)
    ax1.set_ylabel(f"Repetition (#)", fontsize=20)
    ax1.set_xlabel("Cells (#)", fontsize=20)
    ax1.set_ylim(min_rep - 2, max_rep + 2)
    ax1.set_xlim(min_len - 2, max_len + 2)
    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)
    # xticks = np.arange(0, len(data_dict))
    # ax1.set_xticks(xticks)
    # # sce clusters labels
    # ax1.set_xticklabels(labels)

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    if file_name is None:
        file_name = "scatter_significant_seq"
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/{file_name}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())

    plt.close()


def box_plot_data_by_age(data_dict, title, filename,
                         y_label, param, colors=None,
                         path_results=None, y_lim=None,
                         x_label=None, with_scatters=True,
                         y_log=False,
                         scatters_with_same_colors=None,
                         scatter_size=20,
                         scatter_alpha=0.5,
                         n_sessions_dict=None,
                         background_color="black",
                         link_medians=True,
                         color_link_medians="red",
                         labels_color="white",
                         with_y_jitter=None,
                         x_labels_rotation=None,
                         fliers_symbol=None,
                         save_formats="pdf"):
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
    sym = ""
    if fliers_symbol is not None:
        sym = fliers_symbol
    bplot = plt.boxplot(data_list, patch_artist=colorfull,
                        labels=labels, sym=sym, zorder=30)  # whis=[5, 95], sym='+'
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

            if with_y_jitter is not None:
                y_pos = [value + (((np.random.random_sample() - 0.5) * 2) * with_y_jitter) for value in data]
            else:
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
    if link_medians:
        ax1.plot(np.arange(1, len(medians_values) + 1), medians_values, zorder=36, color=color_link_medians,
                 linewidth=2)

    # plt.xlim(0, 100)
    plt.title(title)

    ax1.set_ylabel(f"{y_label}", fontsize=30, labelpad=20)
    if y_lim is not None:
        ax1.set_ylim(y_lim[0], y_lim[1])
    if x_label is not None:
        ax1.set_xlabel(x_label, fontsize=30, labelpad=20)
    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)
    if y_log:
        ax1.set_yscale("log")

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
    if x_labels_rotation is not None:
        for tick in ax1.get_xticklabels():
            tick.set_rotation(x_labels_rotation)

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


def box_joy_plot_data_by_age(data_dict, title, filename, y_label, param, save_formats="pdf"):
    """

    :param data_dict:
    :param title:
    :param filename:
    :param y_label:
    :param param: Contains a field name colors used to color the boxplot
    :param save_formats:
    :return:
    """
    # based on : https://github.com/sbebo/joypy/blob/master/Joyplot.ipynb
    # TODO: data need to be in Pandas format
    pass
    # fig, axes = joypy.joyplot(temp, by="Year", column="Anomaly", ylabels=False, xlabels=False,
    #                           grid=False, fill=False, background='k', linecolor="w", linewidth=1,
    #                           legend=False, overlap=0.5, figsize=(6, 5), kind="counts", bins=80)
    #
    # labels = [y if y % 10 == 0 else None for y in list(temp.Year.unique())]
    # fig, axes = joypy.joyplot(temp, by="Year", column="Anomaly", labels=labels, range_style='own',
    #                           grid="y", linewidth=1, legend=False, fade=True, figsize=(6, 5),
    #                           title="Global daily temperature 1880-2014 \n(°C above 1950-80 average)",
    #                           kind="counts", bins=30)
    #
    # fig, ax1 = plt.subplots(nrows=1, ncols=1,
    #                         gridspec_kw={'height_ratios': [1]},
    #                         figsize=(12, 12))
    # ax1.set_facecolor("black")
    #
    # colorfull = True
    # labels = []
    # data_list = []
    # for age, data in data_dict.items():
    #     data_list.append(data)
    #     labels.append(age)
    # bplot = plt.boxplot(data_list, patch_artist=colorfull,
    #                     labels=labels, sym='', zorder=1)  # whis=[5, 95], sym='+'
    # # color=["b", "cornflowerblue"],
    # # fill with colors
    #
    # # edge_color="silver"
    #
    # for element in ['boxes', 'whiskers', 'fliers', 'caps']:
    #     plt.setp(bplot[element], color="white")
    #
    # for element in ['means', 'medians']:
    #     plt.setp(bplot[element], color="silver")
    #
    # if colorfull:
    #     colors = param.colors[:len(data_dict)]
    #     for patch, color in zip(bplot['boxes'], colors):
    #         patch.set_facecolor(color)
    #
    # # plt.xlim(0, 100)
    # plt.title(title)
    # ax1.set_ylabel(f"{y_label}")
    # ax1.set_xlabel("age")
    # xticks = np.arange(1, len(data_dict) + 1)
    # ax1.set_xticks(xticks)
    # # sce clusters labels
    # ax1.set_xticklabels(labels)
    #
    # if isinstance(save_formats, str):
    #     save_formats = [save_formats]
    # for save_format in save_formats:
    #     fig.savefig(f'{param.path_results}/{filename}'
    #                 f'_{param.time_str}.{save_format}',
    #                 format=f"{save_format}")
    #
    # plt.close()


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
                                                       sce_times_numbers,
                                                       use_only_onsets=False):
    n_cells = len(spike_nums)
    result = np.zeros(n_cells)

    for cell in np.arange(n_cells):
        n_sces = np.max(sce_times_numbers) + 1
        if (not use_only_onsets) and (spike_nums_dur is not None):
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
        max_index_seq = 100  # len(spike_nums_dur_ordered)  # 50

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
                       # spike_shape_size=0.2,
                       spike_shape="|",
                       spike_shape_size=10,
                       span_area_coords=span_area_coords,
                       span_area_colors=span_area_colors,
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
    # TODO: find hubs, but use a list of ms and calculate the thresholds using all data
    n_cells = ms.spike_struct.n_cells
    # print(f"{ms.description}: graph hubs: {n_cells} vs {len(graph)}")
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


def plot_ratio_spikes_on_events_by_cell(spike_nums,
                                        spike_nums_dur,
                                        times_numbers,
                                        param,
                                        use_only_onsets=False,
                                        event_description="",
                                        session_description=""):
    """

    :param spike_nums:
    :param spike_nums_dur:
    :param times_numbers:
    :param use_only_onsets:
    :return:
    """
    ratio_spikes_events = get_ratio_spikes_on_events_vs_total_spikes_by_cell(
        spike_nums=spike_nums,
        spike_nums_dur=spike_nums_dur,
        sce_times_numbers=times_numbers, use_only_onsets=use_only_onsets)

    ratio_spikes_total_events = get_ratio_spikes_on_events_vs_total_events_by_cell(
        spike_nums=spike_nums,
        spike_nums_dur=spike_nums_dur,
        sce_times_numbers=times_numbers, use_only_onsets=use_only_onsets)

    plot_hist_distribution(distribution_data=ratio_spikes_events,
                           description=f"{session_description}_hist_spike_{event_description}_ratio",
                           xlabel=f"spikes in {event_description} vs total spikes (%)",
                           param=param)

    plot_hist_distribution(distribution_data=ratio_spikes_total_events,
                           description=f"{session_description}_hist_spike_total_{event_description}_ratio",
                           xlabel=f"spikes in {event_description} vs total {event_description} (%)",
                           param=param)


def get_ratio_spikes_on_events_vs_total_spikes_by_cell(spike_nums,
                                                       spike_nums_dur,
                                                       sce_times_numbers,
                                                       use_only_onsets=False):
    n_cells = len(spike_nums)
    result = np.zeros(n_cells)

    for cell in np.arange(n_cells):
        n_spikes = np.sum(spike_nums[cell, :])
        if (not use_only_onsets) and (spike_nums_dur is not None):
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
        result += f"{' ' * actual_line}"
        if actual_line > (n_lines / 2):
            result += "\\"
        else:
            result += "|"
        if actual_line == (n_lines // 2):
            result += f"{' ' * (width // 2 - 1)}"
            result += " O "
            result += f"{' ' * (width // 2 - 1)}"
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
            result += f"{' ' * (width // 2 - 1)}"
            result += " O "
            result += f"{' ' * (width // 2 - 1)}"
        else:
            result += f"{' ' * width}"
        result += "|"
    else:
        result += f"{' ' * 5}"
        result += f"{' ' * actual_line}"
        result += f"{'|' * bottom_width}"

    print(f"{result}")


def save_stat_about_mvt_for_each_ms(ms_to_analyse, param):
    file_name = os.path.join(param.path_results, f"ms_stat_mvt_{param.time_str}.txt")
    with open(file_name, "w", encoding='UTF-8') as file:
        for ms in ms_to_analyse:
            if ms.shift_data_dict is None:
                continue
            shift_twitch_bool = ms.shift_data_dict["shift_twitch"]
            shift_long_bool = ms.shift_data_dict["shift_long"]
            shift_unclassified_bool = ms.shift_data_dict["shift_unclassified"]
            n_frames = len(shift_twitch_bool)
            ratio_twitch = str(np.round((np.sum(shift_twitch_bool) / n_frames) * 100, 2))
            ratio_long = str(np.round((np.sum(shift_long_bool) / n_frames) * 100, 2))
            ratio_unclassified = str(np.round((np.sum(shift_unclassified_bool) / n_frames) * 100, 2))
            still_n_frames = n_frames - np.sum(shift_twitch_bool) - np.sum(shift_long_bool) - np.sum(
                shift_unclassified_bool)
            ratio_still = str(np.round((np.sum(still_n_frames) / n_frames) * 100, 2))
            file.write(
                f"{ms.description}: {ratio_twitch} % twitches, {ratio_long} % long mvt,"
                f" {ratio_unclassified} % unclassified, {ratio_still} % still  \n")
            file.write("" + '\n')


def get_peaks_periods_on_sum_of_activity(raster, around_peaks, distance_bw_peaks, min_n_cells=5,
                                         frames_to_exclude=None):
    """

    :param raster:
    :param around_peaks:
    :param distance_bw_peaks:
    :param min_n_cells:
    :param frames_to_exclude: if not None, boolean 1d array of len raster.shape[1], at True when the frame should
    be excluded
    :return:
    """
    n_cells = raster.shape[0]
    n_frames = raster.shape[1]
    sum_activity = np.sum(raster, axis=0)
    peaks, properties = signal.find_peaks(x=sum_activity, height=min_n_cells, distance=distance_bw_peaks)
    peaks_bool = np.zeros(n_frames, dtype="bool")
    for peak in peaks:
        if frames_to_exclude is not None:
            if np.any(frames_to_exclude[max(0, peak - around_peaks): min(n_frames, peak + around_peaks + 1)]):
                continue
        peaks_bool[max(0, peak - around_peaks): min(n_frames, peak + around_peaks + 1)] = True

    peaks_periods = get_continous_time_periods(peaks_bool.astype("int8"))
    cellsinpeak = np.zeros((n_cells, len(peaks_periods)), dtype="int8")
    peaks_numbers = np.zeros(n_frames, dtype="int16")
    for period_index, period in enumerate(peaks_periods):
        peaks_numbers[period[0]:period[1] + 1] = period_index
        sum_spikes = np.sum(raster[:, period[0]:(period[1] + 1)], axis=1)
        active_cells = np.where(sum_spikes)[0]
        cellsinpeak[active_cells, period_index] = 1

    return peaks, peaks_periods, peaks_bool, peaks_numbers, cellsinpeak


def plot_all_cell_assemblies_proportion_on_shift_categories(ms_to_analyse,
                                                            param, save_formats="pdf"):
    # TODO : to finish
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']

    # key is age (p_age) each dict contains a dict with key being a mvt (twitch, shift_long, unidentified, still) and a list of float
    # (from 0 to 100)
    proportion_by_age_dict = dict()

    shift_keys = ["shift_twitch", "shift_long", "shift_unclassified"]

    n_ms = 0
    for ms in ms_to_analyse:
        if ms.shift_data_dict is None:
            continue
        if ms.spike_struct.spike_nums is None:
            continue
        if ms.cell_assemblies is None:
            continue

        if ("p" + str(ms.age)) not in proportion_by_age_dict:
            proportion_by_age_dict[("p" + str(ms.age))] = dict()

        n_shift_keys = len(shift_keys)
        # key is shift_key_descr, value a. For this ms
        assembly_ratio_by_shift = dict()
        n_assemblies = len(ms.cell_assemblies)

        # n_sessions_dict = dict()
        # # we add the string definition the session, so we can have the count of the animals
        # # and not just session by age
        # n_sessions_dict[("p" + str(ms.age))] = set()

        # we ass 2 step in order to add all mvt and also still
        for step in np.arange(n_shift_keys + 2):
            results_dict = dict()
            n_frames = ms.spike_struct.spike_nums.shape[1]
            n_cells = ms.spike_struct.spike_nums.shape[0]
            shift_bool = np.zeros(n_frames, dtype="bool")

            if step < n_shift_keys:
                shift_keys_to_loop = [shift_keys[step]]
                shift_key_descr = shift_keys[step]
            else:
                if step == n_shift_keys:
                    shift_keys_to_loop = shift_keys
                    shift_key_descr = "all_mvt"
                else:
                    shift_keys_to_loop = shift_keys
                    shift_key_descr = "still"

            for shift_key in shift_keys_to_loop:
                shift_bool_tmp = ms.shift_data_dict[shift_key]
                if shift_key == "shift_twitch":
                    extension_frames_after = 15
                    extension_frames_before = 1
                else:
                    extension_frames_after = 0
                    extension_frames_before = 0
                if (extension_frames_after + extension_frames_before) == 0:
                    shift_bool[shift_bool_tmp] = True
                else:
                    # we extend each period, implementation is not the fastest and more elegant way
                    true_frames = np.where(shift_bool_tmp)[0]
                    for frame in true_frames:
                        first_frame = max(0, frame - extension_frames_before)
                        last_frame = min(n_frames - 1, frame + extension_frames_after)
                        shift_bool[first_frame:last_frame + 1] = True

            if shift_key_descr == "still":
                # then we revert the all_mvt result
                shift_bool = np.invert(shift_bool)

            # now for each assemblie, we measure the percentage of reptition in which it is
            assemblies_ratio = []
            for assembly_index in np.arange(n_assemblies):
                # times tuple in which the assembly is active (as single cell_assembly)
                periods = ms.sce_times_in_single_cell_assemblies[assembly_index]
                n_rep_assembly_total = len(periods)
                n_rep_assembly_in_shift = 0
                for period in periods:
                    if shift_key_descr == "still":
                        # for still, all the frames in SCE must be in still
                        in_the_shift = np.all(shift_bool[period[0:period[1] + 1]])
                    else:
                        in_the_shift = np.any(shift_bool[period[0:period[1] + 1]])
                    if in_the_shift:
                        n_rep_assembly_in_shift += 1
                assemblies_ratio.append((n_rep_assembly_in_shift / n_rep_assembly_total) * 100)
            assembly_ratio_by_shift[shift_key_descr] = assemblies_ratio

        # we display a box_plot reprensenting for this session the cell assemblies distribution
        # for each category of shift
        colors_scatter = []
        for cell_assembly_index in np.arange(n_assemblies):
            colors_scatter.append(cm.nipy_spectral(float(cell_assembly_index + 1) / (n_assemblies + 1)))
        box_plot_data_by_age(data_dict=assembly_ratio_by_shift, title="",
                             filename=f"{ms.description}_assemblies_in_shift_category",
                             y_label=f"Cell assemblies activation (%)",
                             colors=None, with_scatters=True,
                             scatter_alpha=1,
                             scatters_with_same_colors=colors_scatter,
                             path_results=param.path_results, scatter_size=240,
                             param=param, save_formats=save_formats)

        # shift_time_periods = get_continous_time_periods(shift_bool.astype("int8"))

        n_ms += 1


def plot_connectivity_graph(ms_to_analyse, param, save_formats="pdf"):
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
              '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
              '#74add1', '#4575b4', '#313695']
    n_ms = 0
    for ms in ms_to_analyse:
        print(f"{ms.description}: detect_n_in_n_out")
        if ms.spike_struct.graph_out is None:
            ms.detect_n_in_n_out()
        if ms.spike_struct.graph_out is not None:
            n_ms += 1

    # we plot one graph for each session
    background_color = "black"
    max_n_lines = 10
    n_lines = n_ms if n_ms <= max_n_lines else max_n_lines
    n_col = math.ceil(n_ms / n_lines)
    # for histogram all events
    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col,
                                          'height_ratios': [1] * n_lines},
                             figsize=(30, 25))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)
    if n_ms == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        axes[ax_index].set_facecolor(background_color)

    real_ms_index = 0
    last_age = None
    n_ages_so_far = 0
    for ms in ms_to_analyse:
        if ms.spike_struct.graph_out is not None:
            if last_age is None:
                last_age = ms.age
            else:
                if ms.age != last_age:
                    last_age = ms.age
                    n_ages_so_far += 1
            plot_graph_using_fa2(graph=ms.spike_struct.graph_out, file_name=f"{ms.description} graph out",
                                 title=f"{ms.description}",
                                 ax_to_use=axes[real_ms_index],
                                 color=colors[n_ages_so_far % len(colors)],
                                 param=param, iterations=15000, save_raster=False, with_labels=False,
                                 save_formats="pdf", show_plot=False)
            real_ms_index += 1
    # ms.spike_struct.graph_out.add_edges_from(ms.spike_struct.graph_in.edges())
    # plot_graph_using_fa2(graph=ms.spike_struct.graph_out, file_name=f"{ms.description} graph in-out",
    #                      title=f"{ms.description} in-out",
    #                      param=param, iterations=5000, save_raster=True, with_labels=False,
    #                      save_formats="pdf", show_plot=False)

    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/connectiviy_graph_by_age'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())


def plot_twitch_ratio_activity(ms_to_analyse, param, time_around=20, save_formats="pdf"):
    """
    For each session, look around twitches, and calculte a ratio definin if the sum of activity is greater
    after or before each twitch
    :param ms_to_analyse:
    :param time_around:
    :return:
    """

    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    # first key age in int, then dict with key: session_id string, then dict with key cell, and results list of ratio
    ratio_dict = dict()
    n_ms = 0
    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums_dur is None:
            continue
        n_ms += 1

        if ms.age not in ratio_dict:
            ratio_dict[ms.age] = dict()

        ratio_dict[ms.age][ms.description] = []

        n_cells = ms.spike_struct.spike_nums_dur.shape[0]
        n_frames = ms.spike_struct.spike_nums_dur.shape[1]
        shift_bool = ms.shift_data_dict["shift_twitch"]
        periods = get_continous_time_periods(shift_bool.astype("int8"))
        # for cell in np.arange(n_cells):
        #     ratio_dict[ms.age][ms.description][cell] = []
        for period in periods:
            start_period = max(0, period[0] - time_around)
            end_period = min(n_frames - 1, period[1] + time_around)
            # sum before the twitch
            sum_before = np.sum(np.sum(ms.spike_struct.spike_nums_dur[:, start_period:period[0]]))

            sum_after = np.sum(np.sum(ms.spike_struct.spike_nums_dur[:, period[0]:end_period + 1]))
            ratio = (sum_after / (sum_after + sum_before)) * 100
            ratio_dict[ms.age][ms.description].append(ratio)

    # we plot one distribution for each session with the ratio
    background_color = "black"
    max_n_lines = 2
    n_lines = n_ms if n_ms <= max_n_lines else max_n_lines
    n_col = math.ceil(n_ms / n_lines)
    # for histogram all events
    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col,
                                          'height_ratios': [1] * n_lines},
                             figsize=(30, 25))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)
    axes = axes.flatten()

    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        axes[ax_index].set_facecolor(background_color)

    real_ms_index = 0
    for ms_index, ms in enumerate(ms_to_analyse):
        if ms.spike_struct.spike_nums_dur is None:
            continue
        print(f"{ms.description} plot_twitch_ratio_activity")

        distribution_ratio = ratio_dict[ms.age][ms.description]
        distribution_ratio = np.array(distribution_ratio)
        threshold = 50
        n_twitches = len(distribution_ratio)
        # number of twitches with ratio sup or inf
        n_twitches_low_ratio = len(np.where(distribution_ratio < threshold)[0])
        n_twitches_high_ratio = len(np.where(distribution_ratio >= threshold)[0])
        if n_twitches > 0:
            perc_sum_activity_after = str(np.round((n_twitches_high_ratio / n_twitches) * 100, 1))
        else:
            perc_sum_activity_after = 50
        if len(distribution_ratio) == 0:
            continue
        display_misc.plot_hist_distribution(distribution_data=distribution_ratio,
                                            description=f"{ms.description} \n {n_twitches} twitches: "
                                                        f"{n_twitches_low_ratio} / {n_twitches_high_ratio} "
                                                        f"({perc_sum_activity_after} %)",
                                            param=param, x_range=(0, 100),
                                            path_results=param.path_results,
                                            tight_x_range=True,
                                            twice_more_bins=True,
                                            ax_to_use=axes[real_ms_index],
                                            color_to_use=colors[real_ms_index % len(colors)],
                                            v_line=threshold,
                                            xlabel="Ratio", save_formats=save_formats)
        real_ms_index += 1
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/ratio_twitches'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())


def get_threshold_sum_activity_for_time_periods(raster, time_periods, perc_threshold=95, n_surrogate=1000):
    """
    Take a raster and a list of periods (tuple of int), and will shuffle the data (rolling) n times and each time
    will measure the sum of activity in each period and return the percentile value of the distribution of all periods
    over the 1000 shuffling
    :param raster:
    :param time_periods:
    :param percentile_threshold:
    :return:
    """
    n_times = raster.shape[1]
    n_periods = len(time_periods)
    count = 0
    n_rand_sum = np.zeros(n_surrogate * n_periods)
    for i in np.arange(n_surrogate):
        copy_raster = np.copy(raster)
        for n, neuron_spikes in enumerate(copy_raster):
            # roll the data to a random displace number
            copy_raster[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
        for period in time_periods:
            n_rand_sum[count] = np.sum(np.sum(copy_raster[:, period[0]:period[1] + 1], axis=0))
            count += 1

    activity_threshold = np.percentile(n_rand_sum, perc_threshold)

    return activity_threshold


def get_thresholds_cell_transient_post_twitches(raster, time_periods_bool, perc_threshold=95, n_surrogate=1000):
    """
    Take a raster and an array of bool, and will shuffle the data (rolling) n times and each time
    will measure how many transient of each cell will be during this time period and return the percentile value of the distribution of all periods
    over the 1000 shuffling for each cell
    :param raster: should be onsets
    :param time_periods:
    :param percentile_threshold:
    :return:
    """
    n_times = raster.shape[1]
    n_cells = raster.shape[0]

    n_rand_sum = np.zeros((n_cells, n_surrogate))
    for i in np.arange(n_surrogate):
        copy_raster = np.copy(raster)
        for n, neuron_spikes in enumerate(copy_raster):
            # roll the data to a random displace number
            copy_raster[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
            # if raster wouldn't be onsets, and rasterdur instead, we could transfer the raster_dur such that
            # each transient for a given cell has a unique id and then count how many unique id are in the bool
            n_rand_sum[n, i] = np.sum(copy_raster[n, time_periods_bool])

    activity_threshold = np.percentile(n_rand_sum, perc_threshold, axis=1)
    print(f"get_thresholds_cell_transient_post_twitches: len(activity_threshold) {len(activity_threshold)}")

    return activity_threshold


def select_cells_that_fire_during_time_periods(raster, time_periods_bool, description="", perc_threshold=95,
                                               n_surrogate=1000):
    """
    Take a raster and a boolean array reprensenting periods.
    :param raster: represents the onsets
    :param time_periods:
    :param description: a string representing the data analysed
    :param perc_threshold:
    :param n_surrogate:
    :return: A list of tuple representing the periods that pass the threshold
    """

    activity_threshold = get_thresholds_cell_transient_post_twitches(raster, time_periods_bool,
                                                                     perc_threshold=perc_threshold,
                                                                     n_surrogate=n_surrogate)
    # print(f"{description} threshold with {perc_threshold} p and {n_surrogate} surrogate: {activity_threshold}")
    significant_cells = []
    n_cells = raster.shape[0]
    for n in np.arange(n_cells):
        n_count = np.sum(raster[n, time_periods_bool])
        if n_count == 0:
            continue
        if n_count >= activity_threshold[n]:
            significant_cells.append(n)

    return significant_cells


def find_hubs_using_all_ms(ms_to_analyse, param):
    print(f"Finding hubs")
    ms_kept = []
    cells_connectivity = []
    bc_values = []
    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums_dur is None:
            print(f"{ms.description} no spike_nums_dur")
            continue
        ms_kept.append(ms)
        if ms.spike_struct.graph_out is None:
            print(f"{ms.description} detect_n_in_n_out")
            ms.detect_n_in_n_out()
        graph = ms.spike_struct.graph_out
        n_cells = ms.spike_struct.n_cells
        for cell in np.arange(n_cells):
            cells_connectivity.append(len(graph[cell]))
        bc_dict = nx.betweenness_centrality(graph)
        bc_values.extend(list(bc_dict.values()))
    # determining hubs for each ms
    print(f"Determining hubs over {len(ms_kept)} animals")
    for ms in ms_kept:
        if ms.spike_struct.graph_out is None:
            continue
        print("")
        print(f"{ms.description}")
        graph = ms.spike_struct.graph_out
        n_cells = ms.spike_struct.n_cells
        # print(f"{ms.description}: graph hubs: {n_cells} vs {len(graph)}")
        # first selecting cells conencted to more than 5% cells
        # we dot it among this particular mouse
        cells_connectivity_perc_threshold = 5
        # step 1
        cells_selected_s1 = []
        local_cells_connectivity = []
        for cell in np.arange(n_cells):
            local_cells_connectivity.append(len(graph[cell]))
            if ((len(graph[cell]) / n_cells) * 100) >= cells_connectivity_perc_threshold:
                cells_selected_s1.append(cell)
        if len(cells_selected_s1) == 0:
            print("Failed at step 1")
            continue
        # step 2
        cells_selected_s2 = []
        connec_treshold = np.percentile(cells_connectivity, 80)
        for cell in cells_selected_s1:
            if local_cells_connectivity[cell] >= connec_treshold:
                cells_selected_s2.append(cell)

        if len(cells_selected_s2) == 0:
            print("Failed at step 2")
            continue

        cells_selected_s3 = []
        local_bc_dict = nx.betweenness_centrality(graph)
        bc_perc_threshold = np.percentile(bc_values, 80)
        for cell in cells_selected_s2:
            if local_bc_dict[cell] >= bc_perc_threshold:
                cells_selected_s3.append(cell)
        if len(cells_selected_s3) == 0:
            print("Failed at step 3")
            continue
        print(f"HUB cells (n={len(cells_selected_s3)}): {cells_selected_s3}")


def do_stat_on_pca(ms_to_analyse, param, save_formats="pdf"):
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
              '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
              '#74add1', '#4575b4', '#313695']
    results_n_cpts_by_age_dict = dict()
    results_variance_by_session_dict = dict()
    n_sessions_dict = dict()
    variance_number = 95

    for ms in ms_to_analyse:
        if ms.raw_traces is None:
            continue
        if "p" + str(ms.age) not in results_n_cpts_by_age_dict:
            results_n_cpts_by_age_dict["p" + str(ms.age)] = []
            n_sessions_dict["p" + str(ms.age)] = set()
        results_variance_by_session_dict[ms.description] = []
        n_sessions_dict["p" + str(ms.age)].add(ms.description[:-4])
        # normalization of raw_traces
        raw_traces = ms.raw_traces
        raw_traces_z_score = np.zeros(raw_traces.shape)
        for cell in np.arange(raw_traces.shape[0]):
            raw_traces_z_score[cell] = (raw_traces[cell] - np.mean(raw_traces[cell])) / np.std(raw_traces[cell])
        pca = PCA(n_components=0.95, svd_solver='full')  #
        pca_result = pca.fit_transform(raw_traces_z_score)
        explained_variance = pca.explained_variance_
        n_components = len(explained_variance)
        results_n_cpts_by_age_dict["p" + str(ms.age)].append(n_components)
        results_variance_by_session_dict[ms.description].extend(explained_variance)
        # sum_activity = np.sum(ms.spike_struct.spike_nums_dur, axis=0)
        # # as percentage of cells
        # sum_activity = (sum_activity / ms.spike_struct.spike_nums_dur.shape[0]) * 100
        # results_n_cpts_by_age_dict["p" + str(ms.age)].append(np.var(sum_activity))

    for age, animals in n_sessions_dict.items():
        n_sessions_dict[age] = len(animals)

    box_plot_data_by_age(data_dict=results_n_cpts_by_age_dict, title="",
                         filename=f"n_component_for {variance_number}_variance_by_age",
                         y_label=f"N componenents explaining {variance_number}% variance",
                         colors=colors, with_scatters=True,
                         n_sessions_dict=n_sessions_dict,
                         path_results=param.path_results, scatter_size=200,
                         param=param, save_formats=save_formats)

    box_plot_data_by_age(data_dict=results_variance_by_session_dict, title="",
                         filename=f"variance_expained_by_session",
                         y_label=f"variance", y_log=True,
                         x_labels_rotation=45,
                         colors=colors, with_scatters=True,
                         path_results=param.path_results, scatter_size=80,
                         param=param, save_formats=save_formats)


def stats_on_graph_on_all_ms(ms_to_analyse, param, save_formats="pdf"):
    print(f"starting to do graph stats on all selected sessions")
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
              '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
              '#74add1', '#4575b4', '#313695']
    density_results_by_age_dict = dict()
    subgraph_len__results_by_age_dict = dict()
    n_sessions_dict = dict()
    for ms in ms_to_analyse:
        print(f"Current mouse session: {ms.description}")
        if ms.spike_struct.graph_out is None:
            print(f"Building graph")
            ms.detect_n_in_n_out()
            print(f"Graph is built")
        if ms.spike_struct.spike_nums is None:
            continue
        if "p" + str(ms.age) not in density_results_by_age_dict:
            density_results_by_age_dict["p" + str(ms.age)] = []
            subgraph_len__results_by_age_dict["p" + str(ms.age)] = []
            n_sessions_dict["p" + str(ms.age)] = set()
        n_sessions_dict["p" + str(ms.age)].add(ms.description[:-4])
        print(f"Computing graph density")
        rho = nx.density(ms.spike_struct.graph_out)
        rho = rho * 100  # Get in % of all possible connections
        print(f"Graph density is computed")
        # largest_subgraph_size = max(nx.connected_component_subgraphs(ms.spike_struct.graph_out), key=len)
        density_results_by_age_dict["p" + str(ms.age)].append(rho)
        # subgraph_len__results_by_age_dict["p" + str(ms.age)].append(largest_subgraph_size)

    for age, animals in n_sessions_dict.items():
        n_sessions_dict[age] = len(animals)

    box_plot_data_by_age(data_dict=density_results_by_age_dict, title="",
                         filename=f"network_density",
                         y_label=f"Density",
                         colors=colors, with_scatters=True,
                         n_sessions_dict=n_sessions_dict,
                         path_results=param.path_results, scatter_size=200,
                         param=param, save_formats=save_formats)

    # box_plot_data_by_age(data_dict=subgraph_len__results_by_age_dict, title="",
    #                      filename=f"largest_subgraph_size",
    #                      y_label=f"largest_subgraph_size",
    #                      colors=colors, with_scatters=True,
    #                      n_sessions_dict=n_sessions_dict,
    #                      path_results=param.path_results, scatter_size=200,
    #                      param=param, save_formats=save_formats)


def twitch_analysis_on_all_ms(ms_to_analyse, param, n_surrogates, before_extension, after_extension,
                              save_formats=["png", "pdf"], option="intersect"):
    print(f"starting twitches analysis on all selected sessions")
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
              '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
              '#74add1', '#4575b4', '#313695']
    results_by_age_dict = dict()
    # n_sessions_dict = dict()
    for ms in ms_to_analyse:
        if (ms.spike_struct.spike_nums_dur is None) or (ms.shift_data_dict is None):
            if ms.shift_data_dict is None:
                print(f"{ms.description} no shift data")
            continue
        if not (np.any(ms.shift_data_dict["shift_twitch"])):
            print(f"{ms.description} has no twitches")
            continue
        print(f"mouse current session: {ms.description}")
        # if "p" + str(ms.age) not in results_by_age_dict:
        #     results_by_age_dict[ms.description] = []
        # n_sessions_dict["p" + str(ms.age)] = set()
        # n_sessions_dict["p" + str(ms.age)].add(ms.description[:-4])
        results = twitch_analysis(ms, n_surrogates=n_surrogates, before_extension=before_extension,
                                  after_extension=after_extension)
        distrib, co_var_matrix, rnd_distrib_list, rnd_co_var_matrix_list = results
        results_by_age_dict[ms.description] = distrib
        results_for_this_ms = dict()
        results_for_this_ms[ms.description] = distrib
        results_for_this_ms["random\nactivation"] = [item for sublist in rnd_distrib_list for item in sublist]
        box_plot_data_by_age(data_dict=results_for_this_ms, title="",
                             filename=f"{ms.description}_{option}_on_twitches_vs_random_activation",
                             y_label=f"{option}",
                             colors=colors, with_scatters=False,
                             fliers_symbol="wo",
                             x_labels_rotation=45,
                             path_results=param.path_results, scatter_size=50,
                             param=param, save_formats=save_formats)
    # for age, animals in n_sessions_dict.items():
    #     n_sessions_dict[age] = len(animals)

    box_plot_data_by_age(data_dict=results_by_age_dict, title="",
                         filename=f"{option}_on_twitches_by_age",
                         y_label=f"{option}",
                         colors=colors, with_scatters=True,
                         x_labels_rotation=45,
                         path_results=param.path_results, scatter_size=50,
                         param=param, save_formats=save_formats)


def plot_variance_according_to_sum_of_activity(ms_to_analyse, param, save_formats="pdf"):
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
              '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
              '#74add1', '#4575b4', '#313695']
    results_by_age_dict = dict()
    n_sessions_dict = dict()
    # one point by cell
    results_by_age_with_all_cells_dict = dict()
    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums_dur is None:
            continue
        if "p" + str(ms.age) not in results_by_age_dict:
            results_by_age_dict["p" + str(ms.age)] = []
            results_by_age_with_all_cells_dict["p" + str(ms.age)] = []
            n_sessions_dict["p" + str(ms.age)] = set()
        n_sessions_dict["p" + str(ms.age)].add(ms.description[:-4])
        sum_activity = np.sum(ms.spike_struct.spike_nums_dur, axis=0)
        # as percentage of cells
        sum_activity = (sum_activity / ms.spike_struct.spike_nums_dur.shape[0]) * 100
        results_by_age_dict["p" + str(ms.age)].append(np.var(sum_activity))

    for age, animals in n_sessions_dict.items():
        n_sessions_dict[age] = len(animals)

    box_plot_data_by_age(data_dict=results_by_age_dict, title="",
                         filename=f"variance_on_sum_activity_by_age",
                         y_label=f"Variance",
                         colors=colors, with_scatters=True,
                         n_sessions_dict=n_sessions_dict,
                         path_results=param.path_results, scatter_size=200,
                         param=param, save_formats=save_formats)


def plot_nb_transients_in_mvt_vs_nb_total_transients(ms_to_analyse, param, save_formats="pdf"):
    # key is p_age, and value a list of the % of cell in that period of time
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']

    results_by_age_dict = dict()
    n_sessions_dict = dict()
    # one point by cell
    results_by_age_with_all_cells_dict = dict()
    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums is None:
            continue
        if "p" + str(ms.age) not in results_by_age_dict:
            results_by_age_dict["p" + str(ms.age)] = []
            results_by_age_with_all_cells_dict["p" + str(ms.age)] = []
            n_sessions_dict["p" + str(ms.age)] = set()
        n_sessions_dict["p" + str(ms.age)].add(ms.description[:-4])
        n_transients_total = 0
        n_transients_during_mvt_total = 0
        list_ratio = []
        n_frames = ms.spike_struct.spike_nums.shape[1]
        n_cells = ms.spike_struct.spike_nums.shape[0]
        shift_bool = np.zeros(n_frames, dtype="bool")
        # creating the boolean array, True means the mouse is moving
        shift_keys_to_loop = ["shift_twitch", "shift_long", "shift_unclassified"]
        for shift_key in shift_keys_to_loop:
            shift_bool_tmp = ms.shift_data_dict[shift_key]
            if shift_key == "shift_twitch":
                extension_frames_after = 15
                extension_frames_before = 1
            else:
                extension_frames_after = 0
                extension_frames_before = 0
            # we extend each period, implementation is not the fastest and more elegant way
            true_frames = np.where(shift_bool_tmp)[0]
            for frame in true_frames:
                first_frame = max(0, frame - extension_frames_before)
                last_frame = min(n_frames - 1, frame + extension_frames_after)
                shift_bool[first_frame:last_frame + 1] = True

        spike_nums_dur_numbers = give_unique_id_to_each_transient_of_raster_dur(ms.spike_struct.spike_nums)
        for cell in np.arange(n_cells):
            # -1 to not take into consideraiton the number attributed to empty frames
            n_transients_in_cell = len(np.unique(spike_nums_dur_numbers[cell])) - 1
            n_transients_total += n_transients_in_cell
            n_transients_in_cell_and_during_mvt = len(np.unique(spike_nums_dur_numbers[cell, shift_bool])) - 1
            n_transients_during_mvt_total += n_transients_in_cell_and_during_mvt
            if n_transients_in_cell > 0:
                list_ratio.append((n_transients_in_cell_and_during_mvt / n_transients_in_cell) * 100)
        print(f"{ms.description}: n_transients_during_mvt_total {n_transients_during_mvt_total}, "
              f"n_transients_total {n_transients_total}")
        ratio_ms = (n_transients_during_mvt_total / n_transients_total) * 100
        results_by_age_dict["p" + str(ms.age)].append(ratio_ms)
        results_by_age_with_all_cells_dict["p" + str(ms.age)].extend(list_ratio)

    for age, animals in n_sessions_dict.items():
        n_sessions_dict[age] = len(animals)

    box_plot_data_by_age(data_dict=results_by_age_dict, title="",
                         filename=f"ratio_transients_in_mvt_by_age",
                         y_label=f"Transients associated to mvt (%)",
                         colors=colors, with_scatters=True,
                         n_sessions_dict=n_sessions_dict,
                         path_results=param.path_results, scatter_size=200,
                         param=param, save_formats=save_formats)
    box_plot_data_by_age(data_dict=results_by_age_with_all_cells_dict, title="",
                         filename=f"ratio_transients_in_mvt_by_age_one_value_by_cell",
                         y_label=f"Transients associated to mvt (%)",
                         colors=colors, with_scatters=False,
                         n_sessions_dict=n_sessions_dict,
                         path_results=param.path_results, scatter_size=40,
                         param=param, save_formats=save_formats)


def plot_cells_that_fire_during_time_periods(ms_to_analyse, shift_keys, param, perc_threshold=95, n_surrogate=1000,
                                             save_formats="pdf"):
    # key is p_age, and value a list of the % of cell in that period of time
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']

    n_shift_keys = len(shift_keys)

    for step in np.arange(n_shift_keys + 2):
        results_dict = dict()
        animals_dict = dict()

        shift_key_descr = None
        for ms in ms_to_analyse:
            if ms.spike_struct.spike_nums is None:
                continue
            if ("p" + str(ms.age)) not in results_dict:
                results_dict[("p" + str(ms.age))] = []
                animals_dict["p" + str(ms.age)] = set()
            animals_dict["p" + str(ms.age)].add(ms.description[:-4])
            n_frames = ms.spike_struct.spike_nums.shape[1]
            n_cells = ms.spike_struct.spike_nums.shape[0]
            shift_bool = np.zeros(n_frames, dtype="bool")

            if step < n_shift_keys:
                shift_keys_to_loop = [shift_keys[step]]
                shift_key_descr = shift_keys[step]
            else:
                if step == n_shift_keys:
                    shift_keys_to_loop = shift_keys
                    shift_key_descr = "all_mvt"
                    # for now we don't want all_mvt
                    continue
                else:
                    shift_keys_to_loop = shift_keys
                    shift_key_descr = "still"

            for shift_key in shift_keys_to_loop:
                shift_bool_tmp = ms.shift_data_dict[shift_key]
                if shift_key == "shift_twitch":
                    extension_frames_after = 15
                    extension_frames_before = 1
                else:
                    extension_frames_after = 0
                    extension_frames_before = 0
                # we extend each period, implementation is not the fastest and more elegant way
                true_frames = np.where(shift_bool_tmp)[0]
                for frame in true_frames:
                    first_frame = max(0, frame - extension_frames_before)
                    last_frame = min(n_frames - 1, frame + extension_frames_after)
                    shift_bool[first_frame:last_frame + 1] = True

                # shift_time_periods = get_continous_time_periods(shift_bool.astype("int8"))

            if shift_key_descr == "still":
                # then we revert the all_mvt result
                shift_bool = np.invert(shift_bool)

            significant_cells = \
                select_cells_that_fire_during_time_periods(raster=ms.spike_struct.spike_nums,
                                                           time_periods_bool=shift_bool,
                                                           description=f"{ms.description}_{shift_key_descr}",
                                                           perc_threshold=perc_threshold,
                                                           n_surrogate=n_surrogate)
            print(f"{ms.description}: significant {len(significant_cells)}, all {len(ms.spike_struct.spike_nums)}")
            if len(significant_cells) == 0:
                continue
            plot_distrib_participation_to_event = True
            if plot_distrib_participation_to_event:
                shift_binary = shift_bool.astype("int8")
                periods_shift = get_continous_time_periods(shift_binary)
                shift_numbers = np.ones(n_frames, dtype="int16")
                shift_numbers = shift_numbers * -1
                for period_index, period in enumerate(periods_shift):
                    shift_numbers[period[0]:period[1] + 1] = period_index
                if len(significant_cells) > 0:
                    ratio_spikes_total_events_significant_cells = get_ratio_spikes_on_events_vs_total_events_by_cell(
                        spike_nums=ms.spike_struct.spike_nums[np.array(significant_cells)],
                        spike_nums_dur=ms.spike_struct.spike_nums_dur[np.array(significant_cells)],
                        sce_times_numbers=shift_numbers,
                        use_only_onsets=False)
                ratio_spikes_total_events = get_ratio_spikes_on_events_vs_total_events_by_cell(
                    spike_nums=ms.spike_struct.spike_nums,
                    spike_nums_dur=ms.spike_struct.spike_nums_dur,
                    sce_times_numbers=shift_numbers,
                    use_only_onsets=False)

                if len(significant_cells) > 0:
                    ratio_spikes_events_significant_cells = get_ratio_spikes_on_events_vs_total_spikes_by_cell(
                        spike_nums=ms.spike_struct.spike_nums[np.array(significant_cells)],
                        spike_nums_dur=ms.spike_struct.spike_nums_dur[np.array(significant_cells)],
                        sce_times_numbers=shift_numbers, use_only_onsets=False)

                ratio_spikes_events = get_ratio_spikes_on_events_vs_total_spikes_by_cell(
                    spike_nums=ms.spike_struct.spike_nums,
                    spike_nums_dur=ms.spike_struct.spike_nums_dur,
                    sce_times_numbers=shift_numbers, use_only_onsets=False)

                n_bins = 30
                values_to_scatter = []
                labels = []
                scatter_shapes = []
                colors = []
                values_to_scatter.append(np.mean(ratio_spikes_total_events_significant_cells))
                labels.extend(["mean"])
                scatter_shapes.extend(["o"])
                colors.extend(["white"])
                plot_hist_distribution(distribution_data=ratio_spikes_total_events_significant_cells,
                                       description=f"{ms.description}_hist_spike_total_{shift_key_descr}_"
                                                   f"ratio_significant_"
                                                   f"{len(ratio_spikes_total_events_significant_cells)}_cells",
                                       xlabel=f"spikes in {shift_key_descr} vs total {shift_key_descr} (%)",
                                       values_to_scatter=np.array(values_to_scatter),
                                       labels=labels,
                                       twice_more_bins=True,
                                       scatter_shapes=scatter_shapes,
                                       colors=colors,
                                       n_bins=n_bins,
                                       param=param)

                values_to_scatter = []
                values_to_scatter.append(np.mean(ratio_spikes_events_significant_cells))
                plot_hist_distribution(distribution_data=ratio_spikes_events_significant_cells,
                                       description=f"{ms.description}_hist_spike_total_{shift_key_descr}_vs_all_spikes"
                                                   f"_ratio_significant_"
                                                   f"{len(ratio_spikes_events_significant_cells)}_cells",
                                       xlabel=f"spikes in {shift_key_descr} vs total spikes (%)",
                                       values_to_scatter=np.array(values_to_scatter),
                                       labels=labels,
                                       twice_more_bins=True,
                                       scatter_shapes=scatter_shapes,
                                       colors=colors,
                                       n_bins=n_bins,
                                       param=param)

                values_to_scatter = []
                values_to_scatter.append(np.mean(ratio_spikes_total_events))
                plot_hist_distribution(distribution_data=ratio_spikes_total_events,
                                       description=f"{ms.description}_hist_spike_total_{shift_key_descr}_ratio_"
                                                   f"{len(ratio_spikes_total_events)}_cells",
                                       xlabel=f"spikes in {shift_key_descr} vs total {shift_key_descr} (%)",
                                       values_to_scatter=np.array(values_to_scatter),
                                       labels=labels,
                                       scatter_shapes=scatter_shapes,
                                       colors=colors,
                                       n_bins=n_bins,
                                       param=param)

                values_to_scatter = []
                values_to_scatter.append(np.mean(ratio_spikes_events))
                plot_hist_distribution(distribution_data=ratio_spikes_events,
                                       description=f"{ms.description}_hist_spike_total_{shift_key_descr}_vs_all_spikes"
                                                   f"_ratio"
                                                   f"{len(ratio_spikes_events)}_cells",
                                       xlabel=f"spikes in {shift_key_descr} vs total spikes (%)",
                                       values_to_scatter=np.array(values_to_scatter),
                                       labels=labels,
                                       scatter_shapes=scatter_shapes,
                                       colors=colors,
                                       n_bins=n_bins,
                                       param=param)

                display_misc.plot_scatters(ratio_spikes_total_events_significant_cells,
                                           ratio_spikes_events_significant_cells,
                                           size_scatter=30, ax_to_use=None, color_to_use=None, legend_str="",
                                           xlabel="spikes on total events", ylabel="spikes on total spikes",
                                           filename_option=f"{ms.description}_spikes_on_all_spikes_vs_all_events_{shift_key_descr}_significant",
                                           param=param, y_lim=(0, 100), x_lim=(0, 100),
                                           save_formats="pdf")

                display_misc.plot_scatters(ratio_spikes_total_events, ratio_spikes_events,
                                           size_scatter=30, ax_to_use=None, color_to_use=None, legend_str="",
                                           xlabel="spikes on total events", ylabel="spikes on total spikes",
                                           filename_option=f"{ms.description}_spikes_on_all_spikes_vs_all_events_{shift_key_descr}",
                                           param=param, y_lim=(0, 100), x_lim=(0, 100),
                                           save_formats="pdf")

                # TODO: Take the mean of each distribution and do a boxplot by age with those values

            print(f"{ms.description}: {shift_key_descr} analysis")
            print(f"{n_cells} cells")
            print(f"{len(significant_cells)} significant cells")
            ratio_significant_cells = (len(significant_cells) / n_cells) * 100
            results_dict[("p" + str(ms.age))].append(ratio_significant_cells)
        # in case no ms would have been analysed
        if shift_key_descr is not None:
            for age, animals in animals_dict.items():
                animals_dict[age] = len(animals)
            box_plot_data_by_age(data_dict=results_dict, title="", filename=f"cells_associated_to_{shift_key_descr}",
                                 n_sessions_dict=animals_dict,
                                 y_label=f"Cells associated to {shift_key_descr} (%)", colors=colors,
                                 with_scatters=True,
                                 path_results=param.path_results, scatter_size=200,
                                 param=param, save_formats=save_formats)


def select_significant_time_periods(raster, time_periods, description="", perc_threshold=95, n_surrogate=1000):
    """
    Take a raster and a list of tuple representing periods.
    :param raster:
    :param time_periods:
    :param description: a string representing the data analysed
    :param perc_threshold:
    :param n_surrogate:
    :return: A list of tuple representing the periods that pass the threshold
    """

    activity_threshold = get_threshold_sum_activity_for_time_periods(raster, time_periods,
                                                                     perc_threshold=perc_threshold,
                                                                     n_surrogate=n_surrogate)
    print(f"{description} threshold with {perc_threshold} p and {n_surrogate} surrogate: {activity_threshold}")
    significant_time_periods = []
    significant_time_periods_indices = []
    for period_index, period in enumerate(time_periods):
        sum_activity = np.sum(np.sum(raster[:, period[0]:period[1] + 1], axis=0))
        if sum_activity >= activity_threshold:
            significant_time_periods.append(period)
            significant_time_periods_indices.append(period_index)

    return significant_time_periods, significant_time_periods_indices


def plot_covnorm_matrix(m_sces, n_clusters, cluster_labels, param, data_descr):
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))
    # display the normlized covariance matrix organized by cluster of SCE such as detected by initial kmeans
    # contains the neurons from the SCE, but ordered by cluster
    ordered_m_sces = np.zeros((np.shape(m_sces)[0], np.shape(m_sces)[1]))
    # to plot line that separate clusters
    cluster_coord_thresholds = []
    cluster_x_ticks_coord = []
    start = 0
    for k in np.arange(-1, n_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        ordered_m_sces[start:start + nb_k, :] = m_sces[e, :]
        ordered_m_sces[:, start:start + nb_k] = m_sces[:, e]
        start += nb_k
        if (k + 1) < n_clusters:
            if k == -1:
                cluster_x_ticks_coord.append(start / 2)
            else:
                cluster_x_ticks_coord.append((start + cluster_coord_thresholds[-1]) / 2)
            cluster_coord_thresholds.append(start)
        else:
            cluster_x_ticks_coord.append((start + cluster_coord_thresholds[-1]) / 2)

    co_var = np.corrcoef(ordered_m_sces)  # cov
    # sns.set()
    result = sns.heatmap(co_var, cmap="Blues", ax=ax1)  # , vmin=0, vmax=1) YlGnBu  cmap="jet" Blues
    # ax1.hlines(cluster_coord_thresholds, 0, np.shape(co_var)[0], color="black", linewidth=1,
    #            linestyles="dashed")
    for n_c, clusters_threshold in enumerate(cluster_coord_thresholds):
        # if (n_c+1) == len(cluster_coord_thresholds):
        #     break
        x_begin = 0
        if n_c > 0:
            x_begin = cluster_coord_thresholds[n_c - 1]
        x_end = np.shape(co_var)[0]
        if n_c < len(cluster_coord_thresholds) - 1:
            x_end = cluster_coord_thresholds[n_c + 1]
        ax1.hlines(clusters_threshold, x_begin, x_end, color="black", linewidth=2,
                   linestyles="dashed")
    for n_c, clusters_threshold in enumerate(cluster_coord_thresholds):
        # if (n_c+1) == len(cluster_coord_thresholds):
        #     break
        y_begin = 0
        if n_c > 0:
            y_begin = cluster_coord_thresholds[n_c - 1]
        y_end = np.shape(co_var)[0]
        if n_c < len(cluster_coord_thresholds) - 1:
            y_end = cluster_coord_thresholds[n_c + 1]
        ax1.vlines(clusters_threshold, y_begin, y_end, color="black", linewidth=2,
                   linestyles="dashed")
    # ax1.xaxis.get_majorticklabels().set_rotation(90)
    # plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    # plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    ax1.set_xticks(cluster_x_ticks_coord)
    ax1.set_xticklabels(np.arange(n_clusters))
    ax1.set_yticks(cluster_x_ticks_coord)
    ax1.set_yticklabels(np.arange(n_clusters))
    ax1.set_title(f"{np.shape(m_sces)[0]} SCEs")
    # ax1.xaxis.set_tick_params(labelsize=5)
    # ax1.yaxis.set_tick_params(labelsize=5)
    ax1.invert_yaxis()

    save_formats = ["pdf"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    path_results = param.path_results
    for save_format in save_formats:
        fig.savefig(f'{path_results}/{data_descr}_covariance_matrix_clustered'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())

    plt.close()


def try_hdbscan(cells_in_sce, param, data_descr, activity_threshold=None,
                spike_nums=None, SCE_times=None, use_co_var=True):
    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']

    m_sces = cells_in_sce
    # m_sces = spike_nums
    # normalized covariance matrix
    if use_co_var:
        m_sces = covnorm(m_sces)
    # m_sces = np.corrcoef(m_sces)
    # print(f"m_sces.shape {m_sces.shape}")

    if use_co_var:
        metric = 'precomputed'
    else:
        metric = 'euclidean'
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                metric=metric, min_cluster_size=2, min_samples=None, p=None)
    # metric='precomputed' euclidean
    clusterer.fit(m_sces)

    labels = clusterer.labels_

    print(f"labels.shape: {labels.shape}")
    print(f"N clusters hdbscan: {labels.max() + 1}")
    print(f"labels: {labels}")

    print(f"With no clusters hdbscan: {len(np.where(labels == -1)[0])}")

    if use_co_var:
        plot_covnorm_matrix(m_sces=m_sces, n_clusters=labels.max() + 1, cluster_labels=labels, param=param,
                            data_descr=data_descr)
        return
    if spike_nums is None:
        return

    n_cells = spike_nums.shape[0]
    clustered_spike_nums = np.zeros(spike_nums.shape, dtype="int8")
    cells_to_highlight = []
    cells_to_highlight_colors = []
    count_cells = 0
    cell_labels = np.zeros(0, dtype="int8")
    for i, cluster_id in enumerate(np.arange(labels.max() + 1)):
        cells_in_clusters = np.where(labels == cluster_id)[0]
        n_cells_in_cluster = len(cells_in_clusters)
        cell_labels = np.concatenate([cell_labels, cells_in_clusters])
        cells_to_highlight.extend(list(np.arange(count_cells, count_cells + n_cells_in_cluster)))
        cells_to_highlight_colors.extend([colors[i % len(colors)]] * n_cells_in_cluster)
        clustered_spike_nums[count_cells:count_cells + n_cells_in_cluster, :] = spike_nums[cells_in_clusters, :]
        count_cells += n_cells_in_cluster

    cells_in_clusters = np.where(labels == -1)[0]
    n_cells_in_cluster = len(cells_in_clusters)
    cells_to_highlight.extend(list(np.arange(count_cells, n_cells)))
    cells_to_highlight_colors.extend(["white"] * n_cells_in_cluster)
    clustered_spike_nums[count_cells:, :] = spike_nums[cells_in_clusters, :]
    cell_labels = np.concatenate([cell_labels, cells_in_clusters])
    count_cells += n_cells_in_cluster

    plot_spikes_raster(spike_nums=clustered_spike_nums, param=param,
                       spike_train_format=False,
                       title=f"hdbscan clusters raster plot {data_descr}",
                       file_name=f"spike_nums_{data_descr}_hdbscan",
                       y_ticks_labels=cell_labels,
                       y_ticks_labels_size=2,
                       save_raster=True,
                       show_raster=False,
                       plot_with_amplitude=False,
                       activity_threshold=activity_threshold,
                       raster_face_color='black',
                       cell_spikes_color='white',
                       span_area_coords=[SCE_times],
                       span_area_colors=['white'],
                       cells_to_highlight=cells_to_highlight,
                       cells_to_highlight_colors=cells_to_highlight_colors,
                       sliding_window_duration=1,
                       show_sum_spikes_as_percentage=True,
                       spike_shape="o",
                       spike_shape_size=0.2,
                       save_formats="pdf",
                       SCE_times=SCE_times)
    raise Exception("NOT TODAY HDBSCAN")


def stat_significant_time_period(ms_to_analyse, shift_key, perc_threshold=95, n_surrogate=1000):
    for ms in ms_to_analyse:
        if ms.spike_struct.spike_nums_dur is None:
            continue
        n_frames = ms.spike_struct.spike_nums_dur.shape[1]
        shift_bool = ms.shift_data_dict[shift_key]
        if shift_key == "shift_twitch":
            extension_frames_after = 15
            extension_frames_before = 0
        else:
            extension_frames_after = 20
            extension_frames_before = 20
        # we extend each period, implementation is not the fastest and more elegant way
        true_frames = np.where(shift_bool)[0]
        for frame in true_frames:
            first_frame = max(0, frame - extension_frames_before)
            last_frame = min(n_frames - 1, frame + extension_frames_after)
            shift_bool[first_frame:last_frame + 1] = True

        shift_time_periods = get_continous_time_periods(shift_bool.astype("int8"))

        significant_time_periods, significant_time_periods_indices = \
            select_significant_time_periods(raster=ms.spike_struct.spike_nums_dur, time_periods=shift_time_periods,
                                            description=f"{ms.description}_{shift_key}", perc_threshold=perc_threshold,
                                            n_surrogate=n_surrogate)

        print(f"{ms.description}: {shift_key} analysis")
        print(f"{len(shift_time_periods)} n {shift_key}")
        print(f"{len(significant_time_periods)} n significant {shift_key}")

        # print(f"significant periods: ")

        significant_shift_bool = np.zeros(len(shift_bool), dtype=bool)
        non_significant_shift_bool = np.zeros(len(shift_bool), dtype=bool)
        for i in np.arange(len(shift_time_periods)):
            if i in significant_time_periods_indices:
                # print(f"{shift_time_periods[i]}")
                significant_shift_bool[shift_time_periods[i][0]:shift_time_periods[i][1] + 1] = True
            else:
                non_significant_shift_bool[shift_time_periods[i][0]:shift_time_periods[i][1] + 1] = True

        new_periods_dict = dict()
        new_periods_dict["significant_twitches"] = significant_shift_bool
        new_periods_dict["non_significant_twitches"] = non_significant_shift_bool
        # plotting the significant twitches
        ms.plot_raster_with_periods(new_periods_dict, bonus_title="_significant",
                                    with_cell_assemblies=False)


def use_rastermap_for_pca(ms, path_results, file_name):
    model = Rastermap(n_components=1, n_X=30, nPC=200, init='pca')


def elephant_cad(ms, param, save_formats="pdf"):
    if ms.spike_struct.spike_nums is None:
        print(f"elephant_cad {ms.description} ms.spike_struct.spike_nums should not be None")

    # first we create a spike_trains in the neo format
    spike_trains = []
    n_cells, n_times = ms.spike_struct.spike_nums.shape
    for cell in np.arange(n_cells):
        cell_spike_nums = ms.spike_struct.spike_nums[cell]
        spike_frames = np.where(cell_spike_nums)[0]
        # convert frames in s
        spike_frames = spike_frames / ms.sampling_rate
        neo_spike_train = neo.SpikeTrain(times=spike_frames, units='s',
                                         t_stop=n_times / ms.sampling_rate)
        spike_trains.append(neo_spike_train)

    binsize = 100 * pq.ms
    spike_trains_binned = elephant_conv.BinnedSpikeTrain(spike_trains, binsize=binsize)
    assembly_bin = cad.cell_assembly_detection(data=spike_trains_binned, maxlag=5, verbose=True)
    print(f"assembly_bin {assembly_bin}")

    """
    assembly_bin
    contains the assemblies detected for the binsize chosen each assembly is a dictionary with attributes: 
    ‘neurons’ : vector of units taking part to the assembly

    (unit order correspond to the agglomeration order)
    
    ‘lag’ : vector of time lags lag[z] is the activation delay between
    neurons[1] and neurons[z+1]
    
    ‘pvalue’ : vector of pvalues. pvalue[z] is the p-value of the
    statistical test between performed adding neurons[z+1] to the neurons[1:z]
    
    ‘times’ : assembly activation time. It reports how many times the
    complete assembly activates in that bin. time always refers to the activation of the first listed assembly element 
    (neurons[1]), that doesn’t necessarily corresponds to the first unit firing. 
    The format is identified by the variable bool_times_format.
    
    ‘signature’ : array of two entries (z,c). The first is the number of
    neurons participating in the assembly (size), the second is number of assembly occurrences.
    """

    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
              '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    cells_to_highlight_colors = []
    cells_to_highlight = []
    cell_new_order = []
    all_cells = np.arange(n_cells)
    cell_index_so_far = 0
    for ca_index, cell_assembly in enumerate(assembly_bin):
        cell_new_order.extend(cell_assembly['neurons'])
        n_cells_in_ca = len(cell_assembly['neurons'])
        cells_to_highlight.extend(np.arange(cell_index_so_far, cell_index_so_far + n_cells_in_ca))
        cell_index_so_far += n_cells_in_ca
        cells_to_highlight_colors.extend([colors[ca_index % len(colors)]] * n_cells_in_ca)

    cell_new_order.extend(list(np.setdiff1d(all_cells, cell_new_order)))
    cell_new_order = np.array(cell_new_order)
    plot_spikes_raster(spike_nums=ms.spike_struct.spike_nums[cell_new_order],
                       param=ms.param,
                       spike_train_format=False,
                       title=f"{ms.description}",
                       file_name=f"{ms.description}_raster_with_elephant_cad",
                       y_ticks_labels=np.arange(n_cells),
                       y_ticks_labels_size=2,
                       save_raster=True,
                       show_raster=False,
                       plot_with_amplitude=False,
                       show_sum_spikes_as_percentage=False,
                       cells_to_highlight=cells_to_highlight,
                       cells_to_highlight_colors=cells_to_highlight_colors,
                       span_area_only_on_raster=False,
                       spike_shape='o',
                       spike_shape_size=0.05,
                       save_formats=["pdf", "png"])


def fca_clustering_on_twitches_activity(ms, param, save_formats="pdf"):
    if ms.spike_struct.spike_nums_dur is None:
        return
    n_frames = ms.spike_struct.spike_nums_dur.shape[1]
    n_cells = ms.spike_struct.spike_nums_dur.shape[0]
    shift_bool = ms.shift_data_dict["shift_twitch"]
    n_twiches = len(shift_bool)
    extension_frames_after = 15
    extension_frames_before = 0
    # we extend each period, implementation is not the fastest and more elegant way
    true_frames = np.where(shift_bool)[0]
    for frame in true_frames:
        first_frame = max(0, frame - extension_frames_before)
        last_frame = min(n_frames - 1, frame + extension_frames_after)
        shift_bool[first_frame:last_frame + 1] = True

    shift_time_periods = get_continous_time_periods(shift_bool.astype("int8"))

    spike_nums = np.zeros((n_cells, n_twiches), dtype="int8")
    shift_time_numbers = np.zeros(n_twiches, dtype="int16")
    for period_index, period in enumerate(shift_time_periods):
        shift_time_numbers[period[0]:period[1] + 1] = period_index
        cells_active_in_twitch = np.where(np.sum(ms.spike_struct.spike_nums_dur[:, period[0]:period[1] + 1], axis=1))[0]
        spike_nums[cells_active_in_twitch, period_index] = 1

    spike_trains = []
    for cell_spikes in spike_nums:
        spike_trains.append(np.where(cell_spikes)[0].astype(float))
    # just for stat, not useful
    perc_threshold = 95
    n_surrogate_activity_threshold = 100
    # useful
    n_surrogate_fca = 20
    sliding_window_duration = 1
    # sigma is the std of the random distribution used to jitter the data
    sigma = 20
    jitter_range = n_twiches // 8
    compute_and_plot_clusters_raster_fca_version(spike_trains=spike_trains,
                                                 spike_nums=spike_nums,
                                                 data_descr=ms.description, param=param,
                                                 sliding_window_duration=sliding_window_duration,
                                                 SCE_times=None,
                                                 sce_times_numbers=shift_time_numbers,
                                                 perc_threshold=perc_threshold,
                                                 n_surrogate_activity_threshold=
                                                 n_surrogate_activity_threshold,
                                                 sigma=sigma, n_surrogate_fca=n_surrogate_fca,
                                                 labels=np.arange(n_cells),
                                                 jitter_range=jitter_range,
                                                 activity_threshold=None,
                                                 fca_early_stop=True,
                                                 with_cells_in_cluster_seq_sorted=
                                                 False,
                                                 use_uniform_jittering=True)


def remove_spike_nums_dur_and_associated_transients(spike_nums_dur, frames_to_keep):
    """

    :param spike_nums_dur: 2-d binary array
    :param frames_to_keep: Array of len n_frames of one dimension, of type bool, True is cell to keep
    :return:
    """
    spike_nums_dur = np.copy(spike_nums_dur)
    spike_nums_dur_transient_id = spike_nums_dur.astype("int16")
    n_cells = len(spike_nums_dur)
    frames_to_remove = np.invert(frames_to_keep)
    # first for each cell, we create a copy of spike_nums_dur in which each transient is identified
    # (unique id for a cell)
    for cell in np.arange(n_cells):
        periods = get_continous_time_periods(spike_nums_dur[cell])
        for period_index, period in enumerate(periods):
            spike_nums_dur_transient_id[cell, period[0]:period[1] + 1] = period_index
        # get transients id that are in the frames to remove
        transients_id_to_remove = np.unique(spike_nums_dur_transient_id[cell, frames_to_remove])
        if len(transients_id_to_remove) > 0:
            for transient_id in transients_id_to_remove:
                indices = np.where(spike_nums_dur_transient_id[cell] == transient_id)[0]
                # removing the transient
                spike_nums_dur[cell, indices] = 0
    spike_nums_dur = spike_nums_dur[:, frames_to_keep]
    return spike_nums_dur


def lexi_loading_process(param, load_traces):
    ms_str_to_load = ["ms"]

    # loading data
    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_str_to_load, param=param,
                                            load_traces=load_traces)
    return ms_str_to_ms_dict


def add_z_shifts_from_file(ms_str_to_ms_dict, param):
    file_name = "Z_movement_movies_to_cut.txt"
    ms_session = None
    z_shifts = None
    with open(os.path.join(param.path_data, file_name), "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            if len(line) < 4:
                continue
            if line[0].lower() == "p":
                if ms_session is not None and z_shifts is not None:
                    ms_str_to_ms_dict[ms_session].z_shift_periods = z_shifts
                line = line.strip("\n")
                line = line.strip(" ")
                line = line.strip("*")
                line = line.strip(" ")
                ms_session = line.lower() + "_ms"
                if ms_session not in ms_str_to_ms_dict:
                    ms_session = None
                    z_shifts = None
                    # print(f"not in ms_str_to_ms_dict {ms_session}")
                else:
                    z_shifts = []
            elif ms_session is not None:
                split_values = line.split("-")
                z_shifts.append((int(split_values[0]), int(split_values[1])))
        if ms_session is not None and z_shifts is not None:
            ms_str_to_ms_dict[ms_session].z_shift_periods = z_shifts


def robin_loading_process(param, load_traces, load_abf=False):
    # all avaialble session
    ms_str_to_load = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms",
                      "p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms",
                      "p7_171012_a000_ms",
                      "p7_17_10_18_a002_ms", "p7_17_10_18_a004_ms",
                      "p7_18_02_08_a000_ms", "p7_18_02_08_a001_ms",
                      "p7_18_02_08_a002_ms", "p7_18_02_08_a003_ms",
                      "p7_19_03_05_a000_ms",
                      "p7_19_03_27_a000_ms", "p7_19_03_27_a001_ms",
                      "p7_19_03_27_a002_ms",
                      "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms",
                      "p8_18_10_17_a000_ms", "p8_18_10_17_a001_ms",
                      "p8_18_10_24_a005_ms", "p8_18_10_24_a006_ms"
                                             "p8_19_03_19_a000_ms",
                      "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms",
                      "p9_18_09_27_a003_ms", "p9_19_02_20_a000_ms",
                      "p9_19_02_20_a001_ms", "p9_19_02_20_a002_ms",
                      "p9_19_02_20_a003_ms", "p9_19_03_14_a000_ms",
                      "p9_19_03_14_a001_ms", "p9_19_03_22_a000_ms",
                      "p9_19_03_22_a001_ms",
                      "p10_17_11_16_a003_ms", "p10_19_02_21_a002_ms",
                      "p10_19_02_21_a003_ms", "p10_19_02_21_a005_ms",
                      "p10_19_03_08_a000_ms", "p10_19_03_08_a001_ms",
                      "p11_17_11_24_a000_ms", "p11_17_11_24_a001_ms",
                      "p11_19_02_15_a000_ms", "p11_19_02_22_a000_ms",
                      "p12_17_11_10_a002_ms", "p12_171110_a000_ms",
                      "p13_18_10_29_a000_ms", "p13_18_10_29_a001_ms",
                      "p13_19_03_11_a000_ms",
                      "p14_18_10_23_a000_ms", "p14_18_10_30_a001_ms",
                      "p16_18_11_01_a002_ms",
                      "p19_19_04_08_a000_ms", "p19_19_04_08_a001_ms",

                      "p41_19_04_30_a000_ms"]
    # ,
    # gadcre_ms= [ ]
    # arnaud_ms = ["p60_arnaud_ms", "p60_a529_2015_02_25_ms"]
    # abf_corrupted = ["p8_18_10_17_a001_ms", "p9_18_09_27_a003_ms"]
    #
    # ms_with_piezo = ["p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms", "p7_18_02_08_a000_ms",
    #                  "p7_18_02_08_a001_ms", "p7_18_02_08_a002_ms", "p7_18_02_08_a003_ms", "p8_18_02_09_a000_ms",
    #                  "p8_18_02_09_a001_ms", "p8_18_10_17_a001_ms",
    #                  "p8_18_10_24_a005_ms", "p9_18_09_27_a003_ms", "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms"]
    # ms_with_run = ["p13_18_10_29_a001_ms", "p13_18_10_29_a000_ms"]
    # run_ms_str = ["p12_17_11_10_a000_ms", "p12_17_11_10_a002_ms", "p13_18_10_29_a000_ms",
    #               "p13_18_10_29_a001_ms"]
    # ms_10000_sampling = ["p8_18_10_17_a001_ms", "p9_18_09_27_a003_ms"]
    #
    # oriens_ms_str = ["p14_18_10_23_a001_ms"]
    #
    # ms_str_to_load = ["p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms",
    #                   "p8_18_10_24_a005_ms", "p8_18_10_17_a001_ms",
    #                   "p8_18_10_17_a000_ms",  # new
    #                   "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms",
    #                   "p9_18_09_27_a003_ms",  # new
    #                   "p10_17_11_16_a003_ms",
    #                   "p11_17_11_24_a001_ms", "p11_17_11_24_a000_ms",
    #                   "p12_17_11_10_a002_ms", "p12_171110_a000_ms",
    #                   "p13_18_10_29_a000_ms",  # new
    #                   "p13_18_10_29_a001_ms",
    #                   "p14_18_10_23_a000_ms",
    #                   "p14_18_10_30_a001_ms",
    #                   "p60_arnaud_ms"]
    # ms_new_from_Robin_2nd_dec = ["p12_171110_a000_ms", "p6_18_02_07_a002_ms"]
    # ms_str_to_load = available_ms_str
    # ms_str_to_load = ms_with_run
    # # ms_str_to_load = ["p60_a529_2015_02_25_v_arnaud_ms"]
    # ms_str_to_load = ["p7_18_02_08_a001_ms"]
    # ms_str_to_load = ["p10_17_11_16_a003_ms"]
    # ms_str_to_load = available_ms_str
    # ms_str_to_load = ["p9_18_09_27_a003_ms", "p10_17_11_16_a003_ms"]
    # ms_str_to_load = ms_with_cell_assemblies
    # ms_str_to_load = ["p6_18_02_07_a001_ms", "p12_17_11_10_a002_ms"]
    # ms_str_to_load = ["p60_arnaud_ms"]
    # ms_str_to_load = available_ms_str
    # ms_str_to_load = ["p6_18_02_07_a002_ms"]
    # ms_str_to_load = ms_with_piezo
    # ms_str_to_load = ms_with_piezo
    # ms_str_to_load = ["p7_17_10_18_a002_ms"]
    # # ms_str_to_load = ["p60_a529_2015_02_25_ms"]
    # ms_str_to_load = ms_new_from_Robin_2nd_dec
    # ms_str_to_load = ["p9_18_09_27_a003_ms"]
    # ms_str_to_load = ["p6_18_02_07_a001_ms"]
    # ms_str_to_load = ["p9_19_02_20_a003_ms"]
    # ms_str_to_load = ["p60_a529_2015_02_25_ms"]
    # no_spike_nums = ["p6_18_02_07_a002_ms", "p12_171110_a000_ms"]
    # ms_str_to_load = ["p13_18_10_29_a000_ms",  # new
    #                   "p13_18_10_29_a001_ms",
    #                   "p14_18_10_23_a000_ms",
    #                   "p14_18_10_30_a001_ms",
    #                   "p60_arnaud_ms", "p60_a529_2015_02_25_ms"]
    # for_graph = ["p6_18_02_07_a001_ms",
    #              "p7_171012_a000_ms", "p7_18_02_08_a000_ms",
    #              "p7_17_10_18_a002_ms", "p7_17_10_18_a004_ms",
    #              "p7_18_02_08_a001_ms", "p7_18_02_08_a002_ms",
    #              "p7_18_02_08_a003_ms",
    #              "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms",
    #              "p8_18_10_24_a005_ms", "p8_18_10_17_a001_ms",
    #              "p8_18_10_17_a000_ms",  # new
    #              "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms",
    #              "p9_18_09_27_a003_ms",  # new
    #              "p10_17_11_16_a003_ms",
    #              "p11_17_11_24_a001_ms", "p11_17_11_24_a000_ms",
    #              "p12_17_11_10_a002_ms",
    #              "p13_18_10_29_a000_ms",  # new
    #              "p13_18_10_29_a001_ms",
    #              "p14_18_10_23_a000_ms",
    #              "p14_18_10_30_a001_ms",
    #              "p60_arnaud_ms", "p60_a529_2015_02_25_ms"]
    # ms_str_to_load = for_graph
    # ms_str_to_load = ["p60_arnaud_ms", "p60_a529_2015_02_25_ms"]
    # ms_str_to_load = ["p6_18_02_07_a002_ms"]
    # ms_str_to_load = ["p6_18_02_07_a001_ms", "p7_171012_a000_ms"]
    # ms_str_to_load = ["p7_18_02_08_a000_ms"]
    # ms_str_to_load = ["p12_171110_a000_ms"]
    # ms_str_to_load = ["p9_18_09_27_a003_ms"]
    # ms_str_to_load = ["p12_171110_a000_ms"]
    # ms_str_to_load = ["p60_a529_2015_02_25_ms"]
    # ms_str_to_load = ["p7_171012_a000_ms"]
    # ms_str_to_load = ["p7_171012_a000_ms"]
    # ms_str_to_load = ["richard_015_D74_P2_ms"]
    # ms_str_to_load = ["richard_015_D89_P2_ms"]
    # ms_str_to_load = ["richard_015_D66_P2_ms"]
    # ms_str_to_load = ["richard_015_D75_P2_ms"]
    # ms_str_to_load = ["richard_018_D32_P2_ms"]
    # ms_str_to_load = ["richard_018_D28_P2_ms"]
    # ms_str_to_load = ["richard_028_D1_P1_ms"]
    # ms_str_to_load = ["richard_028_D2_P1_ms"]
    # ms_str_to_load = ["p12_171110_a000_ms"]
    # ms_str_to_load = ["p8_18_10_24_a005_ms"]
    # ms_str_to_load = ["p5_19_03_25_a001_ms"]
    # ms_str_to_load = ["p9_19_02_20_a002_ms"]
    # ms_str_to_load = ["p5_19_03_20_a000_ms"]
    # ms_str_to_load = ["p6_18_02_07_a002_ms", "p10_17_11_16_a003_ms"]
    # ms_str_to_load = ["p6_18_02_07_a002_ms"]
    # ms_str_to_load = ["p10_17_11_16_a003_ms"]
    # ms_str_to_load = ["p12_19_02_08_a000_ms"]
    # ms_str_to_load = ["p9_19_03_22_a001_ms"]
    # ms_str_to_load = ["p13_18_10_29_a001_ms"]
    # ms_str_to_load = ["p41_19_04_30_a000_ms"]
    # ms_str_to_load = ["p6_18_02_07_a001_ms"]
    # ms_str_to_load = ["p7_19_03_05_a000_ms", "p9_19_02_20_a003"]
    # ms_str_to_load = ["p9_19_02_20_a003_ms"]
    # 4 mice with nice abf + LFP
    # ms_str_to_load = ["p5_19_03_25_a001_ms", "p6_18_02_07_a001_ms", "p7_19_03_05_a000_ms", "p9_19_02_20_a003_ms"]

    # ms_str_to_load = ["p5_19_03_25_a001_ms"]
    # ms_str_to_load = ["p6_18_02_07_a002_ms"]
    # ms_str_to_load = ["p8_19_03_19_a000_ms"]
    # ms_str_to_load = ["p8_18_10_17_a001_ms"]

    # session with mouvements periods (twitch, long mvt etc...) available
    ms_str_to_load = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms", "p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms",
                      "p7_17_10_18_a004_ms", "p7_18_02_08_a000_ms", "p7_18_02_08_a001_ms", "p7_18_02_08_a002_ms",
                      "p7_18_02_08_a003_ms", "p7_19_03_05_a000_ms", "p7_19_03_27_a000_ms", "p7_19_03_27_a001_ms",
                      "p7_19_03_27_a002_ms",
                      "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms", "p8_18_10_17_a000_ms", "p8_18_10_17_a001_ms",
                      "p8_18_10_24_a005_ms", "p8_19_03_19_a000_ms",
                      "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms", "p9_18_09_27_a003_ms", "p9_19_02_20_a000_ms",
                      "p9_19_02_20_a001_ms", "p9_19_02_20_a002_ms", "p9_19_02_20_a003_ms", "p9_19_03_14_a000_ms",
                      "p9_19_03_14_a001_ms", "p9_19_03_22_a000_ms", "p9_19_03_22_a001_ms"]
    # ms_str_to_load = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms", "p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms"]
    # sessions with predictions for graph stats on young animals
    # ms_str_to_load = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms", "p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms",
    #                   "p7_17_10_18_a004_ms", "p7_18_02_08_a000_ms", "p7_18_02_08_a001_ms", "p7_18_02_08_a002_ms",
    #                   "p7_18_02_08_a003_ms", "p7_19_03_05_a000_ms", "p7_19_03_27_a000_ms", "p7_19_03_27_a001_ms",
    #                   "p7_19_03_27_a002_ms",
    #                   "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms", "p8_18_10_17_a000_ms", "p8_18_10_17_a001_ms",
    #                   "p8_18_10_24_a005_ms", "p8_19_03_19_a000_ms",
    #                   "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms", "p9_18_09_27_a003_ms", "p9_19_02_20_a000_ms",
    #                   "p9_19_02_20_a001_ms", "p9_19_02_20_a002_ms", "p9_19_03_14_a000_ms",
    #                   "p9_19_03_14_a001_ms", "p9_19_03_22_a000_ms", "p9_19_03_22_a001_ms"]
    # # #   for test
    # ms_str_to_load = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms",
    #                   "P6_18_02_07_a001_ms", "p6_18_02_07_a002_ms"]
    # ms_str_to_load = ["p5_19_03_25_a001_ms", "P6_18_02_07_a001_ms", "p6_18_02_07_a002_ms"]
    # ms_str_to_load = ["p5_19_03_25_a000_ms"]
    # ms_str_to_load = ["p6_18_02_07_a002_ms"]
    # ms_str_to_load = ["p5_19_03_25_a001_ms", "P6_18_02_07_a001_ms", "p6_18_02_07_a002_ms"]
    #                   "p7_18_02_08_a001_ms", "p7_18_02_08_a003_ms", "p7_18_02_08_a000_ms",
    #                   "p7_19_03_05_a000_ms", "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms",
    #                   "p8_18_10_24_a005_ms", "p9_17_12_06_a001_ms",
    #                   "p9_19_02_20_a003_ms", "p10_17_11_16_a003_ms",
    #                   "p11_17_11_24_a001_ms", "p11_17_11_24_a000_ms",
    #                   "p12_171110_a000_ms",
    #                   "p12_17_11_10_a002_ms",
    #                   "p13_18_10_29_a000_ms",
    #                   "p13_18_10_29_a001_ms",
    #                   "p14_18_10_23_a000_ms",
    #                   "p14_18_10_30_a001_ms"]
    # ms_str_to_load = ["richard_015_D74_P2_ms"]
    # ms_str_to_load = ["p5_19_03_25_a000_ms"]
    # ms_str_to_load = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms", "p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms"]
    # ms_str_to_load = ["p6_18_02_07_a002_ms"]
    # ms_str_to_load = ["p60_a529_2015_02_25_ms"]
    # ms_str_to_load = ["p60_arnaud_ms"]
    # ms_str_to_load = ["p9_19_02_20_a000_ms"]
    # ms_str_to_load = ["p10_19_02_21_a002_ms"]p5
    # ms_str_to_load = ["p8_18_10_24_a005_ms"]
    ## all the ms separated in 5 groups

    # ms_str_to_load = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms",
    #                   "p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms",
    #                   "p7_171012_a000_ms",
    #                   "p7_17_10_18_a002_ms", "p7_17_10_18_a004_ms",
    #                   "p7_18_02_08_a000_ms", "p7_18_02_08_a001_ms",
    #                   "p7_18_02_08_a002_ms", "p7_18_02_08_a003_ms",
    #                   "p7_19_03_05_a000_ms"]
    # # #
    # ms_str_to_load = ["p7_19_03_27_a000_ms", "p7_19_03_27_a001_ms",
    #                   "p7_19_03_27_a002_ms",
    #                   "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms",
    #                    "p8_18_10_17_a000_ms",
    #                   "p8_18_10_17_a001_ms"]
    # #
    # ms_str_to_load = ["p8_18_10_24_a005_ms", "p8_19_03_19_a000_ms",
    #                   "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms",
    #                   "p9_18_09_27_a003_ms", "p9_19_02_20_a000_ms",
    #                   "p9_19_02_20_a001_ms", "p9_19_02_20_a002_ms",
    #                   "p9_19_02_20_a003_ms", "p9_19_03_14_a000_ms",
    #                   "p9_19_03_14_a001_ms", "p9_19_03_22_a000_ms",
    #                   "p9_19_03_22_a001_ms"]
    # # #
    # ms_str_to_load = ["p10_17_11_16_a003_ms", "p10_19_02_21_a002_ms",
    #                   "p10_19_02_21_a003_ms", "p10_19_02_21_a005_ms",
    #                   "p10_19_03_08_a000_ms", "p10_19_03_08_a001_ms",
    #                   "p11_17_11_24_a000_ms", "p11_17_11_24_a001_ms",
    #                   "p11_19_02_15_a000_ms", "p11_19_02_22_a000_ms",
    #                   "p12_17_11_10_a002_ms", "p12_171110_a000_ms",
    #                   "p13_18_10_29_a000_ms", "p13_18_10_29_a001_ms",
    #                   "p13_19_03_11_a000_ms"]
    # # # #
    # ms_str_to_load = ["p14_18_10_23_a000_ms", "p14_18_10_30_a001_ms",
    #                   "p16_18_11_01_a002_ms",
    #                   "p19_19_04_08_a000_ms", "p19_19_04_08_a001_ms",
    #                   "p21_19_04_10_a000_ms", "p21_19_04_10_a001_ms",
    #                   "p41_19_04_30_a000_ms"]

    # ms_str_to_load = ["p5_19_03_25_a001_ms", "p9_18_09_27_a003_ms"]
    # ms_str_to_load = ["p5_19_03_25_a001_ms"]
    # ms_str_to_load = ["p7_18_02_08_a000_ms"]
    # ms_str_to_load = ["p7_18_02_08_a001_ms"]
    # ms_str_to_load = ["p8_18_10_24_a005_ms"]
    # ms_str_to_load = ["p19_19_04_08_a000_ms"]
    # ms_str_to_load = ["p9_19_02_20_a001_ms"]
    # ms_str_to_load = ["p5_19_03_25_a001_ms",  "p41_19_04_30_a000_ms"]
    # ms_str_to_load = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms",
    #                   "p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms", "p41_19_04_30_a000_ms"]
    # ms with good run
    # ms_str_to_load = ["p10_19_03_08_a000_ms", "p10_19_03_08_a001_ms",
    #                   "p13_18_10_29_a000_ms", "p13_18_10_29_a001_ms",
    #                   "p14_18_10_30_a001_ms"] #                       "p14_18_10_30_a001_ms"]

    # ms_str_to_load = ["richard_028_D1_P1_ms"]
    # ms_str_to_load = ["p60_a529_2015_02_25_ms"]
    # ms_str_to_load = ["p21_19_04_10_a000_ms", "p21_19_04_10_a001_ms",
    #                   "p21_19_04_10_a000_j3_ms", "p21_19_04_10_a001_j3_ms"]
    # ms_str_to_load = ["p13_18_10_29_a001_ms"]
    # ms_str_to_load = ["richard_028_D2_P1_ms"]
    # ms_str_to_load = ["p7_18_02_08_a001_ms"]
    # ms_str_to_load = ["p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms",
    #                            "p9_18_09_27_a003_ms", "p10_17_11_16_a003_ms",
    #                            "p11_17_11_24_a000_ms"]
    # Eleonora gad cre
    # ms_str_to_load = ["p6_19_02_18_a000_ms"]
    # ms_str_to_load = ["p11_19_04_30_a001_ms"]
    # ms_str_to_load = ["p7_19_03_05_a000_ms"]
    # ms_str_to_load = ["p60_20160506_gadcre01_01_ms"]
    # to test cilva
    # ms_str_to_load = ["p6_18_02_07_a002_ms"]
    # ms_str_to_load = ["p6_18_02_07_a001_ms"]
    # ms_str_to_load = ["p5_19_03_25_a001_ms"]
    # ms_str_to_load = ["p5_19_03_25_a000_ms"]
    # ms_str_to_load = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms",
    #                   "p6_18_02_07_a001_ms", "p6_18_02_07_a002_ms"]
    # loading data
    # z_shifts_ms = ["p5_19_03_25_a000_ms",
    #                "p5_19_03_25_a001_ms",
    #                "p6_18_02_07_a001_ms",
    #                "p6_18_02_07_a002_ms",
    #                "p7_17_10_18_a001_ms",
    #                "p7_17_10_18_a003_ms",
    #                "p7_18_02_08_a000_ms",
    #                "p7_19_03_05_a000_ms",
    #                "p8_17_11_13_a003_ms",
    #                "p8_18_02_09_a000_ms",
    #                "p8_19_03_19_a000_ms",
    #                "p9_17_12_06_a002_ms",
    #                "p9_17_12_06_a003_ms",
    #                "p9_18_09_27_a003_ms",
    #                "p9_17_12_20_a000_ms",
    #                "p10_17_11_16_a001_ms",
    #                "p11_17_11_17_a000_ms",
    #                "p16_18_11_01_a002_ms",
    #                "p7_19_02_19_a000_ms",
    #                "p10_19_03_04_a000_ms"]
    # ms_str_to_load = ["p5_19_03_25_a000_ms", "p5_19_03_25_a001_ms", "p6_18_02_07_a001_ms", "p6_18_02_07_a001_ms",
    #                    "p6_18_02_07_a002_ms", "p7_18_02_08_a000_ms", "p7_18_02_08_a001_ms", "p7_18_02_08_a002_ms",
    #                    "p7_18_02_08_a003_ms", "p7_19_03_05_a000_ms", "p7_19_03_27_a000_ms", "p7_19_03_27_a001_ms",
    #                    "p8_18_10_17_a001_ms", "p8_18_10_24_a005_ms", "p8_19_03_19_a000_ms",
    #                    "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms", "p9_18_09_27_a003_ms", "p9_19_02_20_a000_ms",
    #                    "p9_19_02_20_a001_ms", "p9_19_02_20_a002_ms", "p9_19_02_20_a003_ms", "p9_19_03_14_a000_ms",
    #                    "p9_19_03_14_a001_ms", "p9_19_03_22_a000_ms", "p9_19_03_22_a001_ms", "p10_17_11_16_a003_ms",
    #                    "p10_19_02_21_a002_ms", "p10_19_02_21_a005_ms",
    #                    "p10_19_03_08_a000_ms", "p10_19_03_08_a001_ms", "p11_17_11_24_a000_ms",
    #                    "p11_17_11_24_a001_ms", "p11_19_02_15_a000_ms", "p12_171110_a000_ms", "p12_17_11_10_a002_ms",
    #                    "p13_18_10_29_a000_ms", "p14_18_10_23_a000_ms",
    #                    "p14_18_10_30_a001_ms", "p16_18_11_01_a002_ms",
    #                    "p19_19_04_08_a000_ms", "p19_19_04_08_a001_ms", "p21_19_04_10_a000_ms",
    #                    "p21_19_04_10_a001_ms",
    #                    "p21_19_04_10_a000_j3_ms", "p41_19_04_30_a000_ms"]
    # with pca matlab
    # ms_str_to_load = [ "p7_18_02_08_a000_ms", "p7_18_02_08_a001_ms",
    #                   "p7_18_02_08_a002_ms", "p7_18_02_08_a003_ms",
    #                   "p7_19_03_27_a000_ms", "p7_19_03_27_a001_ms",
    #                   "p7_19_03_27_a002_ms",
    #                   "p8_18_02_09_a000_ms", "p8_18_02_09_a001_ms",
    #                   "p8_18_10_17_a001_ms",
    #                                          "p8_19_03_19_a000_ms",
    #                   "p9_17_12_06_a001_ms", "p9_17_12_20_a001_ms",
    #                   "p9_19_02_20_a001_ms", "p9_19_02_20_a002_ms",
    #                   "p9_19_02_20_a003_ms", "p9_19_03_14_a000_ms",
    #                   "p9_19_03_14_a001_ms", "p9_19_03_22_a000_ms",
    #                   "p9_19_03_22_a001_ms",
    #                   "p10_17_11_16_a003_ms",
    #                   "p10_19_02_21_a003_ms", "p10_19_02_21_a005_ms",
    #                   "p10_19_03_08_a000_ms", "p10_19_03_08_a001_ms",
    #                   "p11_17_11_24_a000_ms", "p11_17_11_24_a001_ms",
    #                   "p11_19_02_15_a000_ms", "p11_19_02_22_a000_ms",
    #                   "p14_18_10_23_a000_ms", "p14_18_10_30_a001_ms",
    #                   "p16_18_11_01_a002_ms",
    #                   "p19_19_04_08_a000_ms",
    #                   "p21_19_04_10_a000_ms", "p21_19_04_10_a001_ms",
    #                   "p21_19_04_10_a000_j3_ms",
    #                   "p41_19_04_30_a000_ms"]
    # ms_str_to_load = ["p6_18_02_07_a002_ms"]
    # ms_str_to_load = ["p6_19_02_18_a000_ms", "p11_19_04_30_a001_ms"]
    # GAD-CRE with caiman Rois available
    # ms_str_to_load = ["p6_19_02_18_a000_ms", "p11_19_04_30_a001_ms"]
    # 4 GAD-CRE
    # ms_str_to_load = ["p5_19_03_20_a000_ms", "p6_19_02_18_a000_ms",
    #                   "p11_19_04_30_a001_ms", "p12_19_02_08_a000_ms"]
    # ms_str_to_load = ["p5_19_03_20_a000_ms", "p12_19_02_08_a000_ms"]
    # ms_str_to_load = ["p5_19_03_20_a000_ms"]
    # ms_str_to_load = ["p6_19_02_18_a000_ms"]

    # ms_str_to_load = ["p12_171110_a000_ms"]
    # ms_str_to_load = ["p12_17_11_10_a002_ms"]

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_str_to_load, param=param,
                                            load_traces=load_traces, load_abf=load_abf)

    add_z_shifts_from_file(ms_str_to_ms_dict, param)

    return ms_str_to_ms_dict


def main():
    # for line in np.arange(15):
    #     print_surprise_for_michou(n_lines=15, actual_line=line)

    for_lexi = False
    for_richard = False

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
    cell_assemblies_data_path = path_data + "cell_assemblies/v6/"
    best_order_data_path = path_data + "best_order_data/v3/"

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
        compute_stat_about_significant_seq(files_path=f"{path_data}significant_seq/v7_slope/", param=param,
                                           save_formats=["pdf"],
                                           color_option="manual", cmap_name="Reds")
        # use_cmap_gradient
        # color_option="manual"
        return

    just_compute_significant_seq_with_slope_stat = False
    if just_compute_significant_seq_with_slope_stat:
        compute_stat_about_seq_with_slope(files_path=f"{path_data}/seq_slope/v3_70_150_surro/", param=param,
                                          save_formats=["pdf"],
                                          color_option="manual", cmap_name="Reds")
        # use_cmap_gradient
        # color_option="manual"
        return

    just_correlate_global_roi_and_shift = False
    # look in the data file for a params matlab file and a tif movie, and do correlation between shift during motion
    # motion correction and global activity (using a global ROI)
    if just_correlate_global_roi_and_shift:
        correlate_global_roi_and_shift(path_data=os.path.join(path_data), param=param)
        return

    load_traces = True
    load_abf = False

    if for_lexi:
        ms_str_to_ms_dict = lexi_loading_process(param=param, load_traces=load_traces)
    else:
        ms_str_to_ms_dict = robin_loading_process(param=param, load_traces=load_traces, load_abf=load_abf)
    # return
    available_ms = list(ms_str_to_ms_dict.values())
    # for ms in available_ms:
    #     ms.plot_each_inter_neuron_connect_map()
    #     return
    ms_to_analyse = available_ms

    just_plot_all_basic_stats = False
    just_plot_all_sum_spikes_dur = False

    just_do_stat_significant_time_period = False
    just_fca_clustering_on_twitches_activity = False
    just_plot_cell_assemblies_on_map = False
    just_plot_all_cells_on_map = False
    just_plot_all_cells_on_map_with_avg_on_bg = False

    # --------- shift categories analyses
    just_plot_raster_with_periods = False
    just_plot_all_cell_assemblies_proportion_on_shift_categories = False
    just_plot_nb_transients_in_mvt_vs_nb_total_transients = False
    just_plot_cells_that_fire_during_time_periods = False
    just_plot_twitch_ratio_activity = False
    # number of cells active in each type of movement event (normalized by number of cells and length of movement)
    just_plot_movement_activity = False
    just_plot_psth_over_event_time_correlation_graph_style = False
    do_plot_psth_twitches = False
    do_plot_psth_twitches_by_age=True
    # Add weight in legend of long mvt psth
    do_plot_psth_long_mvt = False
    do_twitches_analysis = False
    just_save_stat_about_mvt_for_each_ms = False
    just_plot_all_time_correlation_graph_over_events = False

    # ---------
    just_plot_jsd_correlation = False
    # connectivty graph
    do_plot_graph = False
    do_stats_on_graph = False
    just_plot_cell_assemblies_clusters = False
    just_find_seq_with_pca = False
    just_find_seq_using_graph = False
    just_test_elephant_cad = False
    just_plot_variance_according_to_sum_of_activity = False
    just_cluster_using_grid = False
    just_plot_seq_from_pca_with_map = False
    just_save_raster_as_npy_file = False
    just_do_pca_on_suite2p_spks = False
    just_use_rastermap_for_pca = False
    just_do_stat_on_pca = False
    just_analyse_lfp = False
    just_run_cilva = False
    just_evaluate_overlaps_accuracy = False

    # to merge contour map, like between fiji and caiman
    just_merge_coords_map = False

    just_produce_cell_assemblies_verification = False

    just_plot_raster_with_same_sum_activity_lim = False
    just_plot_raster = False
    just_plot_traces_with_shifts = False
    just_plot_raster_with_z_shift_periods = False
    just_do_stat_on_event_detection_parameters = False
    just_plot_raster_with_sce = False
    # periods such as twitch etc...
    # next one seems to be an old code
    just_plot_raster_with_cells_assemblies_events_and_mvts = False
    # this one works properly
    just_plot_raster_with_cells_assemblies_and_shifts = False
    just_plot_traces_raster = False
    just_plot_piezo_with_extra_info = False
    just_plot_raw_traces_around_each_sce_for_each_cell = False
    just_do_seqnmf = False
    just_generate_artificial_movie_from_rasterdur = False
    just_do_pca_on_raster = False
    just_display_seq_with_cell_assembly = False
    just_produce_animation = False
    just_plot_ratio_spikes_for_shift = False
    just_save_sum_spikes_dur_in_npy_file = False
    do_find_hubs = False
    do_find_hubs_using_all_ms = False

    # for events (sce) detection
    perc_threshold = 95
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
    # #################################### SPOT DIST ###########################################
    # ##########################################################################################
    do_spotdist = False

    # ##########################################################################################
    # #################################### CLUSTERING ###########################################
    # ##########################################################################################
    do_clustering = False
    do_detect_sce_on_traces = False
    do_detect_sce_based_on_peaks_finder = False
    use_hdbscan = False
    # to add in the file title
    clustering_bonus_descr = ""
    # if False, clustering will be done using kmean
    do_fca_clustering = False
    # instead of sce, take the twitches periods
    do_clustering_with_twitches_events = False
    with_cells_in_cluster_seq_sorted = False
    use_richard_option = for_richard
    # wake, sleep, quiet_wake, sleep_quiet_wake, active_wake
    richard_option = "wake"
    if do_clustering:
        # filtering spike_nums_dur using speed info if available
        for ms in ms_to_analyse:
            remove_frames_with_low_speed = False
            if remove_frames_with_low_speed and (ms.speed_by_frame is not None):
                frames_selected = np.where(ms.speed_by_frame >= 1)[0]
                # now we want to fusion frames that are close to each other
                frames_diff = np.diff(frames_selected)
                fusion_thr = 50
                for frame_index in np.arange(len(frames_diff)):
                    if 1 < frames_diff[frame_index] < fusion_thr:
                        frames_selected = np.concatenate(
                            (frames_selected, np.arange(frames_selected[frame_index] + 1,
                                                        frames_selected[frame_index + 1])))
                frames_selected = np.unique(frames_selected)
                frames_to_keep = np.setdiff1d(np.arange(len(ms.speed_by_frame)), frames_selected)
                ms.spike_struct.spike_nums_dur = ms.spike_struct.spike_nums_dur[:, frames_to_keep]
                print(f"Using speed_by_frame {ms.spike_struct.spike_nums_dur.shape}")
                ms.spike_struct.build_spike_nums_and_peak_nums()

        for ms in ms_to_analyse:
            # if not None, filter the frame keeping the kind of mouvements choosen, if available
            # if "no_shift" then select the frame that are not in any period
            # Other keys are: shift_twitch, shift_long, shift_unclassified
            # or a list of those 3 keys and then will take all frames except those
            with_period_mvt_filter = None  # ["shift_twitch","shift_long", "shift_unclassified"]
            if (with_period_mvt_filter is not None) and (ms.shift_data_dict is not None) and \
                    (not do_clustering_with_twitches_events):
                n_frames = ms.spike_struct.spike_nums_dur.shape[1]
                if isinstance(with_period_mvt_filter, list):
                    clustering_bonus_descr = "_all_frames_but_" + '_'.join(with_period_mvt_filter)
                    shift_bool = np.ones(n_frames, dtype="bool")
                    for shift_key in with_period_mvt_filter:
                        shift_bool_tmp = ms.shift_data_dict[shift_key]
                        if shift_key == "shift_twitch":
                            extension_frames_after = 50
                            extension_frames_before = 20
                        else:
                            extension_frames_after = 20
                            extension_frames_before = 20
                        # we extend each period, implementation is not the fastest and more elegant way
                        true_frames = np.where(shift_bool_tmp)[0]
                        for frame in true_frames:
                            first_frame = max(0, frame - extension_frames_before)
                            last_frame = min(n_frames - 1, frame + extension_frames_after)
                            shift_bool_tmp[first_frame:last_frame + 1] = True
                        shift_bool[shift_bool_tmp] = False
                    ms.spike_struct.spike_nums_dur = \
                        remove_spike_nums_dur_and_associated_transients(spike_nums_dur=ms.spike_struct.spike_nums_dur,
                                                                        frames_to_keep=shift_bool)
                    # ms.spike_struct.spike_nums_dur = ms.spike_struct.spike_nums_dur[:, shift_bool]
                    print(f"{ms.description} filtering spike_nums removing all shifts")
                    ms.spike_struct.build_spike_nums_and_peak_nums()
                elif with_period_mvt_filter == "no_shift":
                    clustering_bonus_descr = "_no_shift_"
                    shift_bool = np.ones(ms.spike_struct.spike_nums_dur.shape[1], dtype="bool")
                    for shift_key in ["shift_twitch", "shift_long", "shift_unclassified"]:
                        shift_bool_tmp = ms.shift_data_dict[shift_key]
                        shift_bool[shift_bool_tmp] = False
                    ms.spike_struct.spike_nums_dur = \
                        remove_spike_nums_dur_and_associated_transients(spike_nums_dur=ms.spike_struct.spike_nums_dur,
                                                                        frames_to_keep=shift_bool)
                    # ms.spike_struct.spike_nums_dur = ms.spike_struct.spike_nums_dur[:, shift_bool]
                    print(f"{ms.description} filtering spike_nums removing all shifts")
                    ms.spike_struct.build_spike_nums_and_peak_nums()

                else:
                    clustering_bonus_descr = f"_{with_period_mvt_filter}_"
                    shift_bool = ms.shift_data_dict[with_period_mvt_filter]
                    shift_bool_copy = np.copy(shift_bool)
                    # period_extension
                    if with_period_mvt_filter == "shift_twitch":
                        extension_frames_after = 50
                        extension_frames_before = 20
                        shift_bool_tmp = np.copy(shift_bool_copy)
                        true_frames = np.where(shift_bool_copy)[0]
                        for frame in true_frames:
                            first_frame = max(0, frame - extension_frames_before)
                            last_frame = min(n_frames - 1, frame + extension_frames_after)
                            shift_bool_tmp[first_frame:last_frame + 1] = True
                        shift_bool_copy[shift_bool_tmp] = True
                    ms.spike_struct.spike_nums_dur = \
                        remove_spike_nums_dur_and_associated_transients(spike_nums_dur=ms.spike_struct.spike_nums_dur,
                                                                        frames_to_keep=shift_bool_copy)
                    # ms.spike_struct.spike_nums_dur = ms.spike_struct.spike_nums_dur[:, shift_bool]
                    print(f"{ms.description} filtering spike_nums with shift {with_period_mvt_filter}")
                    ms.spike_struct.build_spike_nums_and_peak_nums()
                print(f"{ms.description} n frames left {ms.spike_struct.spike_nums_dur.shape[1]}")

    # ##### for fca #####
    n_surrogate_fca = 20

    # #### for kmean  #####
    with_shuffling = False
    print(f"use_raster_dur {use_raster_dur}")
    range_n_clusters_k_mean = np.arange(3, 8)
    # range_n_clusters_k_mean = np.array([7])
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
    param.error_rate = 0.25  # 0.25 0.1
    param.max_branches = 10
    param.time_inter_seq = 4  # 30 3
    param.min_duration_intra_seq = 1  # 1
    param.min_len_seq = 4  # 5
    param.min_rep_nb = 5  # 3

    debug_mode = False

    # ------------------------------ end param section ------------------------------
    if just_plot_all_basic_stats:
        plot_all_basic_stats(ms_to_analyse, param, use_animal_weight=True)
        raise Exception("just_plot_all_basic_stats")

    if just_plot_all_sum_spikes_dur:
        plot_all_sum_spikes_dur(ms_to_analyse, param)
        raise Exception("plot_all_sum_spikes_dur")

    if do_plot_psth_twitches_by_age:
        line_mode = True
        duration_option = False
        use_traces = False
        plot_twitches_psth_by_age(ms_to_analyse, param, line_mode=line_mode, use_traces=use_traces,
                                      save_formats="pdf")
        raise Exception("do_plot_psth_twitches_by_age")

    if just_plot_all_time_correlation_graph_over_events:
        # event_str = "shift_twitch" "shift_long"
        plot_all_time_correlation_graph_over_events(event_str="shift_twitch", ms_to_analyse=ms_to_analyse,
                                                    param=param, time_around_events=10)
        raise Exception("just_plot_all_time_correlation_graph_over_events")
    if just_plot_psth_over_event_time_correlation_graph_style:
        plot_psth_over_event_time_correlation_graph_style(event_str="shift_twitch", ms_to_analyse=ms_to_analyse,
                                                          param=param, time_around_events=20)
        raise Exception("just_plot_psth_over_event_time_correlation_graph_style")

    if just_plot_movement_activity:
        plot_movement_activity(ms_to_analyse, param)
        raise Exception("just_plot_movement_activity")

    if just_do_stat_significant_time_period:
        stat_significant_time_period(ms_to_analyse, shift_key="shift_twitch", perc_threshold=95, n_surrogate=1000)
        raise Exception("just_do_stat_significant_time_period")

    if just_plot_cells_that_fire_during_time_periods:
        # take a list of periods, and will determine whichh cells are specific of each
        # and then all and then none (still)
        # "shift_twitch", "shift_long",
        #                                                                            "shift_unclassified"
        plot_cells_that_fire_during_time_periods(ms_to_analyse, shift_keys=["shift_twitch", "shift_long"], param=param,
                                                 perc_threshold=95, n_surrogate=1000)
        raise Exception("just_plot_cells_that_fire_during_time_periods")

    if just_plot_nb_transients_in_mvt_vs_nb_total_transients:
        plot_nb_transients_in_mvt_vs_nb_total_transients(ms_to_analyse, param, save_formats="pdf")
        raise Exception("just_plot_nb_transients_in_mvt_vs_nb_total_transients")

    if just_plot_twitch_ratio_activity:
        plot_twitch_ratio_activity(ms_to_analyse, time_around=20, param=param, save_formats="pdf")
        raise Exception("just_plot_twitch_ratio_activity")

    if do_twitches_analysis:
        twitch_analysis_on_all_ms(ms_to_analyse, param, n_surrogates=0, option="intersect",
                                  before_extension=0, after_extension=20)
        raise Exception("twitches analysed")

    if just_plot_variance_according_to_sum_of_activity:
        plot_variance_according_to_sum_of_activity(ms_to_analyse, param, save_formats="pdf")
        raise Exception("just_plot_variance_according_to_sum_of_activity")

    if just_do_stat_on_pca:
        do_stat_on_pca(ms_to_analyse, param, save_formats="pdf")
        raise Exception("just_do_stat_on_pca")

    if do_find_hubs_using_all_ms:
        find_hubs_using_all_ms(ms_to_analyse, param)
        raise Exception("do_find_hubs_using_all_ms")

    if just_save_stat_about_mvt_for_each_ms:
        save_stat_about_mvt_for_each_ms(ms_to_analyse, param=param)
        raise Exception("just_save_stat_about_mvt_for_each_ms")

    if just_plot_all_cell_assemblies_proportion_on_shift_categories:
        plot_all_cell_assemblies_proportion_on_shift_categories(ms_to_analyse,
                                                                param=param, save_formats="pdf")
        raise Exception("just_plot_all_cell_assemblies_proportion_on_shift_categories")

    if just_plot_jsd_correlation:
        plot_jsd_correlation(ms_to_analyse, param, "Hamming_distance", n_surrogate=20, save_formats=["png", "pdf"])

        raise Exception("just_plot_jsd_correlation")

    if just_plot_raster_with_same_sum_activity_lim:
        max_sum_activity = 0
        show_sum_spikes_as_percentage = False
        for ms_index, ms in enumerate(ms_to_analyse):
            sum_activity = np.sum(ms.spike_struct.spike_nums, axis=0)
            if show_sum_spikes_as_percentage:
                sum_activity = (sum_activity / ms.spike_struct.spike_nums.shape[0]) * 100
            max_sum_activity = max(max_sum_activity, np.max(sum_activity))
        # spike_shape = '|' if use_raster_dur else 'o'

        for ms_index, ms in enumerate(ms_to_analyse):
            spike_shape = 'o'
            if ms.spike_struct.spike_nums is None:
                continue
            n_cells = len(ms.spike_struct.spike_nums)
            y_lim_sum_activity = (0, max_sum_activity)
            bonus_file_name = "_sum"
            if show_sum_spikes_as_percentage:
                bonus_file_name = "_percentage"
            plot_spikes_raster(spike_nums=ms.spike_struct.spike_nums, param=ms.param,
                               spike_train_format=False,
                               title=f"{ms.description}",
                               file_name=f"{ms.description}_raster{bonus_file_name}",
                               y_ticks_labels=np.arange(n_cells),
                               y_ticks_labels_size=2,
                               save_raster=True,
                               show_raster=False,
                               plot_with_amplitude=False,
                               show_sum_spikes_as_percentage=show_sum_spikes_as_percentage,
                               span_area_only_on_raster=False,
                               spike_shape=spike_shape,
                               spike_shape_size=0.5,
                               y_lim_sum_activity=y_lim_sum_activity,
                               save_formats=["pdf", "png"])

        raise Exception("just_plot_raster_with_same_sum_activity_lim")

    if do_plot_graph:
        plot_connectivity_graph(ms_to_analyse, param, save_formats="pdf")
        raise Exception("do_plot_graph")

    if do_stats_on_graph:
        stats_on_graph_on_all_ms(ms_to_analyse, param, save_formats="pdf")
        raise Exception("do_stats_graph")

    ms_by_age = dict()
    for ms_index, ms in enumerate(ms_to_analyse):

        # plot_spikes_raster(spike_nums=ms.spike_struct.spike_nums_dur, param=ms.param,
        #                    spike_train_format=False,
        #                    title=f"{ms.description}",
        #                    file_name=f"{ms.description}_raster",
        #                    y_ticks_labels=np.arange(ms.spike_struct.spike_nums_dur.shape[0]),
        #                    y_ticks_labels_size=2,
        #                    save_raster=True,
        #                    show_raster=False,
        #                    plot_with_amplitude=False,
        #                    activity_threshold=ms.spike_struct.activity_threshold,
        #                    # 500 ms window
        #                    show_sum_spikes_as_percentage=False,
        #                    span_area_only_on_raster=False,
        #                    spike_shape='|',
        #                    spike_shape_size=10,
        #                    save_formats=["pdf", "png"])

        if do_pattern_search or do_clustering:
            break
        print(f"for: ms {ms.description}")
        # np.savez(ms.param.path_data + ms.description + "_rasters_reduced.npz",
        #          spike_nums=ms.spike_struct.spike_nums[:50, :5000],
        #          spike_nums_dur=ms.spike_struct.spike_nums_dur[:50, :5000])
        # raise Exception("ambre")

        if just_plot_raster:
            # spike_shape = '|' if use_raster_dur else 'o'
            spike_shape = 'o'
            if ms.spike_struct.spike_nums_dur is None:
                print(f"{ms.description} spike_struct.spike_nums_dur is None")
                continue
            n_cells = len(ms.spike_struct.spike_nums)
            y_lim_sum_activity = None
            plot_spikes_raster(spike_nums=ms.spike_struct.spike_nums_dur, param=ms.param,
                               spike_train_format=False,
                               title=f"{ms.description}",
                               file_name=f"{ms.description}_raster",
                               y_ticks_labels=np.arange(n_cells),
                               y_ticks_labels_size=2,
                               save_raster=True,
                               show_raster=False,
                               plot_with_amplitude=False,
                               show_sum_spikes_as_percentage=False,
                               span_area_only_on_raster=False,
                               spike_shape=spike_shape,
                               spike_shape_size=0.5,
                               y_lim_sum_activity=None,
                               save_formats=["pdf", "png"])
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("fifi")
            continue

        if just_plot_traces_with_shifts:

            ms.plot_traces_with_shifts()

            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_traces_with_shifts")
            continue

        if just_plot_seq_from_pca_with_map:
            do_test = False

            if ms.pca_seq_cells_order is None:
                continue
            for pc_number, pca_seq_cells_order in ms.pca_seq_cells_order.items():
                if do_test:
                    # pca_seq_cells_order = np.load(f"{param.path_data}/test_pca/"
                    #                               f"P41_19_04_30_a000_spks_suite2p_cells_order_pc_3.npy")
                    data = hdf5storage.loadmat(f"{param.path_data}/test_pca/"
                                               f"p7_18_02_08_a002_order.mat")

                    pca_seq_cells_order = data['used_cells'][0][data['used_cells'][0] > 0] - 1
                    # other_cells = np.setdiff1d(np.arange(ms.coord_obj.n_cells), pca_seq_cells_order)
                    # pca_seq_cells_order = np.concatenate((pca_seq_cells_order, other_cells))
                file_name = f"{ms.description}_pc_{pc_number}_map_and_raster_seq_pca_{len(pca_seq_cells_order)}_cells"
                if ms.speed_by_frame is not None:
                    binary_speed = np.zeros(len(ms.speed_by_frame), dtype="int8")
                    binary_speed[ms.speed_by_frame > 0] = 1
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
                    periods_dict = ms.shift_data_dict
                    if periods_dict is not None:
                        print(f"{ms.description}:")
                        for name_period, period in periods_dict.items():
                            span_area_coords.append(get_continous_time_periods(period.astype("int8")))
                            span_area_colors.append(colors[i % len(colors)])
                            print(f"  Period {name_period} -> {colors[i]}")
                            i += 1
                    elif ms.speed_by_frame is not None:
                        span_area_coords = []
                        span_area_colors = []
                        span_area_coords.append(speed_periods)
                        span_area_colors.append("cornflowerblue")
                    else:
                        print(f"no mvt info for {ms.description}")

                plot_figure_with_map_and_raster_for_sequences(ms=ms,
                                                              cells_in_seq=pca_seq_cells_order[::-1],
                                                              file_name=file_name,
                                                              lines_to_display=None,
                                                              range_around_slope_in_frames=
                                                              0,
                                                              span_area_coords=span_area_coords,
                                                              span_area_colors=span_area_colors,
                                                              without_sum_activity_traces=True,
                                                              save_formats=["pdf", "png"], dpi=300)
                # with z_shifts_mvt
                # print(f"ms.z_shift_periods {ms.z_shift_periods}")
                if len(ms.z_shift_periods) > 0:
                    span_area_coords = []
                    span_area_colors = []
                    span_area_coords.append(ms.z_shift_periods)
                    span_area_colors.append("powderblue")
                    file_name = f"{ms.description}_pc_{pc_number}_map_and_raster_seq_pca_{len(pca_seq_cells_order)}_cells_z_shifts"
                    plot_figure_with_map_and_raster_for_sequences(ms=ms,
                                                                  cells_in_seq=pca_seq_cells_order[::-1],
                                                                  file_name=file_name,
                                                                  lines_to_display=None,
                                                                  range_around_slope_in_frames=
                                                                  0,
                                                                  span_area_coords=span_area_coords,
                                                                  span_area_colors=span_area_colors,
                                                                  without_sum_activity_traces=True,
                                                                  save_formats=["pdf", "png"], dpi=300)

                n_times = ms.spike_struct.spike_nums_dur.shape[1]
                for index_beg in np.arange(0, n_times, 2500):
                    frames_to_display = np.arange(index_beg, index_beg + 2500)
                    file_name = f"{ms.description}_pc_{pc_number}_map_and_raster_seq_pca_{len(pca_seq_cells_order)}_cells_" \
                                f"frame_{index_beg}_to_frame_{index_beg + 2500}"
                    if ms.speed_by_frame is not None:
                        binary_speed = np.zeros(len(ms.speed_by_frame), dtype="int8")
                        binary_speed[ms.speed_by_frame > 0] = 1
                        speed_periods_tmp = get_continous_time_periods(binary_speed)
                        speed_periods = []
                        for speed_period in speed_periods_tmp:
                            if (speed_period[0] not in frames_to_display) and (
                                    speed_period[1] not in frames_to_display):
                                continue
                            elif (speed_period[0] in frames_to_display) and (speed_period[1] in frames_to_display):
                                speed_periods.append((speed_period[0] - index_beg, speed_period[1] - index_beg))
                            elif speed_period[0] in frames_to_display:
                                speed_periods.append((speed_period[0] - index_beg, frames_to_display[-1] - index_beg))
                            else:
                                speed_periods.append((0, speed_period[1] - index_beg))
                    # colors for movement periods
                    span_area_coords = None
                    span_area_colors = None
                    with_mvt_periods = True

                    if with_mvt_periods:
                        colors = ["red", "green", "blue", "pink", "orange"]
                        i = 0
                        span_area_coords = []
                        span_area_colors = []
                        periods_dict = ms.shift_data_dict
                        if periods_dict is not None:
                            print(f"{ms.description}:")
                            for name_period, period in periods_dict.items():
                                mvt_periods_tmp = get_continous_time_periods(period.astype("int8"))
                                mvt_periods = []
                                for mvt_period in mvt_periods_tmp:
                                    if (mvt_period[0] not in frames_to_display) and (
                                            mvt_period[1] not in frames_to_display):
                                        continue
                                    elif (mvt_period[0] in frames_to_display) and (mvt_period[1] in frames_to_display):
                                        mvt_periods.append((mvt_period[0] - index_beg, mvt_period[1] - index_beg))
                                    elif mvt_period[0] in frames_to_display:
                                        mvt_periods.append(
                                            (mvt_period[0] - index_beg, frames_to_display[-1] - index_beg))
                                    else:
                                        mvt_periods.append((0, mvt_period[1] - index_beg))
                                span_area_coords.append(mvt_periods)
                                span_area_colors.append(colors[i % len(colors)])
                                print(f"  Period {name_period} -> {colors[i]}")
                                i += 1
                        elif ms.speed_by_frame is not None:
                            span_area_coords = []
                            span_area_colors = []
                            span_area_coords.append(speed_periods)
                            span_area_colors.append("cornflowerblue")
                        else:
                            print(f"no mvt info for {ms.description}")

                    plot_figure_with_map_and_raster_for_sequences(ms=ms,
                                                                  frames_to_use=frames_to_display,
                                                                  cells_in_seq=pca_seq_cells_order[::-1],
                                                                  file_name=file_name,
                                                                  lines_to_display=None,
                                                                  range_around_slope_in_frames=
                                                                  0,
                                                                  span_area_coords=span_area_coords,
                                                                  span_area_colors=span_area_colors,
                                                                  without_sum_activity_traces=True,
                                                                  save_formats=["pdf", "png"], dpi=300)
                if do_test:
                    break
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_seq_from_pca_with_map")
            continue

        if just_cluster_using_grid:
            cluster_using_grid(ms, param)

            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_cluster_using_grid")
            continue

        if just_merge_coords_map:
            # code to merge to map cells coords, useful to map cells from different segmentation
            # code valid for just some sessions
            merge_coords_map(ms, param)
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_merge_coords_map")
            continue

        if just_plot_raster_with_z_shift_periods:
            # spike_shape = '|' if use_raster_dur else 'o'
            spike_shape = 'o'
            if ms.spike_struct.spike_nums is None:
                continue
            n_cells = len(ms.spike_struct.spike_nums)
            span_area_coords = []
            span_area_colors = []
            span_area_coords.append(ms.z_shift_periods)
            span_area_colors.append("red")
            plot_spikes_raster(spike_nums=ms.spike_struct.spike_nums, param=ms.param,
                               spike_train_format=False,
                               title=f"{ms.description}",
                               file_name=f"{ms.description}_raster",
                               y_ticks_labels=np.arange(n_cells),
                               y_ticks_labels_size=2,
                               save_raster=True,
                               show_raster=False,
                               plot_with_amplitude=False,
                               show_sum_spikes_as_percentage=False,
                               span_area_only_on_raster=False,
                               span_area_coords=span_area_coords,
                               span_area_colors=span_area_colors,
                               spike_shape=spike_shape,
                               spike_shape_size=0.5,
                               save_formats=["pdf", "png"])
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_raster_with_z_shift_periods")
            continue
        if do_find_hubs:
            # for cell_to_map in [61, 73, 130, 138, 142]:
            #     ms.plot_connectivity_maps_of_a_cell(cell_to_map=cell_to_map, cell_descr="", not_in=False,
            #                                         cell_color="red", links_cell_color="cornflowerblue")
            if ms.spike_struct.graph_out is None:
                print(f"{ms.description} detect_n_in_n_out")
                ms.detect_n_in_n_out()
            if ms.spike_struct.graph_out is not None:
                hubs = find_hubs(graph=ms.spike_struct.graph_out, ms=ms)
                print(f"{ms.description} hubs: {hubs}")
            # P13_18_10_29_a001 hubs: [61, 73, 130, 138, 142]
            # P60_arnaud_a_529 hubs: [65, 102]
            # P60_a529_2015_02_25 hubs: [2, 8, 88, 97, 109, 123, 127, 142]
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("do_find_hubs")
            continue
        if just_find_seq_using_graph:
            span_area_coords = None
            span_area_colors = None
            with_mvt_periods = True
            spike_nums_dur = ms.spike_struct.spike_nums_dur
            # spike_nums_dur = tools_misc.bin_raster(raster=spike_nums_dur, bin_size=12, keep_same_dimension=True)
            # if ms.pca_seq_cells_order is not None:
            #     spike_nums_dur = spike_nums_dur[ms.pca_seq_cells_order]
            #     with_mvt_periods = False
            if with_mvt_periods:
                colors = ["red", "green", "blue", "pink", "orange"]
                i = 0
                span_area_coords = []
                span_area_colors = []
                periods_dict = ms.shift_data_dict
                if periods_dict is not None:
                    for name_period, period in periods_dict.items():
                        span_area_coords.append(get_continous_time_periods(period.astype("int8")))
                        span_area_colors.append(colors[i % len(colors)])
                        print(f"Period {name_period} -> {colors[i]}")
                        i += 1
                else:
                    print(f"no shift_data_dict for {ms.description}")
            # used to be 1 and 10
            find_sequences_using_graph_main(spike_nums_dur, param, min_time_bw_2_spikes=2,
                                            max_time_bw_2_spikes=10, max_connex_by_cell=5, min_nb_of_rep=3,
                                            debug_mode=False, descr=ms.description, ms=ms,
                                            error_rate=0.7,
                                            n_surrogates=10, raster_dur_version=True,
                                            span_area_coords=span_area_coords,
                                            span_area_colors=span_area_colors)
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_find_seq_using_graph")
            continue
        if do_spotdist:
            spotdist_function(ms, param)
        if just_find_seq_with_pca:
            # speed = ms.speed_by_frame
            find_seq_with_pca(ms, ms.raw_traces, path_results=param.path_results,
                              file_name=f"{ms.description}_seq_with_pca")
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_find_seq_with_pca")
            continue

        if just_use_rastermap_for_pca:
            use_rastermap_for_pca(ms, path_results=param.path_results,
                                  file_name=f"{ms.description}_pca_with_rastermap")
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_use_rastermap_for_pca")
            continue

        if just_save_raster_as_npy_file:
            if ms.spike_struct.spike_nums_dur is None:
                continue
            np.save(os.path.join(param.path_results, f'{ms.description}_raster_dur.npy'),
                    ms.spike_struct.spike_nums_dur.astype("uint8"))
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_save_raster_as_npy_file")
            continue
        if just_test_elephant_cad:
            elephant_cad(ms, param)
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_test_elephant_cad")
            continue
        if just_save_sum_spikes_dur_in_npy_file:
            ms.save_sum_spikes_dur_in_npy_file()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_save_sum_spikes_dur_in_npy_file")
            continue

        if just_fca_clustering_on_twitches_activity:
            fca_clustering_on_twitches_activity(ms, param, save_formats="pdf")
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_fca_clustering_on_twitches_activity")
            continue

        if do_plot_psth_long_mvt:
            line_mode = True
            duration_option = False
            plot_all_long_mvt_psth_in_one_figure(ms_to_analyse, param, line_mode,
                                                 duration_option=duration_option, save_formats="pdf")
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("do_plot_psth_long_mvt")
            continue

        if do_plot_psth_twitches:
            line_mode = True
            duration_option = False
            use_traces = False
            plot_all_twitch_psth_in_one_figure(ms_to_analyse, param, line_mode, use_traces=use_traces,
                                               duration_option=duration_option, save_formats="pdf")
            # ms.plot_psth_twitches(line_mode=line_mode)
            # ms.plot_psth_twitches(twitches_group=1, line_mode=line_mode)
            # ms.plot_psth_twitches(twitches_group=2, line_mode=line_mode)
            # ms.plot_psth_twitches(twitches_group=3, line_mode=line_mode)
            # ms.plot_psth_twitches(twitches_group=4, line_mode=line_mode)
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("do_plot_psth_twitches")
            continue

        if just_plot_raster_with_periods:
            # frames_selected = ms.richard_dict["Active_Wake_Frames"]
            # frames_selected = frames_selected[frames_selected < ms.spike_struct.spike_nums_dur.shape[1]]
            # ms.spike_struct.spike_nums_dur = ms.spike_struct.spike_nums_dur[:, frames_selected]
            ms.plot_raster_with_periods(ms.shift_data_dict, with_periods=True,
                                        with_cell_assemblies=True, only_cell_assemblies=False)
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("The Lannisters always pay their debts")
            continue

        if just_plot_all_cells_on_map:
            ms.plot_all_cells_on_map()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_all_cells_on_map exception")
            continue

        if just_plot_all_cells_on_map_with_avg_on_bg:
            ms.plot_all_cells_on_map_with_avg_on_bg()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_all_cells_on_map_with_avg_on_bg")
            continue

        if just_produce_animation:
            ms.produce_roi_shift_animation_with_cell_assemblies()
            # ms.produce_roi_shift_animation()
            # ms.produce_animation()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_produce_animation exception")
            continue
        if just_plot_ratio_spikes_for_shift:
            shift_periods = get_continous_time_periods(ms.shift_periods_bool.astype("int8"))
            shift_times_numbers = np.ones(len(ms.shift_periods_bool), dtype="int16")
            shift_times_numbers *= -1
            for shift_index, shift_period in enumerate(shift_periods):
                shift_times_numbers[shift_period[0]:shift_period[1] + 1] = shift_index
            plot_ratio_spikes_on_events_by_cell(spike_nums=ms.spike_struct.spike_nums,
                                                spike_nums_dur=ms.spike_struct.spike_nums_dur,
                                                times_numbers=shift_times_numbers,
                                                param=param,
                                                use_only_onsets=True,
                                                event_description="shift",
                                                session_description=ms.description)
            span_area_coords = []
            span_area_colors = []
            span_area_coords.append(shift_periods)
            span_area_colors.append("red")
            n_cells = len(ms.spike_struct.spike_nums_dur)
            spike_shape = '|' if use_raster_dur else 'o'
            plot_spikes_raster(spike_nums=ms.spike_struct.spike_nums_dur, param=ms.param,
                               spike_train_format=False,
                               title=f"{ms.description}",
                               file_name=f"{ms.description}_raster_with_shifts",
                               y_ticks_labels=np.arange(n_cells),
                               y_ticks_labels_size=2,
                               save_raster=True,
                               show_raster=False,
                               plot_with_amplitude=False,
                               activity_threshold=ms.activity_threshold,
                               # 500 ms window
                               sliding_window_duration=1,
                               show_sum_spikes_as_percentage=False,
                               # vertical_lines=SCE_times,
                               # vertical_lines_colors=['white'] * len(SCE_times),
                               # vertical_lines_sytle="solid",
                               # vertical_lines_linewidth=[0.2] * len(SCE_times),
                               span_area_coords=span_area_coords,
                               span_area_colors=span_area_colors,
                               span_area_only_on_raster=False,
                               spike_shape=spike_shape,
                               spike_shape_size=0.5,
                               save_formats=["pdf", "png"])

        if just_display_seq_with_cell_assembly:
            print("test_seq_detect")
            span_area_coords = None
            span_area_colors = None
            show_richard_active_frames = False
            if show_richard_active_frames:
                active_frames = ms.richard_dict["Active_Wake_Frames"]
                bin_array = np.zeros(ms.spike_struct.spike_nums_dur.shape[1], dtype="int8")
                bin_array[np.unique(active_frames)] = 1
                periods = get_continous_time_periods(bin_array)
                span_area_coords = [periods]
                span_area_colors = ["red"]
            if ms.shift_data_dict is not None:
                colors = ["red", "green", "blue", "pink", "orange"]
                i = 0
                span_area_coords = []
                span_area_colors = []
                for name_period, period in ms.shift_data_dict.items():
                    span_area_coords.append(get_continous_time_periods(period.astype("int8")))
                    span_area_colors.append(colors[i % len(colors)])
                    print(f"Period {name_period} -> {colors[i]}")
                    i += 1

            test_seq_detect(ms, span_area_coords=span_area_coords, span_area_colors=span_area_colors)
            raise Exception("just_display_seq_with_cell_assembly")
        if just_do_pca_on_suite2p_spks:
            ms.pca_on_suite2p_spks()

            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_do_pca_on_suite2p_spks")
            continue

        if just_analyse_lfp:
            ms.analyse_lfp()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_analyse_lfp")
            continue

        if just_run_cilva:
            ms.run_cilva()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_run_cilva")
            continue

        if just_evaluate_overlaps_accuracy:
            ms.evaluate_overlaps_accuracy(path_data=path_data, path_results=param.path_results)
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_evaluate_overlaps_accuracy")
            continue

        if just_do_pca_on_raster:
            spike_nums_to_use = ms.spike_struct.spike_nums_dur
            # sce_detection_result = detect_sce_potatoes_style(spike_nums=spike_nums_to_use, perc_threshold=95,
            #                                                  debug_mode=True)
            #
            # print(f"sce_with_sliding_window detected")
            # # tuple of times
            # SCE_times = sce_detection_result[1]
            #
            # # print(f"SCE_times {SCE_times}")
            # sce_times_numbers = sce_detection_result[3]
            # sce_times_bool = sce_detection_result[0]
            # # useful for plotting twitches
            # ms.sce_bool = sce_times_bool
            # ms.sce_times_numbers = sce_times_numbers
            # ms.SCE_times = SCE_times
            #
            # span_area_coords = [SCE_times]
            # span_area_colors = ['lightgrey']
            # span_area_coords=span_area_coords, span_area_colors=span_area_colors
            if ms.speed_by_frame is not None:
                binary_speed = np.zeros(len(ms.speed_by_frame), dtype="int8")
                binary_speed[ms.speed_by_frame > 0] = 1
                speed_periods = get_continous_time_periods(binary_speed)
                span_area_coords = []
                span_area_colors = []
                span_area_coords.append(speed_periods)
                span_area_colors.append("cornflowerblue")
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

        cellsinpeak = sce_detection_result[2]
        # print(f"SCE_times {SCE_times}")
        sce_times_numbers = sce_detection_result[3]
        sce_times_bool = sce_detection_result[0]
        # useful for plotting twitches
        ms.sce_bool = sce_times_bool
        ms.sce_times_numbers = sce_times_numbers
        ms.SCE_times = SCE_times

        print(f"n_cells {ms.spike_struct.n_cells}, n_sces {len(ms.SCE_times)}")

        if just_plot_cell_assemblies_clusters:
            # use data from txt file that should be loaded
            # don't compute the clusters, just display them
            ms.plot_cell_assemblies_clusters(cellsinpeak=cellsinpeak)
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_cell_assemblies_clusters")
            continue

        if just_plot_raster_with_sce:
            span_area_coords = []
            span_area_colors = []
            span_area_coords.append(ms.SCE_times)
            span_area_colors.append("red")
            n_cells = len(spike_nums_to_use)
            spike_shape = '|' if use_raster_dur else 'o'
            plot_spikes_raster(spike_nums=spike_nums_to_use, param=ms.param,
                               spike_train_format=False,
                               title=f"{ms.description}",
                               file_name=f"{ms.description}_raster",
                               y_ticks_labels=np.arange(n_cells),
                               y_ticks_labels_size=2,
                               save_raster=True,
                               show_raster=False,
                               plot_with_amplitude=False,
                               activity_threshold=ms.spike_struct.activity_threshold,
                               # 500 ms window
                               sliding_window_duration=sliding_window_duration,
                               show_sum_spikes_as_percentage=False,
                               # vertical_lines=SCE_times,
                               # vertical_lines_colors=['white'] * len(SCE_times),
                               # vertical_lines_sytle="solid",
                               # vertical_lines_linewidth=[0.2] * len(SCE_times),
                               span_area_coords=span_area_coords,
                               span_area_colors=span_area_colors,
                               span_area_only_on_raster=False,
                               spike_shape=spike_shape,
                               spike_shape_size=0.5,
                               save_formats=["pdf", "png"])
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("fifi")
            continue

        if just_plot_raster_with_cells_assemblies_and_shifts:
            ms.plot_raster_with_cells_assemblies_and_shifts(only_cell_assemblies=False)
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_raster_with_cells_assemblies_and_shifts")
            continue

        if just_produce_cell_assemblies_verification:
            ms.produce_cell_assemblies_verification()
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_produce_cell_assemblies_verification")
            continue

        if just_plot_traces_raster:
            print("just_plot_traces_raster")
            ms.plot_traces_on_raster(spike_nums_to_use=spike_nums_to_use,
                                     sce_times=None, with_run=True, display_spike_nums=False)

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
            ms.plot_cell_assemblies_on_map(save_formats=["pdf", "png"])
            if ms_index == len(ms_to_analyse) - 1:
                raise Exception("just_plot_cell_assemblies_on_map exception")
            continue

        if ms.age not in ms_by_age:
            ms_by_age[ms.age] = []

        ms_by_age[ms.age].append(ms)

        if do_plot_interneurons_connect_maps or do_plot_connect_hist or do_find_hubs:
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
                                       save_formats=["pdf", "png"])
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

                    plot_hist_distribution(distribution_data=ratio_spikes_events,
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
                    plot_hist_distribution(distribution_data=ratio_spikes_total_events,
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
                # ms_of_this_age[0].plot_psth_twitches(twitches_group=1,
                #                                      line_mode=line_mode, with_other_ms=ms_of_this_age[1:])
                # ms_of_this_age[0].plot_psth_twitches(twitches_group=2,
                #                                      line_mode=line_mode, with_other_ms=ms_of_this_age[1:])
                # ms_of_this_age[0].plot_psth_twitches(twitches_group=3,
                #                                      line_mode=line_mode, with_other_ms=ms_of_this_age[1:])
                # ms_of_this_age[0].plot_psth_twitches(twitches_group=4,
                #                                      line_mode=line_mode, with_other_ms=ms_of_this_age[1:])

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
            plot_hist_distribution(distribution_data=ratio_spikes,
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
            plot_hist_distribution(distribution_data=ratio_spikes,
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
        n_cells = ms.coord_obj.n_cells
        # spike_struct.build_spike_trains()
        # used to keep only some cells
        cells_to_keep = np.arange(ms.coord_obj.n_cells)

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
                fusion_frames = False
                if fusion_frames:
                    # now we want to fusion frames that are close to each other
                    frames_diff = np.diff(frames_selected)
                    fusion_thr = 50
                    for frame_index in np.arange(len(frames_diff)):
                        if 1 < frames_diff[frame_index] < fusion_thr:
                            frames_selected = np.concatenate(
                                (frames_selected, np.arange(frames_selected[frame_index] + 1,
                                                            frames_selected[frame_index + 1])))
                    binary_array = np.zeros(spike_nums_to_use.shape[1], dtype="int8")
                    frames_selected = np.unique(frames_selected)
                    # frames_selected = frames_selected[frames_selected < spike_nums_to_use.shape[1]]
                    binary_array[frames_selected] = 1
                    run_periods = get_continous_time_periods(binary_array)
                    frame_extension = 0
                    for run_period in run_periods:
                        if run_period[0] > frame_extension:
                            frames_selected = np.concatenate(
                                (frames_selected, np.arange(run_period[0] - frame_extension,
                                                            run_period[0])))
                        if run_period[1] < (spike_nums_to_use.shape[1] - frame_extension):
                            frames_selected = np.concatenate((frames_selected, np.arange(run_period[1] + 1,
                                                                                         run_period[
                                                                                             1] + frame_extension + 1)))
                binary_array = np.zeros(spike_nums_to_use.shape[1], dtype="int8")
                frames_selected = np.unique(frames_selected)
                # frames_selected = frames_selected[frames_selected < spike_nums_to_use.shape[1]]
                binary_array[frames_selected] = 1
                run_periods = get_continous_time_periods(binary_array)

                span_area_coords = [run_periods]
                span_area_colors = ["red"]

                plot_spikes_raster(spike_nums=spike_nums_to_use, param=ms.param,
                                   spike_train_format=False,
                                   span_area_only_on_raster=False,
                                   title=f"raster plot {data_descr}",
                                   file_name=f"spike_nums_test_run_{data_descr}",
                                   y_ticks_labels=spike_struct.labels,
                                   y_ticks_labels_size=4,
                                   save_raster=True,
                                   show_raster=False,
                                   plot_with_amplitude=False,
                                   activity_threshold=spike_struct.activity_threshold,
                                   # 500 ms window
                                   sliding_window_duration=sliding_window_duration,
                                   show_sum_spikes_as_percentage=True,
                                   span_area_coords=span_area_coords,
                                   span_area_colors=span_area_colors,
                                   spike_shape="|",
                                   spike_shape_size=1,
                                   save_formats=["png", "pdf"])
                raise Exception("Richard_boyce")

            elif richard_option == "sleep_quiet_wake":
                frames_selected = np.concatenate((ms.richard_dict["REMs_Frames"],
                                                  ms.richard_dict["NREMs_Frames"]))
                frames_selected = np.concatenate((frames_selected,
                                                  ms.richard_dict["Quiet_Wake_Frames"]))
                frames_selected = np.unique(frames_selected)
            # removing frames over the number of frames in the raster dur
            frames_selected = frames_selected[frames_selected < spike_nums_to_use.shape[1]]
            spike_nums_to_use = spike_nums_to_use[:, frames_selected]
            # print(f"spike_nums_to_use n_frames after: {spike_nums_to_use.shape[1]}")
            # raise Exception("test richard")

        if ((ms.activity_threshold is None) or use_richard_option) and (not do_detect_sce_based_on_peaks_finder) \
                and (not do_detect_sce_on_traces):
            # print("kokorico")
            # cells_to_keep = np.arange(ms.coord_obj.n_cells)
            # for pc_number, pca_seq_cells_order in ms.pca_seq_cells_order.items():
            #     cells_to_keep = pca_seq_cells_order

            activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums_to_use,
                                                             window_duration=sliding_window_duration,
                                                             spike_train_mode=False,
                                                             n_surrogate=n_surrogate_activity_threshold,
                                                             perc_threshold=perc_threshold,
                                                             use_max_of_each_surrogate=use_max_of_each_surrogate,
                                                             debug_mode=False)
        elif do_detect_sce_based_on_peaks_finder or do_detect_sce_on_traces:
            activity_threshold = None
        else:
            activity_threshold = ms.activity_threshold
        if activity_threshold is not None:
            print(f"perc_threshold {perc_threshold}, "
                  f"activity_threshold {activity_threshold}, {np.round((activity_threshold / n_cells) * 100, 2)}%")
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

        if do_detect_sce_on_traces:
            # cells_to_keep = np.sum(ms.spike_struct.spike_nums, axis=1) > 2
            # cells_to_keep = np.arange(len(ms.raw_traces))
            # for pc_number, pca_seq_cells_order in ms.pca_seq_cells_order.items():
            #     cells_to_keep = pca_seq_cells_order

            cellsinpeak, sce_loc = detect_sce_on_traces(ms.raw_traces[cells_to_keep],
                                                        speed=ms.speed_by_frame, use_speed=True,
                                                        speed_threshold=None, sce_n_cells_threshold=5,
                                                        sce_min_distance=4, use_median_norm=True,
                                                        use_bleaching_correction=False,
                                                        use_savitzky_golay_filt=True)
            cellsinpeak = cellsinpeak.astype("int8")
            sce_times_bool = np.zeros(spike_nums_to_use.shape[1], dtype="bool")
            print(f"spike_nums_to_use.shape[1] {spike_nums_to_use.shape[1]}")
            sce_times_bool[sce_loc] = True
            SCE_times = get_continous_time_periods(sce_times_bool)
            ms.plot_traces_on_raster(spike_nums_to_use=spike_nums_to_use, sce_times=SCE_times, with_run=True,
                                     display_spike_nums=True, cellsinpeak=cellsinpeak, order_with_cell_assemblies=True)
            raise Exception("STOP AFTER TRACES")
            sce_times_numbers = np.ones(spike_nums_to_use.shape[1], dtype="int16")
            sce_times_numbers *= -1
            for period_index, period in enumerate(SCE_times):
                # if period[0] == period[1]:
                #     print("both periods are equals")
                sce_times_numbers[period[0]:period[1] + 1] = period_index
            ms.sce_bool = sce_times_bool
            ms.sce_times_numbers = sce_times_numbers
            ms.SCE_times = SCE_times
            print(f"n SCE {len(SCE_times)}")
            if len(SCE_times) == 0:
                print("No SCE :(")
                raise Exception("STOP")
        elif do_detect_sce_based_on_peaks_finder:
            for pc_number, pca_seq_cells_order in ms.pca_seq_cells_order.items():
                cells_to_keep = pca_seq_cells_order
            frames_to_exclude = None
            if ms.speed_by_frame is not None:
                frames_to_exclude = ms.speed_by_frame > 0
            results = get_peaks_periods_on_sum_of_activity(raster=spike_nums_to_use[cells_to_keep],
                                                           around_peaks=4, distance_bw_peaks=15, min_n_cells=5,
                                                           frames_to_exclude=frames_to_exclude)
            peaks, SCE_times, sce_times_bool, sce_times_numbers, cellsinpeak = results
            # useful for plotting twitches
            ms.sce_bool = sce_times_bool
            ms.sce_times_numbers = sce_times_numbers
            ms.SCE_times = SCE_times
        else:
            # TODO: detect_sce_with_sliding_window with spike_trains
            sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_nums_to_use[cells_to_keep],
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

        span_area_coords = []
        span_area_colors = []
        if ms.speed_by_frame is not None:
            binary_speed = np.zeros(len(ms.speed_by_frame), dtype="int8")
            binary_speed[ms.speed_by_frame > 1] = 1
            speed_periods = get_continous_time_periods(binary_speed)
            span_area_coords.append(speed_periods)
            span_area_colors.append("cornflowerblue")
        plot_spikes_raster(spike_nums=ms.spike_struct.spike_nums_dur[cells_to_keep],
                           param=param,
                           spike_train_format=False,
                           file_name=f"spike_nums_{data_descr}_with_sce",
                           y_ticks_labels=np.arange
                           (len(ms.spike_struct.spike_nums_dur[cells_to_keep])),
                           y_ticks_labels_size=2,
                           save_raster=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           raster_face_color='black',
                           cell_spikes_color='white',
                           span_area_coords=span_area_coords,
                           span_area_colors=span_area_colors,
                           vertical_lines=SCE_times,
                           vertical_lines_colors=['red'] * len(SCE_times),
                           vertical_lines_sytle="solid",
                           vertical_lines_linewidth=[0.2] * len(SCE_times),
                           alpha_span_area=0.8,
                           span_area_only_on_raster=False,
                           show_sum_spikes_as_percentage=True,
                           spike_shape="o",
                           spike_shape_size=0.3,
                           save_formats="pdf",
                           SCE_times=SCE_times)

        print(f"Nb SCE: {cellsinpeak.shape}")
        # raise Exception("just trying")
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
                if spike_struct.spike_trains is None:
                    ms.spike_struct.set_spike_trains_from_spike_nums()
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
                if False:
                    pass
                # if do_clustering_with_twitches_events:
                #     n_times = len(sce_times_numbers)
                #     ms.define_twitches_events()
                #     for twitch_group in [9]:  # 1, 3, 4, 5, 6, 7, 8
                #         twitches_times = ms.events_by_twitches_group[twitch_group]
                #         cellsinpeak = np.zeros((n_cells, len(twitches_times)), dtype="int16")
                #         for twitch_index, twitch_period in enumerate(twitches_times):
                #             cellsinpeak[:, twitch_index] = np.sum(
                #                 spike_nums_to_use[:, twitch_period[0]:twitch_period[1] + 1], axis=1)
                #             cellsinpeak[cellsinpeak[:, twitch_index] > 0, twitch_index] = 1
                #
                #         twitches_times_numbers = np.ones(n_times, dtype="int16")
                #         twitches_times_numbers *= -1
                #         for twitch_index, twitch_period in enumerate(twitches_times):
                #             twitches_times_numbers[twitch_period[0]:twitch_period[1] + 1] = twitch_index
                #
                #         twitches_times_bool = np.zeros(n_times, dtype="bool")
                #         for twitch_index, twitch_period in enumerate(twitches_times):
                #             twitches_times_bool[twitch_period[0]:twitch_period[1] + 1] = True
                #         descr_twitch = ""
                #         descr_twitch += data_descr
                #         descr_twitch += "_" + ms.twitches_group_title[twitch_group]
                #
                #         print(f"twitch_group {twitch_group}: {len(twitches_times)}")
                #         if len(twitches_times) < 10:
                #             continue
                #         print(f"")
                #         range_n_clusters_k_mean = np.arange(2, np.min((len(twitches_times) // 2, 10)))
                #
                #         compute_and_plot_clusters_raster_kmean_version(labels=ms.spike_struct.labels,
                #                                                        activity_threshold=
                #                                                        ms.spike_struct.activity_threshold,
                #                                                        range_n_clusters_k_mean=range_n_clusters_k_mean,
                #                                                        n_surrogate_k_mean=n_surrogate_k_mean,
                #                                                        with_shuffling=with_shuffling,
                #                                                        spike_nums_to_use=spike_nums_to_use,
                #                                                        cellsinpeak=cellsinpeak,
                #                                                        data_descr=descr_twitch,
                #                                                        param=ms.param,
                #                                                        sliding_window_duration=sliding_window_duration,
                #                                                        SCE_times=twitches_times,
                #                                                        sce_times_numbers=twitches_times_numbers,
                #                                                        sce_times_bool=twitches_times_bool,
                #                                                        perc_threshold=perc_threshold,
                #                                                        n_surrogate_activity_threshold=
                #                                                        n_surrogate_activity_threshold,
                #                                                        debug_mode=debug_mode,
                #                                                        fct_to_keep_best_silhouettes=np.median,
                #                                                        with_cells_in_cluster_seq_sorted=
                #                                                        with_cells_in_cluster_seq_sorted,
                #                                                        keep_only_the_best=
                #                                                        keep_only_the_best_kmean_cluster)
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
                    if (ms.shift_data_dict is not None) and do_clustering_with_twitches_events:
                        # using twitch periods instead of SCE if info is available "shift_twitch"
                        shift_to_use = "shift_twitch"
                        # print(f"shift_to_use {shift_to_use}")
                        sce_times_bool = ms.shift_data_dict[shift_to_use]
                        n_frames = len(sce_times_bool)
                        # extending twitch periods
                        # period_extension
                        if shift_to_use == "shift_twitch":
                            extension_frames_after = 20
                            extension_frames_before = 1
                            shift_bool_tmp = np.copy(sce_times_bool)
                            true_frames = np.where(sce_times_bool)[0]
                            for frame in true_frames:
                                first_frame = max(0, frame - extension_frames_before)
                                last_frame = min(n_frames - 1, frame + extension_frames_after)
                                shift_bool_tmp[first_frame:last_frame + 1] = True
                            sce_times_bool[shift_bool_tmp] = True
                        SCE_times = get_continous_time_periods(sce_times_bool.astype("int8"))
                        sce_times_numbers = np.ones(len(sce_times_bool), dtype="int16")
                        sce_times_numbers *= -1
                        cellsinpeak = np.zeros((n_cells, len(SCE_times)), dtype="int16")
                        for index, period in enumerate(SCE_times):
                            sce_times_numbers[period[0]:period[1] + 1] = index
                            cellsinpeak[:, index] = np.sum(
                                spike_nums_to_use[:, period[0]:period[1] + 1], axis=1)
                            cellsinpeak[cellsinpeak[:, index] > 0, index] = 1
                        data_descr += f"_{shift_to_use}_time_as_sce_"

                    if use_hdbscan:
                        try_hdbscan(cells_in_sce=cellsinpeak, spike_nums=spike_nums_to_use, param=ms.param,
                                    SCE_times=SCE_times,
                                    data_descr=data_descr + clustering_bonus_descr,
                                    activity_threshold=activity_threshold)
                    else:
                        if cells_to_keep is None:
                            cells_to_keep = np.arange(len(spike_nums_to_use))
                        if ms.spike_struct.labels is None:
                            labels = None
                        else:
                            labels = ms.spike_struct.labels[cells_to_keep]
                        compute_and_plot_clusters_raster_kmean_version(labels=labels,
                                                                       activity_threshold=ms.spike_struct.activity_threshold,
                                                                       range_n_clusters_k_mean=range_n_clusters_k_mean,
                                                                       n_surrogate_k_mean=n_surrogate_k_mean,
                                                                       with_shuffling=with_shuffling,
                                                                       spike_nums_to_use=spike_nums_to_use[
                                                                           cells_to_keep],
                                                                       cellsinpeak=cellsinpeak,
                                                                       data_descr=data_descr + clustering_bonus_descr,
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
                                              extra_file_name=f"part_{split_id + 1}",
                                              sce_times_bool=sce_times_bool_to_use,
                                              use_only_uniformity_method=use_only_uniformity_method,
                                              use_loss_score_to_keep_the_best_from_tree=
                                              use_loss_score_to_keep_the_best_from_tree,
                                              spike_shape="|",
                                              spike_shape_size=10
                                              )

            else:
                print("Start of use_new_pattern_package")
                find_significant_patterns(spike_nums=spike_nums_to_use[:5000], param=param,
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
                                          keep_the_longest_seq=keep_the_longest_seq, ms=ms)

    return


main()
