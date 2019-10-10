import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import math
import neo
import quantities as pq
import elephant.conversion as elephant_conv


def plot_psth(cell_dict, frequency_id, threshold, by_cell=None, line_mode=False,
              background_color="white", label_color="black",
              ax_to_use=None, put_mean_line_on_plt=False,
              color_to_use=None, file_name_bonus="",
              with_other_ms=None, path_results="",
              save_formats="pdf"):
    """
    cell_dict key = different frequencies
    #  value -> dict with keys "stim_times_period", "time_stamps", "cells_id", "responses",
    :param sce_bool:
    :param only_in_sce:
    :param time_around:
    group 4: those not in sce-events and not followed by sce
    :param save_formats:
    :return:
    """

    title_option = str(frequency_id)

    max_value = 0
    if ax_to_use is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(30, 30))
        fig.patch.set_facecolor(background_color)

        ax1.set_facecolor(background_color)
    else:
        ax1 = ax_to_use

    values_dict = dict()
    time_stamps = cell_dict["time_stamps"]
    stim_times_period = cell_dict["stim_times_period"]
    if by_cell is not None:
        responses = cell_dict["responses_by_cell"][by_cell]
        mean_values = responses
        median_values = responses
    else:
        responses = cell_dict["responses"]
        mean_values = np.nanmean(responses, axis=0)
        std_values = np.nanstd(responses, axis=0)
        median_values = np.nanmedian(responses, axis=0)
        p_25_values = np.nanpercentile(responses, 25, axis=0)
        p_75_values = np.nanpercentile(responses, 75, axis=0)
        values_dict["mean"] = mean_values
        values_dict["std"] = std_values
        values_dict["median"] = median_values
        values_dict["p25"] = p_25_values
        values_dict["p75"] = p_75_values
    if color_to_use is not None:
        color = color_to_use
    else:
        color = "blue"
    # print(f"len 0: {len(np.where(np.diff(median_values) == 0)[0])}")
    if line_mode:
        mean_version = True
        if mean_version:
            ax1.plot(time_stamps,
                     mean_values, color=color, lw=2, label=f"{frequency_id}")
            if put_mean_line_on_plt:
                plt.plot(time_stamps,
                         mean_values, color=color, lw=2)
            if with_other_ms is None and (not by_cell):
                ax1.fill_between(time_stamps, mean_values - std_values,
                                 mean_values + std_values,
                                 alpha=0.5, facecolor=color)
            # max_value = np.max((max_value, np.max(mean_values + std_values)))
            ax1.hlines(threshold, 0,
                       time_stamps[-1], color="blue", linewidth=2,
                       linestyles="dashed")
        else:
            ax1.plot(time_stamps,
                     median_values, color=color, lw=2, label=f"{frequency_id}")
            if put_mean_line_on_plt:
                plt.plot(time_stamps,
                         median_values, color=color, lw=2)
            if with_other_ms is None and (not by_cell):
                ax1.fill_between(time_stamps, p_25_values, p_75_values,
                                 alpha=0.5, facecolor=color)
            # max_value = np.max((max_value, np.max(p_75_values)))
        ax1.vlines(stim_times_period[0], 0,
                   1, color=label_color, linewidth=2,
                   linestyles="dashed")
        ax1.vlines(stim_times_period[1], 0,
                   1, color=label_color, linewidth=2,
                   linestyles="dashed")
    else:
        edgecolor = color
        ax1.bar(time_stamps,
                median_values, color=color, edgecolor=edgecolor, width=30,
                # yerr=p_75_values-median_values,
                label=f"{frequency_id}")
        # ax1.plot(time_stamps,
        #          p_75_values, color=color, lw=2)
        if not by_cell:
            ax1.fill_between(time_stamps, p_25_values, p_75_values,
                             alpha=0.3, facecolor=color)
        ax1.axvspan(stim_times_period[0], stim_times_period[1], alpha=0.3, facecolor="gray", zorder=2)
        # max_value = np.max((max_value, np.max(mean_values)))
        ax1.hlines(threshold, 0,
                   time_stamps[-1], color="blue", linewidth=2,
                   linestyles="dashed")
    # if put_mean_line_on_plt:
    # ax1.hlines(activity_threshold_percentage, -1 * time_around, time_around,
    #            color="white", linewidth=1,
    #            linestyles="dashed")

    ax1.tick_params(axis='y', colors=label_color, labelsize=20)
    ax1.tick_params(axis='x', colors=label_color, labelsize=20)

    # if with_other_ms is not None:
    ax1.legend()

    # extra_info = ""
    # if line_mode:
    #     extra_info = "lines_"
    # if mean_version:
    #     extra_info += "mean_"
    # else:
    #     extra_info += "median_"

    # ax1.title(f"{descr} {n_twitches} twitches bar chart {title_option} {extra_info}")
    ax1.set_ylabel(f"frequencey (Hz)", fontsize=30, labelpad=20)
    ax1.set_xlabel("time (ms)", fontsize=30, labelpad=20)

    # ax1.set_ylim(0, max_value)
    # ax1.set_ylim(0, 1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax1.xaxis.label.set_color(label_color)
    ax1.yaxis.label.set_color(label_color)
    # xticks = np.arange(0, len(data_dict))
    # ax1.set_xticks(xticks)
    # # sce clusters labels
    # ax1.set_xticklabels(labels)
    if by_cell:
        file_name_bonus = file_name_bonus + f"_cell_{by_cell}"
    if ax_to_use is None:
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{path_results}/psth_{frequency_id}_{file_name_bonus}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())

        plt.close()

    return values_dict


def plot_all_psth_in_one_figure(data_dict, frequencies_id, path_results, threshold_dict, file_name_bonus="",
                                line_mode=True, save_formats="pdf"):
    """
    Will plot in one plot with subplots all  PSTH
    :param data_dict: # data_dict, key = different frequencies
    # cell_dict value -> dict with keys "stim_times_period", "time_stamps", "cells_id", "responses",
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
    colors = ['#1f78b4', '#33a02c', '#e31a1c',
              '#ff7f00', '#6a3d9a', '#b15928']
    colors = ['#1f78b4',
              '#ff7f00', '#b15928']
    color_dict = dict()
    color_dict["10"] = '#1f78b4'
    color_dict["20"] = '#ff7f00'
    color_dict["50"] = '#b15928'

    background_color = "white"
    labels_color = "black"
    n_plots = len(frequencies_id) + 1
    keys_data_dict = list(data_dict.keys())

    max_n_lines = 3
    n_lines = n_plots if n_plots <= max_n_lines else max_n_lines
    n_col = math.ceil(n_plots / n_lines)

    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                             figsize=(50, 70))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)

    avg_values_by_freq = dict()

    axes = axes.flatten()
    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        if ax_index >= n_plots or ax_index >= len(frequencies_id):
            continue
        frequency_id = frequencies_id[ax_index]
        cell_dict = data_dict[frequencies_id[ax_index]]
        if threshold_dict is not None:
            threshold = threshold_dict[frequencies_id[ax_index]]["threshold"]
        else:
            threshold = None
        for key_color, color in color_dict.items():
            if key_color in frequency_id:
                color_to_use = color
        values_dict = plot_psth(cell_dict=cell_dict, frequency_id=frequency_id, line_mode=line_mode,
                                ax_to_use=ax, put_mean_line_on_plt=line_mode, path_results=path_results,
                                file_name_bonus=file_name_bonus, background_color=background_color,
                                label_color=labels_color,
                                color_to_use=color_to_use, threshold=threshold)
        plot_psth(cell_dict=cell_dict, frequency_id=frequency_id, line_mode=line_mode,
                  put_mean_line_on_plt=line_mode, path_results=path_results,
                  file_name_bonus=file_name_bonus, background_color=background_color, label_color=labels_color,
                  color_to_use=color_to_use, threshold=threshold)

        avg_values_by_freq[frequency_id] = values_dict

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{path_results}/psth_erwan_{file_name_bonus}'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()

    return avg_values_by_freq


def plot_all_psth_of_a_cell_in_one_figure(data_dict, frequencies_id, cell_id, responses_by_freq_id_dict,
                                          path_results, threshold_dict, file_name_bonus="",
                                          line_mode=True, save_formats="pdf"):
    """
    Will plot in one plot with subplots all  PSTH
    :param data_dict: # data_dict, key = different frequencies
    # cell_dict value -> dict with keys "stim_times_period", "time_stamps", "cells_id", "responses",
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
    colors = ['#1f78b4', '#33a02c', '#e31a1c',
              '#ff7f00', '#6a3d9a', '#b15928']
    colors = ['#1f78b4',
              '#ff7f00', '#b15928']
    color_dict = dict()
    color_dict["10"] = '#1f78b4'
    color_dict["20"] = '#ff7f00'
    color_dict["50"] = '#b15928'

    background_color = "white"
    labels_color = "black"
    if line_mode:
        n_plots = len(frequencies_id) + 1
    else:
        n_plots = len(frequencies_id)
    # keys_data_dict = list(data_dict.keys())

    max_n_lines = 3
    n_lines = n_plots if n_plots <= max_n_lines else max_n_lines
    n_col = math.ceil(n_plots / n_lines)

    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                             figsize=(50, 70))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)

    avg_values_by_freq = dict()

    axes = axes.flatten()
    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        if ax_index >= n_plots or ax_index >= len(frequencies_id):
            continue
        frequency_id = frequencies_id[ax_index]
        cell_dict = data_dict[frequencies_id[ax_index]]
        if threshold_dict is not None:
            threshold = threshold_dict[cell_id][frequency_id]
        else:
            threshold = None
        for key_color, color in color_dict.items():
            if key_color in frequency_id:
                color_to_use = color
        values_dict = plot_psth(cell_dict=cell_dict, frequency_id=frequency_id, line_mode=line_mode,
                                by_cell=cell_id,
                                ax_to_use=ax, put_mean_line_on_plt=line_mode, path_results=path_results,
                                file_name_bonus=file_name_bonus, background_color=background_color,
                                label_color=labels_color,
                                color_to_use=color_to_use, threshold=threshold)
        plot_psth(cell_dict=cell_dict, frequency_id=frequency_id, line_mode=line_mode,
                  put_mean_line_on_plt=line_mode, path_results=path_results, by_cell=cell_id,
                  file_name_bonus=file_name_bonus, background_color=background_color, label_color=labels_color,
                  color_to_use=color_to_use, threshold=threshold)

        avg_values_by_freq[frequency_id] = values_dict

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{path_results}/psth_erwan_cell_{cell_id}_{file_name_bonus}'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()

    return avg_values_by_freq


def save_avg_values_in_file(values_dict, path_results, time_stamps_dict, bonus_title=""):
    for freq_id, values in values_dict.items():
        writer = pd.ExcelWriter(f'{path_results}/{freq_id}_summary_{bonus_title}.xlsx')
        results_df = pd.DataFrame()
        results_df.insert(loc=0, column="time_stamps",
                          value=time_stamps_dict[freq_id],
                          allow_duplicates=False)
        i = 1
        for key_result, result_values in values.items():
            results_df.insert(loc=i, column=key_result,
                              value=result_values,
                              allow_duplicates=False)
            i += 1
        results_df.to_excel(writer, 'summary', index=False)
        writer.save()


#
# def erwan_old_main():
#     root_path = "/media/julien/Not_today/hne_not_today/"
#     path_data = root_path + "data/erwan/"
#     path_results = root_path + "results_hne/"
#
#     time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
#     path_results = path_results + f"{time_str}"
#     os.mkdir(path_results)
#
#     file_name = "PSTH_Pyramides.xlsx"
#
#     df = pd.read_excel(os.path.join(path_data, file_name))
#
#     # col 0: Temps
#     # col 1: Cellules
#     # col 2: -60_10Hz
#     # col 3: -60_20Hz
#     # col 4: -60_50Hz
#     # col 5: 15_10Hz
#     # col 6: 15_20Hz
#     # col 7: 15_50Hz
#     print(f"len(df) {len(df)}")
#     columns_titles = list(df.columns)
#     print(f"columnsTitles {columns_titles}")
#     times = df.loc[:, "Temps"]
#     # print(f"times {times}")
#     times_iloc = df.iloc[:, 0]
#     # print(f"times_iloc {times_iloc}")
#
#     cells_id = df.iloc[:, 1].unique()
#     print(f"cells_id {cells_id}")
#
#     frequencies_id = columns_titles[2:]
#
#     n_period_to_substract = 5
#     time_to_substract = n_period_to_substract * 50
#
#     # for cell_id in cells_id:
#     one_cell_df = df.loc[df['Cellules'] == cells_id[0]]
#     # binned time
#     time_stamps = one_cell_df.loc[:, "Temps"]
#     # time_stamps = [int(str(x)[:-4]) for x in time_stamps]
#     # print(f'{cell_id} time_stamps {time_stamps}')
#     time_stamps = np.array(time_stamps).astype(int)[n_period_to_substract:]
#     print(f"len(time_stamps) {len(time_stamps)}")
#
#     # data_dict, key = different frequencies
#     # value -> dict with keys "stim_times_period", "time_stamps", "cells_id", "responses",
#     data_dict = dict()
#     threshold_dict = dict()
#     for freq_id in frequencies_id:
#         data_dict[freq_id] = dict()
#         threshold_dict[freq_id] = dict()
#
#         freq_threshold_dict = threshold_dict[freq_id]
#         freq_dict = data_dict[freq_id]
#         if "10" in freq_id:
#             freq_dict["stim_times_period"] = (5178-time_to_substract, 6178-time_to_substract)
#         elif "20" in freq_id:
#             freq_dict["stim_times_period"] = (5170-time_to_substract, 5670-time_to_substract)
#         else:  # 50
#             freq_dict["stim_times_period"] = (5465-time_to_substract, 5670-time_to_substract)
#
#         freq_dict["time_stamps"] = time_stamps
#         freq_threshold_dict["time_stamps"] = time_stamps
#
#         freq_dict["cells_id"] = cells_id
#         freq_threshold_dict["cells_id"] = cells_id
#
#         n_surrogate = 1000
#         surrogate_response_values = []
#         for surrogate_index in range(n_surrogate):
#             responses_surrogate = np.zeros((len(cells_id), len(time_stamps)))
#             for cell_index, cell_id in enumerate(cells_id):
#                 df_cell = df.loc[df['Cellules'] == cell_id]
#                 resp_tmp = df_cell.loc[:, freq_id]
#                 if len(resp_tmp) > 0:
#                     resp_tmp = np.array(resp_tmp)[n_period_to_substract:]
#                     # we roll it
#                     responses_surrogate[cell_index] = np.roll(resp_tmp, np.random.randint(1, len(resp_tmp)))
#                 else:
#                     responses_surrogate[cell_index] = np.repeat(np.nan, len(time_stamps))
#             surrogate_response_values.extend(list(np.nanmean(responses_surrogate, axis=0)))
#         threshold = np.nanpercentile(np.array(surrogate_response_values), 95)
#         freq_threshold_dict["threshold"] = threshold
#
#         responses = np.zeros((len(cells_id), len(time_stamps)))
#         for cell_index, cell_id in enumerate(cells_id):
#             # print(f"cell_id {cell_id}, freq_id {freq_id}")
#             df_cell = df.loc[df['Cellules'] == cell_id]
#             # print(f"len(df_cell) {len(df_cell)}")
#             resp_tmp = df_cell.loc[:, freq_id]
#             if len(resp_tmp) > 0:
#                 resp_tmp = np.array(resp_tmp)[n_period_to_substract:]
#                 # print(f"len(rest_tmp) {len(resp_tmp)}")
#                 responses[cell_index] = np.array(resp_tmp)
#             else:
#                 responses[cell_index] = np.repeat(np.nan, len(time_stamps))
#         freq_dict["responses"] = responses
#
#         # # for each frequency, we collect
#     for line_mode in [True, False]:
#         line_str = "line"
#         if not line_mode:
#             line_str = "bars"
#         avg_values_by_freq = plot_all_psth_in_one_figure(data_dict=data_dict, frequencies_id=frequencies_id,
#                                     file_name_bonus="non_normalized" + "_" + line_str, line_mode=line_mode,
#                                     path_results=path_results, save_formats=["pdf", "eps"],
#                                                          threshold_dict=threshold_dict)
#     save_avg_values_in_file(values_dict=avg_values_by_freq, path_results=path_results,
#                             bonus_title="non_normalized", time_stamps=time_stamps)
#     # normalizing by baseline
#     for freq_id in frequencies_id:
#         freq_dict = data_dict[freq_id]
#         stim_times_period = freq_dict["stim_times_period"]
#         indice_stim = np.searchsorted(time_stamps, stim_times_period[0])
#         baseline_times = np.arange(0, indice_stim)
#         responses = freq_dict["responses"]
#         for cell, cell_responses in enumerate(responses):
#             cell_responses = cell_responses - np.mean(cell_responses[baseline_times])
#             responses[cell] = cell_responses
#     for line_mode in [True, False]:
#         line_str = "line"
#         if not line_mode:
#             line_str = "bars"
#         avg_values_by_freq = plot_all_psth_in_one_figure(data_dict=data_dict, frequencies_id=frequencies_id,
#                                 file_name_bonus="normalized" + "_" + line_str, path_results=path_results,
#                                                          line_mode=line_mode,
#                                                          save_formats=["pdf", "eps"], threshold_dict=None)
#     save_avg_values_in_file(values_dict=avg_values_by_freq, path_results=path_results,
#                             bonus_title="normalized", time_stamps=time_stamps)


def erwan_main():
    root_path = "/media/julien/Not_today/hne_not_today/"
    path_data = root_path + "data/erwan/"
    path_results = root_path + "results_hne/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results + f"{time_str}"
    os.mkdir(path_results)

    file_name = "EventTimes_Pyramides.xlsx"
    # file_name = "PSTH_Pyramides.xlsx"

    df = pd.read_excel(os.path.join(path_data, file_name))

    # col 0: Cellules
    # col 1: -60_10Hz
    # col 2: -60_20Hz
    # col 3: -60_50Hz
    # col 4: 15_10Hz
    # col 5: 15_20Hz
    # col 6: 15_50Hz
    print(f"len(df) {len(df)}")
    columns_titles = list(df.columns)
    print(f"columnsTitles {columns_titles}")

    cells_id = df.iloc[:, 0].unique()
    print(f"cells_id {cells_id}")
    frequencies_id = columns_titles[1:]

    bin_sizes = [50, 100]

    # suing neo package spike_trains to represent the event times
    # first key is the cell, second key is the freq_id, third one is the list of event_time
    event_train_dict = dict()
    range_time_by_freq_dict = dict()
    for cell_index, cell_id in enumerate(cells_id):
        df_cell = df.loc[df['Cellules'] == cell_id]
        event_train_dict[cell_id] = dict()
        # first finding the start and stop time for each frequency
        for freq_id in frequencies_id:
            resp_time_tmp = df_cell.loc[:, freq_id]
            max_time = np.nanmax(np.array(resp_time_tmp))
            if np.isnan(max_time):
                continue
            if freq_id not in range_time_by_freq_dict:
                range_time_by_freq_dict[freq_id] = [0, max_time]
            elif range_time_by_freq_dict[freq_id][1] < max_time:
                range_time_by_freq_dict[freq_id] = [0, max_time]

        # building event trains
        for freq_id in frequencies_id:
            event_train = list(df_cell.loc[:, freq_id])
            event_train_dict[cell_id][freq_id] = event_train

    # raise Exception("TOTO")

    for bin_size in bin_sizes:

        n_period_to_substract = 0
        time_to_substract = n_period_to_substract * bin_size

        # for cell_id in cells_id:
        # one_cell_df = df.loc[df['Cellules'] == cells_id[0]]
        # binned time
        # time_stamps = one_cell_df.loc[:, "Temps"]
        # # time_stamps = [int(str(x)[:-4]) for x in time_stamps]
        # # print(f'{cell_id} time_stamps {time_stamps}')
        # time_stamps = np.array(time_stamps).astype(int)[n_period_to_substract:]
        # print(f"len(time_stamps) {len(time_stamps)}")

        # key is freq_id, vlue is 2d array n_cells * responses
        responses_by_freq_id_dict = dict()
        # first key is cell_id, 2nd key is freq_id
        threshold_by_cell_dict = dict()
        bins_by_freq_dict = dict()
        # for each cell and freq_id we build the bin event train
        for freq_id in frequencies_id:
            t_start, t_stop = range_time_by_freq_dict[freq_id]
            bins = np.arange(t_start, t_stop + bin_size, bin_size)
            bins_by_freq_dict[freq_id] = bins[:-1]
            responses = np.zeros((len(cells_id), len(bins) - 1))
            for cell_index, cell_id in enumerate(cells_id):

                event_train = event_train_dict[cell_id][freq_id]
                # event_trains_binned = elephant_conv.BinnedSpikeTrain(neo_event_train, binsize=bin_size_pq)

                # print(f"bins {bins}")
                hist, bin_edges = np.histogram(np.array(event_train), bins=bins, density=False)
                # print(f"hist {hist}")
                responses[cell_index, :] = np.array(hist)
                # now we z-score it
                if not np.isnan(np.nanstd(responses[cell_index, :])) and (np.nanstd(responses[cell_index, :]) > 0):
                    responses[cell_index, :] = (responses[cell_index, :] - np.nanmean(responses[cell_index, :])) / \
                                               np.nanstd(responses[cell_index, :])
                # now we do the 1000 surrogates to get the threshold for this cell and freq_id
                n_surrogate = 1000
                surrogate_response_values = []
                for surrogate_index in range(n_surrogate):
                    resp_tmp = np.copy(responses[cell_index, :])
                    # we roll it
                    responses_surrogate = np.roll(resp_tmp, np.random.randint(1, len(resp_tmp)))
                    surrogate_response_values.extend(list(responses_surrogate))
                threshold = np.nanpercentile(np.array(surrogate_response_values), 95)
                if cell_id not in threshold_by_cell_dict:
                    threshold_by_cell_dict[cell_id] = dict()
                threshold_by_cell_dict[cell_id][freq_id] = threshold
                # print(f"threshold {cell_id} {freq_id}: {threshold}")
                # print(f"{cell_id}, {freq_id}, hist {hist} bins {bins}")
            responses_by_freq_id_dict[freq_id] = responses
        # raise Exception("TOTO")

        # data_dict, key = different frequencies
        # value -> dict with keys "stim_times_period", "time_stamps", "cells_id", "responses",
        data_dict = dict()
        threshold_dict = dict()
        for freq_id in frequencies_id:
            data_dict[freq_id] = dict()
            threshold_dict[freq_id] = dict()

            freq_threshold_dict = threshold_dict[freq_id]
            freq_dict = data_dict[freq_id]
            if "10" in freq_id:
                freq_dict["stim_times_period"] = (5178 - time_to_substract, 6178 - time_to_substract)
            elif "20" in freq_id:
                freq_dict["stim_times_period"] = (5170 - time_to_substract, 5670 - time_to_substract)
            else:  # 50
                freq_dict["stim_times_period"] = (5465 - time_to_substract, 5670 - time_to_substract)

            freq_dict["time_stamps"] = bins_by_freq_dict[freq_id]
            freq_threshold_dict["time_stamps"] = bins_by_freq_dict[freq_id]

            freq_dict["cells_id"] = cells_id
            freq_threshold_dict["cells_id"] = cells_id

            n_surrogate = 1000
            surrogate_response_values = []
            for surrogate_index in range(n_surrogate):
                real_responses = responses_by_freq_id_dict[freq_id]
                responses_surrogate = np.zeros((len(cells_id), real_responses.shape[1]))
                for cell_index, cell_id in enumerate(cells_id):
                    # df_cell = df.loc[df['Cellules'] == cell_id]
                    resp_tmp = np.copy(real_responses[cell_index, :])
                    if len(resp_tmp) > 0:
                        resp_tmp = np.array(resp_tmp)[n_period_to_substract:]
                        # we roll it
                        responses_surrogate[cell_index] = np.roll(resp_tmp, np.random.randint(1, len(resp_tmp)))
                    else:
                        responses_surrogate[cell_index] = np.repeat(np.nan, real_responses.shape[1])
                surrogate_response_values.extend(list(np.nanmean(responses_surrogate, axis=0)))
            threshold = np.nanpercentile(np.array(surrogate_response_values), 95)
            freq_threshold_dict["threshold"] = threshold

            # responses = np.zeros((len(cells_id), len(time_stamps)))
            # for cell_index, cell_id in enumerate(cells_id):
            #     # print(f"cell_id {cell_id}, freq_id {freq_id}")
            #     df_cell = df.loc[df['Cellules'] == cell_id]
            #     # print(f"len(df_cell) {len(df_cell)}")
            #     resp_tmp = df_cell.loc[:, freq_id]
            #     if len(resp_tmp) > 0:
            #         resp_tmp = np.array(resp_tmp)[n_period_to_substract:]
            #         # print(f"len(rest_tmp) {len(resp_tmp)}")
            #         responses[cell_index] = np.array(resp_tmp)
            #     else:
            #         responses[cell_index] = np.repeat(np.nan, len(time_stamps))
            freq_dict["responses"] = responses_by_freq_id_dict[freq_id]
            freq_dict["responses_by_cell"] = dict()
            for cell_index, cell_id in enumerate(cells_id):
                freq_dict["responses_by_cell"][cell_id] = responses_by_freq_id_dict[freq_id][cell_index, :]

        for cell_id in cells_id:
            for line_mode in [True, False]:
                line_str = "line"
                if not line_mode:
                    line_str = "bars"
                avg_values_by_freq = \
                    plot_all_psth_of_a_cell_in_one_figure(data_dict=data_dict,
                                                          cell_id=cell_id,
                                                          responses_by_freq_id_dict=responses_by_freq_id_dict,
                                                          frequencies_id=frequencies_id,
                                                          file_name_bonus="non_normalized" + "_" +
                                                                          line_str + f"_bin_{bin_size}",
                                                          line_mode=line_mode,
                                                          path_results=path_results,
                                                          save_formats=["pdf"],
                                                          threshold_dict=threshold_by_cell_dict)
            # # for each frequency, we collect
        for line_mode in [True, False]:
            line_str = "line"
            if not line_mode:
                line_str = "bars"
            avg_values_by_freq = plot_all_psth_in_one_figure(data_dict=data_dict, frequencies_id=frequencies_id,
                                                             file_name_bonus="non_normalized" + "_" + line_str
                                                                             + f"_bin_{bin_size}",
                                                             line_mode=line_mode,
                                                             path_results=path_results, save_formats=["pdf"],
                                                             threshold_dict=threshold_dict)
        save_avg_values_in_file(values_dict=avg_values_by_freq, path_results=path_results,
                                bonus_title="non_normalized", time_stamps_dict=bins_by_freq_dict)
        # normalizing by baseline
        for freq_id in frequencies_id:
            real_responses = responses_by_freq_id_dict[freq_id]
            freq_dict = data_dict[freq_id]
            stim_times_period = freq_dict["stim_times_period"]
            indice_stim = np.searchsorted(real_responses[0, :], stim_times_period[0])
            baseline_times = np.arange(0, indice_stim)
            responses = freq_dict["responses"]
            for cell, cell_responses in enumerate(responses):
                cell_responses = cell_responses - np.mean(cell_responses[baseline_times])
                responses[cell] = cell_responses
        for line_mode in [True, False]:
            line_str = "line"
            if not line_mode:
                line_str = "bars"
            avg_values_by_freq = plot_all_psth_in_one_figure(data_dict=data_dict, frequencies_id=frequencies_id,
                                                             file_name_bonus="normalized" + "_" + line_str
                                                                             + f"_bin_{bin_size}",
                                                             path_results=path_results,
                                                             line_mode=line_mode,
                                                             save_formats=["pdf"], threshold_dict=None)
        save_avg_values_in_file(values_dict=avg_values_by_freq, path_results=path_results,
                                bonus_title="normalized", time_stamps_dict=bins_by_freq_dict)
    # eps


erwan_main()
