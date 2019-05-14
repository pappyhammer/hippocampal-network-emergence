import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import math


def plot_psth(cell_dict, frequency_id, line_mode=False,
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

    time_stamps = cell_dict["time_stamps"]
    stim_times_period  = cell_dict["stim_times_period"]
    responses = cell_dict["responses"]
    mean_values = np.nanmean(responses, axis=0)
    std_values = np.nanstd(responses, axis=0)
    median_values = np.nanmedian(responses, axis=0)
    p_25_values = np.nanpercentile(responses, 25, axis=0)
    p_75_values = np.nanpercentile(responses, 75, axis=0)
    values_dict = dict()
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
            if with_other_ms is None:
                ax1.fill_between(time_stamps, mean_values - std_values,
                                 mean_values + std_values,
                                 alpha=0.5, facecolor=color)
            max_value = np.max((max_value, np.max(mean_values + std_values)))
        else:
            ax1.plot(time_stamps,
                     median_values, color=color, lw=2, label=f"{frequency_id}")
            if with_other_ms is None:
                ax1.fill_between(time_stamps, p_25_values, p_75_values,
                                 alpha=0.5, facecolor=color)
            max_value = np.max((max_value, np.max(p_75_values)))

        ax1.vlines(stim_times_period[0], 0,
                   1, color=label_color, linewidth=2,
                   linestyles="dashed")
        ax1.vlines(stim_times_period[1], 0,
                   1, color=label_color, linewidth=2,
                   linestyles="dashed")
    else:
        edgecolor=color
        ax1.bar(time_stamps,
                median_values, color=color, edgecolor=edgecolor, width=30,
                # yerr=p_75_values-median_values,
                label=f"{frequency_id}")
        # ax1.plot(time_stamps,
        #          p_75_values, color=color, lw=2)
        ax1.fill_between(time_stamps, p_25_values, p_75_values,
                         alpha=0.3, facecolor=color)
        ax1.axvspan(stim_times_period[0], stim_times_period[1], alpha=0.3, facecolor="gray", zorder=2)
        max_value = np.max((max_value, np.max(mean_values)))
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
    ax1.set_ylim(0, 1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax1.xaxis.label.set_color(label_color)
    ax1.yaxis.label.set_color(label_color)
    # xticks = np.arange(0, len(data_dict))
    # ax1.set_xticks(xticks)
    # # sce clusters labels
    # ax1.set_xticklabels(labels)
    if ax_to_use is None:
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{path_results}/psth_{frequency_id}_{file_name_bonus}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())

        plt.close()

    return values_dict


def plot_all_psth_in_one_figure(data_dict, frequencies_id, path_results, file_name_bonus="",
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
    colors = ['#1f78b4', '#33a02c',  '#e31a1c',
              '#ff7f00', '#6a3d9a', '#b15928']
    colors = ['#1f78b4',
              '#ff7f00', '#b15928']
    color_dict = dict()
    color_dict["10"] = '#1f78b4'
    color_dict["20"] = '#ff7f00'
    color_dict["50"] = '#b15928'

    background_color = "white"
    labels_color = "black"
    n_plots = len(frequencies_id)
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
        if ax_index >= n_plots:
            continue
        frequency_id = frequencies_id[ax_index]
        cell_dict = data_dict[frequencies_id[ax_index]]
        for key_color, color in color_dict.items():
            if key_color in frequency_id:
                color_to_use = color
        values_dict = plot_psth(cell_dict=cell_dict, frequency_id=frequency_id, line_mode=line_mode,
                  ax_to_use=ax, put_mean_line_on_plt=line_mode, path_results=path_results,
                  file_name_bonus=file_name_bonus, background_color=background_color, label_color=labels_color,
                              color_to_use=color_to_use)
        plot_psth(cell_dict=cell_dict, frequency_id=frequency_id, line_mode=line_mode,
                  put_mean_line_on_plt=line_mode, path_results=path_results,
                  file_name_bonus=file_name_bonus, background_color=background_color, label_color=labels_color,
                  color_to_use=color_to_use)

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

def save_avg_values_in_file(values_dict, path_results, time_stamps, bonus_title=""):

    for freq_id, values in values_dict.items():
        writer = pd.ExcelWriter(f'{path_results}/{freq_id}_summary_{bonus_title}.xlsx')
        results_df = pd.DataFrame()
        results_df.insert(loc=0, column="time_stamps",
                          value=time_stamps,
                          allow_duplicates=False)
        i = 1
        for key_result, result_values in values.items():
            results_df.insert(loc=i, column=key_result,
                                 value=result_values,
                                 allow_duplicates=False)
            i += 1
        results_df.to_excel(writer, 'summary', index=False)
        writer.save()

def erwan_main():
    root_path = "/home/julien/these_inmed/hne_project/"
    path_data = root_path + "data/erwan/"
    path_results = root_path + "results_hne/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results + f"{time_str}"
    os.mkdir(path_results)

    file_name = "PSTH_Pyramides.xlsx"

    df = pd.read_excel(os.path.join(path_data, file_name))

    # col 0: Temps
    # col 1: Cellules
    # col 2: -60_10Hz
    # col 3: -60_20Hz
    # col 4: -60_50Hz
    # col 5: 15_10Hz
    # col 6: 15_20Hz
    # col 7: 15_50Hz
    print(f"len(df) {len(df)}")
    columns_titles = list(df.columns)
    print(f"columnsTitles {columns_titles}")
    times = df.loc[:, "Temps"]
    # print(f"times {times}")
    times_iloc = df.iloc[:, 0]
    # print(f"times_iloc {times_iloc}")

    cells_id = df.iloc[:, 1].unique()
    print(f"cells_id {cells_id}")

    frequencies_id = columns_titles[2:]

    n_period_to_substract = 5
    time_to_substract = n_period_to_substract * 50

    # for cell_id in cells_id:
    one_cell_df = df.loc[df['Cellules'] == cells_id[0]]
    time_stamps = one_cell_df.loc[:, "Temps"]
    # time_stamps = [int(str(x)[:-4]) for x in time_stamps]
    # print(f'{cell_id} time_stamps {time_stamps}')
    time_stamps = np.array(time_stamps).astype(int)[n_period_to_substract:]
    print(f"len(time_stamps) {len(time_stamps)}")

    # data_dict, key = different frequencies
    # value -> dict with keys "stim_times_period", "time_stamps", "cells_id", "responses",
    data_dict = dict()
    for freq_id in frequencies_id:
        data_dict[freq_id] = dict()

        freq_dict = data_dict[freq_id]
        if "10" in freq_id:
            freq_dict["stim_times_period"] = (5178-time_to_substract, 6178-time_to_substract)
        elif "20" in freq_id:
            freq_dict["stim_times_period"] = (5170-time_to_substract, 5670-time_to_substract)
        else:  # 50
            freq_dict["stim_times_period"] = (5465-time_to_substract, 5670-time_to_substract)

        freq_dict["time_stamps"] = time_stamps

        freq_dict["cells_id"] = cells_id

        responses = np.zeros((len(cells_id), len(time_stamps)))
        for cell_index, cell_id in enumerate(cells_id):
            # print(f"cell_id {cell_id}, freq_id {freq_id}")
            df_cell = df.loc[df['Cellules'] == cell_id]
            # print(f"len(df_cell) {len(df_cell)}")
            resp_tmp = df_cell.loc[:, freq_id]
            if len(resp_tmp) > 0:
                resp_tmp = np.array(resp_tmp)[n_period_to_substract:]
                # print(f"len(rest_tmp) {len(resp_tmp)}")
                responses[cell_index] = np.array(resp_tmp)
            else:
                responses[cell_index] = np.repeat(np.nan, len(time_stamps))
        freq_dict["responses"] = responses

        # # for each frequency, we collect
    for line_mode in [True, False]:
        line_str = "line"
        if not line_mode:
            line_str = "bars"
        avg_values_by_freq = plot_all_psth_in_one_figure(data_dict=data_dict, frequencies_id=frequencies_id,
                                    file_name_bonus="non_normalized" + "_" + line_str, line_mode=line_mode,
                                    path_results=path_results, save_formats=["pdf", "eps"])
    save_avg_values_in_file(values_dict=avg_values_by_freq, path_results=path_results,
                            bonus_title="non_normalized", time_stamps=time_stamps)
    # normalizing by baseline
    for freq_id in frequencies_id:
        freq_dict = data_dict[freq_id]
        stim_times_period = freq_dict["stim_times_period"]
        indice_stim = np.searchsorted(time_stamps, stim_times_period[0])
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
                                file_name_bonus="normalized" + "_" + line_str, path_results=path_results,
                                                         line_mode=line_mode,
                                                         save_formats=["pdf", "eps"])
    save_avg_values_in_file(values_dict=avg_values_by_freq, path_results=path_results,
                            bonus_title="normalized", time_stamps=time_stamps)

erwan_main()