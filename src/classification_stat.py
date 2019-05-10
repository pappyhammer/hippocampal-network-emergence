import numpy as np
from pattern_discovery.tools.misc import get_continous_time_periods
import scipy.signal as signal


def compute_stats(spike_nums_dur, predicted_spike_nums_dur, traces, with_threshold=None):
    """
    Compute the stats based on raster dur
    :param spike_nums_dur: should not be "uint" dtype
    :param predicted_spike_nums_dur: should not be "uint" dtype
    :return: two dicts: first one with stats on frames, the other one with stats on transients
    Frames dict has the following keys (as String):
    TP: True Positive
    FP: False Positive
    FN: False Negative
    TN: True Negative
    sensitivity or TPR: True Positive Rate or Recall
    specificity or TNR: True Negative Rate or Selectivity
    FPR: False Positive Rate or Fall-out
    FNR: False Negative Rate or Miss Rate
    ACC: accuracy
    Prevalence: sum positive conditions / total population (for frames only)
    PPV: Positive Predictive Value or Precision
    FDR: False Discovery Rate
    FOR: False Omission Rate
    NPV: Negative Predictive Value
    LR+: Positive Likelihood ratio
    LR-: Negative likelihood ratio

    transients dict has just the following keys:
    TP
    FN
    sensitivity or TPR: True Positive Rate or Recall
    FNR: False Negative Rate or Miss Rate
    """

    if spike_nums_dur.shape != predicted_spike_nums_dur.shape:
        raise Exception("both spike_nums_dur should have the same shape")
    if len(spike_nums_dur.shape) == 1:
        # we transform them in a 2 dimensions array
        spike_nums_dur = spike_nums_dur.reshape(1, spike_nums_dur.shape[0])
        predicted_spike_nums_dur = predicted_spike_nums_dur.reshape(1, predicted_spike_nums_dur.shape[0])
        if traces is not None:
            traces = traces.reshape(1, traces.shape[0])

    frames_stat = dict()
    transients_stat = dict()

    n_frames = spike_nums_dur.shape[1]
    n_cells = spike_nums_dur.shape[0]

    # full raster dur represents the raster dur built from all potential onsets and peaks
    full_raster_dur = None
    if traces is not None:
        full_raster_dur = get_raster_dur_from_traces(traces, with_threshold=with_threshold)

    # positive means active frame, negative means non-active frames
    # condition is the ground truth
    # predicted is the one computed (RNN, CaiMan etc...)

    tp_frames = 0
    fp_frames = 0
    fn_frames = 0
    tn_frames = 0

    tp_transients = 0
    fn_transients = 0
    fp_transients = 0
    tn_transients = 0

    for cell in np.arange(n_cells):
        raster_dur = spike_nums_dur[cell]
        predicted_raster_dur = predicted_spike_nums_dur[cell]

        predicted_positive_frames = np.where(predicted_raster_dur)[0]
        predicted_negative_frames = np.where(predicted_raster_dur == 0)[0]

        tp_frames += len(np.where(raster_dur[predicted_positive_frames] == 1)[0])
        fp_frames += len(np.where(raster_dur[predicted_positive_frames] == 0)[0])
        fn_frames += len(np.where(raster_dur[predicted_negative_frames] == 1)[0])
        tn_frames += len(np.where(raster_dur[predicted_negative_frames] == 0)[0])

        n_fake_transients = 0
        # transients section
        transient_periods = get_continous_time_periods(raster_dur)
        if full_raster_dur is not None:
            full_transient_periods = get_continous_time_periods(full_raster_dur[cell])
            fake_transients_periods = []
            # keeping only the fake ones
            for transient_period in full_transient_periods:
                if np.sum(raster_dur[transient_period[0]:transient_period[1]+1]) > 0:
                    continue
                fake_transients_periods.append(transient_period)
            n_fake_transients = len(fake_transients_periods)
        # positive condition
        n_transients = len(transient_periods)
        tp = 0
        for transient_period in transient_periods:
            frames = np.arange(transient_period[0], transient_period[1] + 1)
            if np.sum(predicted_raster_dur[frames]) > 0:
                tp += 1
        tn = 0
        if full_raster_dur is not None:
            for transient_period in fake_transients_periods:
                frames = np.arange(transient_period[0], transient_period[1] + 1)
                if np.sum(predicted_raster_dur[frames]) == 0:
                    tn += 1
        tp_transients += tp
        fn_transients += (n_transients - tp)
        tn_transients += tn
        fp_transients += (n_fake_transients - tn)

    frames_stat["TP"] = tp_frames
    frames_stat["FP"] = fp_frames
    frames_stat["FN"] = fn_frames
    frames_stat["TN"] = tn_frames

    # frames_stat["TPR"] = tp_frames / (tp_frames + fn_frames)
    if (tp_frames + fn_frames) > 0:
        frames_stat["sensitivity"] = tp_frames / (tp_frames + fn_frames)
    else:
        frames_stat["sensitivity"] = 1
    frames_stat["TPR"] = frames_stat["sensitivity"]

    if (tn_frames + fp_frames) > 0:
        frames_stat["specificity"] = tn_frames / (tn_frames + fp_frames)
    else:
        frames_stat["specificity"] = 1
    frames_stat["TNR"] = frames_stat["specificity"]

    if (tp_frames + tn_frames + fp_frames + fn_frames) > 0:
        frames_stat["ACC"] = (tp_frames + tn_frames) / (tp_frames + tn_frames + fp_frames + fn_frames)
    else:
        frames_stat["ACC"] = 1

    if (tp_frames + fp_frames) > 0:
        frames_stat["PPV"] = tp_frames / (tp_frames + fp_frames)
    else:
        frames_stat["PPV"] = 1
    if (tn_frames + fn_frames) > 0:
        frames_stat["NPV"] = tn_frames / (tn_frames + fn_frames)
    else:
        frames_stat["NPV"] = 1

    frames_stat["FNR"] = 1 - frames_stat["TPR"]

    frames_stat["FPR"] = 1 - frames_stat["TNR"]

    if "PPV" in frames_stat:
        frames_stat["FDR"] = 1 - frames_stat["PPV"]

    if "NPV" in frames_stat:
        frames_stat["FOR"] = 1 - frames_stat["NPV"]

    if frames_stat["FPR"] > 0:
        frames_stat["LR+"] = frames_stat["TPR"] / frames_stat["FPR"]
    else:
        frames_stat["LR+"] = 1

    if frames_stat["TNR"] > 0:
        frames_stat["LR-"] = frames_stat["FNR"] / frames_stat["TNR"]
    else:
        frames_stat["LR-"] = 1

    # transients dict
    transients_stat["TP"] = tp_transients
    transients_stat["FN"] = fn_transients
    if traces is not None:
        transients_stat["TN"] = tn_transients
        transients_stat["FP"] = fp_transients

    if (tp_transients + fn_transients) > 0:
        transients_stat["sensitivity"] = tp_transients / (tp_transients + fn_transients)
    else:
        transients_stat["sensitivity"] = 1

    # print(f'transients_stat["sensitivity"] {transients_stat["sensitivity"]}')
    transients_stat["TPR"] = transients_stat["sensitivity"]

    if traces is not None:
        if (tn_transients + fp_transients) > 0:
            transients_stat["specificity"] = tn_transients / (tn_transients + fp_transients)
        else:
            transients_stat["specificity"] = 1
        transients_stat["TNR"] = transients_stat["specificity"]

        if (tp_transients + tn_transients + fp_transients + fn_transients) > 0:
            transients_stat["ACC"] = (tp_transients + tn_transients) / \
                                 (tp_transients + tn_transients + fp_transients + fn_transients)
        else:
            transients_stat["ACC"] = 1

        if (tp_frames + fp_frames) > 0:
            transients_stat["PPV"] = tp_transients / (tp_transients + fp_transients)
        else:
            transients_stat["PPV"] = 1
        if (tn_frames + fn_frames) > 0:
            transients_stat["NPV"] = tn_transients / (tn_transients + fn_transients)
        else:
            transients_stat["NPV"] = 1

    transients_stat["FNR"] = 1 - transients_stat["TPR"]

    return frames_stat, transients_stat


def get_raster_dur_from_traces(traces, with_threshold=None):
    """

    :param traces:
    :param with_threshold: None or otherwise 1xlen(traces) array with for each cell the threshold under which
    we should not take into account a peak and the transient associated. The value is without normalization.
    :return:
    """
    n_cells = traces.shape[0]
    n_times = traces.shape[1]

    for i in np.arange(n_cells):
        if with_threshold is not None:
            with_threshold[i] = (with_threshold[i] - np.mean(traces[i, :])) / np.std(traces[i, :])
        traces[i, :] = (traces[i, :] - np.mean(traces[i, :])) / np.std(traces[i, :])

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

    spike_nums_dur = build_spike_nums_dur(spike_nums_all, peak_nums,
                                          traces=traces, with_threshold=with_threshold)
    return spike_nums_dur


def build_spike_nums_dur(spike_nums, peak_nums, traces=None, with_threshold=None):
    n_cells = len(spike_nums)
    n_frames = spike_nums.shape[1]
    spike_nums_dur = np.zeros((n_cells, n_frames), dtype="int8")
    for cell in np.arange(n_cells):
        peaks_index = np.where(peak_nums[cell, :])[0]
        onsets_index = np.where(spike_nums[cell, :])[0]

        for onset_index in onsets_index:
            peaks_after = np.where(peaks_index > onset_index)[0]
            if len(peaks_after) == 0:
                continue
            peaks_after = peaks_index[peaks_after]
            peak_after = peaks_after[0]

            if (traces is not None) and (with_threshold is not None):
                # if the peak amplitude is under the threshold, we don't consider it
                if traces[cell, peak_after] < with_threshold[cell]:
                    continue

            spike_nums_dur[cell, onset_index:peak_after + 1] = 1
    return spike_nums_dur


def compute_stats_on_onsets(spike_nums, predicted_spike_nums):
    """
    Compute the stats based on raster dur
    :param spike_nums:
    :param predicted_spike_nums:
    :return: One dicts: with stats on onsets,
    Frames dict has the following keys (as String):
    TP: True Positive
    FP: False Positive
    FN: False Negative
    TN: True Negative
    sensitivity or TPR: True Positive Rate or Recall
    specificity or TNR: True Negative Rate or Selectivity
    FPR: False Positive Rate or Fall-out
    FNR: False Negative Rate or Miss Rate
    ACC: accuracy
    Prevalence: sum positive conditions / total population (for frames only)
    PPV: Positive Predictive Value or Precision
    FDR: False Discovery Rate
    FOR: False Omission Rate
    NPV: Negative Predictive Value
    LR+: Positive Likelihood ratio
    LR-: Negative likelihood ratio

    """

    if spike_nums.shape != predicted_spike_nums.shape:
        raise Exception("both spike_nums should have the same shape")
    if len(spike_nums.shape) == 1:
        # we transform them in a 2 dimensions array
        spike_nums = spike_nums.reshape(1, spike_nums.shape[0])
        predicted_spike_nums = predicted_spike_nums.reshape(1, predicted_spike_nums.shape[0])

    frames_stat = dict()

    # n_frames = spike_nums.shape[1]
    n_cells = spike_nums.shape[0]

    # positive means active frame, negative means non-active frames
    # condition is the ground truth
    # predicted is the one computed (RNN, CaiMan etc...)

    tp_frames = 0
    fp_frames = 0
    fn_frames = 0
    tn_frames = 0


    for cell in np.arange(n_cells):
        raster = spike_nums[cell]
        predicted_raster = predicted_spike_nums[cell]

        predicted_positive_frames = np.where(predicted_raster)[0]
        predicted_negative_frames = np.where(predicted_raster == 0)[0]

        tp_frames += len(np.where(raster[predicted_positive_frames] == 1)[0])
        fp_frames += len(np.where(raster[predicted_positive_frames] == 0)[0])
        fn_frames += len(np.where(raster[predicted_negative_frames] == 1)[0])
        tn_frames += len(np.where(raster[predicted_negative_frames] == 0)[0])

    frames_stat["TP"] = tp_frames
    frames_stat["FP"] = fp_frames
    frames_stat["FN"] = fn_frames
    frames_stat["TN"] = tn_frames

    # frames_stat["TPR"] = tp_frames / (tp_frames + fn_frames)
    frames_stat["sensitivity"] = tp_frames / (tp_frames + fn_frames)
    frames_stat["TPR"] = frames_stat["sensitivity"]

    frames_stat["specificity"] = tn_frames / (tn_frames + fp_frames)
    frames_stat["TNR"] = frames_stat["specificity"]

    frames_stat["ACC"] = (tp_frames + tn_frames) / (tp_frames + tn_frames + fp_frames + fn_frames)

    frames_stat["PPV"] = tp_frames / (tp_frames + fp_frames)

    frames_stat["NPV"] = tn_frames / (tn_frames + fn_frames)

    frames_stat["FNR"] = 1 - frames_stat["TPR"]

    frames_stat["FPR"] = 1 - frames_stat["TNR"]

    frames_stat["FDR"] = 1 - frames_stat["PPV"]

    frames_stat["FOR"] = 1 - frames_stat["NPV"]

    frames_stat["LR+"] = frames_stat["TPR"] / frames_stat["FPR"]

    frames_stat["LR-"] = frames_stat["FNR"] / frames_stat["TNR"]

    return frames_stat