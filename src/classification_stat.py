import numpy as np
from pattern_discovery.tools.misc import get_continous_time_periods


def compute_stats(spike_nums_dur, predicted_spike_nums_dur):
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

    frames_stat = dict()
    transients_stat = dict()

    n_frames = spike_nums_dur.shape[1]
    n_cells = spike_nums_dur.shape[0]

    # positive means active frame, negative means non-active frames
    # condition is the ground truth
    # predicted is the one computed (RNN, CaiMan etc...)

    tp_frames = 0
    fp_frames = 0
    fn_frames = 0
    tn_frames = 0

    tp_transients = 0
    fn_transients = 0

    for cell in np.arange(n_cells):
        raster_dur = spike_nums_dur[cell]
        predicted_raster_dur = predicted_spike_nums_dur[cell]

        predicted_positive_frames = np.where(predicted_raster_dur)[0]
        predicted_negative_frames = np.where(predicted_raster_dur == 0)[0]

        tp_frames += len(np.where(raster_dur[predicted_positive_frames] == 1)[0])
        fp_frames += len(np.where(raster_dur[predicted_positive_frames] == 0)[0])
        fn_frames += len(np.where(raster_dur[predicted_negative_frames] == 1)[0])
        tn_frames += len(np.where(raster_dur[predicted_negative_frames] == 0)[0])

        # transients section
        transient_periods = get_continous_time_periods(raster_dur)
        # positive condition
        n_transients = len(transient_periods)
        tp = 0
        for transient_period in transient_periods:
            frames = np.arange(transient_period[0], transient_period[1] + 1)
            # print(f"np.sum(binary_predicted_as_active[frames] {np.sum(binary_predicted_as_active[frames])}")
            if np.sum(predicted_raster_dur[frames]) > 0:
                tp += 1
        tp_transients += tp
        fn_transients += (n_transients - tp)

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

    # transients dict
    transients_stat["TP"] = tp_transients
    transients_stat["FN"] = fn_transients

    transients_stat["sensitivity"] = tp_transients / (tp_transients + fn_transients)
    transients_stat["TPR"] = transients_stat["sensitivity"]

    transients_stat["FNR"] = 1 - transients_stat["TPR"]

    return frames_stat, transients_stat
