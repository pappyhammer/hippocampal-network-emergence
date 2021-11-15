from pynwb import NWBHDF5IO
import numpy as np
import os
import scipy.io as sio


def find_files_with_ext(dir_to_explore, extension):
    subfiles = []
    for (dirpath, dirnames, filenames) in os.walk(dir_to_explore):
        subfiles = [os.path.join(dirpath, filename) for filename in filenames]
        break
    return [f for f in subfiles if f.endswith(extension)]


def get_frames_timestamps(nwb):
    """

    Returns:

    """

    io = NWBHDF5IO(nwb, 'r')
    nwb_data = io.read()

    if "ci_frames" not in nwb_data.acquisition:
        return None
    frames = nwb_data.acquisition["ci_frames"]
    # print(f"mouse_speed: {mouse_speed}")
    ci_ts = frames.timestamps
    # print(f"speed: {speed}")
    frames_ts = ci_ts[:]

    return frames_ts


def get_continous_time_periods(binary_array):
    """
    take a binary array and return a list of tuples representing the first and last position(included) of continuous
    positive period
    This code was copied from another project or from a forum, but i've lost the reference.
    :param binary_array:
    :return:
    """
    binary_array = np.copy(binary_array).astype("int8")
    # first we make sure it's binary
    if np.max(binary_array) > 1:
        binary_array[binary_array > 1] = 1
    if np.min(binary_array) < 0:
        binary_array[binary_array < 0] = 0
    n_times = len(binary_array)
    d_times = np.diff(binary_array)
    # show the +1 and -1 edges
    pos = np.where(d_times == 1)[0] + 1
    neg = np.where(d_times == -1)[0] + 1

    if (pos.size == 0) and (neg.size == 0):
        if len(np.nonzero(binary_array)[0]) > 0:
            return [(0, n_times-1)]
        else:
            return []
    elif pos.size == 0:
        # i.e., starts on an spike, then stops
        return [(0, neg[0])]
    elif neg.size == 0:
        # starts, then ends on a spike.
        return [(pos[0], n_times-1)]
    else:
        if pos[0] > neg[0]:
            # we start with a spike
            pos = np.insert(pos, 0, 0)
        if neg[-1] < pos[-1]:
            #  we end with aspike
            neg = np.append(neg, n_times - 1)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        # h = np.matrix([pos, neg])
        h = np.zeros((2, len(pos)), dtype="int16")
        h[0] = pos
        h[1] = neg
        if np.any(h):
            result = []
            for i in np.arange(h.shape[1]):
                if h[1, i] == n_times-1:
                    result.append((h[0, i], h[1, i]))
                else:
                    result.append((h[0, i], h[1, i]-1))
            return result
    return []


def main():

    # SET THE PATHS
    root_path = "D:/Robin/data_hne/"
    data_path = os.path.join(root_path, "nwb_files")
    data_id = "p9_211102_211111_1_211111_a001"
    data_path = os.path.join(data_path, data_id)
    print(f"data_path: {data_path}")

    nwb_file = find_files_with_ext(data_path, "nwb")[0]

    matlab_path = os.path.join(root_path, "xy_translation_data")
    matlab_file = data_id + "_xy_translation.mat"
    path_to_mat_file = os.path.join(matlab_path, matlab_file)

    motcorr_distance_name = data_id + "_translated_distance"
    xy_translation_to_save_npz = os.path.join(matlab_path, motcorr_distance_name)

    frames_ts = get_frames_timestamps(nwb_file)

    files = sio.loadmat(path_to_mat_file)
    xy_distance_data = files['xy_translation_data'][0]
    print(f"xy_distance_data: {xy_distance_data}")
    n_frames = len(xy_distance_data)

    print(f"N frames: {n_frames}")
    print(f"Frames timestamps: {frames_ts}")
    print(f"First frame timestamp: {frames_ts[0]}")
    print(f"Last frame timestamp: {frames_ts[-1]}")

    xy_distance = {'translated_distance': xy_distance_data, 'timestamps': frames_ts}
    if os.path.isfile(xy_translation_to_save_npz + 'npz') is False:
        np.savez(xy_translation_to_save_npz, **xy_distance)

    mean_correction = np.mean(xy_distance_data)
    std_correction = np.std(xy_distance_data)
    mot_corr_mvt_thr = mean_correction + 1.55 * std_correction
    print(f"Mot Corr threshold for mvt: {mot_corr_mvt_thr} µm")

    mvt_from_mot_corr = xy_distance_data
    mvt_from_mot_corr[np.where(xy_distance_data <= mot_corr_mvt_thr)[0]] = 0

    mvt_periods = get_continous_time_periods(mvt_from_mot_corr)
    print(f"Mvt-frames periods: {mvt_periods}")

    n_mvts = len(mvt_periods)

    auto_mvt = np.zeros((2, n_mvts), dtype=float)
    for mvt in range(n_mvts):
        start_period = mvt_periods[mvt][0]
        stop_period = mvt_periods[mvt][1]
        start_period_ts = frames_ts[start_period]
        stop_period_ts = frames_ts[stop_period]
        auto_mvt[0, mvt] = start_period_ts
        auto_mvt[1, mvt] = stop_period_ts

    print(f"MotCorr Auto Dectection: {auto_mvt}")

    movements_from_auto_motcorr = {'MotCorr Auto Dectection': auto_mvt}
    motcorr_auto_mvt_name = data_id + "_MotCorrMvt"
    mvt_from_motcorr_to_save_npz = os.path.join(matlab_path, motcorr_auto_mvt_name)
    if os.path.isfile(mvt_from_motcorr_to_save_npz + 'npz') is False:
        np.savez(mvt_from_motcorr_to_save_npz, **movements_from_auto_motcorr)


main()


