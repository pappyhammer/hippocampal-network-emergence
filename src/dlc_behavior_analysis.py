import os
import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
import scipy.signal
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def encode_frames_from_cicada(cicada_file, n_frames, side_to_exclude=None, no_behavior_str = "still"):
    """
    Return a list of len n_frames with str representing each frame behavior
    Args:
        cicada_file:
        n_frames:

    Returns:

    """
    npz_content = np.load(cicada_file)

    tags = []
    tags_index = dict()
    n_tags = 0
    for tag_name, value in npz_content.items():
        if side_to_exclude in tag_name:
            continue
        if value.shape[1] > 0:
            tags.append(tag_name)
            tags_index[tag_name] = 0
            n_tags += 1

    interval_id = 0
    intervals_array = np.zeros((len(tags), n_frames), dtype="int16")
    interval_info_dict = dict()

    for tag_name, value in npz_content.items():
        if side_to_exclude in tag_name:
            continue

        for i in range(value.shape[1]):
            intervals_array[tags_index[tag_name], value[0, i]:value[1, i]+1] = interval_id
            interval_info_dict[interval_id] = value[1, i] - value[0, i] + 1
            interval_id += 1

    tags_by_frame = []
    for frame_index in range(n_frames):
        tags_indices = np.where(intervals_array[:, frame_index] > 0)[0]
        if len(tags_indices) == 0:
            tags_by_frame.append(no_behavior_str)
        if len(tags_indices) == 1:
            tags_by_frame.append(tags[tags_indices[0]])
        else:
            # then we select the shorter one
            for i, tag_index in enumerate(tags_indices):
                if i == 0:
                    shorter_tag = tags[tag_index]
                    interval_id = intervals_array[tag_index, frame_index]
                    shorter_duration = interval_info_dict[interval_id]
                else:
                    interval_id = intervals_array[tag_index, frame_index]
                    duration = interval_info_dict[interval_id]
                    if duration < shorter_duration:
                        shorter_duration = duration
                        shorter_tag = tags[tag_index]
            tags_by_frame.append(shorter_tag)
    return tags_by_frame


def apply_tsne(data, behavior_by_frame=None):
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    palette = sns.color_palette("hls", 10)

    tsne = TSNE(verbose=1, n_components=2) #, perplexity=40, n_iter=300)
    data_embedded = tsne.fit_transform(data)
    print(f"data_embedded.shape {data_embedded.shape}")

    sns.scatterplot(data_embedded[:, 0], data_embedded[:, 1], hue=behavior_by_frame,
                    legend='full', palette=palette)

    plt.show()


def dlc_behavior_analysis():
    root_path = "/media/julien/Not_today/hne_not_today/data/behavior_movies/dlc_predictions/p5_19_12_10_0"

    data_path = os.path.join(root_path, "data")

    results_path = os.path.join(root_path, "results")

    using_h5 = True

    if using_h5:
        pos_file_name = "behavior_p5_19_12_10_0_cam_23109588_cam2_a001_fps_20DLC_resnet50_test_valentinFeb17shuffle1_155000.h5"
        skeketon_file_name = "behavior_p5_19_12_10_0_cam_23109588_cam2_a001_fps_20DLC_resnet50_test_valentinFeb17shuffle1_155000_skeleton.h5"

        pos_h5_file = h5py.File(os.path.join(data_path, pos_file_name), 'r')
        skeketon_h5_file = h5py.File(os.path.join(data_path, skeketon_file_name), 'r')

        config_yaml = os.path.join(data_path, "config.yaml")
        with open(config_yaml, 'r') as stream:
            config_data = yaml.load(stream, Loader=yaml.FullLoader)
        bodyparts = config_data['bodyparts']
        bodyparts_indices = dict()
        n_bodyparts = len(bodyparts)
        print(f"bodyparts {bodyparts}")

        # list of list of 2 part linked
        skeleton_pairs_list = config_data['skeleton']
        # same named as used in the csv
        skeleton_parts = [s1 + "_" + s2 for s1, s2 in skeleton_pairs_list]
        print(f"skeleton_parts {skeleton_parts}")
        n_skeleton_parts = len(skeleton_parts)

        pos_keys = list(pos_h5_file.keys())
        # skeketon_keys = list(skeketon_h5_file.keys())
        # print(f"pos_keys {pos_keys}")
        # print(f"skeketon_keys {skeketon_keys}")

        # print(f"pos df_with_missing keys {list(pos_h5_file['df_with_missing'].keys())}")

        pos_h5_data = pos_h5_file['df_with_missing']['table']
        skeleton_h5_data = skeketon_h5_file['df_with_missing']['table']
        # print(f"pos data {np.array(pos_h5_file['df_with_missing']['table'])}")
        # print(f"pos data {np.array(skeleton_h5_data)}")
        # print(f"pos_data[0, :] {pos_data[0][1][0]}")
        n_frames = len(pos_h5_data)

        # x, y, likelihood
        pos_data_array = np.zeros((n_frames, len(bodyparts) * 3))
        for frame in np.arange(n_frames):
            pos_data_array[frame] = np.array(pos_h5_data[frame][1])

        # length, orientation, likelihood
        skeleton_h5_data_array = np.zeros((n_frames, n_skeleton_parts * 3))
        for frame in np.arange(n_frames):
            skeleton_h5_data_array[frame] = np.array(skeleton_h5_data[frame][1])

        bodyparts_pos_dict = dict()
        body_parts_pos_array = np.zeros((n_bodyparts, 3, n_frames))
        skeleton_parts_pos_array = np.zeros((n_skeleton_parts, 3, n_frames))
        for body_index, bodypart in enumerate(bodyparts):
            bodyparts_indices[bodypart] = body_index
            # y_pos = np.zeros(n_frames)
            # # likelihood
            # lh_pos = np.zeros(n_frames)
            for frame in np.arange(n_frames):
                body_parts_pos_array[body_index, 0, frame] = pos_data_array[frame, body_index * 3]
                body_parts_pos_array[body_index, 1, frame] = pos_data_array[frame, (body_index * 3) + 1]
                body_parts_pos_array[body_index, 2, frame] = pos_data_array[frame, (body_index * 3) + 2]

            bodyparts_pos_dict[bodypart] = body_parts_pos_array[body_index]

        for skeleton_index, skeleton_part in enumerate(skeleton_parts):
            # skeleton_parts_indices[skeleton_part] = skeleton_index

            for frame in np.arange(n_frames):
                skeleton_parts_pos_array[skeleton_index, 0, frame] = skeleton_h5_data_array[frame, skeleton_index * 3]
                skeleton_parts_pos_array[skeleton_index, 1, frame] = skeleton_h5_data_array[
                    frame, (skeleton_index * 3) + 1]
                skeleton_parts_pos_array[skeleton_index, 2, frame] = skeleton_h5_data_array[
                    frame, (skeleton_index * 3) + 2]

        # pcutoff
        # if 1 no thresholding
        likelihood_threshold = 0.1
        replace_by_nan = False
        if likelihood_threshold < 1:
            to_filter = ["bodyparts", "sketleton_parts"]
            for parts_to_filter in to_filter:
                # filling with nan for now
                print(f"### n labels wrong (p_cutoff < {likelihood_threshold})")
                print(f"# For {parts_to_filter}")
                if to_filter == "bodyparts":
                    n_parts = n_bodyparts
                    parts_pos_array = body_parts_pos_array
                    labels = bodyparts
                else:
                    n_parts = n_skeleton_parts
                    parts_pos_array = skeleton_parts_pos_array
                    labels = skeleton_parts
                for part_index in np.arange(n_parts):
                    indices = np.where(parts_pos_array[part_index, 2, :] < likelihood_threshold)[0]
                    print(f"{labels[part_index]}: {len(indices)}")
                    # replacing x, y by NaN
                    if replace_by_nan:
                        parts_pos_array[part_index, 0, indices] = np.nan
                        parts_pos_array[part_index, 1, indices] = np.nan
                    else:
                        # we want to put the previous value or the mean between previous and next if known
                        for frame_index in indices:
                            if frame_index == 0:
                                # looking for next frame_index know
                                if frame_index + 1 in indices:
                                    next_index = frame_index + 1
                                    while next_index in indices:
                                        next_index += 1
                                    parts_pos_array[part_index, 0, frame_index] = parts_pos_array[
                                        part_index, 0, next_index]
                                    parts_pos_array[part_index, 1, frame_index] = parts_pos_array[
                                        part_index, 1, next_index]
                                else:
                                    parts_pos_array[part_index, 0, frame_index] = parts_pos_array[
                                        part_index, 0, frame_index + 1]
                                    parts_pos_array[part_index, 1, frame_index] = parts_pos_array[
                                        part_index, 1, frame_index + 1]
                            else:
                                x_values = parts_pos_array[part_index, 0]
                                y_values = parts_pos_array[part_index, 1]
                                if (frame_index + 1) in indices or frame_index == (n_frames - 1):
                                    # we take previous value
                                    parts_pos_array[part_index, 0, frame_index] = x_values[frame_index - 1]
                                    parts_pos_array[part_index, 1, frame_index] = y_values[frame_index - 1]
                                else:
                                    # we take the mean
                                    parts_pos_array[part_index, 0, frame_index] = (x_values[frame_index - 1] +
                                                                                   x_values[frame_index + 1]) / 2
                                    parts_pos_array[part_index, 1, frame_index] = (y_values[frame_index - 1] +
                                                                                   y_values[frame_index + 1]) / 2
            print("")

        # print(f"pos data _i_table keys() {list(pos_h5_file['df_with_missing']['_i_table']['index'].keys())}")
        #
        # for key in pos_h5_file['df_with_missing']['_i_table']['index'].keys():
        #     print(f"key {key}: {np.array(pos_h5_file['df_with_missing']['_i_table']['index'][key])}")

        # for part_index in range(n_parts):
        #     # using orientation
        #     n_components = 3
        #     pca = PCA(n_components=n_components)
        #     pca.fit(skeleton_parts_pos_array[:, 1, :])
        try_wavelet = False
        if try_wavelet:
            for part_index in range(n_skeleton_parts):
                orientation_values = skeleton_parts_pos_array[part_index, 1, :]
                # widths = np.arange(0.3, 5)
                fs = 60
                freq = np.linspace(1, fs / 2, 100)
                w = 20.
                widths = w * fs / (2 * freq * np.pi)
                cwtm = scipy.signal.cwt(data=orientation_values, wavelet=scipy.signal.morlet2, widths=widths)
                t = np.arange(n_frames)
                plot_it = False
                if plot_it:
                    plt.pcolormesh(t, freq, np.abs(cwtm), cmap='viridis')
                    plt.title(skeleton_parts[part_index])
                    plt.show()
                n_components = 15
                pca = PCA(n_components=n_components)
                pca.fit(np.abs(cwtm))
                print(f"pca.explained_variance_ratio_ {pca.explained_variance_ratio_}")
                print(f"components_ {pca.components_.shape}")
        # apply_tsne(data=skeleton_parts_pos_array[:, 0:2, :])
        # print(f"body_parts_pos_array[:, 1, :] {body_parts_pos_array[:, 1, :].shape}")
        data_to_cluster = np.zeros((n_frames, (n_bodyparts * 2 + n_skeleton_parts * 2)))
        data_to_cluster[:, :n_bodyparts] = body_parts_pos_array[:, 0, :].transpose()
        data_to_cluster[:, n_bodyparts:n_bodyparts * 2] = body_parts_pos_array[:, 1, :].transpose()
        data_to_cluster[:, n_bodyparts * 2:(n_bodyparts * 2 + n_skeleton_parts)] = \
            skeleton_parts_pos_array[:, 0, :].transpose()
        data_to_cluster[:, (n_bodyparts * 2 + n_skeleton_parts):(n_bodyparts * 2 + n_skeleton_parts * 2)] = \
            skeleton_parts_pos_array[:, 1, :].transpose()
        print(f"data_to_cluster.shape {data_to_cluster.shape}")
        behavior_by_frame = None
        use_cicada_file = True
        if use_cicada_file:
            cicada_file = os.path.join(data_path, "p5.npz")
            behavior_by_frame = encode_frames_from_cicada(cicada_file=cicada_file, n_frames=n_frames,
                                                        side_to_exclude="right", no_behavior_str="still")
        apply_tsne(data=data_to_cluster, behavior_by_frame=behavior_by_frame)
        pos_h5_file.close()
        skeketon_h5_file.close()
        raise Exception("H5 OVER")

    pos_file_name = "behavior_p5_19_12_10_0_cam_23109588_cam2_a001_fps_20DLC_resnet50_test_valentinFeb17shuffle1_155000.csv"
    skeketon_file_name = "behavior_p5_19_12_10_0_cam_23109588_cam2_a001_fps_20DLC_resnet50_test_valentinFeb17shuffle1_155000_skeleton.csv"

    pos_df = pd.read_csv(os.path.join(data_path, pos_file_name), low_memory=False)
    skeleton_df = pd.read_csv(os.path.join(data_path, skeketon_file_name), low_memory=False)

    # Preview the first 5 lines of the loaded data
    print(f"pos_df.head() {pos_df.head()}")
    print(f"skeleton_df.head() {skeleton_df.head()}")

if __name__ == "__main__":
    dlc_behavior_analysis()


