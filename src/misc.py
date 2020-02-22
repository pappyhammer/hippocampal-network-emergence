"""
Misc code
"""
import numpy as np
import os


def fusion_time_intervals_file_from_cicada(dir_files):
    """
    Fusion all .npz files in dir_files, produced form cicada badass gui
    Args:
        dir_files:

    Returns:

    """

    file_names = []
    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(dir_files):
        file_names.extend(local_filenames)
        break
    file_names = [f for f in file_names if (not f.startswith(".")) and f.endswith(".npz")]
    print(f"file_names {file_names}")
    time_intervals_dict = dict()
    for npz_file_name in file_names:
        npz_content = np.load(os.path.join(dir_files, npz_file_name))
        for tag_name, value in npz_content.items():
            # print(f"item {tag_name}, value {value}")
            if tag_name not in time_intervals_dict:
                time_intervals_dict[tag_name] = set()
            for i in range(value.shape[1]):
                time_intervals_dict[tag_name].add((value[0, i], value[1, i]))

    # then changing the format so we have numpy array of (2, n_intervals)
    for tag_name, values in time_intervals_dict.items():
        new_values = np.zeros((2, len(values)))
        values = list(values)
        values.sort()
        print(f"tag_name {values}")
        for index, (first_time, last_time) in enumerate(values):
            new_values[0, index] = first_time
            new_values[1, index] = last_time
        time_intervals_dict[tag_name] = new_values

    npz_file_name = os.path.join(dir_files, "fusion.npz")

    np.savez(npz_file_name, **time_intervals_dict)


if __name__ == "__main__":
    fusion_time_intervals_file_from_cicada(dir_files="/media/julien/Not_today/hne_not_today/data/behavior_movies/labels_RD")