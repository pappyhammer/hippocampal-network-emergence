import numpy as np
import os

from datetime import datetime

ABF_DIR = "abf"
ACTIVITY_PRED_DIR = "activity_predictions"
BEHAVIOR_DIR = "behavior"
CELL_TYPE_PRED = "cell_type_predictions"
CI_DATA_DIR = "ci_data"
CONTOURS_FINAL_DIR = "contours_final"
FIJI_MAPS_DIR = "fiji_maps"
SUITE2P_DIR = "suite2p"
TMP_CONTOURS_DIR = "tmp_contours_and_type_predictions_files"


def organize_data_files(path_data):
    # key is the full path containing all the files to sort, value is a list of files
    all_files_by_session = dict()
    main_dir_names = [ABF_DIR, ACTIVITY_PRED_DIR, BEHAVIOR_DIR, CELL_TYPE_PRED, CI_DATA_DIR, CONTOURS_FINAL_DIR,
                      FIJI_MAPS_DIR, SUITE2P_DIR, TMP_CONTOURS_DIR]
    n_levels_root = path_data.count("\\")

    for (dirpath, dirnames, local_filenames) in os.walk(path_data):
        level = dirpath.count("\\") - n_levels_root
        print(f"level: {level}")
        print(f"dirpath : {dirpath}")
        print(f"local filenames: {local_filenames}")

        if level != 3:
            continue

        dirs_at_each_level = dirpath.split("\\")
        animal_id = dirs_at_each_level[-2] + "_" + dirs_at_each_level[-1]
        # continue
        all_files_by_session[dirpath] = local_filenames
        # now we want to create directories and move files accordingly
        # keys is the name of the dir (without the session_id) and value is a list of files to put in
        files_in_dirs = dict()
        for main_dir_name in main_dir_names:
            files_in_dirs[main_dir_name] = []

        for file_name in local_filenames:
            if file_name.startswith("."):
                continue
            if file_name.endswith(".tif") or file_name.endswith(".tiff"):
                files_in_dirs[CI_DATA_DIR].append(file_name)
            elif file_name.endswith(".abf"):
                files_in_dirs[ABF_DIR].append(file_name)
            elif ("cell_type_predictions" in file_name) and (not file_name.endswith("cell_type_predictions.npy")):
                files_in_dirs[CELL_TYPE_PRED].append(file_name)
            # files produces when GT is used
            elif ("cell_type_predictions" in file_name) and file_name.endswith("cell_type_predictions.npy"):
                files_in_dirs[TMP_CONTOURS_DIR].append(file_name)
            elif file_name.endswith("mvts_categories.npz"):
                files_in_dirs[BEHAVIOR_DIR].append(file_name)
            elif file_name.endswith(".npz") and "behavior" in file_name:
                files_in_dirs[BEHAVIOR_DIR].append(file_name)
            elif file_name.endswith("avi"):
                files_in_dirs[BEHAVIOR_DIR].append(file_name)
            elif file_name.endswith("contours_Add_suite2p.npy"):
                files_in_dirs[CONTOURS_FINAL_DIR].append(file_name)
            elif file_name.endswith(".zip") or file_name.endswith(".roi"):
                files_in_dirs[FIJI_MAPS_DIR].append(file_name)
            elif file_name.endswith("INs_coor.mat"):
                files_in_dirs[FIJI_MAPS_DIR].append(file_name)
            elif file_name.endswith(".png") and "cells_map_matching" in file_name:
                files_in_dirs[FIJI_MAPS_DIR].append(file_name)
            elif file_name.endswith("Replace_suite2p.npy"):
                files_in_dirs[TMP_CONTOURS_DIR].append(file_name)

        for dir_name in dirnames:
            if dir_name == "suite2p":
                # then we change the name
                os.rename(os.path.join(dirpath, dir_name),
                          os.path.join(dirpath, dir_name + "_" + animal_id))
                break

        # now we create dirs to put files in
        for main_dir_name in main_dir_names:
            dir_to_create = os.path.join(dirpath, main_dir_name + "_" + animal_id)
            print(f"dir_to_create {dir_to_create}")
            if not os.path.isdir(dir_to_create):
                os.mkdir(dir_to_create)

            for file_name_to_move in files_in_dirs[main_dir_name]:
                os.rename(os.path.join(dirpath, file_name_to_move),
                          os.path.join(dir_to_create, file_name_to_move))


if __name__ == "__main__":
    root_path = "D:/Robin"

    path_data = os.path.join(root_path, "data_hne", "NWB_to_create", "folder_to_order")

    print(f"path data: {path_data}")

    organize_data_files(path_data=path_data)
