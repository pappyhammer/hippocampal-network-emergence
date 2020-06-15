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
    n_levels_root = path_data.count("/")
    print(f"n_levels_root {n_levels_root}")
    for (dirpath, dirnames, local_filenames) in os.walk(path_data):
        level = dirpath.count("/") - n_levels_root
        if level != 3:
            continue
        dirs_at_each_level = dirpath.split("/")
        animal_id = dirs_at_each_level[-2] + "_" + dirs_at_each_level[-1]
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
            elif "cell_type_predictions" in file_name:
                files_in_dirs[CELL_TYPE_PRED].append(file_name)
            elif file_name.endswith("mvts_categories.npz"):
                files_in_dirs[BEHAVIOR_DIR].append(file_name)
            elif file_name.endswith("avi"):
                files_in_dirs[BEHAVIOR_DIR].append(file_name)
            elif file_name.endswith("contours_Add_suite2p.npy"):
                files_in_dirs[CONTOURS_FINAL_DIR].append(file_name)
            elif file_name.endswith(".zip") or file_name.endswith(".roi"):
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
            if not os.path.isdir(dir_to_create):
                os.mkdir(dir_to_create)

            for file_name_to_move in files_in_dirs[main_dir_name]:
                os.rename(os.path.join(dirpath, file_name_to_move),
                          os.path.join(dir_to_create, file_name_to_move))

    print(f"all_files_by_session {all_files_by_session}")

"""
    for (animal_dirpath, animal_dirnames, local_filenames) in os.walk(path_data):

        # each dirname represents an animal
        # make a code that organizes the data files (moving files etc...)
        for animal_dirname in animal_dirnames:
            print(f"animal_dirname {animal_dirname}")
            # recordings_dict[animal_dirname] = dict()
            for (session_dirpath, session_dirnames, session_local_filenames) in \
                    os.walk(os.path.join(animal_dirpath, animal_dirname)):
                for session_dirname in session_dirnames:
                    print(f"session_dirname {session_dirname}")
                    for (segment_dirpath, segment_dirnames, segment_local_filenames) in \
                            os.walk(os.path.join(session_dirpath, session_dirname)):
                        files_by_location = dict()
                        files_by_location["hippocampus"] = dict()
                        files_by_location["pre_frontal"] = dict()
                        for file_name in segment_local_filenames:
                            if not file_name.endswith(".ncs"):
                                continue
                            print(f"file_name {file_name}")
                            segment_number = None
                            if file_name in ["CSC3.ncs", "CSC4.ncs"]:
                                segment_number = 0
                                location = "hippocampus"
                                print("-> to hippo segment 0")
                            elif file_name.startswith("CSC3") or file_name.startswith("CSC4"):
                                # then we get the second part to know the segment number
                                index_ncs = file_name.index(".ncs")
                                segment_number = int(file_name[5:index_ncs])
                                location = "hippocampus"
                                print(f"-> to hippo segment {segment_number}")
                            elif file_name in ["hippoG.ncs", "hippoD.ncs"]:
                                segment_number = 0
                                location = "hippocampus"
                                print("-> to hippo segment 0")
                            elif file_name in ["pfG.ncs", "pfD.ncs"]:
                                segment_number = 0
                                location = "pre_frontal"
                                print("-> to pf segment 0")
                            elif file_name.startswith("hippo"):
                                # then we get the second part to know the segment number
                                index_ncs = file_name.index(".ncs")
                                segment_number = int(file_name[7:index_ncs])
                                location = "hippocampus"
                                print(f"-> to hippo segment {segment_number}")
                            elif file_name.startswith("pf"):
                                # then we get the second part to know the segment number
                                index_ncs = file_name.index(".ncs")
                                segment_number = int(file_name[4:index_ncs])
                                location = "pre_frontal"
                                print(f"-> to pf segment {segment_number}")
                            if segment_number is None:
                                continue
                            if segment_number not in files_by_location[location]:
                                files_by_location[location][segment_number] = []
                            files_by_location[location][segment_number].append(file_name)
                        for location, location_dict in files_by_location.items():
                            if len(location_dict) == 0:
                                continue
                            if not os.path.isdir(os.path.join(segment_dirpath, location)):
                                print(f"mkdir {os.path.join(segment_dirpath, location)}")
                                os.mkdir(os.path.join(segment_dirpath, location))
                            for segment_number, file_names in location_dict.items():
                                segment_number = str(segment_number)
                                if not os.path.isdir(os.path.join(segment_dirpath, location, segment_number)):
                                    print(f"mkdir {os.path.join(segment_dirpath, location, segment_number)}")
                                    os.mkdir(os.path.join(segment_dirpath, location, segment_number))
                                for file_name in file_names:
                                    print(f"from {os.path.join(segment_dirpath, file_name)} to "
                                          f"{os.path.join(segment_dirpath, location, segment_number, file_name)}")
                                    os.rename(os.path.join(segment_dirpath, file_name),
                                              os.path.join(segment_dirpath, location, segment_number, file_name))
                break
        break
"""

if __name__ == "__main__":
    root_path = '/media/julien/Not_today/hne_not_today/'
    path_data = os.path.join(root_path, "data/SWISS_data_test")

    organize_data_files(path_data=path_data)
