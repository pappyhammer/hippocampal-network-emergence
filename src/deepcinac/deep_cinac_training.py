import os
from deepcinac.cinac_model import *
from deepcinac.cinac_predictor import *
# import hdf5storage
from datetime import datetime
import numpy as np


def build_spike_nums_dur(spike_nums, peak_nums):
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
            spike_nums_dur[cell, onset_index:peak_after + 1] = 1
    return spike_nums_dur


if __name__ == '__main__':
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    root_path = "/media/julien/Not_today/hne_not_today/"
    data_path = os.path.join(root_path, "data/")
    results_path = os.path.join(root_path, "results_hne")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    results_path = os.path.join(results_path, time_str)
    os.mkdir(results_path)

    cell_type_classifier_mode = False

    if not cell_type_classifier_mode:
        tiffs_dirname = os.path.join(data_path, "tiffs_for_transient_classifier")
        """
            To start training from a full saved model from another training, it is necessary to:
            - During the previous training, to put as argument: save_only_the_weitghs=False
            - then specify the .h5 containing the model using partly_trained_model
            - finally setting the learning rate so it is the same as the last epoch trained, using learning_rate_start
        """

        partly_trained_model = "/media/julien/Not_today/hne_not_today/data/test_cinac_gui/transient_classifier_full_model_02-0.9817.h5"
        cinac_model = CinacModel(results_path=results_path,
                                 n_epochs=30, verbose=1, batch_size=8, n_gpus=1,
                                 # lstm_layers_size=[32], bin_lstm_size=32,
                                 window_len=100,
                                 max_n_transformations=6,
                                 lstm_layers_size=[128, 256], bin_lstm_size=256,
                                 # main_ratio_balance=(0.65, 0.25, 0.1),
                                 # lstm_layers_size=[128, 256], bin_lstm_size=256,
                                 conv_filters=(64, 64, 128, 128),
                                 # conv_filters=(32, 32, 64, 64),
                                 cell_type_classifier_mode=False,
                                 using_splitted_tiff_cinac_movie=False,
                                   # partly_trained_model=partly_trained_model,
                                   #  learning_rate_start = 0.001,
                                 tiffs_dirname=tiffs_dirname,
                                 save_only_the_weitghs=True
                                 )

        # artificial_movie_1 = os.path.join(data_path, "test_cinac_gui/artificial_movie_1_cell_0.cinac")

        cinac_dir_name = os.path.join(data_path, "cinac_ground_truth/for_training/ins_data")
        # cinac_dir_name = os.path.join(data_path, "cinac_ground_truth/for_training/v2_with_l_y_a")
        # cinac_dir_name = os.path.join(data_path, "cinac_ground_truth/for_benchmarks/to_hide/geco_data_benchmarks")

        cinac_model.add_input_data_from_dir(dir_name=cinac_dir_name, verbose=1)
        # cinac_model.add_input_data(cinac_file_names=artificial_movie_1)

        cinac_model.prepare_model(verbose=1)
        cinac_model.fit()
    else:
        tiffs_dirname = os.path.join(data_path, "tiffs_for_cell_type_classifier")
        """
                To start training from a full saved model from another training, it is necessary to:
                - During the previous training, to put as argument: save_only_the_weitghs=False
                - then specify the .h5 containing the model using partly_trained_model
                - finally setting the learning rate so it is the same as the last epoch trained, using learning_rate_start
        """
        cell_type_categories_yaml_file = "/media/julien/Not_today/hne_not_today/data/cinac_cell_type_ground_truth/cell_type_yaml_files/pyr_vs_ins_binary.yaml"
        cell_type_categories_yaml_file = "/media/julien/Not_today/hne_not_today/data/cinac_cell_type_ground_truth/cell_type_yaml_files/pyr_vs_ins_vs_noise_multi_class.yaml"

        partly_trained_model = "/media/julien/Not_today/hne_not_today/data/test_cinac_gui/"
        cinac_model = CinacModel(results_path=results_path, n_epochs=9, verbose=1, batch_size=4,
                                 cell_type_classifier_mode=True,
                                 max_width=20, max_height=20,
                                 window_len=500, max_n_transformations=0,
                                 with_all_pixels=True,
                                 n_windows_len_to_keep_by_cell=3,
                                 cell_type_categories_yaml_file=cell_type_categories_yaml_file,
                                 # conv_filters=(64, 64, 128, 128),
                                 conv_filters=(32, 32, 64, 64),
                                 lstm_layers_size=[32, 64], bin_lstm_size=64,
                                 # lstm_layers_size=[32], bin_lstm_size=32,
                                 overlap_value=0.5,
                                 dropout_value=0.5,
                                 frames_to_avoid_for_cell_type=[2500, 5000, 7500, 10000],
                                 tiffs_dirname=tiffs_dirname,
                                 # partly_trained_model=partly_trained_model,
                                 #  learning_rate_start = 0.001,
                                 save_only_the_weitghs=True,
                                 without_bidirectional=False,
                                 use_bin_at_al_version=True
                                 )

        # artificial_movie_1 = os.path.join(data_path, "test_cinac_gui/artificial_movie_1_cell_0.cinac")
        # cinac_dir_name = os.path.join(data_path, "cinac_cell_type_ground_truth/for_testing")
        cinac_dir_name = os.path.join(data_path, "cinac_cell_type_ground_truth/for_training")

        cinac_model.add_input_data_from_dir(dir_name=cinac_dir_name, verbose=1, display_cells_count=True)
        # cinac_model.add_input_data(cinac_file_names=artificial_movie_1)

        cinac_model.prepare_model(verbose=1)
        cinac_model.fit()

