import os
from deepcinac.cinac_model import *
from deepcinac.cinac_predictor import *
# import hdf5storage
from datetime import datetime


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

    cell_type_classifier_mode = True

    if not cell_type_classifier_mode:
        tiffs_dirname = os.path.join(data_path, "tiffs_for_cell_activity_classifier")
        """
        To start training from a full saved model from another training, it is necessary to:
        - During the previous training, to put as argument: save_only_the_weitghs=False
        - then specify the .h5 containing the model using partly_trained_model
        - finally setting the learning rate so it is the same as the last epoch trained, using learning_rate_start
        """

        partly_trained_model = "/media/julien/Not_today/hne_not_today/data/test_cinac_gui/transient_classifier_full_model_02-0.9817.h5"
        cinac_model = CinacModel(results_path=results_path, n_epochs=3, verbose=1, batch_size=4,
                                 cell_type_classifier_mode=cell_type_classifier_mode,
                                   # partly_trained_model=partly_trained_model,
                                   #  learning_rate_start = 0.001,
                                 save_only_the_weitghs=False
                                 )

        # artificial_movie_1 = os.path.join(data_path, "test_cinac_gui/artificial_movie_1_cell_0.cinac")

        cinac_dir_name = os.path.join(data_path, "cinac_ground_truth/for_training")

        cinac_model.add_input_data_from_dir(dir_name=cinac_dir_name, verbose=1)
        # cinac_model.add_input_data(cinac_file_names=artificial_movie_1)

        cinac_model.prepare_model(verbose=1)
        # cinac_model.fit()
    else:
        tiffs_dirname = os.path.join(data_path, "tiffs_for_cell_type_classifier")
        """
                To start training from a full saved model from another training, it is necessary to:
                - During the previous training, to put as argument: save_only_the_weitghs=False
                - then specify the .h5 containing the model using partly_trained_model
                - finally setting the learning rate so it is the same as the last epoch trained, using learning_rate_start
                """

        partly_trained_model = "/media/julien/Not_today/hne_not_today/data/test_cinac_gui/"
        cinac_model = CinacModel(results_path=results_path, n_epochs=10, verbose=1, batch_size=8,
                                 cell_type_classifier_mode=True,
                                 window_len=200, max_n_transformations=1,
                                 lstm_layers_size=[128], bin_lstm_size=128,
                                 overlap_value=0,
                                 tiffs_dirname=tiffs_dirname,
                                 # partly_trained_model=partly_trained_model,
                                 #  learning_rate_start = 0.001,
                                 save_only_the_weitghs=False
                                 )

        # artificial_movie_1 = os.path.join(data_path, "test_cinac_gui/artificial_movie_1_cell_0.cinac")

        cinac_dir_name = os.path.join(data_path, "cinac_cell_type_ground_truth/for_training")

        cinac_model.add_input_data_from_dir(dir_name=cinac_dir_name, verbose=1)
        # cinac_model.add_input_data(cinac_file_names=artificial_movie_1)

        cinac_model.prepare_model(verbose=1)
        # cinac_model.fit()

