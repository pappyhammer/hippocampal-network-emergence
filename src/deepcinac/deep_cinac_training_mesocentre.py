from deepcinac.cinac_model import *
from deepcinac.cinac_predictor import *
import os
from datetime import datetime

if __name__ == '__main__':
    root_path = "/scratch/edenis/deepcinac/"

    cinac_dir_name = os.path.join(root_path, "cinac_files_training")
    extra_info = ""
    # cinac_dir_name = os.path.join(root_path, "cinac_files_training/chen_transients_data")
    # extra_info = "_chen_transients_with_p12"
    # cinac_dir_name = os.path.join(root_path, "cinac_files_training/chen_spikes_data")
    # extra_info = "_chen_spikes"
    # cinac_dir_name = os.path.join(root_path, "cinac_files_training/ins_data")
    # extra_info = "_ins"
    # cinac_dir_name = os.path.join(root_path, "cinac_files_training/v2_with_l_y_a")
    # extra_info = "_v2_with_l_y_"
    # cinac_dir_name = os.path.join(root_path, "cinac_files_training/v2_data")
    # extra_info = "_v2_data"
    # cinac_dir_name = os.path.join(root_path, "cinac_files_training/v1_data")
    # extra_info = "_v1_data"
    # cinac_dir_name = os.path.join(root_path, "cinac_files_training/p12_4_cells")
    # extra_info = "_p12_4_cells"
    results_path = os.path.join(root_path, "results_training")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    results_path = os.path.join(results_path, time_str + extra_info)
    os.mkdir(results_path)

    """
    To start training from a full saved model from another training, it is necessary to:
    - During the previous training, to put as argument: save_only_the_weitghs=False
    - then specify the .h5 containing the model using partly_trained_model
    - finally setting the learning rate so it is the same as the last epoch trained, using learning_rate_start
    """
    tiffs_dirname = os.path.join(root_path, "tiffs_for_classifier")
    # tiffs_dirname = None

    # partly_trained_model = "/scratch/edenis/deepcinac/transient_classifier_full_model_02-0.9817.h5"
    cinac_model = CinacModel(results_path=results_path, n_epochs=30, verbose=2,
                             batch_size=16, n_gpus=4,
                             split_values=(0.8, 0.2, 0),
                             # split_values=(0.65, 0.35, 0),
                             # learning_rate_start=0.01,
                             window_len=100,
                             learning_rate_reduction_patience=2,
                             cell_type_classifier_mode=False,
                             tiffs_dirname=tiffs_dirname,
                             max_n_transformations=6,
                             early_stop_patience=10,
                             using_splitted_tiff_cinac_movie=True,
                             # lstm_layers_size=[64, 128], bin_lstm_size=128,
                             # lstm_layers_size=[32, 64], bin_lstm_size=64,
                             lstm_layers_size=[128, 256], bin_lstm_size=256, # usual config
                             # main_ratio_balance=(0.65, 0.25, 0.1),
                             # partly_trained_model=partly_trained_model,
                             #  learning_rate_start = 0.001,
                             save_only_the_weitghs=True
                             )
    # for 4 GPUS, ntasks-per-node=2  is not enough
    cinac_model.add_input_data_from_dir(dir_name=cinac_dir_name, verbose=1)

    cinac_model.prepare_model(verbose=1)
    cinac_model.fit()
