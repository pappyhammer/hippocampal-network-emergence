from deepcinac.cinac_model import *
from deepcinac.cinac_predictor import *
import os
from datetime import datetime


if __name__ == '__main__':
    root_path = "/scratch/mpicardo/deepcinac/"

    cinac_dir_name = os.path.join(root_path, "cinac_files_training/")
    results_path = os.path.join(root_path, "results_training")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    results_path = os.path.join(results_path, time_str)
    os.mkdir(results_path)

    """
    To start training from a full saved model from another training, it is necessary to:
    - During the previous training, to put as argument: save_only_the_weitghs=False
    - then specify the .h5 containing the model using partly_trained_model
    - finally setting the learning rate so it is the same as the last epoch trained, using learning_rate_start
    """
    tiffs_dirname = os.path.join(root_path, "tiffs_for_classifier")

    # partly_trained_model = "/scratch/edenis/deepcinac/transient_classifier_full_model_02-0.9817.h5"
    cinac_model = CinacModel(results_path=results_path, n_epochs=3, verbose=2, batch_size=8,
                             n_gpus=4,
                             tiffs_dirname=tiffs_dirname,
                             # partly_trained_model=partly_trained_model,
                             #  learning_rate_start = 0.001,
                             save_only_the_weitghs=True
                             )

    cinac_model.add_input_data_from_dir(dir_name=cinac_dir_name, verbose=1)

    cinac_model.prepare_model(verbose=1)
    cinac_model.fit()
