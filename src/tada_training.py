from deeptada.tada_model import *
# from deeptada.tada_detector import *
# import hdf5storage
from datetime import datetime

if __name__ == '__main__':
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    root_path = "/media/julien/Not_today/hne_not_today/data/tada_data"
    data_path = os.path.join(root_path, "for_training")
    results_path = os.path.join(root_path, "results_tada")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    results_path = os.path.join(results_path, time_str)
    os.mkdir(results_path)

    action_tags_yaml_file = os.path.join(root_path, "config_tada.yaml")

    images_dirname = os.path.join(root_path, "images_tada")

    """
        To start training from a full saved model from another training, it is necessary to:
        - During the previous training, to put as argument: save_only_the_weitghs=False
        - then specify the .h5 containing the model using partly_trained_model
        - finally setting the learning rate so it is the same as the last epoch trained, using learning_rate_start
    """
    # "config_tada.yaml"
    # partly_trained_model = "/media/julien/Not_today/hne_not_today/data/test_cinac_gui/transient_classifier_full_model_02-0.9817.h5"
    tada_model = TadaModel(results_path=results_path, n_epochs=4, verbose=1, batch_size=4,
                           width_crop=1600, height_crop=1000,
                           window_len=10, max_n_transformations=0,
                           lstm_layers_size=[8], bin_lstm_size=8,
                           action_tags_yaml_file=action_tags_yaml_file,
                           overlap_value=0.5,
                           n_cameras=1,
                           images_dirname=images_dirname,
                           save_only_the_weitghs=False,
                           without_bidirectional=True,
                           use_bin_at_al_version=True)

    # lstm_layers_size=[128, 256], bin_lstm_size=256,

    tada_dir_name = data_path

    tada_model.add_multiple_input_data_from_dir(dir_name=tada_dir_name)

    tada_model.prepare_model()
    tada_model.fit()
