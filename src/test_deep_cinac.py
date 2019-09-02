from deepcinac.cinac_predictor import *

if __name__ == '__main__':
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"

    data_path = os.path.join(root_path, "data/p6/p6_18_02_07_a001")
    results_path = os.path.join(root_path, "results_hne")
    identifier = "p6_18_02_07_a001"
    movie_file_name = os.path.join(data_path, "p6_18_02_07_a001.tif")
    is_cell_suite2p_file_name = os.path.join(data_path, "suite2p", "iscell.npy")
    stat_suite2p_file_name = os.path.join(data_path, "suite2p", "stat.npy")

    weights_file_name = os.path.join(root_path, "data/transient_classifier_model/",
                                  "transient_classifier_weights_11-0.9703_2019_04_13.23-21-27.h5")

    json_file_name = os.path.join(root_path, "data/transient_classifier_model/",
                                  "transient_classifier_model_architecture__2019_04_13.23-21-27.json")
    tiffs_dirname = os.path.join(root_path, "data/cinac_tiffs_test/")
    cinac_predictor = CinacPredictor()

    cinac_recording = CinacRecording(identifier=identifier)
    # cinac_movie = CinacSplitedTiffMovie(identifier=identifier, tiffs_dirname=tiffs_dirname,
    #                                     tiff_file_name=movie_file_name, tiff_movie=None)
    cinac_movie = CinacTiffMovie(tiff_file_name=movie_file_name)
    cinac_recording.set_movie(cinac_movie)
    cinac_recording.set_rois_from_suite_2p(is_cell_file_name=is_cell_suite2p_file_name,
                                           stat_file_name=stat_suite2p_file_name)
    model_files_dict = dict()
    # predicting 2 first cells with this model,  weights and string identifying the network
    model_files_dict[(json_file_name, weights_file_name, "epoch_11_test")] = np.arange(1)
    cinac_predictor.add_recording(cinac_recording=cinac_recording,
                                  removed_cells_mapping=None,
                                  model_files_dict=model_files_dict)

    cinac_predictor.predict(results_path=results_path, output_file_formats="npy")