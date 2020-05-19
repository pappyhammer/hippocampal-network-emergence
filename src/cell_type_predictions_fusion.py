import os
from datetime import datetime
from deepcinac.cinac_predictor import fusion_cell_type_predictions
import numpy as np

if __name__ == "__main__":
    root_path = '/media/julien/Not_today/hne_not_today/'
    path_data = os.path.join(root_path, "data/")
    # result_path = os.path.join(root_path, "results_hne/")
    # time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    # result_path = result_path + "/" + time_str
    # if not os.path.isdir(result_path):
    #     os.mkdir(result_path)

    cell_type_config_file = os.path.join(path_data, "cinac_cell_type_ground_truth",
                                         "cell_type_yaml_files",
                                         "pyr_vs_ins_vs_noise_multi_class.yaml")

    id_session = "191127_191202_191202_a000"
    path_data = os.path.join(path_data, "red_ins", "p5", "191127_191202", "191202_a000")
    filename_to_save = os.path.join(path_data, f"{id_session}_cell_type_predictions_v44_e7_fusion.npy")

    cell_type_pred_1 = os.path.join(path_data,
                                    "cell_type_predictions_tmp",
                                    "p5_191127_191202_191202_a000_cell_type_predictions_v44_e7_without_fusion.npz")
    cell_type_pred_1 = np.load(cell_type_pred_1, allow_pickle=True)
    cell_type_pred_1 = cell_type_pred_1["predictions"]

    cell_type_pred_2 = os.path.join(path_data,
                                    "cell_type_predictions_tmp",
                                    "p5_191127_191202_191202_a000_Add & Replace_cell_type_predictions.npy")
    cell_type_pred_2 = np.load(cell_type_pred_2, allow_pickle=True)

    fusion_cell_type_predictions(cell_type_pred_1=cell_type_pred_1, cell_type_pred_2=cell_type_pred_2,
                                 cell_type_config_file=cell_type_config_file,
                                 filename_to_save=filename_to_save)
