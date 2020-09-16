import os
# from datetime import datetime
from deepcinac.cinac_predictor import fusion_cell_type_predictions, fusion_cell_type_predictions_by_type
import numpy as np

# from deepcinac.utils.cinac_file_utils import read_cell_type_categories_yaml_file
# from collections import Counter

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

    do_it_by_type = True
    fusion_with_gt = True
    age = 5
    animal_id = "191205_191210_0"
    session_id = "191210_a000"

    global_id = animal_id + "_" + session_id

    path_data = os.path.join(path_data, "red_ins", f"p{age}", animal_id, session_id)
    cell_type_dir = f"cell_type_predictions_{global_id}"

    tmp_contours_dir = None

    dir_names = []

    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(path_data):
        dir_names.extend(dirnames)
        break
    tmp_contours_dir = [d for d in dir_names if "tmp_contours" in d]
    if len(tmp_contours_dir) != 1:
        raise Exception(f"No tmp_contours dir found in {path_data}")
    tmp_contours_dir = tmp_contours_dir[0]
    print(f'tmp_contours_dir {tmp_contours_dir}')

    if do_it_by_type:
        path_data_pred = os.path.join(path_data, tmp_contours_dir)
        filename_to_save = os.path.join(path_data_pred, f"{global_id}_cell_type_predictions_fusion_without_GT.npy")

        # ins, pyr, noise
        cell_type_preds_dict = {0: "v58_e7",
                                1: "v61_e8",
                                2: "v60_e4"}

        default_cell_type_pred = "v58_e7"

        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(path_data_pred):
            file_names.extend(local_filenames)
            break

        for file_name in file_names:
            if isinstance(default_cell_type_pred, str):
                if default_cell_type_pred in file_name:
                    default_cell_type_pred = os.path.join(path_data_pred, file_name)
                    default_cell_type_pred = np.load(default_cell_type_pred, allow_pickle=True)
                    default_cell_type_pred = default_cell_type_pred["predictions"]

            for cell_type, file_code in cell_type_preds_dict.items():
                if isinstance(file_code, str):
                    if file_code in file_name:
                        prediction_file = os.path.join(path_data_pred, file_name)
                        data_load = np.load(prediction_file, allow_pickle=True)
                        cell_type_preds_dict[cell_type] = data_load["predictions"]
                        break

        print(f"for {global_id}")
        # skipping noise if conflict
        fusion_cell_type_predictions_by_type(cell_type_preds_dict, default_cell_type_pred,
                                             cell_type_config_file=cell_type_config_file,
                                             filename_to_save=filename_to_save,
                                             cell_type_to_skip_if_conflict=2)
        """
        Allows to associate a cell type to a prediction (a prediction being a 2d array of n_cells * n_classes)
        Args:
            cell_type_preds_dict: key: code of the cell type, value 2d array of n_cells * n_classes representing
            the predictions to associate to this cell type (for cell predicted as so)
            default_cell_type_pred: 2d array of n_cells * n_classes
            Same length as cell_type_preds_dict
            cell_type_config_file:
            filename_to_save:
    
        Returns:
    
        """
    if fusion_with_gt:

        filename_to_save = os.path.join(path_data, cell_type_dir, f"{global_id}_cell_type_predictions_fusion.npy")

        cell_type_pred_1 = os.path.join(path_data,
                                        tmp_contours_dir,
                                        f"{global_id}_cell_type_predictions_fusion_without_GT.npy")
        cell_type_pred_1 = np.load(cell_type_pred_1, allow_pickle=True)
        # cell_type_pred_1 = cell_type_pred_1["predictions"]

        cell_type_pred_2 = os.path.join(path_data,
                                        tmp_contours_dir,
                                        f"p{age}_{global_id}_Add & Replace_cell_type_predictions.npy")
        if not os.path.isfile(cell_type_pred_2):
            cell_type_pred_2 = os.path.join(path_data,
                                            tmp_contours_dir,
                                            f"{global_id}_Add & Replace_cell_type_predictions.npy")
            if not os.path.isfile(cell_type_pred_2):
                raise Exception(f"No pred file 'Add & Replace_cell_type_predictions.npy' "
                                f"in {os.path.join(path_data, tmp_contours_dir)}")
        cell_type_pred_2 = np.load(cell_type_pred_2, allow_pickle=True)

        # cell_types_to_not_fusion_with_gt=[2] means Noise is not replaced by INs
        # with new category (0, 3) means interneurons in GT will be labeled red_ins
        fusion_cell_type_predictions(cell_type_pred_1=cell_type_pred_1, cell_type_pred_2=cell_type_pred_2,
                                     cell_type_config_file=cell_type_config_file,
                                     filename_to_save=filename_to_save,
                                     cell_types_to_not_fusion_with_gt=[2],
                                     with_new_category=[(0, 3)])
