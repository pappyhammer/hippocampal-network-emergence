import os
from datetime import datetime
# from deepcinac.cinac_predictor import fusion_cell_type_predictions, fusion_cell_type_predictions_by_type
import numpy as np
from deepcinac.utils.cinac_file_utils import read_cell_type_categories_yaml_file
from collections import Counter


def fusion_cell_type_predictions_by_type(cell_type_preds_dict, default_cell_type_pred,
                                         cell_type_to_skip_if_conflict=None,
                                         cell_type_config_file=None, filename_to_save=None):
    """
    Allows to associate a cell type to a prediction (a prediction being a 2d array of n_cells * n_classes)
    Args:
        cell_type_preds_dict: key: code of the cell type, value 2d array of n_cells * n_classes representing
        the predictions to associate to this cell type (for cell predicted as so)
        default_cell_type_pred: 2d array of n_cells * n_classes
        Same length as cell_type_preds_dict
        cell_type_config_file:
        filename_to_save:
        cell_type_to_skip_if_conflict: int representing the cell type code, if given and there is a conflict between
        two classifier including this cell type, then the other one in conflict will be chosen. If a conflict with more
        than 2 cell types, then the default one is chosen.

    Returns:

    """
    n_cells = len(default_cell_type_pred)
    cell_type_codes = list(cell_type_preds_dict.keys())

    if cell_type_config_file is not None:
        cell_type_from_code_dict, cell_type_to_code_dict, multi_class_arg = \
            read_cell_type_categories_yaml_file(yaml_file=cell_type_config_file)
        # we display info about the code:
        print(f"fusion_cell_type_predictions_by_type() on {n_cells} cells")
        for code in cell_type_codes:
            print(f"Cell type for code {code}: {cell_type_from_code_dict[code]}")
        print(" ")
    else:
        cell_type_from_code_dict = None

    # key is the cell type code, value is a list of cells (int)
    cells_by_type = dict()
    # key is the cell type code, value is a list of array of lentgh n_cell_types
    cells_predictions_by_type = dict()
    for cell_type_code in cell_type_codes:
        cells_by_type[cell_type_code] = []
        cells_predictions_by_type[cell_type_code] = []

    for cell_type_code, predictions in cell_type_preds_dict.items():
        for cell in range(n_cells):
            if np.argmax(predictions[cell]) == cell_type_code:
                cells_by_type[cell_type_code].append(cell)
                cells_predictions_by_type[cell_type_code].append(predictions[cell])

    # now we look at how much intersects between cell_type
    # without intersects

    # final predictions
    fusion_predictions = np.zeros((n_cells, len(cell_type_codes)))

    n_cells_that_interesect = 0
    n_cells_non_predicted = 0
    interesect_association = []
    if cell_type_from_code_dict is not None:
        print(f"Cells with multiple cell type (-> default that will the final one):")
    for cell in range(n_cells):
        cell_type_code_associated = []
        cell_predictions = None
        for cell_type_code, cells in cells_by_type.items():
            if cell in cells:
                cell_type_code_associated.append(cell_type_code)
                index_cell = np.where(np.array(cells) == cell)[0][0]
                cell_predictions = cells_predictions_by_type[cell_type_code][index_cell]
        if len(cell_type_code_associated) == 1:
            # cell_type_code = cell_type_code_associated[0]
            fusion_predictions[cell] = cell_predictions
        else:
            use_default = True
            if len(cell_type_code_associated) == 0:
                n_cells_non_predicted += 1
            else:
                if cell_type_from_code_dict is not None:
                    new_cell_type_code = np.argmax(default_cell_type_pred[cell])
                    interesect_association.append(tuple([cell_type_from_code_dict[c]
                                                         for c in cell_type_code_associated]))
                    if (cell_type_to_skip_if_conflict is not None) and len(cell_type_code_associated) == 2:
                        if cell_type_to_skip_if_conflict in cell_type_code_associated:
                            use_default = False
                            if cell_type_to_skip_if_conflict == cell_type_code_associated[0]:
                                new_cell_type_code = cell_type_code_associated[1]
                            else:
                                new_cell_type_code = cell_type_code_associated[0]
                            cells = cells_by_type[new_cell_type_code]
                            index_cell = np.where(np.array(cells) == cell)[0][0]
                            fusion_predictions[cell] = cells_predictions_by_type[new_cell_type_code][index_cell]
                    print(f"Cell {cell}: {[cell_type_from_code_dict[c] for c in cell_type_code_associated]} -> "
                          f"{cell_type_from_code_dict[new_cell_type_code]}")
                n_cells_that_interesect += 1
            if use_default:
                fusion_predictions[cell] = default_cell_type_pred[cell]

    if cell_type_config_file is not None:
        print(f"N cells without predictions on first round: {n_cells_non_predicted}")
        print(f"N cells with more than on cell type on first round: {n_cells_that_interesect}")
        print(f"Associations that intersect: {Counter(interesect_association)}")

    if filename_to_save is not None:
        if not filename_to_save.endswith(".npy"):
            filename_to_save = filename_to_save + ".npy"
        np.save(filename_to_save, fusion_predictions)

    print(" ")

    if cell_type_from_code_dict is not None:
        # now we want to look at how many cells were wrongly classified in each pred
        for main_cell_type_code, predictions in cell_type_preds_dict.items():
            wrong_classifications_by_cell = np.zeros(len(cell_type_preds_dict), dtype="int16")
            for cell in range(n_cells):
                if np.argmax(fusion_predictions[cell]) != np.argmax(predictions[cell]):
                    wrong_cell_type = np.argmax(predictions[cell])
                    wrong_classifications_by_cell[wrong_cell_type] += 1
            print(f"Wrong classifications for predictions associated "
                  f"to {cell_type_from_code_dict[main_cell_type_code]} :")
            for cell_type_code in range(len(cell_type_preds_dict)):
                n_wrong = wrong_classifications_by_cell[cell_type_code]
                cell_type_name = cell_type_from_code_dict[cell_type_code]
                print(f"{n_wrong} wrongs {cell_type_name}")

        print(" ")
        # we display to count by cell type in the fusion predictions
        cell_count_by_type = np.zeros(fusion_predictions.shape[1], dtype="int16")
        for cell in np.arange(len(fusion_predictions)):
            cell_type_code = np.argmax(fusion_predictions[cell])
            cell_count_by_type[cell_type_code] += 1
        print(" ")
        print(f"Number of cells for each cell type after fusion:")
        for code in np.arange(len(cell_type_from_code_dict)):
            print(f"{cell_count_by_type[code]} {cell_type_from_code_dict[code]}")
        print(" ")

    return fusion_predictions


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

    if do_it_by_type:
        age = 41
        animal_id = "190320_190410"
        session_id = "190430_a000"
        #

        global_id = animal_id + "_" + session_id
        cell_type_dir = f"cell_type_predictions_{global_id}"
        path_data = os.path.join(path_data, "SWISS_data", f"p{age}", animal_id, session_id, cell_type_dir)
        filename_to_save = os.path.join(path_data, f"{global_id}_cell_type_predictions_fusion.npy")

        # ins, pyr, noise
        cell_type_preds_dict = {0: "v58_e7",
                                1: "v61_e8",
                                2: "v60_e4"}

        default_cell_type_pred = "v58_e7"

        file_names = []

        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(path_data):
            file_names.extend(local_filenames)
            break

        for file_name in file_names:
            if isinstance(default_cell_type_pred, str):
                if default_cell_type_pred in file_name:
                    default_cell_type_pred = os.path.join(path_data, file_name)
                    default_cell_type_pred = np.load(default_cell_type_pred, allow_pickle=True)
                    default_cell_type_pred = default_cell_type_pred["predictions"]

            for cell_type, file_code in cell_type_preds_dict.items():
                if isinstance(file_code, str):
                    if file_code in file_name:
                        prediction_file = os.path.join(path_data, file_name)
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
    else:

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
