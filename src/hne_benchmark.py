import numpy as np
import classification_stat as cs
import hdf5storage
from datetime import datetime
import os


class BenchmarkRasterDur:
    def __init__(self, description, ground_truth_raster_dur, predicted_raster_dur_dict, cells):
        self.description = description
        self.ground_truth_raster_dur = ground_truth_raster_dur
        # cells on which base the ground truth
        self.cells = cells
        # first key describe the data, value is raster_dur (2D array, cells vs frames)
        self.predicted_raster_dur_dict = predicted_raster_dur_dict
        # first key is the cell, value a dict with
        # same keys as raster_dur_dict, value will be a list of dict with results from benchmarks
        self.results_dict_by_cell = dict()
        # same keys as raster_dur_dict, value will be a list of dict with results from benchmarks
        self.results_dict_global = dict()

    def compute_stats(self):
        print(f"{self.description} stats")
        for cell in self.cells:
            print(f"Cell {cell}")
            for key, raster_dur in self.predicted_raster_dur_dict.items():
                gt_rd = self.ground_truth_raster_dur[cell]
                p_rd = raster_dur[cell]
                frames_stat, transients_stat = cs.compute_stats(gt_rd, p_rd)
                # frames stats
                print(f"raster {key}")
                print(f"Frames stat:")
                for k, value in frames_stat.items():
                    print(f"{k}: {str(np.round(value, 4))}")

                print(f"###")
                print(f"Transients stat:")
                for k, value in transients_stat.items():
                    print(f"{k}: {str(np.round(value, 4))}")
                print("")
            print("")
            print("/////////////////")
            print("")
        print("All cells")
        for key, raster_dur in self.predicted_raster_dur_dict.items():
            gt_rd = self.ground_truth_raster_dur[self.cells]
            p_rd = raster_dur[self.cells]
            frames_stat, transients_stat = cs.compute_stats(gt_rd, p_rd)
            # frames stats
            print(f"raster {key}")
            print(f"Frames stat:")
            for k, value in frames_stat.items():
                print(f"{k}: {str(np.round(value, 4))}")

            print(f"###")
            print(f"Transients stat:")
            for k, value in transients_stat.items():
                print(f"{k}: {str(np.round(value, 4))}")
            print("")

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

def main_benchmark():
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")

    path_data = root_path + "data/"
    path_results_raw = root_path + "results_hne/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results_raw + f"{time_str}"
    os.mkdir(path_results)

    ground_truth_raster_dur = None
    ms_to_benchmark = "p12_17_11_10_a000"
    data_dict = dict()
    if ms_to_benchmark == "p12_17_11_10_a000":
        # gt as ground_truth
        data_dict["gt"] = dict()
        # p12
        data_dict["gt"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["gt"]["gui_file"] = "p12_17_11_10_a000_GUI_JD.mat"
        data_dict["gt"]["cnn"] = "p12_17_11_10_a000_cell_to_suppress_ground_truth.txt"
        data_dict["gt"]["cells"] = np.arange(7)
        data_dict["caiman_raw"] = dict()
        data_dict["caiman_raw"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["caiman_raw"]["file_name"] = "p12_17_11_10_a000_RasterDur.mat"
        data_dict["caiman_raw"]["var_name"] = "rasterdur"
        data_dict["caiman_filt"] = dict()
        data_dict["caiman_filt"]["path"] = "p12/p12_17_11_10_a000"
        data_dict["caiman_filt"]["file_name"] = "p12_17_11_10_a000_filt_RasterDur_caiman.mat"
        data_dict["caiman_filt"]["var_name"] = "rasterdur"

    data_file = hdf5storage.loadmat(os.path.join(path_data, data_dict["gt"]["path"], data_dict["gt"]["gui_file"]))
    peak_nums = data_file['LocPeakMatrix_Python'].astype(int)
    spike_nums = data_file['Bin100ms_spikedigital_Python'].astype(int)
    inter_neurons = data_file['inter_neurons'].astype(int)
    cells_to_remove = data_file['cells_to_remove'].astype(int)
    ground_truth_raster_dur = build_spike_nums_dur(spike_nums, peak_nums)

    cell_cnn_predictions = []
    with open(os.path.join(path_data, data_dict["gt"]["path"], data_dict["gt"]["cnn"]), "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split()
            cells_list = [float(i) for i in line_list]
            cell_cnn_predictions.extend(cells_list)
    cell_cnn_predictions = np.array(cell_cnn_predictions)
    cells_for_benchmark = data_dict["gt"]["cells"]
    cells_for_benchmark = np.setdiff1d(cells_for_benchmark, cells_to_remove)

    # not taking into consideration cells that are not predicted as true from the cell classifier
    cells_predicted_as_false = np.where(cell_cnn_predictions < 0.5)[0]
    cells_for_benchmark = np.setdiff1d(cells_for_benchmark, cells_predicted_as_false)
    predicted_raster_dur_dict = dict()
    for key, value in data_dict.items():
        if key == "gt":
            continue
        data_file = hdf5storage.loadmat(os.path.join(path_data, value["path"], value["file_name"]))
        raster_dur = data_file[value['var_name']].astype(int)
        predicted_raster_dur_dict[key] = raster_dur

    benchmarks = BenchmarkRasterDur(description=ms_to_benchmark, ground_truth_raster_dur=ground_truth_raster_dur,
                       predicted_raster_dur_dict=predicted_raster_dur_dict, cells=cells_for_benchmark)

    benchmarks.compute_stats()

main_benchmark()