import os
from mouse_session_loader import load_mouse_sessions
from datetime import datetime
import pattern_discovery.tools.param as p_disc_tools_param


# ## aim at converting the first version of ground truth to the new format, .cinac


class DataForMs(p_disc_tools_param.Parameters):
    def __init__(self, path_data, result_path, time_str=None):
        if time_str is None:
            self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        else:
            self.time_str = time_str
        super().__init__(path_results=result_path, time_str=self.time_str, bin_size=1)
        self.path_data = path_data
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None


def main_convert_gt_to_cinac():
    import pyqtgraph.examples
    pyqtgraph.examples.run()
    return
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    path_data = os.path.join(root_path, "data")
    result_path = os.path.join(root_path, "results")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    result_path = result_path + "/" + time_str
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    param = DataForMs(path_data=path_data, result_path=result_path, time_str=time_str)

    # first key will be the ms id
    data_dict = dict()

    data_dict["p12_171110_a000_ms"] = dict()
    data_dict["p12_171110_a000_ms"]["id"] = "1"
    data_dict["p12_171110_a000_ms"]["path"] = "p12/p12_17_11_10_a000"
    # data_dict["p12_171110_a000_ms"]["gt_file"] = "p12_17_11_10_a000_GUI_fusion_validation.mat"
    data_dict["p12_171110_a000_ms"]["gt_cells"] = [9, 10, 17, 22, 24, 25, 29, 30, 33]
    data_dict["p12_171110_a000_ms"]["segmentation_tool"] = "suite2p"

    data_dict["p8_18_10_24_a005_ms"] = dict()
    data_dict["p8_18_10_24_a005_ms"]["id"] = "2"
    data_dict["p8_18_10_24_a005_ms"]["path"] = "p8/p8_18_10_24_a005"
    # data_dict["p8_18_10_24_a005_ms"]["gt_file"] = "p8_18_10_24_a005_fusion_validation.mat"
    data_dict["p8_18_10_24_a005_ms"]["gt_cells"] = [0, 1, 9, 10, 13, 15, 28, 41, 42, 110, 207, 321]
    data_dict["p8_18_10_24_a005_ms"]["segments_folder"] = "transients_to_add_for_rnn"
    data_dict["p8_18_10_24_a005_ms"]["segmentation_tool"] = "caiman"

    data_dict["p5_19_03_25_a001_ms"] = dict()
    data_dict["p5_19_03_25_a001_ms"]["id"] = "10"
    data_dict["p5_19_03_25_a001_ms"]["path"] = "p5/p5_19_03_25_a001"
    data_dict["p5_19_03_25_a001_ms"]["segments_folder"] = "transients_to_add_for_rnn"
    data_dict["p5_19_03_25_a001_ms"]["segmentation_tool"] = "suite2p"

    # ms_to_use = ["p7_171012_a000_ms",
    #              "p8_18_10_24_a006_ms",
    #              "p8_18_10_24_a005_ms",
    #              "p11_17_11_24_a000_ms", "p12_171110_a000_ms"]

    # we need to change in mouse_session_loader the type of segmentation to use

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=list(data_dict.keys()),
                                            param=param,
                                            load_traces=True, load_abf=False,
                                            for_transient_classifier=True)



if __name__ == "__main__":
    main_convert_gt_to_cinac()