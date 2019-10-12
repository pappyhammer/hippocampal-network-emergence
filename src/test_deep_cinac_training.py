import os
from deepcinac.cinac_model import *

if __name__ == '__main__':
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    root_path = "/media/julien/Not_today/hne_not_today/"

    data_path = os.path.join(root_path, "data/")
    results_path = os.path.join(root_path, "results_hne")

    cinac_model = CinacModel(results_path=results_path)

