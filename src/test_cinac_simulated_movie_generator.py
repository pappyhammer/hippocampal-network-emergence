from deepcinac.cinac_simulated_movie_generator import SimulatedMovieGenerator
import os
from datetime import datetime

def main():
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")
    path_data = root_path + "data/"

    path_results = root_path + "results_hne/"
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results + "/" + time_str
    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    movie_generator = SimulatedMovieGenerator(dimensions=(120, 120), with_mvt=False,
                                              time_str=time_str,
                                              n_frames=2500, path_results=path_results)

    vessels_dir = os.path.join(path_data, "artificial_movie_generator")

    # movie_generator.load_vessels(vessels_dir=vessels_dir, n_vessels_max=2)

    vessels_imgs_dir = os.path.join(path_data, "artificial_movie_generator/vessels_tiff_imgs")

    # movie_generator.produce_and_load_vessels(vessels_imgs_dir=vessels_imgs_dir, n_vessels_max=2, path_results=None)

    coord_data_file = os.path.join(path_data,
                                            "artificial_movie_generator",
                                   "coords_artificial_movie_suite2p_p7_17_10_12_a000-p8_18_10_24_a0005_p12_17_11_10_a000.mat")
                                   # "coords_artificial_movie.mat")
    movie_generator.load_cell_coords(data_file=coord_data_file, from_matlab=False)

    movie_generator.generate_movie()

if __name__ == '__main__':
    main()