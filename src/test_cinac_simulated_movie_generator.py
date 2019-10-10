from deepcinac.cinac_simulated_movie_generator import SimulatedMovieGenerator, produce_vessels
import os

def main():
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")
    path_data = root_path + "data/"

    movie_generator = SimulatedMovieGenerator(dimensions=(120, 120), with_mvt=False)

    vessels_dir = os.path.join(path_data, "artificial_movie_generator")

    # movie_generator.load_vessels(vessels_dir=vessels_dir, n_vessels_max=2)

    vessels_imgs_dir = os.path.join(path_data, "artificial_movie_generator/vessels_tiff_imgs")

    movie_generator.produce_and_load_vessels(vessels_imgs_dir=vessels_imgs_dir, n_vessels_max=2, path_results=None)

    coord_data_file = os.path.join(path_data,
                                            "artificial_movie_generator",
                                   "coords_artificial_movie.mat")
                                            # "coords_artificial_movie_suite2p_p7_17_10_12_a000-p8_18_10_24_a0005_p12_17_11_10_a000.mat")
    movie_generator.load_cell_coords(data_file=coord_data_file)

    movie_generator.generate_movie()

    # path_results = root_path + "results_hne/"
    # path_results = path_results + "/" + movie_generator.time_str
    # if not os.path.isdir(path_results):
    #     os.mkdir(path_results)

    # ------------- OLD CODE ---------------
    n_frames = 2500

    # we need to generate a raster_dur, with some synchronicity between overlapping cells
    raster_dur = build_raster_dur(map_coords=map_coords, cells_with_overlap=cells_with_overlap,
                                  overlapping_cells=overlapping_cells, n_frames=n_frames, param=param)

    save_raster_dur_for_gui(raster_dur, param)

    # then we build the movie based on cells_coords and the raster_dur
    shaking_frames = produce_movie(map_coords=map_coords, raster_dur=raster_dur, param=param,
                                   cells_with_overlap=cells_with_overlap,
                                   overlapping_cells=overlapping_cells, padding=padding,
                                   vessels=vessels)

    # saving cells' number of interest

    file_name_txt = 'artificial_cells_listing.txt'

    with open(os.path.join(param.path_results, file_name_txt), "w", encoding='UTF-8') as file:
        file.write(f"Targets cells: {', '.join(list(map(str, cells_with_overlap)))}" + '\n')
        file.write(f"Shaking frames: {', '.join(list(map(str, shaking_frames)))}" + '\n')

    coords_matlab_style = np.empty((len(map_coords),), dtype=np.object)
    for i in range(len(map_coords)):
        coords_matlab_style[i] = map_coords[i]
    sio.savemat(os.path.join(param.path_results, "map_coords.mat"), {"coord_python": coords_matlab_style})

if __name__ == '__main__':
    main()