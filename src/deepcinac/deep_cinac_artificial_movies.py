"""
We're going to guide you on how using DeepCINAC to generate a simulated version of 2-photon calcium imaging
(the goal is not to be super realistic).

Here is a link to our gitlab page (https://gitlab.com/cossartlab/deepcinac) for more information about our package.


So far, to run this code, you will need some cell contours model
(you can download a set from our gitlab at this address(https://gitlab.com/cossartlab/deepcinac/tree/master/demos/data/simulated_movies/cell_models_simulated_movie_suite2p.mat)
or follow the instructions below to generate your own.
You can also add vessels to your movie (see instructions below, you can dowload some here
(https://gitlab.com/cossartlab/deepcinac/tree/master/demos/data//vessel_pics) as an example ).
We haven't use vessels on our simulated data to train our classifier

You can check examples of simulated movie at this address:
(https://gitlab.com/cossartlab/deepcinac/tree/master/demos/data/simulated_movies).
"""

from deepcinac.cinac_simulated_movie_generator import SimulatedMovieGenerator
import os
from datetime import datetime

# root path, just used to avoid copying the path everywhere
root_path = '/media/julien/Not_today/hne_not_today/'

# path to calcium imaging data
data_path = os.path.join(root_path, "data")

# file containing cell models, based on suite2p segmentation from a few
# 2-photon calcium imaging recorded in the CA1 pyramidal layer on pups
cell_models_file = os.path.join(data_path,
                                "simulated_movie_generator",
                                "cell_models_simulated_movie_suite2p.mat")

# path of the directory where the results will be save
# a directory will be created each time the prediction is run
# the directory name will be the date and time at which the analysis has been run
# the predictions will be in this directory.
path_results = os.path.join(root_path, "results_hne")
time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
path_results = os.path.join(path_results, time_str)
if not os.path.isdir(path_results):
    os.mkdir(path_results)

# #### First step: create an instance of SimulatedMovieGenerator and decide of the parameters
# #### (dimensions, number of frames, transients' rate etc...)

"""
dimensions: tuple of 2 int, represent the dimension of the movie in pixels, (x, y)
n_overlap_by_cell_range: (tuple of 2 int) bottom and upper range of how many cells will overlap
            our cell "of interest" aimed to train the classifier
non_overlap_by_cell_range:  (tuple of 2 int) bottom and upper range of  how many cells will be near
            our cells "of interest" aimed to train the classifier, but without any overlap
n_cells_of_interest: (int) cells of interest, are the cell for which we want to control how many overlaps,
             their activity the aim is to use them to train the classifier
range_n_transient_cells_of_interest: bottom and upper range of the number of transients over 1000 frames
            for our cells of interest
range_n_transient_overlapping_cells: bottom and upper range of the number of transients over 1000 frames
            for cells that overlap our cell of interest
range_n_transient_other_cells: bottom and upper range of the number of transients over 1000 frames
            for cells that are not of interest or overlap them
with_mvt: boolean, is True, means some motion will be added, otherwise it would produce a movie with 
            perfect motion correction  
path_results: Directory in which to save the results
time_str (string): id (meant to be a timestamps string representation) in order to  give a unique id to files. 
"""
geco_simulation = True

if geco_simulation:
    range_duration_transient = (1, 2)
    decay_factor = 5
    max_decay = 10
else:
    range_duration_transient = (1, 8)
    decay_factor = 10
    max_decay = 12

movie_generator = SimulatedMovieGenerator(dimensions=(120, 120),
                                          n_cells_of_interest=16,
                                          n_overlap_by_cell_range=(1, 4),
                                          non_overlap_by_cell_range=(2, 10),
                                          range_n_transient_cells_of_interest=(2, 4),
                                          range_n_transient_overlapping_cells=(8, 16),
                                          range_n_transient_other_cells=(2, 16),
                                          range_duration_transient=range_duration_transient,
                                          decay_factor=decay_factor,
                                          max_decay=max_decay,
                                          n_frames=2500,
                                          with_mvt=False,
                                          time_str=time_str,
                                          path_results=path_results)

# ## OPTIONAL CODE FOR INCLUDING VESSELS ###
"""
(optional) This code allows you to create vessels based on tiff images. You can find examples on our gitlab, 
it should be a picture with a white background and the vessel should be in black. 
"""
load_vessels=False

if load_vessels:

    vessels_imgs_dir = os.path.join(data_path, "simulated_movie_generator/vessels_tiff_imgs")

    # you can indicate a path where to save the newly created vessels using path_results to load
    # them directly another time
    # n_vessels_max is the number of vessels to add, if the directory contains more
    # than n_vessels_max

    movie_generator.produce_and_load_vessels(vessels_imgs_dir=vessels_imgs_dir,
                                             n_vessels_max=2, path_results=None)
    # ## Code to load vessels already created

    # vessels_dir is the directory that contains the vessels
    # n_vessels_max is the number of vessels to add, if the directory contains more
    # than n_vessels_max
    vessels_dir = os.path.join(data_path, "simulated_movie_generator")
    movie_generator.load_vessels(vessels_dir=vessels_dir, n_vessels_max=2)

# ## END VESSELS ###


# ## Then it is necessary to load some cell models that will be use to generate the contour map.
# from_matlab: boolean that indicates if the coordinates have been generated (indexing starts at 1 then)
# using matlab code and so starting from 1 and not 0.
movie_generator.load_cell_coords(data_file=cell_models_file, from_matlab=False)

"""
Finally, we generate the movie.

The automatic steps are:

1. Generating a contour map (a figure will be created to represent it as well as a file with the coordinates)
2. Generating a for each transients and building a raster 
(figure will be created as well as files to be opened in the GUI to check the activity)
3. Generating the traces (figure created as well)
4. Generating the movie (tiff file created)

"""

movie_generator.generate_movie()
