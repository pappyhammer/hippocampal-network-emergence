from deepcinac.utils.cinac_file_utils import create_tiffs_from_movie

import os

# root path, just used to avoid copying the path everywhere
root_path = '/media/julien/Not_today/hne_not_today/data/'

tiffs_dirname = os.path.join(root_path, "tiffs_for_transient_classifier")
tiffs_to_convert_dir = os.path.join(root_path, "movies_to_split")
file_names = []
# look for filenames in the fisrst directory, if we don't break, it will go through all directories
for (dirpath, dirnames, local_filenames) in os.walk(tiffs_to_convert_dir):
    file_names.extend([x for x in local_filenames])
    break

file_names = [f for f in file_names if f.endswith(".tif")]

for file_name in file_names:
    movie_id = file_name[:-4]
    create_tiffs_from_movie(path_for_tiffs=tiffs_dirname,
                            movie_identifier=movie_id,
                            movie_file_name=os.path.join(tiffs_to_convert_dir, file_name),
                            movie_data=None)
