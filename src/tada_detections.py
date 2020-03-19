from deeptada.tada_model import *
from deeptada.tada_detector import *
import tensorflow as tf
import numpy as np
import hdf5storage
import os
from datetime import datetime


# ------------------------- #
#     Setting file names
# ------------------------- #

# root path, just used to avoid copying the path everywhere
root_path = "/media/julien/Not_today/hne_not_today/data/tada_data"

model_path = os.path.join(root_path, "models")

results_path = os.path.join(root_path, "results_tada")
time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
results_path = os.path.join(results_path, time_str)
os.mkdir(results_path)

# path to data
data_path = os.path.join(root_path, "for_testing")

# Path to your model data. It's possible to have more than one model, and use
# each for different cell of the same recording (for exemple, one
# network could be specialized for interneurons and the other one for pyramidal
# cells)
file_names = []
for (dirpath, dirnames, local_filenames) in os.walk(dir_name):
    file_names.extend(local_filenames)
    break
weights_file_name = os.path.join(model_path, [f for f in file_names if f.endswith(".h5") and not f.startswith(".")][0])
json_file_name = os.path.join(model_path, [f for f in file_names if f.endswith(".json") and not f.startswith(".")][0])

# not mandatory, just to test if you GPU is accessible
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

cinac_dir_name = "/media/julien/Not_today/hne_not_today/data/cinac_ground_truth/for_benchmarks"

evaluate_action_detections(cinac_dir_name, results_path,
                           json_file_name, weights_file_name, save_activity_distribution=True)
