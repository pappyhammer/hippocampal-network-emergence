from deeptada.tada_detector import *
import tensorflow as tf
import os
from datetime import datetime

# ------------------------- #
#     Setting file names
# ------------------------- #

# root path, just used to avoid copying the path everywhere
root_path = "/scratch/edenis/deeptada/"


model_path = os.path.join(root_path, "models")

results_path = os.path.join(root_path, "results_tada")
time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
results_path = os.path.join(results_path, time_str)
os.mkdir(results_path)

# path to data
data_path = os.path.join(root_path, "for_testing")

config_yaml_file = os.path.join(data_path, "config_tada_detections.yaml")

# Path to your model data. It's possible to have more than one model, and use
# each for different cell of the same recording (for exemple, one
# network could be specialized for interneurons and the other one for pyramidal
# cells)
file_names = []
for (dirpath, dirnames, local_filenames) in os.walk(model_path):
    file_names.extend(local_filenames)
    break
weights_file_name = os.path.join(model_path, [f for f in file_names if f.endswith(".h5") and not f.startswith(".")][0])
json_file_name = os.path.join(model_path, [f for f in file_names if f.endswith(".json") and not f.startswith(".")][0])

# not mandatory, just to test if you GPU is accessible
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

predictions_results = evaluate_action_detections(tada_dir_name=data_path, results_path=results_path,
                                                 config_yaml_file=config_yaml_file,
                                                 json_file_name=json_file_name, weights_file_name=weights_file_name,
                                                 save_activity_distribution=True)
