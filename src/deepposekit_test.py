import tensorflow as tf

# might be removed ?
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.gpu.set_per_process_memory_growth(True)

import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from deepposekit.io import VideoReader, DataGenerator, initialize_dataset, TrainingGenerator
from deepposekit.annotate import KMeansSampler
import tqdm
import glob
import pandas as pd
from deepposekit import Annotator
from os.path import expanduser
import glob
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, destroyAllWindows, VideoCapture
import os

from os.path import expanduser



import numpy as np
import matplotlib.pyplot as plt

from deepposekit.augment import FlipAxis
import imgaug.augmenters as iaa
import imgaug as ia

from deepposekit.models import DeepLabCut, StackedDenseNet, StackedHourglass, LEAP
from deepposekit.models import load_model

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from deepposekit.callbacks import Logger, ModelCheckpoint

import time
from os.path import expanduser


def change_frame_resolution(frame, new_width, new_height, using_croping=True):
    if using_croping:
        if frame.shape[1] > new_width and frame.shape[0] > new_height:
            return frame[:new_height, :new_width]
        return frame
    else:
        return cv2.resize(np.squeeze(frame), (new_width, new_height))



def change_video_resolution(new_width=1792, new_height=1024, using_croping=True):
    """
    Change a video resolution from 1920*1200 to 1792*1024
    Or add black pixels and extend to : 2048*1280
    Returns:

    """
    fpath = "/media/julien/Not_today/hne_not_today/data/deepposekit_test/"
    file_name = "behavior_p8_19_09_29_1_cam_22983298_cam1_a000_fps_20.avi"
    new_avi_file_name = f"res_dpk_{new_width}_{new_height}_" + file_name
    file_name = os.path.join(fpath, file_name)
    new_avi_file_name = os.path.join(fpath, new_avi_file_name)

    # TO READ
    cap = VideoCapture(file_name)
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        return
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_avi = cap.get(cv2.CAP_PROP_FPS)

    # for writing
    is_color = True
    # put fourcc to 0 for no compression
    # fourcc = 0
    fourcc = VideoWriter_fourcc(*"XVID")

    size_avi = (new_width, new_height)
    vid_avi = VideoWriter(new_avi_file_name, fourcc, fps_avi, size_avi, is_color)

    n_frames = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if n_frames == 0:
            print(f"frame.shape {frame.shape}")

        if ret == True:
            frame = change_frame_resolution(frame=frame, new_width=new_width, new_height=new_height,
                                            using_croping=using_croping)
            if n_frames == 0:
                print(f"new frame.shape {frame.shape}")
            n_frames += 1
            vid_avi.write(frame)
        # Break the loop
        else:
            break
        if n_frames % 5000 == 0:
            print(f"{n_frames} converted over {length}")
    print(f"{n_frames} frames converted")
    # Closes all the frames
    cv2.destroyAllWindows()
    # When everything done, release the video capture object
    vid_avi.release()
    cap.release()

    print(f"new_avi_file_name {new_avi_file_name}")
    return new_avi_file_name

def prepare_annotation(h5_file, fpath, skeleton_file, sampled_frames):
    # skeleton = pd.read_csv(os.path.join(fpath, skeleton_file))

    initialize_dataset(
        images=sampled_frames,
        datapath=os.path.join(fpath, h5_file),
        skeleton=os.path.join(fpath, skeleton_file),
        overwrite=True # This overwrites the existing datapath
    )

def sample_video_frames(fpath, movie_file_name):
    """

    Args:
        fpath:
        movie_file_name:

    Returns:

    """
    reader = VideoReader(os.path.join(fpath, movie_file_name), batch_size=100, gray=True)

    randomly_sampled_frames = []
    for idx in tqdm.tqdm(range(len(reader) - 1)):
        batch = reader[idx]
        random_sample = batch[np.random.choice(batch.shape[0], 10, replace=False)]
        randomly_sampled_frames.append(random_sample)
    reader.close()

    randomly_sampled_frames = np.concatenate(randomly_sampled_frames)
    randomly_sampled_frames.shape

    kmeans = KMeansSampler(n_clusters=10, max_iter=1000, n_init=10, batch_size=100, verbose=True)
    kmeans.fit(randomly_sampled_frames)

    plot_kmeans = False
    if plot_kmeans:
        kmeans.plot_centers(n_rows=2)
        plt.show()

    kmeans_sampled_frames, kmeans_cluster_labels = kmeans.sample_data(randomly_sampled_frames, n_samples_per_label=10)
    print(f"kmeans_sampled_frames.shape {kmeans_sampled_frames.shape}")

    return kmeans_sampled_frames

def annotate(fpath, skeleton_file, h5_file):
    # ANNOTATE
    """

    Annotation Hotkeys
    +- = rescale image by ±10%
    left mouse button = move active keypoint to cursor location
    WASD = move active keypoint 1px or 10px
    space = change WASD mode (swaps between 1px or 10px movements)
    JL = next or previous image
    <> = jump 10 images forward or backward
    I,K or tab, shift+tab = switch active keypoint
    R = mark image as unannotated ("reset")
    F = mark image as annotated ("finished")
    esc or Q = quit
    """

    app = Annotator(datapath=os.path.join(fpath, h5_file),
                    dataset='images',
                    skeleton=os.path.join(fpath, skeleton_file),
                    shuffle_colors=False,
                    text_scale=0.5)
    app.run()

def get_callbacks(fpath, model_file):
    """

    Returns:

    """

    """
        Define callbacks to enhance model training
        Here you can define callbacks to pass to the model for use during training. 
        You can use any callbacks available in deepposekit.callbacks or tensorflow.keras.callbacks

        Remember, if you set validation_split=0 for your TrainingGenerator, 
        which will just use the training set for model fitting, make sure to set monitor="loss" instead of monitor="val_loss".

        Logger evaluates the validation set (or training set if validation_split=0 in the TrainingGenerator) 
        at the end of each epoch and saves the evaluation data to a HDF5 log file (if filepath is set).
        """

    logger = Logger(validation_batch_size=10,
                    # filepath saves the logger data to a .h5 file
                    # filepath=HOME + "/deepposekit-data/datasets/fly/log_densenet.h5"
                    )

    """
    ReduceLROnPlateau automatically reduces the learning rate of the optimizer when the validation loss stops improving. 
    This helps the model to reach a better optimum at the end of training.
    """

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1, patience=20)

    """
    ModelCheckpoint automatically saves the model when the validation loss improves at the end of each epoch. 
    This allows you to automatically save the best performing model during training, 
    without having to evaluate the performance manually.
    """

    model_checkpoint = ModelCheckpoint(
        os.path.join(fpath, model_file),
        monitor="val_loss",
        # monitor="loss" # use if validation_split=0
        verbose=1,
        save_best_only=True,
    )

    """
    EarlyStopping automatically stops the training session when the validation 
    loss stops improving for a set number of epochs, 
    which is set with the patience argument. This allows you to save time when 
    training your model if there's not more improvment.
    """

    early_stop = EarlyStopping(
        monitor="val_loss",
        # monitor="loss" # use if validation_split=0
        min_delta=0.001,
        patience=100,
        verbose=1
    )

    # Create a list of callbacks to pass to the model
    callbacks = [early_stop, reduce_lr, model_checkpoint, logger]

    return callbacks

def get_augmenter(data_generator):
    """

    Returns:

    """

    """
        DeepPoseKit works with augmenters from the imgaug package. 
        This is a short example using spatial augmentations with axis flipping and 
        affine transforms See https://github.com/aleju/imgaug for more documentation on augmenters.

        deepposekit.augment.FlipAxis takes the DataGenerator as an argument to get the keypoint swapping information 
        defined in the annotation set. When the images are mirrored keypoints for left and 
        right sides are swapped to avoid "confusing" the model during training.
        """

    augmenter = []

    # augmenter.append(FlipAxis(data_generator, axis=0))  # flip image up-down
    augmenter.append(FlipAxis(data_generator, axis=1))  # flip image left-right

    sometimes = []
    sometimes.append(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                                shear=(-8, 8),
                                order=ia.ALL,
                                cval=ia.ALL,
                                mode=ia.ALL)
                     )
    sometimes.append(iaa.Affine(scale=(0.8, 1.2),
                                mode=ia.ALL,
                                order=ia.ALL,
                                cval=ia.ALL)
                     )
    augmenter.append(iaa.Sometimes(0.75, sometimes))
    augmenter.append(iaa.Affine(rotate=(-180, 180),
                                mode=ia.ALL,
                                order=ia.ALL,
                                cval=ia.ALL)
                     )
    augmenter = iaa.Sequential(augmenter)
    return augmenter

def train_model(fpath, labeled_data_h5_file, model_file):
    # creating data generator
    data_generator = DataGenerator(os.path.join(fpath, labeled_data_h5_file))

    """
    Indexing the generator, e.g. data_generator[0] returns an image-keypoints pair, which you can then visualize.
    """
    image, keypoints = data_generator[0]

    plot_key_points = False

    if plot_key_points:
        plt.figure(figsize=(5, 5))
        image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
        cmap = None if image.shape[-1] is 3 else 'gray'
        plt.imshow(image, cmap=cmap, interpolation='none')
        for idx, jdx in enumerate(data_generator.graph):
            if jdx > -1:
                plt.plot(
                    [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
                    [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
                    'r-'
                )
        plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50,
                    cmap=plt.cm.hsv, zorder=3)

        plt.show()

    augmenter = get_augmenter(data_generator=data_generator)

    """
    Create a TrainingGenerator
    This creates a TrainingGenerator from the DataGenerator for training the model with annotated data. 
    The TrainingGenerator uses the DataGenerator to load image-keypoints pairs and then applies the augmentation 
    and draws the confidence maps for training the model.
    
    If you're using StackedDenseNet, StackedHourglass, or DeepLabCut you should set downsample_factor=2 
    for 1/4x outputs or downsample_factor=3 for 1/8x outputs (1/8x is faster). Here it is set to downsample_factor=3 
    to maximize speed. If you are using LEAP you should set the downsample_factor=0 for 1x outputs.
    
    The validation_split argument defines how many training examples to use for validation during training. 
    If your dataset is small (such as initial annotations for active learning), 
    you can set this to validation_split=0, which will just use the training set for model fitting. 
    However, when using callbacks, make sure to set monitor="loss" instead of monitor="val_loss".
    
    Visualizing the outputs in the next section also works best with downsample_factor=0.
    """
    # if validation_split is too small, an error occurs, see https://github.com/jgraving/DeepPoseKit/issues/9
    train_generator = TrainingGenerator(generator=data_generator,
                                        downsample_factor=3,
                                        augmenter=augmenter,
                                        sigma=5,
                                        validation_split=0.2,
                                        use_graph=True,
                                        random_seed=1,
                                        graph_scale=1)
    train_generator.get_config()

    """
    Define a model
    Here you can define a model to train with your data. You can use our StackedDenseNet model, StackedHourglass model, 
    DeepLabCut model, or the LEAP model. The default settings for each model 
    should work well for most datasets, but you can customize the model architecture. 
    The DeepLabCut model has multiple pretrained (on ImageNet) backbones available for using transfer learning, 
    including the original ResNet50 (He et al. 2015) as well as the faster MobileNetV2 (Sandler et al. 2018; 
    see also Mathis et al. 2019) and DenseNet121 (Huang et al. 2017). 
    We'll select StackedDenseNet and set n_stacks=2 for 2 hourglasses, 
    with growth_rate=32 (32 filters per convolution). Adjust the growth_rate and/or n_stacks to 
    change model performance (and speed). You can also set pretrained=True to use transfer learning 
    with StackedDenseNet, which uses a DenseNet121 pretrained on ImageNet to encode the images.
    """
    model = StackedDenseNet(train_generator, n_stacks=2, growth_rate=32, pretrained=True)

    # model = DeepLabCut(train_generator, backbone="resnet50")
    # model = DeepLabCut(train_generator, backbone="mobilenetv2", alpha=0.35) # Increase alpha to improve accuracy
    # model = DeepLabCut(train_generator, backbone="densenet121")

    # model = LEAP(train_generator)
    # model = StackedHourglass(train_generator)

    print(f"model config() {model.get_config()}")

    """
    Test the prediction speed
    This generates a random set of input images for the model to test how 
    fast the model can predict keypoint locations.
    """
    test_speed_prediction = False

    if test_speed_prediction:
        data_size = (10000,) + data_generator.image_shape
        x = np.random.randint(0, 255, data_size, dtype="uint8")
        # used to be batch_size = 100, but memory issue occured
        y = model.predict(x[:100], batch_size=32)  # make sure the model is in GPU memory
        t0 = time.time()
        y = model.predict(x, batch_size=32, verbose=1)
        t1 = time.time()
        print(x.shape[0] / (t1 - t0)) # 27.47681073503684

    callbacks = get_callbacks(fpath=fpath, model_file=model_file)

    """
    Fit the model
    This fits the model for a set number of epochs with small batches of data. 
    If you have a small dataset initially you can set batch_size to a small value 
    and manually set steps_per_epoch to some large value, e.g. 500, 
    to increase the number of batches per epoch, otherwise this is automatically 
    determined by the size of the dataset.
    
    The number of epochs is set to epochs=200 for demonstration purposes. 
    Increase the number of epochs to train the model longer, for example epochs=1000. 
    The EarlyStopping callback will then automatically end training if there is no 
    improvement. See the doc string for details:
    """

    model.fit(
        batch_size=2,
        validation_batch_size=2,
        callbacks=callbacks,
        # epochs=1000, # Increase the number of epochs to train the model longer
        epochs=20,
        n_workers=8,
        steps_per_epoch=None,
    )


def resume_training(fpath, original_model_file, new_model_file, labeled_data_h5_file):
    data_generator = DataGenerator(os.path.join(fpath, labeled_data_h5_file))
    augmenter = get_augmenter(data_generator=data_generator)
    model = load_model(
        os.path.join(fpath, original_model_file),
        augmenter=augmenter,
        generator=data_generator,
    )

    callbacks = get_callbacks(fpath, model_file=new_model_file)

    model.fit(
        batch_size=8,
        validation_batch_size=2,
        callbacks=callbacks,
        epochs=500,
        n_workers=8,
        steps_per_epoch=None,
    )

def initialize_annotations():
    pass

if __name__ == "__main__":

    do_change_video_resolution = False
    do_annotation = False
    do_train_model = True
    do_resume_training = False
    do_initialize_annotations = False

    movie_file_name = "res_dpk_1024_640_behavior_p8_19_09_29_1_cam_22983298_cam1_a000_fps_20.avi"
    fpath = "/media/julien/Not_today/hne_not_today/data/deepposekit_test/"
    labeled_data_h5_file = "test_dpk.h5"
    # TIPS: put it in the same order as the order you want to annotate
    skeleton_file = "dpk_right_side_video_simple.csv"
    model_file = "best_model_densenet.h5"

    # STEP 0
    if do_change_video_resolution:
        new_width = 1024
        new_height = 640
        new_avi_file_name = change_video_resolution(new_width=new_width, new_height=new_height, using_croping=False)

    if do_annotation:
        # cam1 == "right_side"
        # movie_file_name = "new_res_behavior_p8_19_09_29_1_cam_22983298_cam1_a000_fps_20.avi"
        # STEP 1
        sampled_frames = sample_video_frames(movie_file_name=movie_file_name, fpath=fpath)
        prepare_annotation(h5_file=labeled_data_h5_file, fpath=fpath, skeleton_file=skeleton_file,
                           sampled_frames=sampled_frames)
        # STEP 2
        annotate(fpath, skeleton_file, labeled_data_h5_file)

    # STEP 3
    if do_train_model:
        train_model(fpath=fpath, labeled_data_h5_file=labeled_data_h5_file, model_file=model_file)

    # STEP 3 bis
    if do_resume_training:
        resume_training(fpath=fpath, original_model_file=model_file, new_model_file="new_" + model_file,
                        labeled_data_h5_file=labeled_data_h5_file)

    # STEP 4a Initialize annotations
    if do_initialize_annotations:
        # TODO: continue this
        initialize_annotations()

