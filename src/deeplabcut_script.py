import deeplabcut
import os
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, VideoCapture

def change_video_resolution(file_name, new_width=1024, new_height=None, using_croping=True):
    """
    Change a video resolution from 1920*1200 to 1792*1024
    Or add black pixels and extend to : 2048*1280
    Returns:

    """
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

    # if new_width or new_height is None, then choose the one at None so we keep the ratio
    if new_width is None and new_height is None:
        raise Exception("Width and height are None")
    if new_width is None:
        new_width = (width * new_height) / height
    if new_height is None:
        new_height = (height * new_width) / width
    new_avi_file_name = f"res_dpk_{new_width}_{new_height}_" + file_name
    new_avi_file_name = os.path.join(fpath, new_avi_file_name)

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

 """
    Installing DLC using conda :
    https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md

    Guide: 
    https://github.com/AlexEMG/DeepLabCut/blob/master/docs/UseOverviewGuide.md
     @article{NathMathisetal2019,
    title={Using DeepLabCut for 3D markerless pose estimation across species and behaviors},
    author = {Nath*, Tanmay and Mathis*, Alexander and Chen, An Chi and Patel, Amir and Bethge, Matthias and Mathis, Mackenzie W},
    journal={Nature Protocols},
    year={2019},
    url={https://doi.org/10.1038/s41596-019-0176-0}
    
"""

"""
Protocol timing
    Steps 1 and 2, Stage I, starting Python and creation of a new project: ~3 min
    Step 3, Stage II, configuration of the config.yaml file: ~5 min
    Step 4, Stage III, extraction of frames to label: variable, ranging from 10 to 15 min
    Steps 5 and 6, Stage IV, manual labeling of frames: variable, ranging from 1 to 10 h
    Step 7, Stage V, checking labels: 5 min, depending on how many frames were labeled
    Step 8, Stage VI, creation of training set: ~1 min
    Step 9, Stage VII, training the network: ~1–12 h (~6 h on a 1080-Ti GPU, and dependent on frame size)
    Step 10, Stage VIII, evaluation of the network: variable, but typically 2–10 min
    Steps 11–13, Stage IX, novel video analysis: variable, (10–1,000 FPS); i.e., a 20-min video collected at 100
    FPS, frame size 138 × 138, and batch size set to 64 will take ~3.5 min on a GPU
    Steps 14–16, (optional) Stage X, refinement: timing is equivalent timing to that for Steps 4–10
    Step 17, Stage XI, working with the output files of DeepLabCut: variable
"""

if __name__ == "__main__":
    root_path = "/media/julien/Not_today/hne_not_today/data/deeplabcut_test/"

    working_dir = os.path.join(root_path, "test_29_12_19")
    movie_file = os.path.join(root_path, "res_dpk_1024_640_behavior_p8_19_09_29_1_cam_22983298_cam1_a000_fps_20.avi")

    new_videos_to_add = []
    config_path = os.path.join(working_dir, "test_2-Robin-2019-12-29", "config.yaml")

    # ['Full path of video or videofolder']
    videos_to_analyze = []
    # full video paths for those 3 variables
    videos_to_filter_predictions = []
    videos_to_plot_trajectories = []
    videos_to_create_with_labels = []
    videos_to_extract_outlier_frames = []

    try_project_manager_gui = False

    if try_project_manager_gui:
        deeplabcut.launch_dlc()
        raise Exception('try_project_manager_gui OVER')

    do_change_video_resolution = False
    if do_change_video_resolution:
        fpath = "/media/julien/Not_today/hne_not_today/data/deepposekit_test/"
        file_name = "behavior_p8_19_09_29_1_cam_22983298_cam1_a000_fps_20.avi"
        file_name = os.path.join(fpath, file_name)
        change_video_resolution(file_name=file_name, new_width=1024, new_height=None, using_croping=True)
        raise Exception('do_change_video_resolution OVER')

    # keys are:
    # "step_2" or "create_new_project"
    # "step_2_bis" or "add_new_videos"
    # "step_4 or "data_selection"
    # "step_5" or "labeling_frames"
    # "step_7" or "checking_annotated_frames"
    # "step_8" or "create_training_dataset"
    # "step_9" or "training_network"
    # "step_10" or "evaluate_network"
    # "step_11" or "analyze_videos" (set videos_to_analyze to analyse)
    # "filterpredictions" (set videos_to_filter_predictions variable)
    # "step_12" or "plot_trajectories" (set videos_to_plot_trajectories variable)
    # "step_13" or "create_labeled_video" (set videos_to_create_with_labels variable)
    # "step_14" or "extract_outlier_frames" (set videos_to_extract_outlier_frames variable)
    # "step_15" or "refine_labels"
    # "step_16" or "merge_datasets"
    stages_to_run = ["create_new_project"]

    # ------------------------------------------------------------------
    # STAGE I: Stage I: opening DeepLabCut and creation of a new project
    # Steps 1, 2
    # ------------------------------------------------------------------
    if "step_2" in stages_to_run or "create_new_project" in stages_to_run:
        config_path = deeplabcut.create_new_project(project="test_2", experimenter="Robin", videos=[movie_file],
                                      copy_videos=True,
                                      videotype='.avi',
                                      working_directory=working_dir)



    if "step_2_bis" in stages_to_run or "add_new_videos" in stages_to_run:
        deeplabcut.add_new_videos(config=config_path, videos=new_videos_to_add, copy_videos=True)


    # ------------------------------------------------------------------
    # STAGE II: configuration of the project, open the config.yaml file
    # Step 3
    # ------------------------------------------------------------------

    """
    Exemple of config:
    bodyparts:
    - forepaw
    - foreleg_joint
    - foreleg_body_jonction
    - hindlepaw
    - hindleleg_joint
    - hindleleg_body_jonction
    - tail_prox

    skeleton:
    - - forepaw
      - foreleg_joint
      - foreleg_body_jonction
    - - hindlepaw
      - hindleleg_joint
      - hindleleg_body_jonction
    
    start: 0
    stop: 0.5
    numframes2pick: 20
    """

    # ------------------------------------------------------------------
    # Stage III: data selection
    # Step 4
    # ------------------------------------------------------------------
    """
    Select videos from which to grab frames:
        Use videos with images from
        -Different sessions reflecting (if the case) varying light conditions, backgrounds, setups, and camera angles
        -Different individuals, especially if they look different (i.e., brown and black mice)
    
    In our case: different ages !
    
    The toolbox contains three methods for extracting frames, namely, by clustering based on visual content, by
    randomly sampling in a uniform way across time, or by manually grabbing frames of interest using a custom GUI.
    
    For the behaviors we have tested so far, a dataset of 50–200 frames gave good results
    """

    if "step_4" in stages_to_run or "data_selection" in stages_to_run:
        deeplabcut.extract_frames(config=config_path,
                                  mode='automatic', algo='kmeans', crop=False, userfeedback=True, cluster_step=1,
                                  cluster_resizewidth=30, cluster_color=False, opencv=True, slider_width=25)
        """
        The extracted frames from all the videos are stored in a separate subdirectory named after the
        video file’s name under the ‘labeled-data’ directory.
        
        When running the function extract_frames, if the parameter crop=True, then
        frames will be cropped to the interactive user feedback provided (which is then written to the
        config.yaml file). Upon calling extract_frames, it will ask the user to draw a boundingbox
        in the GUI
        
        he provided function selects frames from the videos in a temporally uniformly distributed way
        (uniform), by clustering based on visual appearance (kmeans), or by manual selection (Fig. 3).
        Uniform selection of frames works best for behaviors in which the postures vary in a temporally
        independent way across the video. However, some behaviors might be sparse, as in a case of
        reaching in which the reach and pull are very fast and the mouse is not moving much between
        trials. In such a case, visual information should be used for selecting different frames If the user
        chooses to use kmeans as a method to cluster the frames, then this function downsamples the
        video and clusters the frames. Frames from different clusters are then selected. This procedure
        ensures that the frames look different and is generally preferable. However, on large and long
        videos, this code is slow due to its computational complexity.
        
        If users feel that specific frames are lacking, they can extract hand-picked frames
        of interest using the interactive GUI provided along with the toolbox. This can be launched by
        using the following (optional) command: >> deeplabcut.extract_frames(config_path,
        ‘manual’). The user can use the ‘Load Video’ button to load one of the videos in the project
        configuration file, use the scroll bar to navigate across the video, and select ‘Grab a Frame’ to extract
        the frame. The user can also look at the extracted frames and, e.g., delete frames (from the directory)
        that are too similar before manually annotating them. The methods can be used in a mix-and-match
        way to create a diverse set of frames. The user can also choose to select frames for only specific videos.
        
        CRITICAL STEP It is advisable to keep the frame size small, as large frames increase the training
        and inference times.
        """

    # ------------------------------------------------------------------
    # Stage IV: labeling of the frames
    # Steps 5, 6
    # ------------------------------------------------------------------
    if "step_5" in stages_to_run or "labeling_frames" in stages_to_run:
        # The following command invokes the labeling toolbox
        deeplabcut.label_frames(config_path)

        """
        Next, use the ‘Load Frames’ button to select the directory that stores the extracted frames from one
        of the videos. A right click places the first body part, and, subsequently, you can either select one of
        the radio buttons (top right) to select a body part to label, or use the built-in mechanism that
        automatically advances to the next body part. If a body part is not visible, simply do not label the
        part and select the next body part you want to label. Each label will be plotted as a dot in a unique
        color (see Fig. 4 for more details).
        
        You can also move the label around by left-clicking and dragging. Once the position is
        satisfactory, you can select another radio button (in the top right) to switch to another label (it also
        auto-advances, but you can manually skip labels if needed). Once all the visible body parts are
        labeled, then you can click ‘Next’ to load the following frame, or ‘Previous’ to look at and/or adjust
        the labels on previous frames. You need to save the labels after all the frames from one of the videos
        are labeled by clicking the ‘Save’ button. You can save at intermediate points, and then relaunch the
        GUI to continue labeling (or refine your already-applied labels). Saving the labels will create a
        labeled dataset in a hierarchical data format (HDF) file and comma-separated (CSV) file in the
        subdirectory corresponding to the particular video in ‘labeled-data’.
        
        CRITICAL STEP It is advisable to consistently label similar spots (e.g., on a wrist that is very large,
        try to label the same location). In general, invisible or occluded points should not be labeled. Simply
        skip the hidden part by not applying the label anywhere on the frame, or guess the location of body
        parts, in order to train a neural network that does that as well.
        
        CRITICAL STEP Occasionally, the user might want to label additional body parts. In such a case,
        the user needs to append the new labels to the bodyparts list in the config.yaml file. Thereafter,
        the user can call the function label_frames and will be asked if he or she wants to display only
        the new labels or all labels before loading the frames. Saving the labels after all the images are
        labeled will append the new labels to the existing labeled dataset.
        """

    # ------------------------------------------------------------------
    # Stage V: (optional) checking of annotated frames
    # Step 7
    # ------------------------------------------------------------------
    if "step_7" in stages_to_run or "checking_annotated_frames" in stages_to_run:
        """
        CRITICAL Checking whether the labels were created and stored correctly is beneficial for training, as
        labeling is one of the most critical parts of a supervised learning algorithm such as DeepLabCut.
        Nevertheless, this section is optional.
        """
        deeplabcut.check_labels(config_path)
        """
        For each directory in ‘labeled-data’, this function creates a subdirectory with ‘labeled’ as a suffix.
        These directories contain the frames plotted with the annotated body parts. You can then double-
        check whether the body parts are labeled correctly. If they are not correct, use the labeling GUI
        (stage 4) and adjust the location of the labels.
        """

    # ------------------------------------------------------------------
    # Stage VI: creation of a training dataset
    # Step 8
    # ------------------------------------------------------------------
    """
    CRITICAL Combining the labeled datasets from all the videos and splitting them will create train and
    test datasets. The training data will be used to train the network, whereas the test dataset will be used to
    test the generalization of the network (during evaluation). The function
    create_training_dataset performs these steps.
    """
    if "step_8" in stages_to_run or "create_training_dataset" in stages_to_run:
        deeplabcut.create_training_dataset(config_path, num_shuffles=1, Shuffles=None,
                                           windows2linux=False,userfeedback=False,
                                           trainIndexes=None,testIndexes=None,
                                           net_type=None,augmenter_type=None)

    """
    The set of arguments in the function will shuffle the combined labeled dataset and split it to
    create a train and a test set. The subdirectory with the suffix ‘iteration-#’ under the directory
    ‘training-datasets’ stores the dataset and meta information, where the ‘#’ is the value of the
    iteration variable stored in the project’s configuration file (this number keeps track of how
    often the dataset is refined; see Stage X). If you wish to benchmark the performance of DeepLabCut,
    create multiple splits by specifying an integer value in the num_shuffles parameter.
    Each iteration of the creation of a training dataset will create a .mat file, which contains the
    address of the images as well as the target postures, and a .pickle file, which contains the meta
    information about the training dataset. This step also creates a directory for the model, including
    two subdirectories within ‘dlc-models’ called ‘test’ and ‘train’, each of which has a configuration file
    called pose_cfg.yaml. Specifically, you can edit the pose_cfg.yaml within the ‘train’ subdirectory
    before starting the training. These configuration files contain meta information to configure feature
    detectors, as well as the training procedure. Typically, these do not need to be edited for most
    applications
    
    
    """

    # ------------------------------------------------------------------
    # Stage VII: training the network
    # Step 9
    # ------------------------------------------------------------------
    """
    CRITICAL It is recommended to train for thousands of iterations (typically >100,000) until the loss
    plateaus. The variables display_iters and save_iters in the pose_cfg.yaml file allow the user to
    alter how often the loss is displayed and how often the (intermediate) weights are stored.
    """
    if "step_9" in stages_to_run or "training_network" in stages_to_run:
        deeplabcut.train_network(config_path, shuffle=1,trainingsetindex=0,
                                 max_snapshots_to_keep=5, displayiters=10000, saveiters=None, maxiters=None,
                                 allow_growth=False, gputouse=None, autotune=False, keepdeconvweights=True)  # gputouse=0
    # saveiters is at 50000 by default

    """
    During training, checkpoints are stored in the subdirectory ‘train’ under the respective iteration
    directory at user-specified iterations (‘save_iters’). If you wish to restart the training from a specific
    checkpoint, specify the full path of the checkpoint for the variable init_weights in the pose_cfg.
    yaml file under the ‘train’ subdirectory before starting to train
    """

    # ------------------------------------------------------------------
    # Stage VIII: evaluation of the trained network
    # Step 10
    # ------------------------------------------------------------------
    """
    It is important to evaluate the performance of the trained network. This performance is
    measured by computing the mean average Euclidean error (MAE; which is proportional to the average
    root mean square error) between the manual labels and the ones predicted by DeepLabCut. The MAE is
    saved as a comma-separated file and displayed for all pairs and only likely pairs (>p-cutoff). This
    helps to exclude, for example, occluded body parts. One of the strengths of DeepLabCut is that, owing to
    the probabilistic output of the scoremap, it can, if sufficiently trained, also reliably report whether a body
    part is visible in a given frame (see discussions of fingertips in reaching and the Drosophila legs during
    3D behavior in Mathis et al. 12 ).
    """
    if "step_10" in stages_to_run or "evaluate_network" in stages_to_run:
        deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=True, trainingsetindex=0,
                                    show_errors = True,comparisonbodyparts="all",gputouse=None, rescale=False)
    """
    Setting plotting to True plots all the testing and training frames with the manual and
    predicted labels. You should visually check the labeled test (and training) images that are created in
    the ‘evaluation-results’ directory. Ideally, DeepLabCut labeled the unseen (test) images according to
    your required accuracy, and the average train and test errors will be comparable (good
    generalization). What (numerically) constitutes an acceptable MAE depends on many factors
    (including the size of the tracked body parts and the labeling variability). Note that the test error
    can also be larger than the training error because of human variability in labeling (see Fig. 2 in
    Mathis et al. 12 ).
    
    If desired, customize the plots by editing the config.yaml file (i.e., the colormap, marker size
    (dotsize), and transparency of labels (alphavalue) can be modified). By default, each body part is
    plotted in a different color (governed by the colormap) and the plot labels indicate their source.
    Note that, by default, the human labels are plotted as a plus symbol (‘+’) and DeepLabCut’s
    predictions are plotted either as a dot (for confident predictions with likelihood >p-cutoff) or as
    an ‘x’ (for likelihood ≤p-cutoff). Example test and training plots from various projects are
    depicted in Fig. 5.
    
    The evaluation results for each shuffle of the training dataset are stored in a unique subdirectory in
    a newly created ‘evaluation-results’ directory in the project directory. You can visually inspect whether
    the distance between the labeled and the predicted body parts is acceptable. In the event of
    benchmarking with different shuffles of the same training dataset, you can provide multiple shuffle
    indices to evaluate the corresponding network. If the generalization is not sufficient, you might want to:
    
    ● Check if the body parts were labeled correctly, i.e., invisible points are not labeled and the points
    of interest are labeled accurately (Step 7);
    ● Make sure that the loss has already converged (Step 9; i.e., make sure the loss displayed has
    plateaued);
    ● Change the augmentation parameters (see Box 2);
    ● Consider labeling additional images and make another iteration of the training dataset (optional
    Stage X).
    """

    # ------------------------------------------------------------------
    # Stage IX: video analysis and plotting of results
    # Steps 11, 12, 13
    # ------------------------------------------------------------------
    """
    The trained network can be used to analyze new videos. The user needs to first choose
    a checkpoint with the best evaluation results for analyzing the videos. In this case, the user can specify
    the corresponding index of the checkpoint in the variable snapshotindex in the config.yaml file.
    By default, the most recent checkpoint (i.e., last) is used for analyzing the video.
    """
    # deeplabcut.analyze_videos(os.path.join(path_config, "config.yaml"),
    #                           [os.path.join(path_config, "test_2.avi")], videotype='avi', shuffle=1,
    #                           trainingsetindex=0, gputouse=0, save_as_csv=True, destfolder=None, cropping=None)
    if "step_11" in stages_to_run or "analyze_videos" in stages_to_run:
        deeplabcut.analyze_videos(config_path, videos_to_analyze,
                                  shuffle=1, save_as_csv=True, videotype='.avi', trainingsetindex=0,
                                  gputouse=None, destfolder=None, batchsize=None,
                                  cropping=None, get_nframesfrommetadata=True,
                                  TFGPUinference=True,
                                  dynamic=(False, .5, 10))
        # TODO: check if by putting save_as_csv to True, the .h5 file will still be created
    """
    The labels are stored in a multi-index Pandas array 43 , which contains the name of the network,
    body part name, (x, y) label position in pixels, and the likelihood for each frame per body part.
    These arrays are stored in an efficient HDF format in the same directory where the video is stored.
    However, if the flag save_as_csv is set to True, the data can also be exported in CSV format
    files, which in turn can be imported into many programs, such as MATLAB, R, and Prism; this flag
    is set to False by default. Instead of the video path, one can also pass a directory, in which case all
    videos of the type ‘videotype’ in that folder will be analyzed. For some projects, time-lapsed images
    are taken, for which each frame is saved independently. Such data can be analyzed with the function
    deeplabcut.analyze_time_lapse_frames.
    """

    """
    The labels for each body part across the video (‘trajectories’) can also be filtered and plotted after
    analyze_videos is run (which has many additional options (seedeeplabcut.filterpredictions?)).
    We also provide a function to plot the data overtime and pixels in frames. The provided plotting 
    function in this toolbox utilizes matplotlib 44 ; therefore, these plots can easily be customized. To call
    this function, type the following:

    The ouput files can also be easily imported into many programs for further behavioral analysis
    (see Stage XI and ‘Anticipated results’).
    """

    if "filterpredictions" in stages_to_run:
        deeplabcut.filterpredictions(config_path, videos_to_filter_predictions, videotype='avi', shuffle=1, trainingsetindex=0,
                                     filtertype='median', windowlength=5,
                                     p_bound=.001, ARdegree=3, MAdegree=1, alpha=.01,
                                     save_as_csv=True, destfolder=None)

    if "step_12" in stages_to_run or "plot_trajectories" in stages_to_run:
        deeplabcut.plot_trajectories(config_path, videos_to_plot_trajectories, videotype='.avi', shuffle=1,
                                     trainingsetindex=0, filtered=False, showfigures=False, destfolder=None)

    """
    In addition, the toolbox provides a function to create labeled videos based on the extracted poses by
    plotting the labels on top of the frame and creating a video. To use it to create multiple labeled
    videos (provided either as each video path or as a folder path), type the following:
    >> deeplabcut.create_labeled_video(config_path,[‘Full path of video 1’, ‘Full path of video 2’])
    This function has various parameters; in particular, the user can set the colormap, the
    dotsize, and the alphavalue of the labels in the config.yaml file, and can pass a variable called
    displayedbodyparts to select only a subset of parts to be plotted. The user can also save
    individual frames in a temp-directory by passing save_frames=True (this also creates a higher-
    quality video).
    """
    if "step_13" in stages_to_run or "create_labeled_video" in stages_to_run:
        deeplabcut.create_labeled_video(config_path, videos_to_create_with_labels, videotype='avi',
                                        shuffle=1 ,trainingsetindex=0, filtered=False, save_frames=False,
                                        Frames2plot=None, delete=False, displayedbodyparts='all', codec='mp4v',
                                        outputframerate=None, destfolder=None, draw_skeleton=False, trailpoints = 0,
                                        displaycropped=False)

    # --------------------------------------------------------------------
    # Stage X: (optional) network refinement—extraction of outlier frames
    # Step 14
    # --------------------------------------------------------------------
    """
    Although DeepLabCut typically generalizes well across datasets, one might want to optimize its
    performance in various, perhaps unexpected, situations. For generalization to large datasets, images
    with insufficient labeling performance can be extracted and manually corrected by adjusting the
    labels to increase the training set and iteratively improve the feature detectors. Such an active
    learning framework can be used to achieve a predefined level of confidence for all images with
    minimal labeling cost (discussed in Mathis et al. 12 ). Then, owing to the large capacity of the neural
    network that underlies the feature detectors, one can continue training the network with these
    additional examples. One does not necessarily need to correct all errors, as common errors can be
    eliminated by relabeling a few examples and then retraining. A priori, given that there is no ground
    truth data for analyzed videos, it is challenging to find putative ‘outlier frames’. However, one can
    use heuristics such as the continuity of body part trajectories to identify images where the decoder
    might make large errors. We provide various frame-selection methods for this purpose. In
    particular, the user can do the following:
    ● Select frames if the likelihood of a particular or all body parts lies below p
    bound (note this could
    also be due to occlusions rather than errors);
    ● Select frames in which a particular body part or all body parts jumped more than epsilon pixels
    from the last frame;
    ● Select frames if the predicted body part location deviates from a state-space model
    fit to the time series of individual body parts. Specifically, this method fits an AutoRegressive Integrated
    Moving Average (ARIMA) model to the time series for each body part. Thereby, each body part
    detection with a likelihood smaller than p bound is treated as missing data. An example fit for one
    body part can be found in Fig. 6a. Putative outlier frames are then identified as time points, at
    which the average body part estimates are at least epsilon pixels away from the fits. The
    parameters of this method are epsilon, p bound , the ARIMA parameters, and the list of body parts
    to consider (can also be ‘all’).
    """
    if "step_14" in stages_to_run or "extract_outlier_frames" in stages_to_run:
        deeplabcut.extract_outlier_frames(config_path, videos_to_extract_outlier_frames, videotype='avi',
                                          shuffle=1, trainingsetindex=0, outlieralgorithm='jump',
                                          comparisonbodyparts='all', epsilon=20, p_bound=.01, ARdegree=3,
                                          MAdegree=1, alpha=.01, extractionalgorithm='kmeans' ,automatic=False,
                                          cluster_resizewidth=30, cluster_color=False, opencv=True, savelabeled=True,
                                          destfolder=None)
    """
    This step has many parameters that can be set.
    
    In general, depending on the parameters, these methods might return many more frames than the
    user wants to extract (numframes2pick). Thus, this list is then used to select outlier frames
    either by randomly sampling from this list (‘uniform’) or by performing ‘k-means’ clustering on the
    corresponding frames (same methodology and parameters as in Step 4). Furthermore, before this
    second selection happens, you are informed about the number of frames satisfying the criteria and
    asked if the selection should proceed. This step allows you to perhaps change the parameters of the
    frame-selection heuristics first. The extract_outlier_frames command can be run
    iteratively and can (even) extract additional frames from the same video. Once enough outlier
    frames are extracted, use the refinement GUI to adjust the labels based on any user feedback.
    
    """

    # --------------------------------------------------------------------
    # Stage X continued: (optional) refinement of labels—augmentation of the training dataset
    # Steps 15, 16
    # --------------------------------------------------------------------
    """
    Based on the performance of DeepLabCut, four scenarios are possible:
    ● A visible body part with an accurate DeepLabCut prediction. These labels do not need any
    modification.
    ● A visible body part, but the wrong DeepLabCut prediction. Move the label’s location to the actual
    position of the body part.
    ● An invisible, occluded body part. Remove the predicted label by DeepLabCut with a right click.
    Every predicted label is shown, even when DeepLabCut is uncertain. This is necessary, so that
    the user can potentially move the predicted label. However, to help the user to remove all
    invisible body parts, the low-likelihood predictions are shown as open circles (rather than disks).
    ● Invalid image. in the unlikely event that there are any invalid images, the user should remove
    such images and their corresponding predictions, if any. Here, the GUI will prompt the user to
    remove an image identified as invalid.
    """
    if "step_15" in stages_to_run or "refine_labels" in stages_to_run:
        # open a gui
        deeplabcut.refine_labels(config_path, multianimal=False)
        """
        This will launch a GUI with which you can refine the labels (Fig. 6). Use the ‘Load Labels’ button to
        select one of the subdirectories where the extracted frames are stored. Each label will be identified
        by a unique color. To identify low-confidence labels, specify the threshold of the likelihood. This
        causes the body parts with likelihood below this threshold to appear as circles and the ones above
        the threshold to appear as solid disks while retaining the same color scheme. Next, to adjust the
        position of the label, hover the mouse over the label to identify the specific body part, then left-click
        it and drag it to a different location. To delete a label, right-click on the label (once a label is deleted,
        it cannot be retrieved).
        """

    """
    After correcting the labels for all the frames in each of the subdirectories, merge the dataset to
    create a new dataset. To do this, type the following:
    """
    if "step_16" in stages_to_run or "merge_datasets" in stages_to_run:
        deeplabcut.merge_datasets(config_path, forceiterate=None)
    """
    The iteration parameter in the config.yaml file will be automatically updated.
    Once the datasets are merged, you can test if the merging process was successful by
    plotting all the labels (Step 7). Next, with this expanded image set, you can now create a
    novel training set and train the network as described in Steps 8 and 9. The training dataset will be
    stored in the same place as before but under a different ‘iteration- #’ subdirectory, where the ‘#’ is
    the new value of iteration variable stored in the project’s configuration file (this is
    automatically done).
    If, after training, the network generalizes well to the data (i.e., run evaluate_network in
    Step 10), proceed to Step 11 to analyze new videos. Otherwise, consider labeling more data
    (optional Stage X).
    """

    # --------------------------------------------------------------------
    # Stage XI: working with the output files of DeepLabCut
    # Step 17
    # --------------------------------------------------------------------
    """
    Once you have a trained network with your desired level of performance (i.e., Steps 1–10 with/
    without optional refinement) and have used DeepLabCut to analyze videos (i.e., used the trained
    network snapshot weights to analyze novel videos by running Step 11), you will have files that
    contain the predicted x and y pixel coordinates of each body part and the network confidence
    likelihood (per frame). These files can then be loaded into many programs for further analysis. For
    example, you can compute general movement statistics (e.g., position, velocity) or interactions, e.g.,
    with (tracked) objects in the environment. The output from DeepLabCut can also interface with
    behavioral clustering tools such as JAABA 47 , MotionMapper 48 , an AR-HMM 49,50 , or other
    clustering approaches such as iterative denoising tree (IDT) methods 51,52 ; for a review, see Todd
    et al. 53 . In addition, users are contributing analysis code for the outputs of DeepLabCut here:
    https://github.com/AlexEMG/DLCutils.
    Beyond extracting poses from videos (Step 11), there are options to generate several plots
    (by running Step 12) and to generate a labeled video (by running Step 13). The following list
    summarizes the main result files and directories generated:
    ● Step 9 creates network weights, which are stored as snapshot-#.meta, snapshot-#.index and
    snapshot-#.data-00000-of-00001; these TensorFlow files contain the weights and can be used to
    analyze videos (Step 13). If the training was interrupted, it can be restarted from a particular
    snapshot (Box 2). These files are periodically saved in a subdirectory (‘train’) in the ‘dlc-models’
    directory. The ‘#’ in the filename specifies the training iteration index.
    ● Step 11 generates predicted DeepLabCut labels, which are stored as .HDF and/or .CSV files
    named <name_of_new_video>DeepCut_resnet<#>_<Task><date>shuffle<num_shuffles>_
    <snapshotindex>.h5; this file is saved in the same location where the video is located. This
    file contains the name of the network, body part names, x- and y-label positions in pixels, and
    the prediction likelihood for each body part per frame of the video. The filename is based on the
    name of the new video (<name_of_new_video>), the number of ResNet layers used (<#>), the
    task and date of the experiment (<Task> and <date>, respectively), shuffle index of the training
    dataset (<num_shuffle>), and the training snapshot file index (<snapshotindex) used for making
    the predictions.
    ● Step 12 generates several plots for an analyzed video. It creates a folder called ‘plot-poses’
    (in the directory of the video). The plots display the coordinates of body parts versus time,
    likelihoods versus time, the x versus y coordinates of the body parts, and histograms of
    consecutive coordinate differences. These plots help the user to quickly assess the tracking
    performance for a video. Ideally, the likelihood stays high and the histogram of consecutive
    coordinate differences has values close to zero (i.e., no jumps in body-part detections
    across frames).
    ● Step
    13 creates a labeled video that will be named <name_of_new_video>DeepCut_
    resnet<#>_<Task><date>shuffle<num_shuffle>_<snapshotindex>_labeled.mp4; This file is
    saved in the same location as that of the original video. The properties of the labels (color,
    dotsize, and others) can be changed in the config.yaml file (Step 3), and video compression can
    be used.
    """