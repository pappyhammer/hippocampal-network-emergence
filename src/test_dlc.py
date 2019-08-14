import deeplabcut
import os

def main():

    dir_data = "test_2-Robin" + "-2019-07-19"
    path_config = f"/media/julien/Not_today/hne_not_today/tmp/{dir_data}/"
    # deeplabcut.create_new_project("test_2", "Robin", ["/media/julien/Not_today/hne_not_today/tmp/test_2.avi"],
    #                               copy_videos=True,
    #                               working_directory="/media/julien/Not_today/hne_not_today/tmp/")

    # deeplabcut.extract_frames(os.path.join(path_config, "config.yaml"),
    #                           mode='automatic', algo='kmeans', crop=False, userfeedback=True, cluster_step=1,
    #                           cluster_resizewidth=30, cluster_color=False, opencv=True)

    # deeplabcut.label_frames(os.path.join(path_config, "config.yaml"))

    # deeplabcut.check_labels(os.path.join(path_config, "config.yaml"))

    # deeplabcut.create_training_dataset(os.path.join(path_config, "config.yaml"),
    #                                    num_shuffles=1, Shuffles=None, windows2linux=False, trainIndexes=None,
    #                                    testIndexes=None)

    # deeplabcut.train_network(os.path.join(path_config, "config.yaml"), shuffle=1,
    #                          trainingsetindex=0, gputouse=0, max_snapshots_to_keep=5, autotune=False,
    #                          displayiters=1000, saveiters=None, maxiters=None)
    #
    # deeplabcut.evaluate_network(os.path.join(path_config, "config.yaml"), Shuffles=[1],
    #                             plotting=None, show_errors=True, comparisonbodyparts="all", gputouse=0)

    # deeplabcut.analyze_videos(os.path.join(path_config, "config.yaml"),
    #                           [os.path.join(path_config, "test_2.avi")], videotype='avi', shuffle=1,
    #                           trainingsetindex=0, gputouse=0, save_as_csv=True, destfolder=None, cropping=None)

    # deeplabcut.create_labeled_video(os.path.join(path_config, "config.yaml"),
    #                                 [os.path.join(path_config, "test_2.avi")],
    #                                 videotype='avi', shuffle=1, trainingsetindex=0, filtered=False, save_frames=False,
    #                                 Frames2plot=None, delete=False, displayedbodyparts='all', codec='mp4v',
    #                                 outputframerate=None, destfolder=None)

    deeplabcut.plot_trajectories(os.path.join(path_config, "config.yaml"),
                                 [os.path.join(path_config, "test_2.avi")], videotype='.avi', shuffle=1,
                                 trainingsetindex=0, filtered=False, showfigures=False, destfolder=None)


main()
