import deeplabcut

def main():

    #deeplabcut.create_new_project("test_2", "Robin", ["D:/Robin/test _movie_DLC/test_training/test_2.avi"], copy_videos=True)

    # deeplabcut.extract_frames("D:/Robin/test _movie_DLC/test_training/test_2-Robin-2019-06-03/config.yaml",
    #                           mode='automatic', algo='kmeans', crop=False, userfeedback=True, cluster_step=1,
    #                           cluster_resizewidth=30, cluster_color=False, opencv=True)

    # deeplabcut.label_frames("D:/Robin/test _movie_DLC/test_training/test_2-Robin-2019-06-03/config.yaml")

    # deeplabcut.check_labels("D:/Robin/test _movie_DLC/test_training/test_2-Robin-2019-06-03/config.yaml")

    # deeplabcut.create_training_dataset("D:/Robin/test _movie_DLC/test_training/test_2-Robin-2019-06-03/config.yaml",
    #                                    num_shuffles=1, Shuffles=None, windows2linux=False, trainIndices=None,
    #                                    testIndices=None)

    # deeplabcut.train_network("D:/Robin/test _movie_DLC/test_training/test_2-Robin-2019-06-03/config.yaml", shuffle=1,
    #                          trainingsetindex=0, gputouse=0, max_snapshots_to_keep=5, autotune=False,
    #                          displayiters=1000, saveiters=None, maxiters=None)

    # deeplabcut.evaluate_network("D:/Robin/test _movie_DLC/p11_test/test-Robin-2019-05-31/config.yaml", Shuffles=[1],
    #                             plotting=None, show_errors=True, comparisonbodyparts="all", gputouse=0)

    # deeplabcut.analyze_videos("D:/Robin/test _movie_DLC/p11_test/test-Robin-2019-05-31/config.yaml",
    #                           ["D:/Robin/test _movie_DLC/p11_test/p11_test.avi"], videotype='avi', shuffle=1,
    #                           trainingsetindex=0, gputouse=0, save_as_csv=False, destfolder=None, cropping=None)

    # deeplabcut.create_labeled_video("D:/Robin/test _movie_DLC/p11_test/test-Robin-2019-05-31/config.yaml",
    #                                 ["D:/Robin/test _movie_DLC/p11_test/p11_test.avi"],
    #                                 videotype='avi', shuffle=1, trainingsetindex=0, filtered=False, save_frames=False,
    #                                 Frames2plot=None, delete=False, displayedbodyparts='all', codec='mp4v',
    #                                 outputframerate=None, destfolder=None)

    deeplabcut.plot_trajectories("D:/Robin/test _movie_DLC/p11_test/test-Robin-2019-05-31/config.yaml",
                                 ["D:/Robin/test _movie_DLC/p11_test/p11_test.avi"], videotype='.avi', shuffle=1,
                                 trainingsetindex=0, filtered=False, showfigures=False, destfolder=None)


main()
