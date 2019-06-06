import numpy as np
import scipy.stats as scipy_stats
from matplotlib import pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.manifold import TSNE as t_sne
import pandas as pd
from pattern_discovery.tools.misc import get_continous_time_periods


def load_data():
    data_path = "D:/Robin/data_hne/data/p41/p41_19_04_30_a000/predictions/P41_19_04_30_a000_filtered_predicted_raster_dur_meso_v1_epoch_9.npy"
    spike_nums_dur = np.load(data_path)
    return spike_nums_dur


def build_spike_nums_and_peak_nums(spike_nums_dur):
    n_cells, n_frames = spike_nums_dur.shape
    spike_nums = np.zeros((n_cells, n_frames), dtype="int8")
    peak_nums = np.zeros((n_cells, n_frames), dtype="int8")
    for cell in np.arange(n_cells):
        transient_periods = get_continous_time_periods(spike_nums_dur[cell])
        for transient_period in transient_periods:
            onset = transient_period[0]
            peak = transient_period[1]
            # if onset == peak:
            #     print("onset == peak")
            spike_nums[cell, onset] = 1
            peak_nums[cell, peak] = 1
    return spike_nums, peak_nums


def firing_feature(spike_nums, window_length):
    n_cells, n_frames = spike_nums.shape
    n_epochs = n_frames // window_length
    print(f"n epochs is {n_epochs}")
    # to make things easy for now, the number of frames should be divisible by the length of epochs
    if (n_frames % window_length) != 0:
        raise Exception("number of frames {n_frames} not divisible by {len_epoch}")
    nb_activation = np.zeros((n_cells, n_epochs))
    for i in np.arange(n_epochs):
        nb_activation[:, i] = np.sum(spike_nums[:, np.arange(i*window_length, (i+1)*window_length)], axis=1)
    # print(f"{np.where(nb_activation)[0]}")
    return nb_activation


def get_pearson_correlation_matrix(nb_activation):
    n_epochs = nb_activation.shape[1]
    corr_matrix = np.zeros((n_epochs, n_epochs))
    for i in np.arange(n_epochs):
        for j in np.arange(n_epochs):
            corr_matrix[i, j] = scipy_stats.pearsonr(nb_activation[:, i], nb_activation[:, j])[0]
    return corr_matrix


def main():
    spike_nums_dur = load_data()
    spike_nums = build_spike_nums_and_peak_nums(spike_nums_dur)[0]
    nb_activation = firing_feature(spike_nums, 50)
    # print(f"nb_activation size is {nb_activation.shape}")
    corr_matrix = get_pearson_correlation_matrix(nb_activation)
    # svm = sns.heatmap(corr_matrix)
    # fig = svm.get_figure()
    # save_formats = ["pdf"]
    # if isinstance(save_formats, str):
    #     save_formats = [save_formats]
    #
    # path_results = "D:/Robin/data_hne/data/p41/p41_19_04_30_a000"
    # for save_format in save_formats:
    #     fig.savefig(f'{path_results}/test_batta'
    #                 f'.{save_format}',
    #                 format=f"{save_format}",
    #                 facecolor=fig.get_facecolor())


    ## DO HDBSCAN CLUSTERING ON CORRELATION MATRIX ( ACTIVATION FEATURE) ##

    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                metric='precomputed', min_cluster_size=3, min_samples=None, p=None)
    # metric='precomputed' euclidean
    clusterer.fit(corr_matrix)

    labels = clusterer.labels_
    # print(f"labels.shape: {labels.shape}")
    print(f"N clusters hdbscan: {labels.max()+1}")
    print(f"labels: {labels}")
    print(f"With no clusters hdbscan: {len(np.where(labels == -1)[0])}")
    n_clusters = 0
    if labels.max() + 1 > 0:
        n_clusters = labels.max() + 1

    if n_clusters > 0:
        n_epoch_by_cluster = [len(np.where(labels == x)[0]) for x in np.arange(n_clusters)]
        print(f"Number of epochs by clusters hdbscan: {' '.join(map(str, n_epoch_by_cluster))}")

    corr_matrix_order = np.copy(corr_matrix)
    labels_indices_sorted = np.argsort(labels)
    corr_matrix_order = corr_matrix_order[labels_indices_sorted, :]
    corr_matrix_order = corr_matrix_order[:, labels_indices_sorted]

    mean_corr_values = np.zeros(n_clusters)
    for i in np.arange(0, n_clusters - 1):
        tmp = corr_matrix_order[np.where(labels == i)[0], :]
        tmp = tmp[:, np.where(labels == i)[0]]
        mean_corr_values[i] = np.mean(tmp)
    print(f" {mean_corr_values}")
    # print(f"{np.where(mean_corr_values>0.6)}")
    # print(f"{np.max(mean_corr_values[np.where(n_epoch_by_cluster>5)])}")
    # print(f"{np.where(labels==7)}")
    # print(f"{tmp}")


    # Generate figure: correlation matrix ordered by cluster
    svm = sns.heatmap(corr_matrix_order)
    svm.set_yticklabels(labels_indices_sorted)
    svm.set_xticklabels(labels_indices_sorted)
    fig = svm.get_figure()

    save_formats = ["pdf"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    path_results = "D:/Robin/data_hne/data/p41/p41_19_04_30_a000/clawson_battaglia_paper"
    for save_format in save_formats:
        fig.savefig(f'{path_results}/test_hdbscan'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())

    ## DO T-SNE CLUSTERING ON CORRELATION MATRIX  ##

    tsne = t_sne(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(corr_matrix)

    # first figure: plot t-sne without color
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['color'] = labels
    plt.figure(figsize=(16, 10))
    svm = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        data=df_subset,
        legend="full",
        alpha=1
    )
    fig = svm.get_figure()

    path_results = "D:/Robin/data_hne/data/p41/p41_19_04_30_a000/clawson_battaglia_paper"
    for save_format in save_formats:
        fig.savefig(f'{path_results}/tsne_cluster'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()

    # second figure: plot t-sne with color from previous hdbscan result
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['color'] = labels

    plt.figure(figsize=(16, 10))
    svm = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="color",
        palette=sns.color_palette("hls", labels.max() + 2),
        data=df_subset,
        legend="full",
        alpha=1
    )
    fig = svm.get_figure()

    path_results = "D:/Robin/data_hne/data/p41/p41_19_04_30_a000/clawson_battaglia_paper"
    for save_format in save_formats:
        fig.savefig(f'{path_results}/tsne_colors_from_previous_hdbscan_clustering'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()

    # DO CLUSTERING ON T-SNE RESULTS TO COLOR THE T-SNE FIGURE ##

    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                metric='euclidean', min_cluster_size=3, min_samples=None, p=None)
    clusterer.fit(tsne_results)
    labels_hdbscan_on_tsne = clusterer.labels_
    print(f"N clusters hdbscan on t-sne results: {labels_hdbscan_on_tsne.max()+1}")
    # print(f"labels: {labels_hdbscan_on_tsne}")

    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['color'] = labels_hdbscan_on_tsne

    plt.figure(figsize=(16, 10))
    svm = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="color",
        palette=sns.color_palette("hls", labels_hdbscan_on_tsne.max() + 2),
        data=df_subset,
        legend="full",
        alpha=1
    )
    # plt.show()
    fig = svm.get_figure()

    path_results = "D:/Robin/data_hne/data/p41/p41_19_04_30_a000/clawson_battaglia_paper"
    for save_format in save_formats:
        fig.savefig(f'{path_results}/tsne_colors_from_post_tsne_clustering'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()


main()




