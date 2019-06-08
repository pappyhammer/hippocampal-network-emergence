import numpy as np
import scipy.stats as scipy_stats
import scipy.spatial.distance as sci_sp_dist
from matplotlib import pyplot as plt
import seaborn as sns
import hdbscan
import hdf5storage
from sklearn.manifold import TSNE as t_sne
import pandas as pd
from pattern_discovery.tools.misc import get_continous_time_periods


def load_data():
    data_path = 'D:/Robin/data_hne/data/p9/p9_19_02_20_a002/predictions/' \
                'P9_19_02_20_a002_predictions_meso_v1_epoch_9.mat'
    data = hdf5storage.loadmat(data_path)
    predictions = data["predictions"]
    predictions[predictions >= 0.5] = 1
    spike_nums_dur = predictions
    # spike_nums_dur = np.load(data_path)
    return spike_nums_dur


def get_hamming_distance_matrix(spike_nums_dur):
    n_cells = spike_nums_dur.shape[0]
    hamm_dist_matrix = np.zeros((n_cells, n_cells))
    for i in np.arange(n_cells):
        for j in np.arange(n_cells):
            hamm_dist_matrix[i, j] = sci_sp_dist.hamming(spike_nums_dur[i, :], spike_nums_dur[j, :])
    return hamm_dist_matrix


def main():
    spike_nums_dur = load_data()
    hamm_dist_matrix = get_hamming_distance_matrix(spike_nums_dur)

    ## DO HDBSCAN CLUSTERING ON HAMMING DISTANCE MATRIX  ##

    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                metric='precomputed', min_cluster_size=3, min_samples=None, p=None)
    # metric='precomputed' euclidean
    clusterer.fit(hamm_dist_matrix)

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

    hamm_dist_matrix_order = np.copy(hamm_dist_matrix)
    labels_indices_sorted = np.argsort(labels)
    hamm_dist_matrix_order = hamm_dist_matrix_order[labels_indices_sorted, :]
    hamm_dist_matrix_order = hamm_dist_matrix_order[:, labels_indices_sorted]

    mean_corr_values = np.zeros(n_clusters)
    for i in np.arange(0, n_clusters - 1):
        tmp = hamm_dist_matrix_order[np.where(labels == i)[0], :]
        tmp = tmp[:, np.where(labels == i)[0]]
        mean_corr_values[i] = np.mean(tmp)
    print(f" {mean_corr_values}")
    # print(f"{np.where(mean_corr_values>0.6)}")
    # print(f"{np.max(mean_corr_values[np.where(n_epoch_by_cluster>5)])}")
    # print(f"{np.where(labels==7)}")
    # print(f"{tmp}")

    # Generate figure: correlation matrix ordered by cluster
    svm = sns.heatmap(hamm_dist_matrix_order)
    svm.set_yticklabels(labels_indices_sorted)
    svm.set_xticklabels(labels_indices_sorted)
    fig = svm.get_figure()

    save_formats = ["pdf"]
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    path_results = 'D:/Robin/data_hne/data/p9/p9_19_02_20_a002/predictions'
    for save_format in save_formats:
        fig.savefig(f'{path_results}/test_hamming'
                    f'.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())


main()
