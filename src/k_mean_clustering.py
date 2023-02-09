import numpy as np
from owid_data import get_data
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def k_mean_clustering_data_in(data, window_size, weights=None):
    data_shape = window_size * len(data.columns)

    data = data.groupby('iso_code').rolling(window=window_size, center=True)
    data = filter(lambda item: len(item) == window_size, data)

    index, columns = [], []
    for df in data:
        index.append(df.index[0] + (df.index[-1][1],))
        columns.append(df.values)
    columns = np.array(columns)

    if weights is not None:
        columns *= weights.reshape((1, 1, -1))

    index = pd.MultiIndex.from_tuples(index, names=['iso_code', 'date_start', 'date_end'])
    return index, columns.reshape((-1, data_shape))


def get_k_mean_results(columns, n_clusters):
    return KMeans(
        n_clusters=n_clusters,
        random_state=0,
    ).fit(columns)


def plot_k_mean_results(fig, columns, results, limit=25, ylim=None):
    n_clusters = len(results.cluster_centers_)
    n_rows = int(np.ceil(np.sqrt(n_clusters)))
    n_cols = int(np.ceil(np.sqrt(n_clusters)))
    for i in range(n_clusters):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.plot(results.cluster_centers_[i], color='orange', linewidth=3)
        indices, = np.where(results.labels_ == i)
        indices = np.random.choice(indices, limit)
        ax.plot(columns[indices].T, color='k', alpha=0.1)
        ax.set_ylim(ylim)


if __name__ == '__main__':
    import sys

    columns = [
        # 'new_cases_smoothed_per_million',
        # 'icu_patients_per_million',
        'new_deaths_smoothed_per_million',
    ]

    iso_codes = sys.argv[1:]
    print(f'{iso_codes=}')

    data = get_data(
        iso_codes=iso_codes,
        columns=columns,
        fill_small_gaps=True,
        max_gap_size=8,
        truncate_to_contiguous=True
    )


    window_size = 90
    n_clusters = 16
    index, columns = k_mean_clustering_data_in(data, window_size, weights=None)
    results = get_k_mean_results(columns, n_clusters)

    fig = plt.figure()
    plot_k_mean_results(fig, columns, results, ylim=[-1, 20])
    plt.show()
