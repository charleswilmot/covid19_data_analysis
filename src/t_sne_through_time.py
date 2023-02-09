import os
from owid_data import get_data
import numpy as np
from sklearn.manifold import TSNE
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial


warnings.filterwarnings(action='ignore', category=FutureWarning)


def t_sne_through_time_data_in(data, window_size, weights=None):
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


def get_normalization_weights(data):
    return 1. / data.max().values


def t_sne_through_time_data_out(index, columns, perplexity):
    embedding = TSNE(
        n_components=2,
        init='pca',
        perplexity=perplexity,
    ).fit_transform(columns)

    results = pd.DataFrame(
        data={
            "component_1": embedding[:, 0],
            "component_2": embedding[:, 1],
        },
        index=index,
    )
    return results


def plot_t_sne_through_time(ax, results):
    iso_codes = tuple(sorted(results.index.unique(level='iso_code')))

    for iso_code in iso_codes:
        ax.scatter(
            results.loc[iso_code, 'component_1'],
            results.loc[iso_code, 'component_2'],
            label=iso_code
        )

    for _, df in results.groupby('date_start'):
        ax.plot(
            df['component_1'],
            df['component_2'],
            color='k',
            alpha=0.1
        )

    ax.legend()



def t_sne_through_time_interactive(data, window_size, perplexity):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    weights = get_normalization_weights(data)
    index, columns = t_sne_through_time_data_in(data, window_size, weights)
    results = t_sne_through_time_data_out(index, columns, perplexity)
    plot_t_sne_through_time(ax, results)

    index = 0
    by_date_start = results.groupby('date_start')
    dates = results.index.unique(level='date_start')
    line, = ax.plot(
        by_date_start.get_group(dates[index])['component_1'],
        by_date_start.get_group(dates[index])['component_2'],
        color='r',
        marker='o',
        linewidth=5,
        markersize=15,
    )
    date_label = ax.set_xlabel(str(dates[index]))

    @partial(fig.canvas.mpl_connect, 'key_press_event')
    def on_press(event):
        nonlocal index
        if event.key == 'right':
            index += 1
        elif event.key == 'left':
            index -= 1
        elif event.key == 'pageup':
            index += 10
        elif event.key == 'pagedown':
            index -= 10
        if index < 0: index = 0
        if index >= len(dates): index = len(dates) - 1
        X = by_date_start.get_group(dates[index])['component_1']
        Y = by_date_start.get_group(dates[index])['component_2']
        line.set_data(X, Y)
        date_label.set_text(str(dates[index]))
        fig.canvas.draw()

    plt.show()
    plt.close(fig)


# def t_sne_through_time_movie(data, window_size, perplexity):
#     iso_codes = tuple(sorted(data.index.unique(level='iso_code')))
#     filename = '_'.join(iso_codes) + f'_WS{window_size}_P{perplexity}'
#     os.mkdir(f"../plots/t_sne_through_time/movie_{filename}.png")

#     fig = plt.figure(figsize=(10, 5), dpi=120)
#     ax_tsne = fig.add_subplot(121)
#     ax_timeline = fig.add_subplot(122)

#     results = t_sne_through_time(ax_tsne, data, window_size, perplexity)

#     index = 0
#     by_date_start = results.groupby('date_start')
#     dates_start = results.index.unique(level='date_start')
#     dates_end = results.index.unique(level='date_end')
#     line, = ax_tsne.plot(
#         by_date_start.get_group(dates_start[index])['component_1'],
#         by_date_start.get_group(dates_start[index])['component_2'],
#         color='r',
#         marker='o',
#         linewidth=5,
#         markersize=15,
#     )
#     date_label = ax_tsne.set_xlabel(str(dates_start[index]))

#     by_iso_codes = data.groupby('iso_code')
#     for iso_code in iso_codes:
#         color = None
#         df = by_iso_codes.get_group(iso_code)
#         X = df.index.get_level_values('date')
#         Ys = df.values.T
#         labels = [f'{iso_code}_{column}' for column in data.columns]
#         for Y, ls, label in zip(Ys, ['-', ':', '--', '-.'], labels):
#             timeline, = ax_timeline.plot(X, Y, label=label, linestyle=ls, color=color)
#             color = timeline.get_color()

#     span = ax_timeline.axvspan(dates_start[index], dates_end[index], alpha=0.1, color='k')
#     ax_timeline.xaxis.set_tick_params(rotation=25)
#     ax_timeline.legend()

#     for index in range(len(dates_start)):
#         print(index, '/', len(dates_start))
#         #
#         X = by_date_start.get_group(dates_start[index])['component_1']
#         Y = by_date_start.get_group(dates_start[index])['component_2']
#         line.set_data(X, Y)
#         date_label.set_text(str(dates_start[index]))
#         #
#         s, e = float(dates_start[index].value), float(dates_end[index].value)
#         t, b = 1000.0, -1000.0
#         span.set_xy(np.array([[s, b], [s, t], [e, t], [e, b], [s, b]]))
#         #
#         fig.canvas.draw()
#         fig.savefig(f"../plots/t_sne_through_time/movie_{filename}.png/{index:04d}_{filename}.png")

#     plt.close(fig)


def t_sne_through_time_movie(data, window_size, perplexity):
    iso_codes = tuple(sorted(data.index.unique(level='iso_code')))
    filename = '_'.join(iso_codes) + f'_WS{window_size}_P{perplexity}'
    os.mkdir(f"../plots/t_sne_through_time/movie_{filename}.png")

    fig = plt.figure(figsize=(10, 5), dpi=120)

    weights = get_normalization_weights(data)
    index, columns = t_sne_through_time_data_in(data, window_size, weights)
    results = t_sne_through_time_data_out(index, columns, perplexity)

    by_date_start = results.groupby('date_start')
    by_iso_codes = data.groupby('iso_code')
    dates_start = results.index.unique(level='date_start')
    dates_end = results.index.unique(level='date_end')

    def _plot_frame(index):
        print(f'{index: 4d} / {len(dates_start)}')

        fig.clf()
        ax_tsne = fig.add_subplot(121)
        ax_timeline = fig.add_subplot(122)

        plot_t_sne_through_time(ax_tsne, results)
        ax_tsne.plot(
            by_date_start.get_group(dates_start[index])['component_1'],
            by_date_start.get_group(dates_start[index])['component_2'],
            color='r',
            marker='o',
            linewidth=5,
            markersize=15,
        )
        ax_tsne.set_xlabel(str(dates_start[index]))

        for iso_code in iso_codes:
            color = None
            df = by_iso_codes.get_group(iso_code)
            X = df.index.get_level_values('date')
            Ys = df.values.T
            labels = [f'{iso_code}_{column}' for column in data.columns]
            for Y, ls, label in zip(Ys, ['-', ':', '--', '-.'], labels):
                timeline, = ax_timeline.plot(X, Y, label=label, linestyle=ls, color=color)
                color = timeline.get_color()

        ax_timeline.axvspan(dates_start[index], dates_end[index], alpha=0.1, color='k')
        ax_timeline.xaxis.set_tick_params(rotation=25)
        ax_timeline.legend()

        fig.savefig(f"../plots/t_sne_through_time/movie_{filename}.png/{index:04d}_{filename}.png")

    for index in range(len(dates_start)):
        _plot_frame(index)

    plt.close(fig)



def t_sne_through_time_batch(columns, iso_codes_list, window_sizes, perplexities):
    for iso_codes in iso_codes_list:
        data = get_data(
            iso_codes=iso_codes,
            columns=columns,
            fill_small_gaps=True,
            max_gap_size=8,
            truncate_to_contiguous=True
        )

        for window_size in window_sizes:
            for perplexity in perplexities:
                print('generating t-sne with params: ', iso_codes, window_size, perplexity)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plot_t_sne_through_time(ax, data, window_size, perplexity)
                filename = '_'.join(iso_codes) + f'_WS{window_size}_P{perplexity}'
                fig.savefig(f"../plots/t_sne_through_time/{filename}.png")
                plt.close(fig)


def batch_1():
    columns = [
        # 'new_cases_smoothed_per_million',
        'icu_patients_per_million',
        'new_deaths_smoothed_per_million',
    ]

    iso_codes_list = [
        ['FRA', 'DEU'],
        ['FRA', 'BEL'],
        ['FRA', 'OWID_EUR'],
        ['FRA', 'OWID_EUN'],
        ['USA', 'OWID_EUN'],
        ['FRA', 'DEU', 'OWID_EUN']
    ]

    window_sizes = [7, 14, 21, 28]
    perplexities = [30, 50]
    t_sne_through_time_batch(columns, iso_codes_list, window_sizes, perplexities)



if __name__ == '__main__':
    import sys

    columns = [
        # 'new_cases_smoothed_per_million',
        'icu_patients_per_million',
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

    t_sne_through_time_movie(data, window_size=28, perplexity=30)
