import pandas as pd
import numpy as np


ERROR_CANNOT_TRUNCATE = "Cannot truncate to a common timeline (some countries have non-contiguous dates)"
ERROR_NO_WINDOW_OVERLAP = "Cannot truncate to a common timeline (the intersection of all timelines is empty)"

europe = ['AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE']

data = pd.read_csv('../data/owid-covid-data.csv', delimiter=',')
data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
# with pd.option_context('display.max_rows', None):
#     print(data.dtypes)



def check_data_contiguous(data):
    def time_window_is_contiguous(window):
        return (max(window) - min(window)).days == len(window) - 1

    data_by_iso_code = data.groupby(by='iso_code')
    data_contiguous = data_by_iso_code.aggregate(
        func={'date': time_window_is_contiguous},
    )
    data_contiguous = data_contiguous.rename(
        columns={'date': 'dates_are_contiguous'},
    )
    return data_contiguous


def make_dates_contiguous(data):
    data_by_iso_code = data.groupby(by='iso_code')
    start_date = data['date'].min()
    end_date = data['date'].max()
    new_index = pd.date_range(start=start_date, end=end_date)

    def _reindex(df):
        return df.set_index('date').reindex(new_index).reset_index(names='date')

    return data_by_iso_code.apply(_reindex)


def filter_contiguous_only(data):
    data_contiguous = check_data_contiguous(data)
    valid_iso_codes = data_contiguous.index[data_contiguous['dates_are_contiguous']]
    return data[data['iso_code'].isin(valid_iso_codes)]


def truncate_to_common_time(data):
    if not check_data_contiguous(data).all().bool():
        raise ValueError(ERROR_CANNOT_TRUNCATE)

    data_by_iso_code = data.groupby(by='iso_code')
    data_start_end = data_by_iso_code.aggregate(
        func={'date': ['min', 'max']},
    )
    window_start = data_start_end['date']['min'].max()
    window_end = data_start_end['date']['max'].min()

    if window_end < window_start:
        raise ValueError(ERROR_NO_WINDOW_OVERLAP)

    return data[(data['date'] >= window_start) & (data['date'] <= window_end)]


def filter_by_iso_code(data, iso_codes):
    return data[data['iso_code'].isin(iso_codes)]


def plot_column_by_iso_code(ax, data, column):
    data = data.filter(items=['iso_code', 'date', column])

    for iso_code, df in data.groupby('iso_code'):
        ax.plot(df['date'], df[column], label=iso_code)

    ax.set_title(column)
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel(column)


def fill_small_gaps(data, max_gap_size, columns=None, inplace=False):

    def _fill_small_gaps_series(series):
        if not pd.api.types.is_float_dtype(series):
            return series
        index_nans = [i for i, v in enumerate(series) if np.isnan(v)]
        while len(index_nans):
            current = index_nans[0]
            cursor = current + 1

            # increasingly
            while cursor in index_nans:
                cursor += 1
            end = cursor

            cursor = current

            # decreasingly
            while cursor in index_nans:
                cursor -= 1
            start = cursor + 1

            if end - start <= max_gap_size and start != 0 and end != len(series):
                series.iloc[start:end] = np.linspace(
                    series.iloc[start - 1],
                    series.iloc[end],
                    end - start + 2
                )[1:-1]

            for i in range(start, end): index_nans.remove(i)

        return series


    def _fill_small_gaps_df(df):
        # df represents the data for one country (based on iso_code)
        df = df.sort_values('date')
        if columns is None:
            df = df.apply(_fill_small_gaps_series)
        else:
            df[columns] = df[columns].apply(_fill_small_gaps_series)
        return df


    if not inplace:
        data = data.copy()

    return data.groupby('iso_code', group_keys=False).apply(_fill_small_gaps_df)


# def truncate_to_common_nonnan(data):
#     float_columns = [col for col in data.columns if pd.api.types.is_float_dtype(data[col])]
#     pass


def test_1():
    data = filter_by_iso_code(data, europe)
    data = filter_contiguous_only(data)
    data = truncate_to_common_time(data)
    data = fill_small_gaps(data, max_gap_size=10)


    import matplotlib.pyplot as plt


    fig, ax = plt.subplots()
    # plot_column_by_iso_code(ax, data, 'new_deaths_smoothed')
    # plot_column_by_iso_code(ax, data, 'new_cases_smoothed_per_million')
    # plot_column_by_iso_code(ax, data, 'new_deaths_smoothed_per_million')
    # plot_column_by_iso_code(ax, data, 'icu_patients_per_million')
    # plot_column_by_iso_code(ax, data, 'weekly_icu_admissions_per_million')


    plot_column_by_iso_code(ax, data, 'people_fully_vaccinated_per_hundred')
    # plot_column_by_iso_code(ax, data, 'new_people_vaccinated_smoothed_per_hundred')
    # plot_column_by_iso_code(ax, data, 'total_deaths_per_million')
    plt.show()


def test_2():
    global data
    data = make_dates_contiguous(data)
    print(data.index)


test_2()