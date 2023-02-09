import pandas as pd


fill_top = {
    'total_cases': 0.0,
    'new_cases': 0.0,
    'new_cases_smoothed': 0.0,
    'total_deaths': 0.0,
    'new_deaths': 0.0,
    'new_deaths_smoothed': 0.0,
    'total_cases_per_million': 0.0,
    'new_cases_per_million': 0.0,
    'new_cases_smoothed_per_million': 0.0,
    'total_deaths_per_million': 0.0,
    'new_deaths_per_million': 0.0,
    'new_deaths_smoothed_per_million': 0.0,
    # 'reproduction_rate': 1.0,
    'icu_patients': 0.0,
    'icu_patients_per_million': 0.0,
    'hosp_patients': 0.0,
    'hosp_patients_per_million': 0.0,
    'weekly_icu_admissions': 0.0,
    'weekly_icu_admissions_per_million': 0.0,
    'weekly_hosp_admissions': 0.0,
    'weekly_hosp_admissions_per_million': 0.0,
    'total_tests': 0.0,
    'new_tests': 0.0,
    'total_tests_per_thousand': 0.0,
    'new_tests_per_thousand': 0.0,
    'new_tests_smoothed': 0.0,
    'new_tests_smoothed_per_thousand': 0.0,
    'positive_rate': 0.0,
    'tests_per_case': 0.0,
    # 'tests_units': 0.0,
    'total_vaccinations': 0.0,
    'people_vaccinated': 0.0,
    'people_fully_vaccinated': 0.0,
    'total_boosters': 0.0,
    'new_vaccinations': 0.0,
    'new_vaccinations_smoothed': 0.0,
    'total_vaccinations_per_hundred': 0.0,
    'people_vaccinated_per_hundred': 0.0,
    'people_fully_vaccinated_per_hundred': 0.0,
    'total_boosters_per_hundred': 0.0,
    'new_vaccinations_smoothed_per_million': 0.0,
    'new_people_vaccinated_smoothed': 0.0,
    'new_people_vaccinated_smoothed_per_hundred': 0.0,
}
europe = ['AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE']


def _read_data(path):
    data = pd.read_csv(path, delimiter=',')
    data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
    data.drop(columns=['continent', 'location'], inplace=True)
    data.set_index(['iso_code', 'date'], inplace=True)
    return data


def _reindex_common_timeline(data):
    all_dates = data.index.get_level_values('date')
    start_date = all_dates.min()
    end_date = all_dates.max()
    new_index = pd.Index(pd.date_range(start=start_date, end=end_date), name='date')

    def _reindex(df):
        df.index = df.index.droplevel('iso_code')
        return df.reindex(new_index)

    data = data.groupby(level='iso_code', group_keys=True).apply(_reindex)
    return data


def _filter_by_iso_code(data, iso_codes):
    return data[data.index.isin(iso_codes, level='iso_code')]


def _fill_small_gaps(data, max_gap_size=8, inplace=False):
    half = max_gap_size // 2
    if 2 * half != max_gap_size:
        raise ValueError("The max gap size must be divisible by 2.")
    return data.interpolate(limit=half, limit_direction='both', inplace=inplace)


def _fill_column_top_by_iso_code(df, column, value):
    index = df[column].first_valid_index()
    df[column][:index] = value
    return df


def _fill_column_top(data, column, value):
    return data.groupby('iso_code', group_keys=False).apply(_fill_column_top_by_iso_code, column=column, value=value)


def _truncate_to_contiguous(data):
    nulls = data.isna().any(axis='columns')
    nulls = nulls.groupby('date').any()
    blocks = nulls[~nulls].groupby(nulls.cumsum())
    largest_block = blocks.get_group(blocks.size().agg('idxmax'))
    return data[data.index.isin(largest_block.index, level='date')]


def get_data(
        path='../data/owid-covid-data.csv',
        iso_codes=None,
        columns=None,
        fill_small_gaps=False,
        max_gap_size=8,
        truncate_to_contiguous=False):
    # read data from hard drive
    data = _read_data(path=path)

    # filter out some iso_codes
    if iso_codes is not None:
        data = _filter_by_iso_code(data, iso_codes=iso_codes)

    # filter out some columns
    if columns is not None:
        data = data[columns]

    # reindex such that all countries share the same time index
    data = _reindex_common_timeline(data)

    # replace the NAN values at the beggining of some columns
    for column, value in fill_top.items():
        if columns is None or column in columns:
            data = _fill_column_top(data, column, value)

    # fill gaps in the data (ie when a few days in a row are missing data)
    if fill_small_gaps:
        _fill_small_gaps(data, max_gap_size=max_gap_size, inplace=True)

    # get only valid data, leaving out NANs
    if truncate_to_contiguous:
        data = _truncate_to_contiguous(data)

    return data


if __name__ == '__main__':

    columns = [
        'new_cases_smoothed_per_million',
        'icu_patients_per_million',
        'new_deaths_smoothed_per_million',
    ]

    test = ['FRA', 'DEU', 'ESP']

    data = get_data(columns=columns, iso_codes=test, fill_small_gaps=True, truncate_to_contiguous=True)

    with pd.option_context('display.max_rows', None):
        print(data.iloc[:100])