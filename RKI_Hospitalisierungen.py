import pandas as pd
import geopandas as gpd


columns_ger_2_eng = {
    'Datum': 'date',
    'Bundesland': 'bundesland',
    'Bundesland_Id': 'bundesland_id',
    'Altersgruppe': 'age_group',
    'fixierte_7T_Hospitalisierung_Faelle': '7days_hospitalization_cases_fixed',
    'aktualisierte_7T_Hospitalisierung_Faelle': '7days_hospitalization_cases_updated',
    'PS_adjustierte_7T_Hospitalisierung_Faelle': '7days_hospitalization_cases_PS_adjusted',
    'UG_PI_adjustierte_7T_Hospitalisierung_Faelle': '7days_hospitalization_cases_UG_PI_adjusted',
    'OG_PI_adjustierte_7T_Hospitalisierung_Faelle': '7days_hospitalization_cases_OG_PI_adjusted',
    'Bevoelkerung': 'population',
    'fixierte_7T_Hospitalisierung_Inzidenz': '7days_hospitalization_incidence_fixed',
    'aktualisierte_7T_Hospitalisierung_Inzidenz': '7days_hospitalization_incidence_updated',
    'PS_adjustierte_7T_Hospitalisierung_Inzidenz': '7days_hospitalization_incidence_PS_adjusted',
    'UG_PI_adjustierte_7T_Hospitalisierung_Inzidenz': '7days_hospitalization_incidence_UG_PI_adjusted',
    'OG_PI_adjustierte_7T_Hospitalisierung_Inzidenz': '7days_hospitalization_incidence_OG_PI_adjusted',
}


def _read_data(path):
    data = pd.read_csv(path, delimiter=',')
    data.rename(columns=columns_ger_2_eng, inplace=True)
    data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
    data.drop(columns=['age_group'], inplace=True)
    data.set_index(['bundesland_id', 'bundesland', 'date'], inplace=True)
    return data


def _plot_one_column(ax, data, column_name, *args, **kwargs):
    ax.plot(data.index.get_level_values('date'), data[column_name], *args, **kwargs)


def plot_one_column(ax, data, column_name, *args, **kwargs):
    for bundesland, df in data.groupby('bundesland'):
        _plot_one_column(ax, df, column_name, label=bundesland)


def plot_map(ax):
    plz_shape_df = gpd.read_file('../data/plz-2stellig.shp', dtype={'plz': int})
    plz_shape_df.plot(ax=ax, categorical=True, alpha=0.8, cmap='tab20')

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = _read_data("../data/Aktuell_Deutschland_adjustierte-COVID-19-Hospitalisierungen.csv")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_one_column(ax, data, '7days_hospitalization_cases_fixed', linewidth=3)
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_map(ax)
    plt.legend()
    plt.show()