wget -P ./data/ https://covid.ourworldindata.org/data/owid-covid-data.csv
wget -P ./data/ https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv
wget -P ./data/ https://raw.githubusercontent.com/robert-koch-institut/COVID-19-Hospitalisierungen_in_Deutschland/master/Aktuell_Deutschland_adjustierte-COVID-19-Hospitalisierungen.csv

# https://juanitorduz.github.io/germany_plots/
# here are some cool viz of germany, plus some relevant data. To be investigated
wget -P ./data/ https://downloads.suche-postleitzahl.org/v2/public/plz-2stellig.shp.zip
unzip data/plz-2stellig.shp.zip -d ./data
wget -P ./data/ https://downloads.suche-postleitzahl.org/v2/public/plz-1stellig.shp.zip
unzip data/plz-1stellig.shp.zip -d ./data

wget -P ./data/ https://downloads.suche-postleitzahl.org/v2/public/plz_einwohner.csv
wget -P ./data/ https://downloads.suche-postleitzahl.org/v2/public/zuordnung_plz_ort.csv
