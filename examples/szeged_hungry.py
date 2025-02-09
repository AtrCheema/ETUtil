"""
=========================
Szeged Hungary
=========================
"""

import os
import site
# add parent directory to path
et_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(et_dir)
site.addsitedir(et_dir)

import pandas as pd

from ETUtil.et_methods import PenmanMonteith

# This should be a good example for missing data/interpolation of input data

# https://www.kaggle.com/budincsevity/szeged-weather?select=weatherHistory.csv
df = pd.read_csv(
    "/mnt/datawaha/hyex/atr/weatherHistory.csv", 
    #na_values="-", 
    delimiter=','
    )
idx_str = df['Formatted Date'].astype(str)

x = []
for i in idx_str:
    x.append(i[0:19])
df.index = pd.to_datetime(x)


df = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']]
df = df.rename(columns={'Temperature (C)': 'temp',
                        'Humidity': 'rel_hum',
                        'Wind Speed (km/h)': 'wind_speed',
                        })

df = df.apply(pd.to_numeric)

units = {'temp': 'Centigrade',
         'rel_hum': 'percent',
         'wind_speed': 'MilesPerHour'}

constants = dict()
constants['lat_dec_deg'] = 46.2529984
constants['altitude'] = 76
# These values are not accurate
constants['a_s'] = 0.23
constants['albedo'] = 0.23
constants['b_s'] = 0.5
constants['wind_z'] = 2

eto_model = PenmanMonteith(df, units=units, constants=constants, verbosity=2, calculate_at='60mins')
eto = eto_model()
