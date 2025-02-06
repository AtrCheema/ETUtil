import os
import site
# add parent directory to path
et_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(et_dir)
site.addsitedir(et_dir)

import pandas as pd

from ETUtil import PenmanMonteith


# This example shows when solar radiation is not given then it can be calculated from temperature data.

fpath = "/mnt/datawaha/hyex/atr/datasets_1923_3359_austin_weather.csv"
# https://www.kaggle.com/grubenm/austin-weather?select=austin_weather.csv
df = pd.read_csv(fpath, na_values="-")
df.index = pd.to_datetime(df['Date'])
df.index.freq = pd.infer_freq(df.index)

df = df[['TempHighF', 'TempLowF', 'DewPointAvgF', 'HumidityAvgPercent', 'WindAvgMPH']]

df = df.rename(columns={'TempHighF': 'tmax',
                        'TempLowF': 'tmin',
                        'HumidityAvgPercent': 'rel_hum',
                        'DewPointAvgF': 'tdew',
                        'WindAvgMPH': 'wind_speed',
                        })

df = df.apply(pd.to_numeric)

units = {'tmin': 'Fahrenheit',
         'tmax': 'Fahrenheit',
         'rel_hum': 'percent',
         'tdew': 'Fahrenheit',
         'wind_speed': 'MilesPerHour'}

constants = dict()
constants['lat_dec_deg'] = 30.266666
constants['altitude'] = 305
# These values are not accurate
constants['a_s'] = 0.23
constants['albedo'] = 0.23
constants['b_s'] = 0.5
constants['wind_z'] = 2

eto_model = PenmanMonteith(df, units=units, constants=constants, verbosity=2)
pet = eto_model()
eto_model.plot_outputs()
eto_model.plot_inputs()
