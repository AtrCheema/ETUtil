from ETUtil.ETUtil import ReferenceET

import pandas as pd


df = pd.read_csv( '../data/jen_obj.txt', na_values=-9999.0)
df.index = pd.to_datetime(df['index'])
df.pop('index')
df.index.freq = pd.infer_freq(df.index)


print(df.columns)
print(type(df.index), df.index.freq, df.shape)


df.columns = ['temp', 'rel_hum', 'tdew', 'es', 'ea', 'vp_def', 'uz', 'rain', 'solar_rad', 'rn']
df['es'] = df['es'] * 0.1
df['ea'] = df['ea'] * 0.1
df['vp_def'] = df['vp_def'] * 0.1
rad_fac = (60*10) / 1e6
df['solar_rad'] = df['solar_rad'] * rad_fac
df['rn'] = df['rn'] * rad_fac
rain = df.pop('rain')

df = df["20021218":]       # initial wind data contains nan values
print(df.isna().sum())

df = df.interpolate()
print(df.isna().sum())

constants = {'altitude': 143,
             'lat': 50.92722222,
            'long': 11.419
}

units={'temp': 'centigrade', 'rel_hum':'percent','uz':'MeterPerSecond', 'tdew':'centigrade',
       'es': 'KiloPascal', 'ea': 'KiloPascal', 'vp_def': 'KiloPascal',
       'solar_rad': 'MegaJoulePerMeterSquarePerHour', 'rn': 'MegaJoulePerMeterSquare'}


etp = ReferenceET(df, units,
                  constants=constants)

etp.Penman()