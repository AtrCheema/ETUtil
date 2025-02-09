"""
=================
Daily Methods
=================
"""

if __name__ == '__main__':
    import os
    import site
    # add parent directory to path
    et_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(et_dir)
    site.addsitedir(et_dir)

# %%
import pandas as pd

from ETUtil import PenmanMonteith

_df = pd.read_excel(os.path.join(et_dir, 'data/y_river.xlsx'))

df = pd.DataFrame()
df[['sol_rad', 'rel_hum', 'wind_speed', 'tmin', 'tmax']] = _df[['solar rad', 'humidity', 'wind', 'tmin', 'tmax']]
df.index = pd.to_datetime(_df['date'])

constants = {'lat_dec_deg': 35.34440, 'altitude': 300}
units = {'sol_rad': '', 'rel_hum': 'percentage', 'wind_speed': 'MeterPerSecond', 'tmin': 'Centigrade',
         'tmax': 'Centigrade'}

# following values are wild guess
constants['albedo'] = 0.23
constants['a_s'] = 0.23
constants['b_s'] = 0.5
constants['wind_z'] = 2

pm_pet = PenmanMonteith(df, units, constants)

et = pm_pet()
pm_pet.plot_outputs()
pm_pet.plot_inputs()
pm_pet.summary()
