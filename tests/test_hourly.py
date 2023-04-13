import numpy as np
import pandas as pd
from SeqMetrics import RegressionMetrics
import unittest

from ETUtil.ETUtil.et_methods import PenmanMonteith

constants = {
    'altitude': 1208.5,
    'lat_dec_deg': 39.4575,
    'long_dec_deg': -118.77388,
    'wind_z': 3.0,
}

df = pd.DataFrame([[91.80, 49.36, 3.33, 2.56064, 78],
                   [91.80, 49.36, 3.33, 2.56064, 78],
                   [91.80, 49.36, 3.33, 2.56064, 78]],
                  columns=['temp', 'tdew', 'wind_speed', 'sol_rad', 'rel_hum'],
                  index=pd.date_range('20150701 11:00', '20150701 13:00', periods=3))

units = {'temp': 'Fahrenheit',
         'tdew': 'Fahrenheit',
         'wind_speed': 'MilesPerHour',
         'sol_rad': 'Mega_Joule_per_Meter_square'}


pm = PenmanMonteith(df, units=units, constants=constants)

et = pm()