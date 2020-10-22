import numpy as np
import pandas as pd
from TSErrors import FindErrors
import unittest

from new_api.NewETUtil.et_methods import PenmanMonteith

constants = {
    'altitude': 1208.5,
    'lat_dec_deg': 39.4575,
    'long_dec_deg': -118.77388,
    'wind_z': 3.0,
}

df = pd.DataFrame([[91.80, 49.36, 3.33, 2.56064]],
                  columns=['temp', 'tdew', 'wind_speed', 'sol_rad'],
                  index=pd.date_range('20150701 11:00', '20150702 12:00', periods=1))

units = {'temp': 'Fahrenheit',
         'tdew': 'Fahrenheit',
         'wind_speed': 'Miles_per_hour',
         'sol_rad': 'Mega_Joule_per_Meter_square'}


