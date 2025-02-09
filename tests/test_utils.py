import os
import site
# add parent directory to path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(wd_dir)
site.addsitedir(wd_dir)

import unittest

import numpy as np
import pandas as pd

from ETUtil.utils import freq_in_mins_from_string
from ETUtil.utils import Utils


for freq in ['1D', '20D', '5days',  'Daily', 'hourly',
             '5hours', '10H', '3Hour', 'min', 'minute',
             '6min', '6mins', '10minutes',  'Hourly', '3Day']:
    print(freq, freq_in_mins_from_string(freq))


data = pd.DataFrame({
    'tmax': [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    'tmin': [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    'rh_max': [60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
    'rh_min': [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    'wind_speed': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'sol_rad': [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    'tdew': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  
},
index=pd.date_range(start='1/1/2020', periods=10, freq='D')
)


units={'tmax': 'Centigrade',
       'tmin': 'Centigrade',
       'rh_max': 'percent',
       'rh_min': 'percent',
       'wind_speed': 'MeterPerSecond',
       'sol_rad': 'MegaJoulePerMeterSquarePerDay',
       'tdew': 'Centigrade',
       }

u = Utils(
    data =data,
        constants={'altitude': 1800, 'lat_dec_deg': 35.183},
        units=units,
)

class TestUtils(unittest.TestCase):

    def test_atm_pres(self):
        # Example 8 in Allen et al. (1998)
        atm_press = u.atm_pressure()

        np.testing.assert_almost_equal(atm_press, 81.8, 1)
        return

    def test_psy_constant(self):
        # Example 8 in Allen et al. (1998)
        psy = u.psy_const()

        np.testing.assert_almost_equal(psy, 0.054, 2)
        return

    def test_ea_from_rel_hum_min_max(self):
        # Example 5 in Allen et al. (1998)
        data = pd.DataFrame({
            'tmax': [25, 30, 30],
            'tmin': [18, 20, 20],
            'rh_max': [82, 60, 60],
            'rh_min': [54, 30, 30],
        },
        index=pd.date_range(start='1/1/2020', periods=3, freq='D')
        )

        u = Utils(
            data =data,
                constants={'altitude': 1800, 'lat_dec_deg': 35.183},
                units=units,
        )

        ea = u.avp_from_rel_hum()

        np.testing.assert_almost_equal(ea[0], 1.70, 2)

        return

    def test_ea_from_mean_rel_hum(self):
        # Example 5 in Allen et al. (1998)
        data = pd.DataFrame({
            'tmax': [25, 30, 30],
            'tmin': [18, 20, 20],
            'rel_hum': [68, 0.6, 0.6],
        },
        index=pd.date_range(start='1/1/2020', periods=3, freq='D')
        )

        u = Utils(
            data =data,
                constants={'altitude': 1800, 'lat_dec_deg': 35.183},
                units=units,
        )

        ea = u.avp_from_rel_hum()

        np.testing.assert_almost_equal(ea[0], 1.78, 2)

        return


if __name__ == "__main__":
    unittest.main()
