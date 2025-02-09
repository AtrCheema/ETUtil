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

    def test_et_rad(self):
        # Test Extra Terrestrial Radiation
        # Example 8 in Allen et al. (1998)
        data = pd.DataFrame({
            'tmax': [30, 30, 30],
        },
        index=pd.date_range(start='9/3/2021', periods=3, freq='D')
        )

        u = Utils(
            data =data,
                constants={'altitude': 1800, 'lat_dec_deg': -20.0},
                units=units,
        )

        ra = u._et_rad()
        np.testing.assert_almost_equal(ra[0], 32.2, 1)
        return

    def test_sol_rad_from_sun_hours(self):
        # solar radiation from measured duration of sunshine
        # Example 10 in Allen et al. (1998)
        data = pd.DataFrame({
            'sunshine_hrs': [7.1, 7.1, 7.1],
        },
        index=pd.date_range(start='5/15/2021', periods=3, freq='D')
        )

        u = Utils(
            data =data,
                constants={'a_s': 0.25, 'lat_dec_deg': -22.9, 'b_s': 0.50},
                units=units,
        )

        sol_rad = u.sol_rad_from_sun_hours()

        np.testing.assert_almost_equal(sol_rad.iloc[0], 14.5, 1)

        u.input['sol_rad'] = sol_rad
        evap = u.rad_to_evap()

        np.testing.assert_almost_equal(evap.iloc[0], 5.9, 1)

        return

    def test_net_out_lw_rad(self):
        # EXAMPLE 11. Determination of net longwave radiation
        data = pd.DataFrame({
            'tmax': [25.1, 30, 30],
            'tmin': [19.1, 20, 20],
            'sunshine_hrs': [7.1, 7.1, 7.1],
        },
        index=pd.date_range(start='5/15/2021', periods=3, freq='D')
        )

        u = Utils(
            data =data,
                constants={'a_s': 0.25, 'lat_dec_deg': -22.70, 'b_s': 0.50, 'altitude': 1800},
                units=units,
        )

        # rso = u._cs_rad(method='a_s')
        # ra = u._et_rad()
        lwd = u.net_out_lw_rad(ea=[2.1, 2.1, 2.1], rs=u.rs(), rso_method='a_s')

        np.testing.assert_almost_equal(lwd.iloc[0], 3.5, 1)

        return


if __name__ == "__main__":
    unittest.main()
