import pandas as pd
import numpy as np

from ETUtil.ETUtil.utils import Util


"""
Test for extra-terresterial radiation i.e. Rn calculation
reproducting example 8 from http://www.fao.org/3/X0490E/x0490e07.htm
"""
dr = pd.date_range('20110903 00:00', '20110903 23:59', freq='D')
sol_rad = np.array([0.45 ])
df = pd.DataFrame(np.stack([sol_rad],axis=1), columns=['solar_rad'], index=dr)
constants = {'lat' : -20             }
units={'solar_rad': 'MegaJoulePerMeterSquarePerHour'}
eto = Util(df,units,constants=constants, verbose=0)
ra = eto._et_rad()
np.testing.assert_almost_equal(ra, 32.17, 2, "extraterresterial radiation calculation failing")



"""
Test for daylight hours calculation
reproducing example 9 from http://www.fao.org/3/X0490E/x0490e07.htm
"""
dr = pd.date_range('20110903 00:00', '20110903 23:59', freq='H')
sol_rad = np.array([0.45 for _ in range(len(dr))])
df = pd.DataFrame(np.stack([sol_rad], axis=1), columns=['solar_rad'], index=dr)
constants = {'lat': -20}
units = {'solar_rad': 'MegaJoulePerMeterSquarePerHour'}
eto = Util(df, units, constants=constants, verbose=0)
N = np.unique(eto.daylight_fao56())
np.testing.assert_almost_equal(N, 11.669, 2, "Daylihght hours calculation failing")