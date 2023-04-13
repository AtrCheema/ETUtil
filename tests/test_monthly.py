
import numpy as np
import pandas as pd
from SeqMetrics import RegressionMetrics
import unittest

from ETUtil import Thornthwait


temp = np.array([2.1, 2.5, 4.8, 7.1, 8.3, 10.7,
                 13.4, 14.5, 11.1, 8.2, 5.4, 3.7])
dl_hrs = np.array([9.4, 10.6, 11.9, 13.4, 14.6, 15.2,
                   14.9, 13.9, 12.6, 11.1, 9.8, 9.1])

obs = [10.9558, 13.0699, 29.3479, 45.608, 59.159, 75.0385,
       93.2112, 93.403, 64.307, 44.4857, 26.0252, 17.7248]

units = {'temp': 'Centigrade', 'daylight_hrs': 'hour'}

constants = {'lat_dec_deg': 20.0,
             'wind_z': 2}


class TestThornthwaite(unittest.TestCase):

    def test_thornthwaite(self):

        df = pd.DataFrame(np.stack([temp, dl_hrs]).transpose(),
                          columns=['temp', 'daylight_hrs'],
                          index=pd.date_range('20110101', periods=12, freq='M'))

        thornthwaite = Thornthwait(df, units, constants)
        pet = thornthwaite()

        errors = RegressionMetrics(obs, pet)
        self.assertAlmostEqual(errors.mae(), 0.00018, 2, "Thornthwaite Failling")

        # check for leap year
        df = pd.DataFrame(np.stack([temp, dl_hrs]).transpose(),
                          columns=['temp', 'daylight_hrs'],
                          index=pd.date_range('20120101', periods=12, freq='M'))

        thornthwaite = Thornthwait(df, units, constants)
        pet_leap = thornthwaite()

        idx = 0
        for m, n in zip(pet, pet_leap):
            if idx == 1:  # February of leap year
                self.assertGreater(n, m)
            else:
                self.assertEqual(m, n)
            idx += 1


if __name__ == "__main__":
    unittest.main()
