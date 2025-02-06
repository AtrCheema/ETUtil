
import os
import site
# add parent directory to path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(wd_dir)
site.addsitedir(wd_dir)

import json
import unittest

import pandas as pd
import numpy as np
from SeqMetrics import RegressionMetrics

from ETUtil.et_methods import Kharrufa
from ETUtil import PenmanMonteith, Romanenko, Hamon, HargreavesSamani
from ETUtil import ChapmanAustralia, Turc, Linacre, Makkink, McGuinnessBordne
from ETUtil import Abtew, PriestleyTaylor, Penman, ETBase
from ETUtil import PenPan, MattShuttleworth, GrangerGray
from ETUtil import BrutsaertStrickler, SzilagyiJozsa, BlaneyCriddle


def get_daily_observed_data():
    _obs = pd.read_csv(os.path.join(wd_dir, 'data/obs.txt'))
    _obs.index = pd.to_datetime(_obs['index'])
    _obs.index.freq = pd.infer_freq(_obs.index)
    return _obs


observed = get_daily_observed_data()

# #**** DAILY TEST for PenmanMonteith
with open(os.path.join(wd_dir, 'data/constants.json'), 'r') as fp:
    constants = json.load(fp)

data = pd.read_csv(os.path.join(wd_dir, 'data/data.txt'), index_col=0, comment='#')
data.index = pd.to_datetime(data.index)
data.index.freq = pd.infer_freq(data.index)
units = {'tmin': 'Centigrade', 'tmax': 'Centigrade', 'sunshine_hrs': 'hour', 'rh_min': 'percent',
         'rh_max': 'percent', 'wind_speed': 'MeterPerSecond', 'tdew': 'Centigrade'}
# although 0.55 is used in original, but I do not divide by 100 in Hamon method so 0.0055 here
constants['cts'] = 0.0055
constants['pen_ap'] = 2.4
constants['pan_ap'] = 2.4
constants['turc_k'] = 0.013
constants['wind_f'] = 'pen48'
constants['albedo'] = 0.23
constants['a_s'] = 0.23
constants['b_s'] = 0.5
constants['abtew_k'] = 0.52
constants['ct'] = 0.025
constants['tx'] = 3
constants['pan_coeff'] = 0.71
constants['pan_over_est'] = False
constants['pan_est'] = 'pot_et'
constants['CH'] = 0.12
constants['Ca'] = 0.001013
constants['surf_res'] = 70
constants['alphaPT'] = 1.28


class ETTests(unittest.TestCase):

    def test_kharufa(self):
        eto_model = Kharrufa(data, units=units, constants=constants, verbosity=0)
        et = eto_model()
        return

    def test_PenmanMonteith(self):
        eto_model = PenmanMonteith(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'PenmanMonteith', 0.001)
        return

    def test_Romanenko(self):
        eto_model = Romanenko(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'Romanenko', 1e-6)
        return

    def test_HargreavesSamani(self):
        eto_model = HargreavesSamani(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'HargreavesSamani', 0.002)
        return

    def test_Hamon(self):
        eto_model = Hamon(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'Hamon', 1e-6)
        return

    def test_Turc(self):
        eto_model = Turc(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'Turc', 0.002)

    def test_ChapmanAustralia(self):
        eto_model = ChapmanAustralia(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'ChapmanAustralia', 0.03)
        return

    def test_Linacre(self):
        eto_model = Linacre(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'Linacre', 1e-6)
        return

    def test_Makkink(self):
        eto_model = Makkink(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'Makkink', 0.0009)
        return

    def test_McGuinnessBordne(self):
        eto_model = McGuinnessBordne(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'McGuinnessBordne', 0.002)
        return

    def test_Abtew(self):
        eto_model = Abtew(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'Abtew', 0.002)
        return

    def test_PriestleyTaylor(self):
        eto_model = PriestleyTaylor(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'PriestleyTaylor', 0.002)
        return

    def test_Penman(self):
        cons = constants.copy()
        cons['albedo'] = 0.08
        eto_model = Penman(data, units=units, constants=cons, verbosity=0)
        self.do_test(eto_model, 'Penman', 0.002)
        return

    def test_JensenHaise(self):
        eto_model = ETBase(data, units=units, constants=constants, verbosity=0)
        eto_model()
        errors = RegressionMetrics(observed['ET_' + 'JensenHaise' + '_Daily'],
                                   eto_model.output['et_' + 'ETBase' + '_Daily'])
        self.assertLess(errors.mae(), 0.002, 'JensenHaise Failling')
        return

    def test_PenPan(self):
        eto_model = PenPan(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'PenPan', 0.05)
        return

    def test_MattShuttleworth(self):
        eto_model = MattShuttleworth(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'MattShuttleworth', 0.002)
        return

    def test_GrangerGray(self):
        eto_model = GrangerGray(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'GrangerGray', 0.002)
        return

    def test_BrutsaertStrickler(self):
        eto_model = BrutsaertStrickler(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'BrutsaertStrickler', 0.003)
        return

    def test_SzilagyiJozsa(self):
        eto_model = SzilagyiJozsa(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'SzilagyiJozsa', 1.1)
        return

    def test_BlaneyCriddle(self):
        eto_model = BlaneyCriddle(data, units=units, constants=constants, verbosity=0)
        self.do_test(eto_model, 'BlaneyCriddle', 0.56)
        return

    def do_test(self, et_model, method: str, tol: float):
        et_model()
        errors = RegressionMetrics(observed['ET_' + method + '_Daily'],
                                   et_model.output['et_' + method + '_Daily'])
        self.assertLess(errors.mae(), tol, method + ' Failling')
        return

    def test_pm_daily(self):
        """
        Daily Penman-Monteith FAO56
        reproducing daily example from http://www.fao.org/3/X0490E/x0490e08.htm
        location:
        lat: 16.217 deg (15 48 N)
        """
        dr = pd.date_range('20110706 00:00', '20110708 23:00', freq='D')
        tmin = np.array([12.3, 12., 12.])
        tmax = np.array([21.5, 20, 20])
        rh_min = np.array([63.0, 63, 63])
        rh_max = np.array([84.0, 80, 89])
        uz = np.array([10.0, 10, 10])
        sunshine_hrs = np.array([9.25, 9, 9])
        cons = {'lat_dec_deg': 50.80,
                'altitude': 100.0,
                'a_s': 0.25,
                'wind_z': 10.0}
        df = pd.DataFrame(np.stack([tmin, tmax, sunshine_hrs, uz, rh_min, rh_max, ], axis=1),
                          columns=['tmin', 'tmax', 'sunshine_hrs', 'wind_speed', 'rh_min', 'rh_max'],
                          index=dr)
        _units = {'tmin': 'Centigrade', 'tmax': 'Centigrade',
                  'wind_speed': 'KiloMeterPerHour', 'sunshine_hrs': 'hour',
                  'rh_min': 'percent', 'rh_max': 'percent'}
        eto = PenmanMonteith(df, _units, constants=cons, verbosity=0)
        et_penman = eto()
        self.assertAlmostEqual(et_penman.iloc[0], 3.88, 2, "Daily PenmanMonteith Failling")
        return

    def test_pm_hourly(self):
        """Hourly Penman-Monteith FAO56
        # # reproducing hourly example from http://www.fao.org/3/X0490E/x0490e08.htm
        # location:
        # lat: 16.217 deg (16 13 N), long: -16.25 deg (16 15 W)"""
        dr = pd.date_range('20111001 02:00', '20111001 15:00', freq='h')
        uz = np.array([1.9 for _ in range(len(dr))])
        uz[-1] = 3.3
        temp = np.array([28 for _ in range(len(dr))])
        temp[-1] = 38
        rel_hum = np.array([90 for _ in range(len(dr))])
        rel_hum[-1] = 52
        sol_rad = np.array([0.45 for _ in range(len(dr))])
        sol_rad[-1] = 2.45
        df = pd.DataFrame(np.stack([uz, temp, rel_hum, sol_rad], axis=1),
                          columns=['wind_speed', 'temp', 'rel_hum', 'sol_rad'],
                          index=dr)

        cons = {'lat_dec_deg': 16.217,
                'altitude': 8.0,
                'long_dec_deg': -16.25}
        _units = {'wind_speed': 'MeterPerSecond', 'temp': 'Centigrade',
                  'sol_rad': 'MegaJoulePerMeterSquarePerHour',
                  'rel_hum': 'percent'}
        eto = PenmanMonteith(df, _units, constants=cons, verbosity=0)
        pet_penman = eto()
        self.assertAlmostEqual(pet_penman.iloc[-1], 0.6269, 2, "hourly PenmanMonteith Failling")
        return


if __name__ == "__main__":
    unittest.main()
