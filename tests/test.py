
import json
import random
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ETUtil import Hamon, JensenHaiseBasins, PenmanMonteith, Thornthwait
from ETUtil import BlaneyCriddle, Abtew, BrutsaertStrickler, Romanenko
from ETUtil import PriestleyTaylor, MattShuttleworth, Penman, PenPan
from ETUtil import McGuinnessBordne, Makkink, HargreavesSamani, ETBase
from ETUtil import Linacre, Turc, SzilagyiJozsa, GrangerGray, ChapmanAustralia


class TestMisc(unittest.TestCase):

    def test_hamon(self):
        """
        Daily Hamon
         """
        tmin =  np.array([27.3, 25.9, 26.1, 26.1, 24.6, 18.1, 19.9, 22.6, 16.0, 18.9])
        tmax =  np.array([47.8, 38.5, 38.5, 42.6, 36.1, 37.4, 43.5, 38.3, 36.7, 39.7 ])
        dr = pd.date_range('20110101', '20110110', freq='D')
        df = pd.DataFrame(np.stack([tmin,tmax, ],axis=1),
                           columns=['tmin', 'tmax'],
                          index=dr)
        lat = 45.2  # 45 12 15
        units={'tmin':'fahrenheit', 'tmax':'fahrenheit'}
        eto = Hamon(df, units,
                    constants=dict(lat_dec_deg=lat, altitude=20, cts=1.2))
        pet_hamon = eto()
        return

    def test_blaney_cridle(self):
        # # Blaney-Criddle test
        # # http://www.fao.org/3/S2022E/s2022e07.htm#3.1.3%20blaney%20criddle%20method
        latitude = -35.0
        tmin =  np.array([19.4, 20.4, 20])
        tmax =  np.array([29.5, 30.5, 30])
        dr = pd.date_range('20110401', '20110701', freq='M')
        df = pd.DataFrame(np.stack([tmin,tmax, ],axis=1),
                            columns=['tmin', 'tmax'],
                           index=dr)
        units={'tmin':'fahrenheit', 'tmax':'fahrenheit'}
        eto = BlaneyCriddle(df,
                            units,
                            constants=dict(lat_dec_deg=latitude,
                                           e0=2,
                                           e2=2,
                                           e3=2,
                                           e4=2,
                                           e1=2))
        #eto()
        return

    def test_thornthwait(self):
        """ Thornthwaite
        Sellers (1969), Physical Climatology, pg 173,
        https://www.ncl.ucar.edu/Document/Functions/Built-in/thornthwaite.shtml"""
        temp = np.array([23.3, 21.1, 19.6, 17.2, 12.6, 10.9,
                         10. , 11. , 13. , 15.8, 17.8,  20.1])
        dr = pd.date_range('20110101', '20111231', freq='M')
        df = pd.DataFrame(temp ,  columns=['temp'], index=dr)
        df.index = pd.to_datetime(df.index)
        units = {'temp': 'centigrade', 'daylight_hrs':'hour'}
        constants = {'lat_dec_deg' : -38.0}
        etp = Thornthwait(df, units, constants=constants)
        pet = etp()
        return

    def test_jensenHaise(self):
        """
        Jensen and Haise
        """
        tmin = np.array([27.3,      25.9,      26.1,      26.1,      24.6,
                         18.1,      19.9,      22.6,      16.0,    18.9])
        tmax = np.array([47.8,      38.5,      38.5,      42.6,      36.1,
                         37.4,      43.5,      38.3,      36.7,   39.7])
        sol_rad = np.array([256.,     306.,     193.,     269.,     219.,
                            316.,     318.,     320.,    289.,     324.])
        dr = pd.date_range('20110101', '20110110', freq='D')
        df = pd.DataFrame(np.stack([tmin, tmax, sol_rad], axis=1),
                          columns=['tmin', 'tmax', 'sol_rad'],
                          index=dr)

        Units = {'tmin': 'Centigrade', 'tmax': 'Centigrade', 'solar_rad': 'LangleysPerDay'}
        etp = JensenHaiseBasins(
            df,
            units=Units,
            constants={'lat_dec_deg': 24.0, 'altitude': 100, 'cts_jh': 0.3, 'ctx_jh': 0.3})
        pet = etp()
        o = np.array([0.159, 0.165, 0.104, 0.153, 0.112,
                      0.149, 0.169, 0.164, 0.130, 0.160])
        # np.testing.assert_array_almost_equal(
        # etp.output['ET_JensenHaiseBASINS_Daily'].values.reshape(-1,), o, 3)
        return

    def test_daily_penmanMonteith(self):
        """
        Daily Penman-Monteith FAO56
        reproducing daily example from http://www.fao.org/3/X0490E/x0490e08.htm
        location:
        lat: 16.217 deg (15 48 N)
         """
        dr = pd.date_range('20110706 00:00', '20110708 23:00', freq='D')
        tmin = np.array([12.3, 12, 12])
        tmax = np.array([21.5, 21, 21])
        rh_min = np.array([63.0, 63, 63])
        rh_max = np.array([84.0, 84, 84])
        uz = np.array([10.0, 10., 10.])
        sunshine_hrs = np.array([9.25, 9, 9])
        constants = {'lat_dec_deg': 50.80,
                     'altitude': 100.0,
                     'a_s': 0.25,
                     'wind_z': 10.0}
        df = pd.DataFrame(
            np.stack([tmin, tmax, sunshine_hrs, uz, rh_min, rh_max, ], axis=1),
            columns=['tmin', 'tmax', 'sunshine_hrs', 'wind_speed', 'rh_min', 'rh_max'],
            index=dr)
        units = {'tmin': 'Centigrade', 'tmax': 'Centigrade',
                 'wind_speed': 'KiloMeterPerHour', 'sunshine_hrs': 'hour',
                 'rh_min': 'percent', 'rh_max': 'percent'}
        eto = PenmanMonteith(df, units, constants=constants, verbosity=0)
        et_penman = eto()
        np.testing.assert_almost_equal(
            et_penman[0], 3.88, 2, "Daily PenmanMonteith Failling")
        return

    def test_hourly_pm(self):
        """Hourly Penman-Monteith FAO56
        # # reproducing hourly example from http://www.fao.org/3/X0490E/x0490e08.htm
        # location:
        # lat: 16.217 deg (16 13 N), long: -16.25 deg (16 15 W)"""
        dr = pd.date_range('20111001 02:00', '20111001 15:00', freq='H')
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

        constants = {'lat_dec_deg': 16.217,
                     'altitude': 8.0,
                     'long_dec_deg': -16.25}
        units = {'wind_speed': 'MeterPerSecond', 'temp': 'Centigrade',
                 'sol_rad': 'MegaJoulePerMeterSquarePerHour',
                 'rel_hum': 'percent'}
        eto = PenmanMonteith(df, units, constants=constants, verbosity=0)
        pet_penman = eto()
        np.testing.assert_almost_equal(
            pet_penman[-1], 0.6269, 2, "hourly PenmanMonteith Failling")
        return


def get_daily_observed_data():
    obs = pd.read_csv('data/obs.txt', date_parser=['index'])
    obs.index = pd.to_datetime(obs['index'])
    obs.index.freq = pd.infer_freq(obs.index)
    return obs

class TestDaily(unittest.TestCase):
    st = '20010301'
    en = '20040831'

    with open('data/constants.json', 'r') as fp:
        constants = json.load(fp)
    constants['alphaPT'] = 1.28
    constants['wind_f'] = 'pen48'
    constants['Roua'] = 1.2
    constants['Ca'] = 0.001013
    constants['CH'] = 0.12
    constants['surf_res'] = 70
    constants['turc_k'] = 0.013
    constants['albedo'] = 0.23
    # although 0.55 is used in original, but I do not divide by 100
    # in Hamon method so 0.0055 here
    constants['cts'] = 0.0055
    data = pd.read_csv('data/data.txt', index_col=0, comment='#')
    data.index = pd.to_datetime(data.index)
    data.index.freq = pd.infer_freq(data.index)
    data = data[st:en]
    data = data.rename(columns={'uz': 'wind_speed'})
    units = {'tmin': 'Centigrade', 'tmax': 'Centigrade',
              'sunshine_hrs': 'hour', 'rh_min': 'percent',
              'rh_max': 'percent', 'wind_speed': 'MeterPerSecond',
             'tdew': 'Centigrade'}

    def test_ET_ChapmanAustralia_Daily(self):
        etp = ChapmanAustralia(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_GrangerGray_Daily(self):
        etp = GrangerGray(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_SzilagyiJozsa_Daily(self):
        etp = SzilagyiJozsa(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_Turc_Daily(self):
        etp = Turc(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_Linacre_Daily(self):
        etp = Linacre(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_Hamon_Daily(self):
        etp = Hamon(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_HargreavesSamani_Daily(self):
        constants = self.constants.copy()
        constants['ct'] = 0.2
        etp = HargreavesSamani(self.data, units=self.units,
                               constants=constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_JensenHaise_Daily(self):
        constants = self.constants.copy()
        constants['ct'] = 0.2
        constants['tx'] = 0.2
        etp = ETBase(self.data, units=self.units,
                               constants=constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_Makkink_Daily(self):
        etp = Makkink(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_MattShuttleworth_Daily(self):
        etp = MattShuttleworth(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_McGuinnessBordne_Daily(self):
        etp = McGuinnessBordne(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_Penman_Daily(self):
        constants = self.constants.copy()
        constants['albedo'] = 0.08
        etp = Penman(self.data, units=self.units,
                               constants=constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_PenPan_Daily(self):
        constants = self.constants.copy()
        constants['pan_over_est'] = 0.4
        constants['pan_est'] = 0.3
        etp = PenPan(self.data, units=self.units,
                               constants=constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_PriestleyTaylor_Daily(self):
        etp = PriestleyTaylor(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_Romanenko_Daily(self):
        etp = Romanenko(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_PenmanMonteith_Daily(self):
        etp = PenmanMonteith(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_Abtew_Daily(self):
        etp = Abtew(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_BlaneyCriddle_Daily(self):
        etp = BlaneyCriddle(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def test_ET_BrutsaertStrickler_Daily(self):
        etp = BrutsaertStrickler(self.data, units=self.units,
                               constants=self.constants, verbosity=0)
        self.obs = get_daily_observed_data()
        sim = etp()
        self.plot_erros()
        return

    def plot_erros(self):
        # plot whole result in two plots
        # to_plot1 = random.sample(self.to_test, k=int(len(self.to_test)/2))
        # to_plot2 = [m for m in methods_to_test if m not in to_plot1]
        # all_plots = {'a': to_plot1, 'b': to_plot2}
        #
        # for _plot, _methods in all_plots.items():
        #     figure, axs = plt.subplots(len(_methods), sharex='all')
        #     figure.set_figwidth(9)
        #     figure.set_figheight(12)
        #
        #     for axis, method in zip(axs, _methods):
        #         diff = pd.DataFrame(data=np.abs(self.diff[method]),
        #                             index=self.obs.loc[self.data.index].index,
        #                             columns=[method])
        #         axis.plot(diff, label=method)
        #         axis.legend(loc="best")
        #
        #     plt.savefig('diff'+_plot, dpi=300, bbox_inches='tight')
        return



if __name__ == "__main__":
    unittest.main()
