from main import ReferenceET

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


# temp = np.arange(10)
# person = Temp(temp, 'centigrade')
# print(person.kelvin)


# wind = np.arange(10)
# W = Wind(wind, 'KilometerPerHour')
# print(W.MeterPerSecond)
#
# print(W.KilometerPerHour)
#
# print(W.MilesPerHour)
#
# print(W.InchesPerSecond)
#
# print(W.FeetPerSecond)

"""
Daily Hamon
 """
# tmin =  np.array([27.3, 25.9, 26.1, 26.1, 24.6, 18.1, 19.9, 22.6, 16.0, 18.9])
# tmax =  np.array([47.8, 38.5, 38.5, 42.6, 36.1, 37.4, 43.5, 38.3, 36.7, 39.7 ])
# dr = pd.date_range('20110101', '20110110', freq='D')
# df = pd.DataFrame(np.stack([tmin,tmax, ],axis=1),
#                    columns=['tmin', 'tmax'],
#                   index=dr)
# lat = 45.2  # 45 12 15
# units={'tmin':'fahrenheit', 'tmax':'fahrenheit'}
# eto = ReferenceET(df,units,lat=lat)
# pet_hamon = eto.Hamon()



# # Blaney-Criddle test
# # http://www.fao.org/3/S2022E/s2022e07.htm#3.1.3%20blaney%20criddle%20method
# latitude = -35.0
# tmin =  np.array([19.4, 20.4])
# tmax =  np.array([29.5, 30.5])
# dr = pd.date_range('20110401', '20110501', freq='M')
# df = pd.DataFrame(np.stack([tmin,tmax, ],axis=1),
#                     columns=['tmin', 'tmax'],
#                    index=dr)
# units={'tmin':'fahrenheit', 'tmax':'fahrenheit'}
# eto = ReferenceET(df,units,lat=latitude)


""" Thornthwaite
Sellers (1969), Physical Climatology, pg 173,
https://www.ncl.ucar.edu/Document/Functions/Built-in/thornthwaite.shtml"""
# temp = np.array([23.3, 21.1, 19.6, 17.2, 12.6, 10.9, 10. , 11. , 13. , 15.8, 17.8,  20.1])
# dr = pd.date_range('20110101', '20111231', freq='M')
# df = pd.DataFrame(temp ,  columns=['temp'], index=dr)
# units = {'temp': 'centigrade', 'daylight_hrs':'hour'}
# constants = {'lat' : -38.0}
# etp = ReferenceET(df, units, constants=constants)
# pet = etp.Thornthwait()



"""
Jensen and Haise
"""
tmin =  np.array([27.3,      25.9,      26.1,      26.1,      24.6,      18.1,      19.9,      22.6,      16.0,      18.9])
tmax =  np.array([ 47.8,      38.5,      38.5,      42.6,      36.1,      37.4,      43.5,      38.3,      36.7,      39.7])
sol_rad = np.array([256.,     306.,     193.,     269.,     219.,     316.,     318.,     320.,    289.,     324.])
dr = pd.date_range('20110101', '20110110', freq='D')
df = pd.DataFrame(np.stack([tmin,tmax, sol_rad],axis=1),
                    columns=['tmin', 'tmax', 'solar_rad'],
                   index=dr)

Units = {'tmin': 'centigrade', 'tmax':'centigrade', 'solar_rad': 'LangleysPerDay'}
etp = ReferenceET(df, units=Units, constants={'lat': 24.0})
pet = etp.JesnsenBASINS()
    # 0.159
    # 0.165
    # 0.104
    # 0.153
    # 0.112
    # 0.149
    # 0.169
    # 0.164
    # 0.130
    # 0.160



class  Tests(object):

    et_methods = ['ET_PenmanMonteith_Daily', 'ET_Abtew_Daily', 'ET_BlaneyCriddle_Daily',
       'ET_BrutsaertStrickler_Daily', 'ET_ChapmanAustralia_Daily',
       'ET_GrangerGrey_Daily', 'ET_SzilagyiJozsa_Daily', 'ET_Turc_Daily',
       'ET_Hamon_Daily', 'ET_HargreavesSamani_Daily', 'ET_JensenHaise_Daily',
       'ET_Linacre_Daily', 'ET_Makkink_Daily', 'ET_MattShuttleworth_Daily',
       'ET_McGuinnessBordne_Daily', 'ET_Penman_Daily', 'ET_Penpan_Daily',
       'ET_PriestleyTaylor_Daily', 'ET_Romanenko_Daily', 'ET_CRWE_Mon',
       'ET_CRAE_Mon']

    def __init__(self, to_test, st='20010301', en = '20040831'):

        with open('data/constants.json', 'r') as fp:
            self.constants = json.load(fp)
        data = pd.read_csv('data/data.txt', index_col=0, comment='#')
        data.index = pd.to_datetime(data.index)
        data.index.freq = pd.infer_freq(data.index)
        self.data = data[st:en]
        units={'tmin': 'centigrade', 'tmax':'centigrade', 'sunshine_hrs': 'hour', 'rh_min':'percent',
       'rh_max':'percent','uz':'MeterPerSecond', 'tdew':'centigrade'}
        self.etp = ReferenceET(self.data, units,
                  constants=self.constants)

        self.obs = self.get_observed_data()
        self.to_test = to_test
        self.diff = {}


    def get_observed_data(self):
        obs = pd.read_csv('data/obs.txt', date_parser=['index'])
        obs.index = pd.to_datetime(obs['index'])
        obs.index.freq = pd.infer_freq(obs.index)
        return obs

    def run(self, plot_diff=False):
        etp_methods = [method for method in dir(self.etp) if callable(getattr(self.etp, method)) if
                               not method.startswith('_')]
        for method in self.to_test:
            _method = method.split('_')[1]
            if _method in etp_methods:
                print('calling: ', _method)
                getattr(self.etp, _method)()  # call
                out_et = self.etp.output[method].values
                obs_et = self.obs[method].loc[self.data.index].values.reshape(-1,1)
                self.diff[method] = np.subtract(out_et, obs_et)

        if plot_diff:
            self.plot_erros()


    def plot_erros(self):
        figure, axs = plt.subplots(len(self.to_test), sharex='all')
        figure.set_figwidth(9)
        figure.set_figheight(12)

        for axis, method in zip(axs, self.to_test):
            diff = pd.DataFrame(data=np.abs(self.diff[method]), index=self.obs.loc[self.data.index].index, columns=[method])
            axis.plot(diff, label=method)
            axis.legend(loc="best")

        plt.show()
        return



methods_to_test = ['ET_PenmanMonteith_Daily', 'ET_Hamon_Daily', 'ET_HargreavesSamani_Daily', 'ET_JensenHaise_Daily',
           'ET_Penman_Daily', 'ET_PriestleyTaylor_Daily', 'ET_Abtew_Daily', 'ET_McGuinnessBordne_Daily',
           'ET_Makkink_Daily', 'ET_Linacre_Daily', 'ET_Turc_Daily', 'ET_ChapmanAustralia_Daily', 'ET_Romanenko_Daily']
start = '20020110'
end = '20020120'
#test = Tests(methods_to_test, st=start, en=end)
#test.run(plot_diff=True)


# Daily FAO Penman-Monteith
# tmin =  np.array([12.3, 27.3, 25.9, 26.1, 26.1, 24.6, 18.1, 19.9, 22.6, 16.0, 18.9])
# tmax =  np.array([21.5, 47.8, 38.5, 38.5, 42.6, 36.1, 37.4, 43.5, 38.3, 36.7, 39.7 ])
# rh_min = np.array([63 for _ in range(len(tmin))])
# rh_max = np.array([84 for _ in range(len(tmin))])
# uz = np.array([3.3 for _ in range(len(tmin))])
# sunshine_hrs = np.array([9.25 for i in range(len(tmin))])
# lat = 50.
# altitude = 100.0
# wind_z = 10.0
# dr = pd.date_range('20110706', '20110716', freq='D')
# df = pd.DataFrame(np.stack([tmin,tmax, sunshine_hrs, uz, rh_min, rh_max, ],axis=1),
#                    columns=['tmin', 'tmax', 'sunshine_hrs', 'uz', 'rh_min', 'rh_max'],
#                    index=dr)
# units={'tmin':'centigrade', 'tmax':'centigrade', 'uz': 'MeterPerSecond', 'sunshine_hrs':'hour',
#         'rh_min':'percent', 'rh_max':'percent'}
# eto = ReferenceET(df,units,lat, altitude, wind_z)
# pet_penman = eto.Penman_Monteith()



# # Hourly Penman-Monteith FAO56
# N = 11
# uz = np.array([3.3 for _ in range(N)])
# temp = np.array([38 for _ in range(N)])
# rel_hum = np.array([52 for _ in range(N)])
# sol_rad = np.array([2.45 for _ in range(N)])
# dr = pd.date_range('20111001 15:00', '20111002 01:00', freq='H')
# df = pd.DataFrame(np.stack([uz, temp, rel_hum, sol_rad],axis=1),
#                   columns=['uz', 'temp', 'rel_hum', 'solar_rad'],
#                   index=dr)
# lat = 16.25
# altitude = 8.0
# units={'uz': 'MeterPerSecond', 'temp':'centigrade', 'solar_rad': 'MegaJoulePerMeterSquarePerHour', 'rel_hum':'percent'}
# eto = ReferenceET(df,units,lat, altitude)
# pet_penman = eto.Penman_Monteith()
