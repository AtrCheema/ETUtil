from convert import Wind
from main import ReferenceET
import numpy as np
import pandas as pd


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

# # Daily Hamon
tmin =  np.array([27.3, 25.9, 26.1, 26.1, 24.6, 18.1, 19.9, 22.6, 16.0, 18.9])
tmax =  np.array([47.8, 38.5, 38.5, 42.6, 36.1, 37.4, 43.5, 38.3, 36.7, 39.7 ])
dr = pd.date_range('20110101', '20110110', freq='D')
df = pd.DataFrame(np.stack([tmin,tmax, ],axis=1),
                   columns=['tmin', 'tmax'],
                  index=dr)
lat = 45.2  # 45 12 15
units={'tmin':'fahrenheit', 'tmax':'fahrenheit'}
eto = ReferenceET(df,units,lat=lat)
pet_hamon = eto.Hamon()



# Daily FAO Penman-Monteith
# tmin =  np.array([12.3, 27.3, 25.9, 26.1, 26.1, 24.6, 18.1, 19.9, 22.6, 16.0, 18.9])
# tmax =  np.array([21.5, 47.8, 38.5, 38.5, 42.6, 36.1, 37.4, 43.5, 38.3, 36.7, 39.7 ])
# rh_min = np.array([63 for _ in range(len(tmin))])
# rh_max = np.array([84 for _ in range(len(tmin))])
# wind = np.array([3.3 for _ in range(len(tmin))])
# sunshine_hrs = np.array([9.25 for i in range(len(tmin))])
# lat = 50.
# altitude = 100.0
# wind_z = 10.0
# dr = pd.date_range('20110706', '20110716', freq='D')
# df = pd.DataFrame(np.stack([tmin,tmax, sunshine_hrs, wind, rh_min, rh_max, ],axis=1),
#                    columns=['tmin', 'tmax', 'sunshine_hrs', 'wind', 'rh_min', 'rh_max'],
#                    index=dr)
# units={'tmin':'centigrade', 'tmax':'centigrade', 'wind': 'MeterPerSecond', 'sunshine_hrs':'hour',
#         'rh_min':'percent', 'rh_max':'percent'}
# eto = ReferenceET(df,units,lat, altitude, wind_z)
# pet_penman = eto.Penman_Monteith()



# # Hourly Penman-Monteith FAO56
# N = 11
# wind = np.array([3.3 for _ in range(N)])
# temp = np.array([38 for _ in range(N)])
# rel_hum = np.array([52 for _ in range(N)])
# sol_rad = np.array([2.45 for _ in range(N)])
# dr = pd.date_range('20111001 15:00', '20111002 01:00', freq='H')
# df = pd.DataFrame(np.stack([wind, temp, rel_hum, sol_rad],axis=1),
#                   columns=['wind', 'temp', 'rel_hum', 'solar_rad'],
#                   index=dr)
# lat = 16.25
# altitude = 8.0
# units={'wind': 'MeterPerSecond', 'temp':'centigrade', 'solar_rad': 'MegaJoulePerMeterSquarePerHour', 'rel_hum':'percent'}
# eto = ReferenceET(df,units,lat, altitude)
# pet_penman = eto.Penman_Monteith()
