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

# Hamon
tmin =  np.array([12.3, 27.3, 25.9, 26.1, 26.1, 24.6, 18.1, 19.9, 22.6, 16.0, 18.9])
tmax =  np.array([21.5, 47.8, 38.5, 38.5, 42.6, 36.1, 37.4, 43.5, 38.3, 36.7, 39.7 ])
temp = np.array([])
sunshine_hrs = np.array([9.25 for i in range(len(tmin))])
wind = np.array([10 for _ in range(len(tmin))])
rh_min = np.array([63 for _ in range(len(tmin))])
rh_max = np.array([84 for _ in range(len(tmin))])
#pet = np.array([0.128, 0.0971,0.0979,0.110,0.0892,0.0777,0.0967,0.0909,0.0728,0.0862])

dr = pd.date_range('20110706', '20110716', freq='D')
df = pd.DataFrame(np.stack([tmin,tmax, sunshine_hrs, wind, rh_min, rh_max],axis=1),
                  columns=['tmin', 'tmax', 'sunshine_hrs', 'wind', 'rh_min', 'rh_max'],index=dr)
lat = 50.48
units={'tmin':'centigrade', 'tmax':'centigrade', 'wind': 'KilometerPerHour', 'sunshine_hrs':'hour',
       'rh_min':'percent', 'rh_max':'percent'}

eto = ReferenceET(df,units,lat=lat, altitude=100.0, wind_z=10.0)

pet = eto.Hamon()


pet_penman = eto.Penman_Monteith()
