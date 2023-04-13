
import unittest
import pandas as pd

from ETUtil.ETUtil.et_methods import PenmanMonteith
# The BetC software comes with some example data.
# This is to test reference ET with respect to that

fpath = "C:\\Users\\ather\\OneDrive\\Desktop\\sub_daily_etp\\data\\betc_sample_data.xlsx"
df = pd.read_excel(fpath, sheet_name='daily')
df.index = pd.to_datetime(df['index'])
df.index.freq = pd.infer_freq(df.index)

df = df[['tmin_c', 'tmax_c', 'tdew_c', 'rel_hum', 'sol_rad_MegaJoulePerMeterSquarePerDay',
         'wind_speed_mps']]
df = df.rename(columns={'tmin_c': 'tmin',
                        'tmax_c': 'tmax',
                        'tdew_c': 'tdew',
                        'rel_hum': 'rel_hum',
                        'sol_rad_MegaJoulePerMeterSquarePerDay': 'sol_rad',
                        'wind_speed_mps': 'wind_speed'})
units = {'tmin': 'Centigrade',
         'tmax': 'Centigrade',
         'tdew': 'Centigrade',
         'rel_hum': 'Percentage',
         'wind_speed': 'MeterPerSecond'}

constants = {'lat_dec_deg': 35.183,
             'long_dec_deg': 102.12,
             'altitude': 1170,
             'albedo': 0.23,
             'a_s': 0.23,
             'b_s': 0.5,
             'wind_z': 2}

class TestBetc(unittest.TestCase):

    def test_penmanMonteith(self):
        eto = PenmanMonteith(df, units, constants)
        pet = eto()

        df = pd.read_excel(fpath, sheet_name='hourly')
        df.index = pd.to_datetime(df['index'])
        df.index.freq = pd.infer_freq(df.index)
        df = df[['temp_c', 'tdew_c', 'rel_hum', 'sol_rad_Wm2',
                 'wind_speed_mps']]

        df['sol_rad'] = df['sol_rad_Wm2'] * 0.0036
        x = df.pop('sol_rad_Wm2')
        df = df.rename(columns={'temp_c': 'temp',
                                'tdew_c': 'tdew',
                                'rel_hum': 'rel_hum',
                                'sol_rad': 'sol_rad',
                                'wind_speed_mps': 'wind_speed'})

        units = {'temp': 'Centigrade',
                 'tdew': 'Centigrade',
                 'rel_hum': 'Percentage',
                 'wind_speed': 'MeterPerSecond'}
        eto = PenmanMonteith(df, units, constants)
        pet = eto()
        return


if __name__ == "__main__":
    unittest.main()