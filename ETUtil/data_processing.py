
import numpy as np
import pandas as pd
import os
from numpy import array, mean, where

from ETUtil.ETUtil.convert import Speed, Temp, Pressure

sb_cons = 4.903e-9

# default constants
# description, default value, min, max
def_cons = {
    'lat': ['latitude in decimal degrees', None, -1, 1],
    'long': ['longitude in decimal degrees'],
    'a_s': ['fraction of extraterrestrial radiation reaching earth on sunless days', 0.23],
    'b_s': ['difference between fracion of extraterrestrial radiation reaching full-sun days and that on sunless days',
            0.5],
    'albedo': ["""a numeric value between 0 and 1 (dimensionless), albedo of evaporative surface representing the
    portion of the incident radiation that is reflected back at the surface. Default is 0.23 for
    surface covered with short reference crop, which is for the calculation of Matt-Shuttleworth
     reference crop evaporation.""", 0.23, 0, 1],
    'alpha_pt': ['Priestley-Taylor coefficient', 1.26],
    'altitude': ['Elevation of station'],
    'wind_z': ['height at which wind speed is measured'],
    'CH': ['crop height', 0.12],
    'Ca': ['specific heat of air', 0.001013],
    'Roua': ['mean air density', 1.20],
    'surf_res': ["""surface resistance (s/m) depends on the type of reference crop. 
                    Default is 70 for short reference crop""", 70, 0, 9999],
    'wind_f' : ["wind function", 'pen48'],
    'pan_over_est': ["""Must be T or F, indicating if adjustment for the overestimation (i.e. divided by 1.078) of
                  Class-A pan evaporation for Australian data is applied in PenPan formulation.""", False],
    'pan_coef': ["""Only required if argument est has value of potential ET, which defines the pan coefficient 
                 used to adjust the estimated pan evaporation to the potential ET required""", 0.711],
    'pan_est': ["""Must be either `pan` or `pot_et` to specify if estimation for the Class-A pan evaporation or
                potential evapotranspriation is performed.""", 'pot_et'],
    'pen_ap': ['a constant in PenPan', 2.4],
    'alphaA': ['albedo for class-A pan'],
    'cts'   : [' float, or array of 12 values for each month of year', 0.0055],
    'ct'    : ['a coefficient in Jensen and Haise', 0.025],
    'tx'    : ['a coefficient in Jensen and Haise', 3],
    'abtew_k': ['a coefficient used defined by Abtew', 0.52],
    'turc_k' : ['crop coefficient to be used in Turc method', 0.013],
    'cts_jensen':  ['used for JensenHaise method', 0.012],
    'ctx_jensen': ['used for JensenHaise method', 24.0],
    'ap'     : ['', 2.4],
    'alphaPT': ["Brutsaert and Strickler (1979) constant", 1.28],
    'e0':      ["a variable used in BlaneyCridle formulation", 0.819],
    'e1':      ["a variable used in BlaneyCridle formulation", -0.00409],
    'e2':      ["a variable used in BlaneyCridle formulation", 1.0705],
    'e3':      ["a variable used in BlaneyCridle formulation", 0.065649],
    'e4':      ["a variable used in BlaneyCridle formulation", -0.0059684],
    'e5':      ["a variable used in BlaneyCridle formulation", -0.0005967]
}


class process_input(object):

    def __init__(self,input_df,units, constants, calculate_at_freq=None, verbose=1):
        self.verbose = verbose
        self.input = input_df
        self.cons = constants
        self.def_cons = def_cons
        self.SB_CONS = None
        self.daily_index=None
        self.no_of_hours = None
        self.units = units
        self.freq_in_min = None
        self.freq = self.set_freq(at_freq=calculate_at_freq)
        self._check_compatibility()
        self.lat_rad = self.cons['lat'] * 0.0174533 if 'lat' in  self.cons else None  # degree to radians
        self.wind_z = constants['wind_z'] if 'wind_z' in constants else None
        self.output = {}

    def set_freq(self, at_freq=None):

        self.check_nans()

        in_freq = self.get_in_freq()
        setattr(self, 'input_freq', in_freq)

        if at_freq is not None:

            if not hasNumbers(at_freq):
                at_freq = "1" + at_freq

            out_freq_in_min, at_freq = split_freq(at_freq)

            if at_freq not in ['H',  'D', 'M', 'min']:
                raise ValueError("unknown frequency {} is provided".format(at_freq))

            at_freq = str(out_freq_in_min) + str(at_freq)

            if int(out_freq_in_min) < 60:
                freq = 'sub_hourly'
            elif 60 <= int(out_freq_in_min) < 1440:
                freq = 'Hourly'
            elif int(out_freq_in_min) >= 1440:
                freq = 'Daily'
            else:
                freq = 'Monthly'

            if freq != in_freq:
                self.resample_data(in_freq, out_freq_in_min)

            freq = freq
        else:
            freq = in_freq
            out_freq_in_min = int(pd.to_timedelta(self.input.index.freq).seconds / 60.0)

        freq_in_min = int(out_freq_in_min)
        setattr(self, 'freq_in_min', freq_in_min)

        self.get_additional_ts()

        if 'D' in freq:
            setattr(self, 'SB_CONS', 4.903e-9)   #  MJ m-2 day-1.
        elif 'H' in freq:    #  (4.903/24) 10-9
            setattr(self, 'SB_CONS', 2.043e-10)   # MJ m-2 hour-1.
        elif 'T' in freq or freq == 'sub_hourly':
            setattr(self, 'SB_CONS', sb_cons/freq_in_min)  # MJ m-2 per timestep.
        elif 'M' in freq:
            start_year = str(self.input.index[0].year)
            end_year = str(self.input.index[-1].year)
            start_month = str(self.input.index[0].month)
            if len(start_month) < 2:
                start_month = '0' + start_month
            end_month = str(self.input.index[-1].month)
            if len(end_month) < 2:
                end_month = '0' + start_month
            start_day = str(self.input.index[0].day)
            if len(start_day) < 2:
                start_day = '0' + start_day
            end_day = str(self.input.index[-1].day)
            if len(end_day) < 2:
                end_day = '0' + start_day
            st = start_year + start_month + '01'
            en = end_year + end_month + end_day
            dr = pd.date_range(st, en, freq='D')
            setattr(self, 'daily_index', dr)
        return freq


    def check_nans(self):
        for col in self.input.columns:
            nans = self.input[col].isna().sum()
            if nans>0:
                raise ValueError("""Columns {} in input data contains {} nan values. Input dataframe should not have
                                 any nan values""".format(col, nans))

    def get_in_freq(self):
        freq = self.input.index.freqstr
        freq_in_min = int(pd.to_timedelta(self.input.index.freq).seconds / 60.0)
        setattr(self, 'in_freq_in_min', freq_in_min)
        if freq is None:
            idx = self.input.index.copy()
            _freq = pd.infer_freq(idx)
            if self.verbose>1: print('Frequency inferred from input data is', _freq)
            freq = _freq
            data = self.input.copy()
            data.index.freq = _freq
            self.input = data

        if 'D' in freq:
            return 'Daily'
        elif 'H' in freq:
            return 'Hourly'
        elif 'T' in freq:
            return 'sub_hourly'
        elif 'M' in freq:
            return 'Monthly'
        else:
            raise ValueError('unknown frequency of input data')


    def get_additional_ts(self):
        if self.input_freq in ['sub_hourly', 'Hourly'] and self.freq_in_min>=1440:
            # find tmax and tmin
            temp = pd.DataFrame(self.orig_input['temp'])
            self.input['tmax'] = temp.groupby(pd.Grouper(freq='D'))['temp'].max()
            self.input['tmin'] = temp.groupby(pd.Grouper(freq='D'))['temp'].max()
            self.units['tmax'] = self.units['temp']
            self.units['tmin'] = self.units['temp']
            self.input.pop('temp')
        return


    def resample_data(self, data_frame, desired_freq_in_min):
        self.orig_input = self.input.copy()
        _input = self.input.copy()

        for data_name in _input:
            data_frame = pd.DataFrame(_input[data_name])
            orig_tstep = int(_input.index.freq.delta.seconds/60)  # in minutes

            # if not hasNumbers(desired_freq):
            #     desired_freq = '1' + desired_freq

            #out_tstep = int((pd.Timedelta(desired_freq).seconds/60))  # in minutes
            out_tstep = desired_freq_in_min #str(out_tstep) + 'min'

            if out_tstep > orig_tstep:  # from low timestep to high timestep i.e from 1 hour to 24 hour
                # from low timestep to high timestep
                data_frame = self.downsample_data(data_frame, data_name, out_tstep)

            elif out_tstep < orig_tstep:  # from larger timestep to smaller timestep
                data_frame = self.upsample_data(data_frame, data_name, out_tstep)

            _input[data_name] = data_frame

        self.input = _input.dropna()
        return


    def upsample_data(self, data_frame, data_name, out_freq):
        out_freq = str(out_freq) + 'min'

        old_freq = data_frame.index.freqstr
        nan_idx = data_frame.isna()  # preserving indices with nan values

        nan_idx_r = nan_idx.resample(out_freq).ffill() #
        data_frame = data_frame.copy()


        if self.verbose>1: print('upsampling {} data from {} to {}'.format(data_name, old_freq, out_freq))
        # e.g from monthly to daily or from hourly to sub-hourly
        if data_name in ['temp', 'rel_hum', 'rh_min', 'rh_max', 'uz', 'u2', 'q_lps']:
            data_frame = data_frame.resample(out_freq).interpolate(method='linear')
            data_frame[nan_idx_r] = np.nan  # filling those interpolated values with NaNs which were NaN before interpolation

        elif data_name in ['rain_mm', 'ss_gpl', 'solar_rad', 'pet', 'pet_hr']:
            # distribute rainfall equally to smaller time steps. like hourly 17.4 will be 1.74 at 6 min resolution
            idx = data_frame.index[-1] + pd.offsets.Hour(1)
            data_frame = data_frame.append(data_frame.iloc[[-1]].rename({data_frame.index[-1]: idx}))
            data_frame = add_freq(data_frame)
            df1 = data_frame.resample(out_freq).ffill().iloc[:-1]
            df1[data_name] /= df1.resample(data_frame.index.freqstr)[data_name].transform('size')
            data_frame = df1
            data_frame[nan_idx_r] = np.nan  #filling those interpolated values with NaNs which were NaN before interpolation

        return data_frame


    def downsample_data(self, data_frame, data_name, out_freq):
        out_freq = str(out_freq) + 'min'
        data_frame = data_frame.copy()
        old_freq = data_frame.index.freq
        if self.verbose>1: print('downsampling {} data from {} min to {}'.format(data_name, old_freq, out_freq))
        # e.g. from hourly to daily
        if data_name in ['temp', 'rel_hum', 'rh_min', 'rh_max', 'uz', 'u2', 'wind_speed_kph', 'q_lps']:
            return data_frame.resample(out_freq).mean()
        elif data_name in ['rain_mm', 'ss_gpl', 'solar_rad']:
            return data_frame.resample(out_freq).sum()


    def _check_compatibility(self):
        """units are also converted here."""

        self.validate_constants()

        if not isinstance(self.input, pd.DataFrame):
            raise TypeError('input must be a pandas dataframe')

        for col in self.input.columns:
            if col not in self.units.keys():
                raise ValueError('units for input {} are not given'.format(col))

        if 'tmin' in self.input.columns and 'tmax' in self.input.columns:
            if 'temp' in self.input.columns:
                raise ValueError(""" Don't provide both Min Max temp and Mean temperatures. This is confusing.
                if tmin and tmax are given, don't provide temp, that is of no use and confusing.""")

        allowed_units = {'temp': ['Centigrade', 'Fahrenheit', 'Kelvin'],
                         'tmin': ['Centigrade', 'Fahrenheit', 'Kelvin'],
                         'tmax': ['Centigrade', 'Fahrenheit', 'Kelvin'],
                         'tdew': ['Centigrade', 'Fahrenheit', 'Kelvin'],
                         'uz':  ['MeterPerSecond', 'KiloMeterPerHour', 'MilesPerHour', 'InchesPerSecond',
                                   'FeetPerSecond'],
                         'daylight_hrs': ['hour'],
                         'sunshine_hrs': ['hour'],
                         'rel_hum': ['percent'],
                         'rh_min': ['percent'],
                         'rh_max': ['percent'],
                         'solar_rad': ['MegaJoulePerMeterSquarePerHour', 'LangleysPerDay'], #TODO
                         'ea': ['KiloPascal'],  # actual vapour pressure
                         'es': ['KiloPascal'], # saturation vapour pressure
                         'vp_def': ['KiloPascal'], # vapour pressure deficit
                         'rns': ['MegaJoulePerMeterSquare'],  # net incoming shortwave radiation
                         'rn' : ['MegaJoulePerMeterSquare'],  # net radiation
                         'cloud': ['']}

        for _input, _unit in self.units.items():
            try:
                if _unit not in allowed_units[_input]:
                    raise ValueError('unit {} of input data {} is not allowed. Use any of {}'
                                     .format(_unit, _input, allowed_units[_input]))
            except KeyError:
                raise KeyError("Unnecessary input data {} provided. Remove it from input".format(_input))

        self._preprocess_temp()

        self._preprocess_rh()

        self._check_wind_units()

        self._cehck_pressure_units()

        # getting julian day
        self.input['jday'] = self.input.index.dayofyear

        if self.freq == 'Hourly':
            a = self.input.index.hour
            ma = np.convolve(a, np.ones((2,)) / 2, mode='same')
            ma[0] = ma[1] - (ma[2] - ma[1])
            self.input['half_hr'] = ma
            freq = self.input.index.freqstr
            if len(freq)>1:
                setattr(self, 'no_of_hours', int(freq[0]))
            else:
                setattr(self, 'no_of_hours', 1)

            self.input['t1'] = np.zeros(len(self.input)) + self.no_of_hours

        elif self.freq == 'sub_hourly':
            a = self.input.index.hour
            b = (self.input.index.minute + self.freq_in_min / 2.0) / 60.0
            self.input['half_hr'] = a + b

            self.input['t1'] = np.zeros(len(self.input)) + self.freq_in_min/60.0

        if 'solar_rad' in self.input:
            if self.freq in ['Hourly', 'sub_hourly']:
                self.input['is_day'] = where(self.input['solar_rad'].values > 0.1, 1, 0)

        return


    def _preprocess_rh(self):
         # make sure that we mean relative humidity calculated if possible
        if 'rel_hum' in self.input.columns:
            self.input['rh_mean'] = self.input['rel_hum']
        else:
            if 'rh_min' in self.input.columns:
                self.input['rh_mean'] = mean(array([self.input['rh_min'].values, self.input['rh_max'].values]), axis=0)
        return

    def _preprocess_temp(self):
        """ converts temperature related input to units of Centigrade if required. """
        # converting temperature units to celsius
        for val in ['tmin', 'tmax', 'temp', 'tdew']:
            if val in self.input:
                t = Temp(self.input[val].values, self.units[val])
                self.input[val] = t.Centigrade

        # if 'temp' is given, it is assumed to be mean otherwise calculate mean and put it as `temp` in input dataframe.
        if 'temp' not in self.input.columns:
            if 'tmin' in self.input.columns and 'tmax' in self.input.columns:
                self.input['temp'] = mean(array([self.input['tmin'].values, self.input['tmax'].values]), axis=0)
        return

    def _check_wind_units(self):
        # check units of wind speed and convert if needed
        if 'uz' in self.input:
            w = Speed(self.input['uz'].values, self.units['uz'])
            self.input['uz'] = w.MeterPerSecond
        return

    def _cehck_pressure_units(self):
        """ converts pressure related input to units of KiloPascal if required. """
        for pres in ['ea', 'es', 'vp_def']:
            if pres in self.input:
                p = Pressure(self.input[pres].values, self.units[pres])
                self.input[pres] = p.KiloPascal

    @property
    def seconds(self):
        """finds number of seconds between two steps of input data"""
        if len(self.input)>1:
            return  (self.input.index[1]-self.input.index[0])/np.timedelta64(1, 's')


    def check_constants(self, method):
        _cons = {
            'PenPan': {'opt': ['pan_over_est', 'albedo', 'pan_coef', 'pen_ap', 'alphaA'],
                       'req': ['lat']},

            'PenmanMonteith': {'opt': ['albedo', 'a_s', 'b_s'],
                               'req': ['lat', 'altitude']},

            'Abtew': {'opt': ['a_s', 'b_s', 'abtew_k'],
                      'req': ['lat']},

            'BlaneyCriddle': {'opt': ['e0', 'e1', 'e2', 'e3', 'e4', 'e5'],
                              'req': ['lat']},

            'BrutsaertStrickler': {'opt': [None],
                                   'req': ['alphaPT']},

            'ChapmanAustralia': {'opt': ['ap', 'albedo', 'alphaA'],
                                 'req': ['lat', 'long']},

            'GrangerGray': {'opt': ['wind_f', 'albedo'],
                            'req': ['lat']},

            'SzilagyiJozsa': {'opt': ['wind_f', 'alpha_pt'],
                              'req': ['lat']},

            'Turc': {'opt': ['a_s', 'b_s', 'turc_k'],
                    'req':  ['lat', 'long']},

            'Hamon': {'opt': ['cts'],
                      'req': ['lat', 'long']},

            'HargreavesSamani': {'opt': [''],
                                 'req': ['lat', 'long']},

            'JensenHaise': {'opt': ['a_s', 'b_s', 'ct', 'tx'],
                            'req': ['lat', 'long']},

            'JensenHaiseBASINS':{'opt': ['cts_jensen', 'ctx_jensen'],
                                 'req': ['lat']},

            'Linacre': {'opt': ['altitude'],
                        'req': ['lat', 'long']},

            'Makkink': {'opt': ['a_s', 'b_s'],
                        'req': ['lat', 'long']},

            'MattShuttleworth': {'opt': ['CH', 'Roua', 'Ca', 'albedo', 'a_s', 'b_s', 'surf_res'],
                                 'req': ['lat', 'long']},

            'McGuinnessBordne': {'opt': [None],
                                 'req': ['lat', 'long', 'long']},

            'Penman': {'opt': ['wind_f', 'a_s', 'b_s', 'albedo'],
                       'req': ['lat', 'long']},

            'Penpan': {'opt': [''],
                       'req': ['lat', 'long']},

            'PriestleyTaylor': {'opt': ['a_s', 'b_s', 'alpha_pt', 'albedo'],
                                'req': ['lat', 'long']},

            'Romanenko': {'opt': [None],
                          'req': ['lat', 'long']},

            'CRWE': {'opt': [''],
                     'req': ['lat', 'long']},

            'CRAE': {'opt': [''],
                     'req': ['lat', 'long']},

            'Thornthwait': {'opt':[None],
                             'req': ['lat']}
        }

        # checking for optional input variables
        for opt_v in _cons[method]['opt']:
            if opt_v is not None:
                if opt_v not in self.cons:
                    self.cons[opt_v] = self.def_cons[opt_v][1]
                    if self.verbose>0:
                        print('WARNING: value of {} which is {} is not provided as input and is set to default value of {}'
                      .format(opt_v, self.def_cons[opt_v][0], self.def_cons[opt_v][1]))

        # checking for compulsory input variables
        for req_v in _cons[method]['req']:
            if req_v not in self.cons:
                raise ValueError("""Insufficient input Error: value of {} which is {} is not provided and is required"""
                      .format(req_v, self.def_cons[req_v][0]))

        return


    def validate_constants(self):
        """
        validates whether constants are provided correctly or no
        """



def add_freq(dataframe,  name=None, _force_freq=None, method=None):
    """Add a frequency attribute to idx, through inference or directly.
    Returns a copy.  If `freq` is None, it is inferred.
    """
    idx = dataframe.index
    idx = idx.copy()
    #if freq is None:
    if idx.freq is None:
        freq = pd.infer_freq(idx)
        idx.freq = freq

        if idx.freq is None:
            if _force_freq is not None:
                dataframe = force_freq(dataframe, _force_freq, name, method=method)
            else:

                raise AttributeError('no discernible frequency found in {} for {}.  Specify'
                                     ' a frequency string with `freq`.'.format(name, name))
        else:
            print('frequency {} is assigned to {}'.format(idx.freq, name))
            dataframe.index = idx

    return dataframe


def force_freq(data_frame, freq_to_force, name, method=None):
    #TODO make method work
    #print('name is', name)
    old_nan_counts = data_frame.isna().sum()
    dr = pd.date_range(data_frame.index[0], data_frame.index[-1], freq=freq_to_force)

    df_unique = data_frame[~data_frame.index.duplicated(keep='first')] # first remove duplicate indices if present
    if method:
        df_idx_sorted = df_unique.sort_index()
        df_reindexed = df_idx_sorted.reindex(dr, method='nearest')
    else:
        df_reindexed = df_unique.reindex(dr, fill_value=np.nan)

    df_reindexed.index.freq = pd.infer_freq(df_reindexed.index)
    new_nan_counts = df_reindexed.isna().sum()
    print('Frequency {} is forced in file {} while working with {}, NaN counts changed from {} to {}'
          .format(df_reindexed.index.freq, os.path.basename(name), name, old_nan_counts.values, new_nan_counts.values))
    return df_reindexed


def split_freq(freq_str):
    match = re.match(r"([0-9]+)([a-z]+)", freq_str, re.I)
    if match:
        minutes, freq = match.groups()
        if freq == 'H':
            minutes  = int(minutes) * 60
        elif freq == 'D':
            minutes = int(minutes) * 1440
        return minutes, 'min'


import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))