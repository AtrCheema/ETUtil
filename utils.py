
from numpy import multiply, divide, add, subtract, power, sin, cos, tan, array, where, mean, sqrt
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.style"] = 'normal'   # normal/italic/oblique
import matplotlib.dates as mdates

from .convert import Temp, Wind

#: Solar constant [ MJ m-2 min-1]
SOLAR_CONSTANT = 0.0820

DegreesToRadians = 0.01745329252
MetComputeLatitudeMax = 66.5
MetComputeLatitudeMin = -66.5

# default constants
# description, default value, min, max
def_cons = {
    'lat': ['latitude in decimal degrees', None, -1, 1],
    'long': ['longitude in decimal degrees'],
    'a_s': ['fraction of extraterrestrial radiation reaching earth on sunless days', 0.23],
    'b_s': ['difference between fracion of extraterrestrial radiation reaching full-sun days and that on sunless days',
            0.5],
    'albedo': ["""Any numeric value between 0 and 1 (dimensionless), albedo of evaporative surface representing the'
           ' portion of the incident radiation that is reflected back at the surface. Default is 0.23 for'
           ' surface covered with short reference crop, which is for the calculation of Matt-Shuttleworth'
           ' reference crop evaporation.""", 0.23, 0, 1],
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
    'ap'     : ['', 2.4]
}

class Util(object):

    def __init__(self,input_df,units, constants, verbose=True):

        self.input = input_df
        self.cons = constants
        self.def_cons = def_cons
        self.SB_CONS = None
        self.daily_index=None
        self.no_of_hours = None
        self.input_freq = self.get_in_freq()
        self.units = units
        self._check_compatibility()
        self.lat_rad = self.cons['lat'] * 0.0174533 if self.cons['lat'] is not None else None  # degree to radians
        self.wind_z = constants['wind_z'] if 'wind_z' in constants else None
        self.verbose = verbose
        self.output = {}



    def get_in_freq(self):
        freq = self.input.index.freqstr
        if freq is None:
            idx = self.input.index.copy()
            _freq = pd.infer_freq(idx)
            print('Frequency inferred from input data is', _freq)
            freq = _freq
            data = self.input.copy()
            data.index.freq = _freq
            self.input = data

        if 'D' in freq:
            setattr(self, 'SB_CONS', 4.903e-9)   #  MJ m-2 day-1.
            return 'Daily'
        elif 'H' in freq:    #  (4.903/24) 10-9
            setattr(self, 'SB_CONS', 2.043e-10)   # MJ m-2 hour-1.
            return 'Hourly'
        elif 'T' in freq:
            return 'sub_hourly'
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
            return 'Monthly'
        else:
            raise ValueError('unknown frequency of input data')


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

        allowed_units = {'temp': ['centigrade', 'fahrenheit', 'kelvin'],
                         'tmin': ['centigrade', 'fahrenheit', 'kelvin'],
                         'tmax': ['centigrade', 'fahrenheit', 'kelvin'],
                         'tdew': ['centigrade', 'fahrenheit', 'kelvin'],
                         'uz':  ['MeterPerSecond', 'KilometerPerHour', 'MilesPerHour', 'InchesPerSecond',
                                   'FeetPerSecond'],
                         'daylight_hrs': ['hour'],
                         'sunshine_hrs': ['hour'],
                         'rel_hum': ['percent'],
                         'rh_min': ['percent'],
                         'rh_max': ['percent'],
                         'solar_rad': ['MegaJoulePerMeterSquarePerHour', 'LangleysPerDay'],
                         'cloud': ['']}

        for _input, _unit in self.units.items():
            if _unit not in allowed_units[_input]:
                raise ValueError('unit {} of input data {} is not allowed. Use any of {}'
                                 .format(_unit, _input, allowed_units[_input]))

        # converting temperature units to celsius
        for val in ['tmin', 'tmax', 'temp']:
            if val in self.input:
                t = Temp(self.input[val].values, self.units[val])
                self.input[val] = t.celsius

        # if 'temp' is given, it is assumed to be mean otherwise calculate mean and put it as `temp` in input dataframe.
        if 'temp' not in self.input.columns:
            if 'tmin' in self.input.columns and 'tmax' in self.input.columns:
                self.input['temp'] = mean(array([self.input['tmin'].values, self.input['tmax'].values]), axis=0)

         # make sure that we mean relative humidity calculated if possible
        if 'rel_hum' in self.input.columns:
            self.input['rh_mean'] = self.input['rel_hum']
        else:
            if 'rh_min' in self.input.columns:
                self.input['rh_mean'] = mean(array([self.input['rh_min'].values, self.input['rh_max'].values]), axis=0)

        # check units of wind speed and convert if needed
        if 'uz' in self.input:
            w = Wind(self.input['uz'].values, self.units['uz'])
            self.input['uz'] = w.MeterPerSecond

        # getting julian day
        self.input['jday'] = self.input.index.dayofyear

        if self.input_freq == 'Hourly':
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

            if 'solar_rad' in self.input.columns:
                self.input['is_day'] = where(self.input['solar_rad'].values>0.1, 1, 0)
        return


    @property
    def seconds(self):
        """finds number of seconds between two steps of input data"""
        return  (self.input.index[1]-self.input.index[0])/np.timedelta64(1, 's')


    def check_constants(self, method):
        _cons = {
            'PenPan': {'opt': ['over_est', 'albedo', 'pan_coef', 'pen_ap', 'alphaA'],
                       'req': ['lat']},

            'PenmanMonteith': {'opt': ['albedo'],
                               'req': ['lat']},

            'Abtew': {'opt': ['a_s', 'b_s', 'abtew_k'],
                      'req': ['lat']},

            'BlaneyCriddle': {'opt': [''],
                              'req': ['lat']},

            'BrutsaertStrickler': {'opt': [''],
                                   'req': ['']},

            'ChapmanAustralia': {'opt': ['ap', 'albedo', 'alphaA'],
                                 'req': ['lat', 'long']},

            'GrangerGrey': {'opt': [''],
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
                    print('WARNING: value of {} which is {} is not provide as input and is set to default value of {}'
                      .format(opt_v, self.def_cons[opt_v][0], self.def_cons[opt_v][1]))

        # checking for compulsory input variables
        for req_v in _cons[method]['req']:
            if req_v not in self.cons:
                raise ValueError("""Insufficient input Error: value of {} which is {} is not provide and is required"""
                      .format(req_v, self.def_cons[req_v][0]))

        return


    def validate_constants(self):
        """
        validates whether constants are provided correctly or no
        """


    def sol_rad_from_sun_hours(self):
        """
        Calculate incoming solar (or shortwave) radiation, *Rs* (radiation hitting a horizontal plane after
        scattering by the atmosphere) from relative sunshine duration.

        If measured radiation data are not available this method is preferable to calculating solar radiation from
        temperature. If a monthly mean is required then divide the monthly number of sunshine hours by number of
        days in the month and ensure that *et_rad* and *daylight_hours* was calculated using the day of the year
        that corresponds to the middle of the month.

        Based on equations 34 and 35 in Allen et al (1998).

        uses: Number of daylight hours [hours]. Can be calculated  using ``daylight_hours()``.
              Sunshine duration [hours]. Can be calculated  using ``sunshine_hours()``.
              Extraterrestrial radiation [MJ m-2 day-1]. Can be estimated using ``et_rad()``.
        :return: Incoming solar (or shortwave) radiation [MJ m-2 day-1]
        :rtype: float
        """

        # 0.5 and 0.25 are default values of regression constants (Angstrom values)
        # recommended by FAO when calibrated values are unavailable.
        n = self.input['sunshine_hrs']  # sunshine_hours
        N = self.daylight_fao56()       # daylight_hours
        return multiply( add(self.cons['a_s'] , multiply(divide(n , N) , self.cons['b_s'])) , self._et_rad())


    def rs(self):
        """calculate solar radiation either from temperature (as first preference) or from daily_sunshine hours
        (as second preference). Sunshine hours is given second preference because sunshine hours will remain
        same for all years if sunshine hours data is not provided (which is difficult to obtain), but temperature data
        which is easy to obtain and thus will be different for different years"""
        rs = None
        if 'solar_rad' not in self.input.columns:
            if 'tmin' in self.input.columns and 'tmax' in self.input.columns:
                rs = self._sol_rad_from_t()
                if self.verbose:
                    print("solar radiation is calculated from temperature")
            elif 'sunshine_hrs' in self.input.columns:
                rs = self.sol_rad_from_sun_hours()
                if self.verbose:
                    print("Sunshine hour data is used for calculating incoming solar radiation")
        else:
            rs = self.input['solar_rad']
        if rs is None:
            raise ValueError("Unable to calculate solar radiation data")
        self.input['solar_rad'] = rs
        return rs


    def daylight_fao56(self):
        """get number of maximum hours of sunlight for a given latitude using equation 34 in Fao56.
        Annual variation of sunlight hours on earth are plotted in figre 14 in ref 1.

        1) http://www.fao.org/3/X0490E/x0490e07.htm"""
        ws = self.sunset_angle()
        hrs = (24/3.14) * ws
        # if self.input_freq == 'Monthly':
        #     df = pd.DataFrame(hrs, index=self.daily_index)
        #     hrs = df.resample('M').mean().values.reshape(-1,)
        return hrs


    def _et_rad(self):
        """
        Estimate extraterrestrial radiation (*Ra*, 'top of the atmosphere radiation').

        For daily, it is based on equation 21 in Allen et al (1998). If monthly mean radiation is required make sure *sol_dec*. *sha*
         and *irl* have been calculated using the day of the year that corresponds to the middle of the month.

        **Note**: From Allen et al (1998): "For the winter months in latitudes greater than 55 degrees (N or S), the equations have limited validity.
        Reference should be made to the Smithsonian Tables to assess possible deviations."

        :return: extraterrestrial radiation [MJ m-2 timestep-1]
        :rtype: float
        """
        ra = -9999
        if self.input_freq=='Hourly':
            j = (3.14/180) * self.cons['lat']  # eq 22  phi
            dr = self.inv_rel_dist_earth_sun() # eq 23
            d = self.dec_angle  # eq 24    # gamma
            w1,w2 = self.solar_time_angle()
            t1 = (12*60)/math.pi
            t2 = multiply(t1, multiply(SOLAR_CONSTANT, dr))
            t3 = multiply(subtract(w2,w1), multiply(sin(j), sin(d)))
            t4 = subtract(sin(w2), sin(w1))
            t5 = multiply(multiply(cos(j), cos(d)), t4)
            t6 = add(t5, t3)
            ra = multiply(t2, t6)   # eq 28

        elif self.input_freq == 'Daily':
            sol_dec = self.dec_angle  # based on julian day
            sha = self.sunset_angle()   # sunset hour angle[radians], based on latitude
            ird = self.inv_rel_dist_earth_sun()
            tmp1 = (24.0 * 60.0) / math.pi
            tmp2 = multiply(sha , multiply(math.sin(self.lat_rad) , sin(sol_dec)))
            tmp3 = multiply(math.cos(self.lat_rad) , multiply(cos(sol_dec) , sin(sha)))
            ra = multiply(tmp1 , multiply(SOLAR_CONSTANT , multiply(ird , add(tmp2 , tmp3)))) # eq 21
        self.input['ra'] = ra
        return ra


    def _sol_rad_from_t(self, coastal=False):
        """Estimate incoming solar (or shortwave) radiation  [Mj m-2 day-1] , *Rs*, (radiation hitting  a horizontal plane after
        scattering by the atmosphere) from min and max temperature together with an empirical adjustment coefficient for
        'interior' and 'coastal' regions.

        The formula is based on equation 50 in Allen et al (1998) which is the Hargreaves radiation formula (Hargreaves
        and Samani, 1982, 1985). This method should be used only when solar radiation or sunshine hours data are not
        available. It is only recommended for locations where it is not possible to use radiation data from a regional
        station (either because climate conditions are heterogeneous or data are lacking).

        **NOTE**: this method is not suitable for island locations due to the
        moderating effects of the surrounding water. """

        # Determine value of adjustment coefficient [deg C-0.5] for
        # coastal/interior locations
        if coastal:     # for 'coastal' locations, situated on or adjacent to the coast of a large l
            adj = 0.19  # and mass and where air masses are influenced by a nearby water body,
        else:           #  for 'interior' locations, where land mass dominates and air
            adj = 0.16  # masses are not strongly influenced by a large water body

        et_rad = None
        cs_rad = None
        if 'et_rad' not in self.input:
            et_rad = self._et_rad()
            self.input['et_rad'] = et_rad
        if 'cs_rad' not in self.input:
            cs_rad = self._cs_rad()
            self.input['cs_rad'] = cs_rad
        sol_rad = multiply(adj , multiply(sqrt(subtract(self.input['tmax'].values , self.input['tmin'].values)) , et_rad))

        # The solar radiation value is constrained by the clear sky radiation
        return np.min( array([sol_rad, cs_rad]), axis=0)


    def sunset_angle(self):
        """calculates sunset hour angle in radians given by Equation 25  in Fao56 (1)

        1): http://www.fao.org/3/X0490E/x0490e07.htm"""
        if 'sha' not in self.input:
            j = (3.14/180.0) * self.cons['lat']           # eq 22
            d = self.dec_angle       # eq 24, declination angle
            angle = np.arccos(-tan(j)*tan(d))      # eq 25
            self.input['sha'] = angle
        else:
            angle = self.input['sha'].values
        return angle


    def inv_rel_dist_earth_sun(self):
        """
        Calculate the inverse relative distance between earth and sun from day of the year.
        Based on FAO equation 23 in Allen et al (1998).
        ird = 1.0 + 0.033 * cos( [2pi/365] * j )

        :return: Inverse relative distance between earth and the sun
        :rtype: np array
        """
        if 'ird' not in self.input:
            inv1 = multiply(2*math.pi/365.0 ,  self.input['jday'].values)
            inv2 = cos(inv1)
            inv3 = multiply(0.033, inv2)
            ird = add(1.0, inv3)
            self.input['ird'] = ird
        else:
            ird = self.input['ird']
        return ird


    @property
    def dec_angle(self):
        """finds solar declination angle"""
        if 'sol_dec' not in self.input:
            if self.input_freq == 'monthly':
                solar_dec =  array(0.409 * sin(2*3.14 * self.daily_index.dayofyear/365 - 1.39))
            else:
                solar_dec = 0.409 * sin(2*3.14 * self.input['jday'].values/365 - 1.39)       # eq 24, declination angle
            self.input['solar_dec'] = solar_dec
        else:
            solar_dec = self.input['solar_dec']
        return solar_dec


    def solar_time_angle(self):
        """
        returns solar time angle at start, mid and end of period using equation 29, 31 and 30 respectively in Fao
        w = pi/12 [(t + 0.06667 ( lz-lm) + Sc) -12]
        lm = longitude of the measurement site [degrees west of Greenwich]
        lz = longitude of the centre of the local time zone [degrees west of Greenwich]
        """

        #TODO find out how to calculate lz
        lz = 15 * round(self.cons['long'] / 15)  # https://github.com/djlampert/PyHSPF/blob/c3c123acf7dba62ed42336f43962a5e4db922422/src/pyhspf/preprocessing/etcalculator.py#L610
        lm = self.cons['long']
        t1 = 0.0667*(lz-lm)
        t2 = self.input['half_hr'].values + t1 + self.solar_time_cor()
        t3 = subtract(t2, 12)
        w = multiply((math.pi/12.0) , t3)     # eq 31

        w1 = subtract(w, divide(multiply(math.pi , self.input['t1']).values, 24.0))  # eq 29
        w2 = add(w, divide(multiply(math.pi, self.input['t1']).values, 24.0))   # eq 30
        return w1,w2


    def _cs_rad(self):
        """
        Estimate clear sky radiation from altitude and extraterrestrial radiation.

        Based on equation 37 in Allen et al (1998) which is recommended when calibrated Angstrom values are not available.
        et_rad is Extraterrestrial radiation [MJ m-2 day-1]. Can be estimated using ``et_rad()``.

        :return: Clear sky radiation [MJ m-2 day-1]
        :rtype: float
        """
        return (0.00002 * self.cons['altitude'] + 0.75) * self._et_rad()


    def solar_time_cor(self):
        """seasonal correction for solar time by implementation of eqation 32"""
        upar = multiply((2*math.pi), subtract(self.input['jday'].values, 81))
        b =  divide(upar, 364)   # eq 33
        t1 = multiply(0.1645, sin(multiply(2, b)))
        t2 = multiply(0.1255, cos(b))
        t3 = multiply(0.025, sin(b))
        return t1-t2-t3   # eq 32


    def _wind_2m(self, method='fao56',z_o=0.001):
        """
        converts wind speed (m/s) measured at height z to 2m using either FAO 56 equation 47 or McMohan eq S4.4.
         u2 = uz [ 4.87/ln(67.8z-5.42) ]         eq 47 in [1], eq S5.20 in [2].
         u2 = uz [ln(2/z_o) / ln(z/z_o)]         eq S4.4 in [2]

        :param `method` string, either of `fao56` or `mcmohan2013`. if `mcmohan2013` is chosen then `z_o` is used
        :param `z_o` float, roughness height. Default value is from [2]

        :return: Wind speed at 2 m above the surface [m s-1]

        [1] http://www.fao.org/3/X0490E/x0490e07.htm
        [2] McMahon, T., Peel, M., Lowe, L., Srikanthan, R. & McVicar, T. 2012. Estimating actual, potential, reference crop
            and pan evaporation using standard meteorological data: a pragmatic synthesis. Hydrology and Earth System
            Sciences Discussions, 9, 11829-11910. https://www.hydrol-earth-syst-sci.net/17/1331/2013/hess-17-1331-2013-supplement.pdf
        """

        if self.wind_z is None:  # if value of height at which wind is measured is not given, then don't convert
            if self.verbose:
                print("WARNING: givn wind data is not at 2 meter but `wind_z` is also not given is assuming wind given"
                      " as measured at 2m height")
            return self.input['uz'].values
        else:
            if method == 'fao56':
                return multiply(self.input['uz'] , (4.87 / math.log((67.8 * self.wind_z) - 5.42)))
            else:
                return multiply(self.input['uz'].values, math.log(2/z_o) / math.log(self.wind_z/z_o))


    def net_rad(self,rs,ea):
        """
            Calculate daily net radiation at the crop surface, assuming a grass reference crop.

        Net radiation is the difference between the incoming net shortwave (or solar) radiation and the outgoing net
        longwave radiation. Output can be converted to equivalent evaporation [mm day-1] using ``energy2evap()``.

        Based on equation 40 in Allen et al (1998).

        :uses rns: Net incoming shortwave radiation [MJ m-2 day-1]. Can be
                   estimated using ``net_in_sol_rad()``.
              rnl: Net outgoing longwave radiation [MJ m-2 day-1]. Can be
                   estimated using ``net_out_lw_rad()``.
        :return: net radiation [MJ m-2 timestep-1].
        :rtype: float
        """
        rns = self.net_in_sol_rad(rs)
        rnl = self.net_out_lw_rad(rs=rs, ea=ea)

        return subtract(rns, rnl)


    def net_in_sol_rad(self,rs):
        """
        Calculate net incoming solar (or shortwave) radiation (*Rns*) from gross incoming solar radiation, assuming a grass
         reference crop.

        Net incoming solar radiation is the net shortwave radiation resulting from the balance between incoming and
         reflected solar radiation. The output can be converted to equivalent evaporation [mm day-1] using
        ``energy2evap()``.

        Based on FAO equation 38 in Allen et al (1998).
        Rns = (1-a)Rs

        uses Gross incoming solar radiation [MJ m-2 day-1]. If necessary this can be estimated using functions whose name
            begins with 'solar_rad_from'.
        :param rs: solar radiation
        albedo: Albedo of the crop as the proportion of gross incoming solar
            radiation that is reflected by the surface. Default value is 0.23,
            which is the value used by the FAO for a short grass reference crop.
            Albedo can be as high as 0.95 for freshly fallen snow and as low as
            0.05 for wet bare soil. A green vegetation over has an albedo of
            about 0.20-0.25 (Allen et al, 1998).
        :return: Net incoming solar (or shortwave) radiation [MJ m-2 day-1].
        :rtype: float
        """
        return multiply((1 - self.cons['albedo']) ,rs)


    def net_out_lw_rad(self,rs, ea ):
        """
        Estimate net outgoing longwave radiation.

        This is the net longwave energy (net energy flux) leaving the earth's surface. It is proportional to the
        absolute temperature of the surface raised to the fourth power according to the Stefan-Boltzmann law. However,
        water vapour, clouds, carbon dioxide and dust are absorbers and emitters of longwave radiation. This function
        corrects the Stefan- Boltzmann law for humidity (using actual vapor pressure) and cloudiness (using solar
        radiation and clear sky radiation). The concentrations of all other absorbers are assumed to be constant.

        The output can be converted to equivalent evaporation [mm timestep-1] using  ``energy2evap()``.

        Based on FAO equation 39 in Allen et al (1998).

        uses: Absolute daily minimum temperature [degrees Kelvin]
              Absolute daily maximum temperature [degrees Kelvin]
              Solar radiation [MJ m-2 day-1]. If necessary this can be estimated using ``sol+rad()``.
              Clear sky radiation [MJ m-2 day-1]. Can be estimated using  ``cs_rad()``.
              Actual vapour pressure [kPa]. Can be estimated using functions with names beginning with 'avp_from'.
        :param ea: actual vapour pressure, can be calculated using method avp_from
        :param rs: solar radiation
        :return: Net outgoing longwave radiation [MJ m-2 timestep-1]
        :rtype: float
        """
        if 'tmin' in self.input.columns and 'tmax' in self.input.columns:
            added = add(power(self.input['tmax'].values+273.16, 4), power(self.input['tmin'].values+273.16, 4))
            divided = divide(added, 2.0)
        else:
            divided = power(self.input['temp'].values+273.16, 4.0)

        tmp1 = multiply(self.SB_CONS , divided)
        tmp2 = subtract(0.34 , multiply(0.14 , sqrt(ea)))
        tmp3 = subtract(multiply(1.35 , divide(rs , self._cs_rad())) , 0.35)
        return multiply(tmp1 , multiply(tmp2 , tmp3))  # eq 39


    def soil_heat_flux(self, rn=None):
        if self.input_freq=='Daily':
            return 0.0
        elif self.input_freq == 'Hourly':
            Gd = multiply(0.1, rn)
            Gn = multiply(0.5, rn)
            return where(self.input['is_day']==1, Gd, Gn)
        elif self.input_freq == 'Monthly':
            pass


    def cleary_sky_rad(self, a_s=None, b_s=None):
        """clear sky radiation Rso"""

        if a_s is None:
            rso = multiply(0.75 + 2e-5*self.cons['altitude'], self._et_rad())  # eq 37
        else:
            rso = multiply(a_s+b_s, self._et_rad())  # eq 36
        return rso


    def sat_vp_fao56(self, temp):
        """calculates saturation vapor pressure (*e_not*) as given in eq 11 of FAO 56 at a given temp which must be in
         units of centigrade.
        using Tetens equation
        es = 0.6108 * exp((17.26*temp)/(temp+273.3))

        Murray, F. W., On the computation of saturation vapor pressure, J. Appl. Meteorol., 6, 203-204, 1967.
        """
        e_not_t = multiply(0.6108, np.exp( multiply(17.26939, temp) / add(temp , 237.3)))
        return e_not_t


    def mean_sat_vp_fao56(self):
        """ calculates mean saturation vapor pressure (*es*) for a day, weak or month according to eq 12 of FAO 56 using
        tmin and tmax which must be in centigrade units
        """
        es_tmax = self.sat_vp_fao56(self.input['tmin'].values)
        es_tmin = self.sat_vp_fao56(self.input['tmax'].values)
        es = mean(array([es_tmin, es_tmax]), axis=0)
        return es


    def sat_vpd(self, temp):
        """calculates saturated vapor density at the given temperature.
        """
        esat = self.sat_vp_fao56(temp)
        # multiplying by 10 as in WDMUtil nad Lu et al, they used 6.108 for calculation of saturation vapor pressura
        # while the real equation for calculation of vapor pressure has '0.6108'. I got correct result for Hamon etp when
        # I calculated sat_vp_fao56 with 6.108. As I have put 0.6108 for sat_vp_fao56 calculation, so multiplying with 10
        # here, although not sure why to multiply with 10.
        return multiply(divide(multiply(216.7, esat), add(temp, 273.3)), 10)


    @property
    def atm_pressure(self):
        """
        Estimate atmospheric pressure from altitude.

        Calculated using a simplification of the ideal gas law, assuming 20 degrees Celsius for a standard atmosphere.
         Based on equation 7, page 62 in Allen et al (1998).

        :return: atmospheric pressure [kPa]
        :rtype: float
        """
        tmp = (293.0 - (0.0065 * self.cons['altitude'])) / 293.0
        return math.pow(tmp, 5.26) * 101.3


    def slope_sat_vp(self, t):
        """
        slope of the relationship between saturation vapour pressure and temperature for a given temperature
        according to equation 13.

        delta = 4098 [0.6108 exp(17.27T/T+237.3)] / (T+237.3)^2

        :param t: Air temperature [deg C]. Use mean air temperature for use in Penman-Monteith.
        :return: Saturation vapour pressure [kPa degC-1]
        """
        to_exp = divide(multiply(17.27, t), add(t, 237.3))
        tmp = multiply(4098 , multiply(0.6108 , np.exp(to_exp)))
        return divide(tmp , power( add(t , 237.3), 2))


    @property
    def psy_const(self):
        """
        Calculate the psychrometric constant.

        This method assumes that the air is saturated with water vapour at the minimum daily temperature. This
        assumption may not hold in arid areas.

        Based on equation 8, page 95 in Allen et al (1998).

        uses Atmospheric pressure [kPa].
        :return: Psychrometric constant [kPa degC-1].
        :rtype: array
        """
        return multiply(0.000665 , self.atm_pressure)


    def avp_from_rel_hum(self):
        """
        Estimate actual vapour pressure (*ea*) from saturation vapour pressure and relative humidity.

        Based on FAO equation 17 in Allen et al (1998).
        ea = [ e_not(tmin)RHmax/100 + e_not(tmax)RHmin/100 ] / 2

        uses  Saturation vapour pressure at daily minimum temperature [kPa].
              Saturation vapour pressure at daily maximum temperature [kPa].
              Minimum relative humidity [%]
              Maximum relative humidity [%]
        :return: Actual vapour pressure [kPa]
        :rtype: float
        http://www.fao.org/3/X0490E/x0490e07.htm#TopOfPage
        """
        avp = 0.0
        if self.input_freq=='Hourly': # use equation 54
            avp = multiply(self.sat_vp_fao56(self.input['temp'].values), divide(self.input['rel_hum'].values, 100.0))
        elif self.input_freq=='Daily':
            if 'rh_min' in self.input.columns and 'rh_max' in self.input.columns:
                tmp1 = multiply(self.sat_vp_fao56(self.input['tmin'].values) , divide(self.input['rh_max'].values , 100.0))
                tmp2 = multiply(self.sat_vp_fao56(self.input['tmax'].values) , divide(self.input['rh_min'].values , 100.0))
                avp = divide(add(tmp1 , tmp2) , 2.0)
            elif 'rel_hum' in self.input.columns:
                # calculation actual vapor pressure from mean humidity
                # equation 19
                t1 = divide(self.input['rel_hum'].values, 100)
                t2 = divide(add(self.sat_vp_fao56(self.input['tmax'].values), self.sat_vp_fao56(self.input['tmin'].values)), 2.0)
                avp = multiply(t1,t2)

        self.input['ea'] = avp
        return avp


    def tdew_from_t_rel_hum(self):
        """Calculates the dew point temperature given temperature and relative humidity.  """

    def plot_etp(self, freq='Daily'):

        fig, ax = plt.subplots(1)
        for k, v in self.output.items():
            if freq in k:
                to_plot = v
                ax.plot(to_plot, label=k)
                ax.legend(loc="best",fontsize=12)
                ax.set_xlabel("Time Period", fontsize=12)
                ax.set_ylabel("Evapotranspiration (mm)", fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=12)
                loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
                ax.xaxis.set_major_locator(loc)
                fmt = mdates.AutoDateFormatter(loc)
                ax.xaxis.set_major_formatter(fmt)
                ax.tick_params(axis="both", which='major', labelsize=15)
                ax.set_title('{} Evapotranspiration'.format(freq), fontsize=20)
        plt.show()


    def check_output_freq(self, method, et):
        """calculate ET at all frequencies other than `input_freq` but based on `input_freq` and method."""
        if not isinstance(et, np.ndarray):
            et = et.values
        et = pd.DataFrame(et, index=self.input.index, columns=['pet'])
        if self.input_freq == 'Daily':
            self.output['ET_' + method + '_Daily'] = et
            self.output['ET_' + method + '_Hourly'] = self.resample(et, out_freq='Hourly')
            self.output['ET_' + method + '_Monthly'] = self.resample(et, out_freq='Monthly')
            self.output['ET_' + method + '_Annualy'] = self.resample(et, out_freq='Annualy')

        elif self.input_freq == 'Hourly':
            self.output['ET_' + method + '_Hourly'] = et
            self.output['ET_' + method + '_Daily'] = self.resample(et, out_freq='Daily')
            self.output['ET_' + method + '_Monthly'] = self.resample(et, out_freq='Monthly')
            self.output['ET_' + method + '_Annualy'] = self.resample(et, out_freq='Annualy')

        elif self.input_freq == 'Monthly':
            self.output['ET_' + method + '_Monthly'] = et
            self.output['ET_' + method + '_Hourly'] = self.resample(et, out_freq='Hourly')
            self.output['ET_' + method + '_Daily'] = self.resample(et, out_freq='Daily')
            self.output['ET_' + method + '_Annualy'] = self.resample(et, out_freq='Annualy')


    def resample(self, df, out_freq):
        df = df.copy()
        out_df = pd.DataFrame()
        in_freq = self.input_freq
        if in_freq == 'Daily':
            if out_freq == 'Hourly':
                out_df = self.dis_sol_pet(df, 2)
            elif out_freq == 'sub-hourly':
                pass  # increase time step
            elif out_freq == 'Monthly':
                out_df = df.resample('M').sum()
            elif out_freq == 'Annualy':
                out_df = df.resample('365D').sum()

        elif in_freq == 'Hourly':
            if out_freq == 'Daily':
                out_df = df.resample('D').sum()
            if out_freq == 'Monthly':
                out_df = df.resample('M').sum()
            if out_freq == 'Annualy':
                out_df = df.resample('A').sum()

        elif in_freq == 'Monthly':
            if out_freq == 'Daily':
                out_df = divide(df['pet'].values, self.input.index.days_in_month)
            elif out_freq == 'Hourly':
                daily = pd.DataFrame(divide(df['pet'].values, self.input.index.days_in_month), index=self.input.index, columns=['pet'])
                out_df = self.dis_sol_pet(daily, 2)
            elif out_freq == 'sub-hourly':
                pass
            elif out_freq == 'Annualy':
                out_df = df.resample('365D').sum()

        df.pop('pet') # removing columns which was already present in df
        return out_df


    def dis_sol_pet(self, InTs, DisOpt):
        """
        Follows the code from [1] to disaggregate solar radiation and PET from daily to hourly time step.
        uses Latitude `float` latitude in decimal degrees, should be between -66.5 and 66.5
        :param InTs a pandas dataframe of series which contains hourly data with a column named `pet` to be disaggregated
        :param DisOpt `int` 1 or 2, 1 means solar radiation, 2 means PET

        ``example
        Lat = 45.2
        dis_opt = 2
        in_ts = pd.DataFrame(np.array([20.0, 30.0]), index=pd.date_range('20110101', '20110102', freq='D'), columns=['pet'])
        hr_pet = dis_sol_pet(in_ts, dis_opt, Lat)
        array([0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.44944907, 1.88241394,
               3.09065427, 3.09065427, 3.09065427, 3.09065427, 3.09065427,
               1.88241394, 0.44944907, 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.68743966, 2.82968249, 4.62820549,
               4.62820549, 4.62820549, 4.62820549, 4.62820549, 2.82968249,
               0.68743966, 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        ])

        # solar radiation
        in_ts = pd.DataFrame(np.array([2388.6, 2406.9]), index=pd.date_range('20111201', '20111202', freq='D'), columns=['sol_rad'])
        hr_pet = dis_sol_pet(in_ts, dis_opt, Lat)
        hr_pet['sol_rad_hr'].values
        array([  0.        ,   0.        ,   0.        ,   0.        ,
                 0.        ,   0.        ,   0.        ,   0.        ,
                61.82288753, 228.50842499, 364.2825187 , 364.2825187 ,
               364.2825187 , 364.2825187 , 364.2825187 , 228.50842499,
                61.82288753,   0.        ,   0.        ,   0.        ,
                 0.        ,   0.        ,   0.        ,   0.        ,
                 0.        ,   0.        ,   0.        ,   0.        ,
                 0.        ,   0.        ,   0.        ,   0.        ,
                60.84759744, 229.60755169, 367.94370722, 367.94370722,
               367.94370722, 367.94370722, 367.94370722, 229.60755169,
                60.84759744,   0.        ,   0.        ,   0.        ,
                 0.        ,   0.        ,   0.        ,   0.        ])

        ``


        *Note There is a small bug in disaggregation error. The disaggregated time series is slightly more than input time
        series. Don't fret, the error/overestimation is not more than 0.1% unless you are using unrealistic values. This
        accuracy can be found by using  `disagg_accuracy` attribute of this class. The output values are same as those
         obtained from using SARA timeseries utility, however, hourly pet calculated from SARA is also slightly
        more than input.

        [1] https://github.com/respec/BASINS/blob/4356aa9481eb7217cb2cbc5131a0b80a932907bf/atcMetCmp/modMetCompute.vb#L653
        """

        HrVals = np.full(24, np.nan)
        InumValues = len(InTs)    # number of days
        OutTs = np.full(InumValues*24, np.nan)
        HrPos = 0

        if MetComputeLatitudeMin > self.cons['lat'] > MetComputeLatitudeMax:
            raise ValueError('Latitude should be between -66.5 and 66.5')

        LatRdn = self.lat_rad #Latitude * DegreesToRadians

        if DisOpt == 2:
            InCol = 'pet'
            OutCol = 'pet_hr'
        else:
            InCol = 'sol_rad'
            OutCol = 'sol_rad_hr'

        for i in range(InumValues):

            # This formula for Julian Day which is slightly different what exact julian day obtained from pandas datetime
            # index.If  pandas datetime dayofyear is used, this gives more error in disaggregation.
            JulDay = 30.5 * (InTs.index.month[i] - 1) + InTs.index.day[i]

            Phi = LatRdn
            AD = 0.40928 * cos(0.0172141 * (172.0 - JulDay))
            SS = sin(Phi) * sin(AD)
            CS = cos(Phi) * cos(AD)
            X2 = -SS / CS
            Delt = 7.6394 * (1.5708 - np.arctan(X2 / sqrt(1.0 - X2 ** 2.0)))
            SunR = 12.0 - Delt / 2.0

            # develop hourly distribution given sunrise, sunset and length of day(DELT)
            DTR2 = Delt / 2.0
            DTR4 = Delt / 4.0
            CRAD = 0.6666 / DTR2
            SL = CRAD / DTR4
            TRise = SunR
            TR2 = TRise + DTR4
            TR3 = TR2 + DTR2
            TR4 = TR3 + DTR4

            if DisOpt ==1:
                RADDST(TRise, TR2, TR3, TR4, CRAD, SL, InTs[InCol].values[i], HrVals)
            else:
                PETDST(TRise, TR2, TR3, TR4, CRAD, SL, InTs[InCol].values[i], HrVals)

            for j in range(24):
                OutTs[HrPos + j] = HrVals[j]

            HrPos = HrPos + 24



        ndf = pd.DataFrame(data=OutTs, index=pd.date_range(InTs.index[0], periods=len(OutTs), freq='H'),
                           columns=[OutCol])
        ndf[InCol] = InTs
        accuracy = ndf[InCol].sum() / ndf[OutCol].sum() * 100
        setattr(self, 'disagg_accuracy', accuracy)
        return ndf


def RADDST(TRise, TR2, TR3, TR4, CRAD, SL, DayRad, HrRad):
    """distributes daily solar radiation to hourly, baed on HSP (Hydrocomp, 1976).

    Hydrocomp, Inc. (1976). Hydrocomp Simulation Programming Operations Manual.
    """

    for ik in range(24):
        rk = ik
        if rk>TRise:
            if rk > TR2:
                if rk > TR3:
                    if rk > TR4:
                        HrRad[ik] = 0.0
                    else:
                        HrRad[ik] = (CRAD - (rk-TR3) * SL) * DayRad
                else:
                    HrRad[ik] = CRAD * DayRad
            else:
                HrRad[ik] = (rk - TRise) * SL * DayRad
        else:
            HrRad[ik] = 0.0

    return


def PETDST(TRise, TR2, TR3, TR4, CRAD, SL, DayPet, HrPet):
    """
    Distributes PET from daily to hourly scale. The code is adopted from [1] which uses method of [2].
    DayPet float, input daily pet
    HrPet = ouput array of hourly PET

    [1]  https://github.com/respec/BASINS/blob/4356aa9481eb7217cb2cbc5131a0b80a932907bf/atcMetCmp/modMetCompute.vb#L1001
    [2] Hydrocomp, Inc. (1976). Hydrocomp Simulation Programming Operations Manual.
    """

    CURVE = np.full(24, np.nan)

    # calculate hourly distribution curve
    for ik in range(24):
        RK = ik
        if RK > TRise:
            if RK > TR2:
                if RK > TR3:
                    if RK > TR4:
                        CURVE[ik] = 0.0
                        HrPet[ik] = CURVE[ik]
                    else:
                        CURVE[ik] = (CRAD - (RK-TR3) * SL)
                        HrPet[ik] = CURVE[ik] * DayPet
                else:
                    CURVE[ik] = CRAD
                    HrPet[ik] = CURVE[ik] * DayPet
            else:
                CURVE[ik] = (RK - TRise) * SL
                HrPet[ik] = CURVE[ik] * DayPet
        else:
            CURVE[ik] = 0.0
            HrPet[ik] = CURVE[ik]

        if HrPet[ik]>40.0:
            print('bad Hourly Value ', HrPet[ik])


class MortonRadiation(object):
    """
    Calculate radiation variables
    """
    def __init__(self,input_df, model, constants):
        self.input = input_df
        self.model = model
        self.cons = constants
        self.t_mo = self.get_monthly_t()

    def __call__(self, *args, **kwargs):

        delta = multiply( 4098, divide(multiply(0.6108, np.exp(add(17.27, divide(self.t_mo, add(self.t_mo, 237.3))))), power(add(self.t_mo, 237.3), 2)))
        deltas = sin(multiply(2, multiply(divide(3.14, 365), self.J)))
        omegas = 'cal'
        PA = 'aggregate precip'

        if self.model in ['CRLE', 'CRWE']:
            epsilonMo = 0.97
            fz = 25
            b0 = 1.12
            b1 = 13
            b2 = 1.12

        ptops = ((288 - 0.0065 * self.cons['altitude'])/288)**5.256


    def get_monthly_t(self):
        """get monthly temperature"""
        t_mo = self.input['temp'].resample('M').mean()
        return t_mo

    def tdew(self):
        if 'tdew' in self.input.columns:
            tdew_mon = None  # aggreate
        elif 'va' in self.input.columns:
            vabar_mo =None
            tdew_mon = vabar_mo
        else:
            vabar = None
            vabar_mo = 'aggregate'
            tdew_mon = 'from vabar_mo'

    def vas(self):
        if 'vs' not in self.input.columns:
            vs_tmax = 'from tmax'
            vs_tmin = 'from tmin'
            vas = mean(vs_tmax, vs_tmin)

