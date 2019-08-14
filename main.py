# https://www.intechopen.com/books/advanced-evapotranspiration-methods-and-applications
# https://www.intechopen.com/books/current-perspective-to-predict-actual-evapotranspiration
# https://rdrr.io/cran/Evapotranspiration/man/
# https://www.ncl.ucar.edu/Document/Functions/index.shtml
import pandas as pd
import numpy as np
from numpy import multiply, divide, add, subtract, power, sin, cos, tan, array, where, mean, sqrt
import math

from convert import Temp, Wind

DegreesToRadians = 0.01745329252
MetComputeLatitudeMax = 66.5
MetComputeLatitudeMin = -66.5

#: Solar constant [ MJ m-2 min-1]
SOLAR_CONSTANT = 0.0820

# Latent heat of vaporisation [MJ.Kg-1]
LAMBDA = 2.45

# Stefan Boltzmann constant [MJ K-4 m-2 day-1]
STEFAN_BOLTZMANN_CONSTANT = 0.000000004903
"""Stefan Boltzmann constant [MJ K-4 m-2 day-1]"""

class ReferenceET(object):
    """calculates reference evapotranspiration using the `input_method`
    # Arguments
     :param `input_df`: must be a pandas dataframe with some or all following values.
            temp: air temperature
            uz: wind speed
            rel_hum: relative humidity
            solar_rad: solar radiation
            daylight_hrs: number of daylight hours in a day.
            sunshine_hrs: actual sunshine hours
            cloud:   cloud cover
            rh_max: maximum relative humidty
            rh_min: minimum relative humidity
            u2: wind speed at measured at two meters
            tdew: dew point temperature

     :param `input_units`: a dictionary containing units for all input time series data.
              it must have one or all of following keys and corresponding values
            temp -- centigrade, fahrenheit, kelvin
            tmin -- centigrade, fahrenheit, kelvin
            tmax -- centigrade, fahrenheit, kelvin
            tdew -- centigrade, fahrenheit, kelvin
            uz -- 'MeterPerSecond', 'KilometerPerHour', 'MilesPerHour', 'InchesPerSecond',  'FeetPerSecond'
            rel_hum: relative humidity
            rh_max:
            rh_min:
            solar_rad:
            cloud:
            daylight_hrs: hour
            sunshine_hrs: hour
    :param `lat` float, latitude of measured data in degree decimals. May not be always required. Depends upon the
             method used and input data.
    :param `long` float, logitude of measurement site in decimal degrees [degrees west of Greenwich]. It is required for
             hourly PET calculation using Penman-Monteith method.
    :param `alatitude` float, Elevation/altitude above sea level [m]
    :param `wind_z` float Height of wind measurement above ground surface [m]
     'method': str, method to be employed to calculated reference evapotranspiration. Must be one of following
        `pm`: penman-monteith fao 56 method
        `thornwait`:
            """

    def __init__(self, input_df, units,  lat = None, altitude=None, wind_z=None, long=None, verbose=True):
        self.input = input_df
        self.input_freq = self.get_in_freq()
        self.units = units
        self._check_compatibility()
        self.lat = lat
        self.lat_rad = self.lat * 0.0174533 if self.lat is not None else None  # degree to radians
        self.altitude = altitude
        self.wind_z = wind_z
        self.long = long
        self.verbose = verbose


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
            return 'daily'
        elif 'H' in freq:    #  (4.903/24) 10-9
            setattr(self, 'SB_CONS', 2.043e-10)   # MJ m-2 hour-1.
            return 'hourly'
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
            return 'monthly'
        else:
            raise ValueError('unknown frequency of input data')

    def _check_compatibility(self):
        """units are also converted here."""

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

        if self.input_freq == 'hourly':
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


    def Abtew(self, k=0.52, a_s=0.23, b_s=0.5):
        """daily etp using equation 3 in [1]. `k` is a dimentionless coefficient.

        :param `k` coefficient, default value taken from [1]
        :param `a_s` fraction of extraterrestrial radiation reaching earth on sunless days
        :param `b_s` difference between fracion of extraterrestrial radiation reaching full-sun days
                 and that on sunless days.

         [1] Abtew, W. (1996). EVAPOTRANSPIRATION MEASUREMENTS AND MODELING FOR THREE WETLAND SYSTEMS IN
             SOUTH FLORIDA 1. JAWRA Journal of the American Water Resources Association, 32(3),
             465-473. https://doi.org/10.1111/j.1752-1688.1996.tb04044.x

         """

        if 'solar_rad' not in self.input.columns:
            if 'sunshine_hrs' in self.input.columns:
                rs = self.sol_rad_from_sun_hours(a_s=a_s, b_s=b_s)
                if self.verbose:
                    print("Sunshine hour data is used for calculating incoming solar radiation")
            else:
                rs = self._sol_rad_from_t()
                if self.verbose:
                    print("solar radiation is calculated from temperature")
            self.input['solar_rad'] = rs
        else:
            rs = self.input['solar_rad']
        pet = multiply(k, divide(rs, LAMBDA))
        self.input['pet_abtew'] = pet
        return pet


    def Blaney_Criddle(self):
        """using formulation of Blaney-Criddle for daily reference crop ETP using monthly mean tmin and tmax.
        Inaccurate under extreme climates. underestimates in windy, dry and sunny conditions and overestimates under calm, humid
        and clouded conditions.

        [2] Allen, R. G. and Pruitt, W. O.: Rational use of the FAO Blaney-Criddle Formula, J. Irrig. Drain. E. ASCE,
             112, 139–155, 1986."""
        N = self.daylight_fao56()  # mean daily percentage of annual daytime hours
        u2 = self._wind_2m 
        return multiply(N, add(multiply(0.46, self.input['temp'].values), 8.0))


    def Chapman_Australia(self):
        """using formulation of [1],

        [1] Chapman, T. 2001, Estimation of evaporation in rainfall-runoff models,
            in F. Ghassemi, D. Post, M. SivapalanR. Vertessy (eds), MODSIM2001: Integrating models for Natural
             Resources Management across Disciplines, Issues and Scales, MSSANZ, vol. 1, pp. 293-298. """

    def Turc(self):
        """
        using Turc 1961 formulation, originaly developed for southern France and Africa.


        [1] Turc, L. (1961). Estimation of irrigation water requirements, potential evapotranspiration: a simple climatic
             formula evolved up to date. Ann. Agron, 12(1), 13-49.
        """

    @property
    def atm_pressure(self):
        """
        Estimate atmospheric pressure from altitude.

        Calculated using a simplification of the ideal gas law, assuming 20 degrees Celsius for a standard atmosphere.
         Based on equation 7, page 62 in Allen et al (1998).

        :return: atmospheric pressure [kPa]
        :rtype: float
        """
        tmp = (293.0 - (0.0065 * self.altitude)) / 293.0
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
        if self.input_freq=='hourly': # use equation 54
            avp = multiply(self.sat_vp_fao56(self.input['temp'].values), divide(self.input['rel_hum'].values, 100.0))
        elif self.input_freq=='daily':
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


    def sol_rad_from_sun_hours(self, a_s=0.25, b_s=0.5):
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
        return multiply( add(a_s , multiply(divide(n , N) , b_s)) , self._et_rad())


    def net_in_sol_rad(self, albedo=0.23):
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
        :param albedo: Albedo of the crop as the proportion of gross incoming solar
            radiation that is reflected by the surface. Default value is 0.23,
            which is the value used by the FAO for a short grass reference crop.
            Albedo can be as high as 0.95 for freshly fallen snow and as low as
            0.05 for wet bare soil. A green vegetation over has an albedo of
            about 0.20-0.25 (Allen et al, 1998).
        :return: Net incoming solar (or shortwave) radiation [MJ m-2 day-1].
        :rtype: float
        """
        if 'solar_rad' not in self.input.columns:
            raise KeyError('first calculate solar radiation and then use this method')
        return multiply((1 - albedo) , self.input['solar_rad'].values)

    def net_out_lw_rad(self ):
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
        :return: Net outgoing longwave radiation [MJ m-2 timestep-1]
        :rtype: float
        """
        if 'tmin' in self.input.columns and 'tmax' in self.input.columns:
            added = add(power(self.input['tmax'].values+273.16, 4), power(self.input['tmin'].values+273.16, 4))
            divided = divide(added, 2.0)
        else:
            divided = power(self.input['temp'].values+273.16, 4.0)

        tmp1 = multiply(self.SB_CONS , divided)
        tmp2 = subtract(0.34 , multiply(0.14 , sqrt(self.input['ea'].values)))
        tmp3 = subtract(multiply(1.35 , divide(self.input['solar_rad'].values , self._cs_rad())) , 0.35)
        return multiply(tmp1 , multiply(tmp2 , tmp3))  # eq 39


    def soil_heat_flux(self, rn=None):
        if self.input_freq=='daily':
            return 0.0
        elif self.input_freq == 'hourly':
            Gd = multiply(0.1, rn)
            Gn = multiply(0.5, rn)
            return where(self.input['is_day']==1, Gd, Gn)
        elif self.input_freq == 'monthly':
            pass


    def cleary_sky_rad(self, a_s=None, b_s=None):
        """clear sky radiation Rso"""

        if a_s is None:
            rso = multiply(0.75 + 2e-5*self.altitude, self._et_rad())  # eq 37
        else:
            rso = multiply(a_s+b_s, self._et_rad())  # eq 36
        return rso


    def Mcguinnes_bordne(self):
        """
        calculates evapotranspiration [mm/day] using Mcguinnes Bordne formulation [1].

        [1] McGuinness, J. L., & Bordne, E. F. (1972). A comparison of lysimeter-derived potential evapotranspiration
            with computed values (No. 1452). US Dept. of Agriculture.
        """

        ra = self._et_rad()
        # latent heat of vaporisation, MJ/Kg
        _lambda = LAMBDA # multiply((2.501 - 2.361e-3), self.input['temp'].values)
        tmp1 = multiply((1/_lambda), ra)
        tmp2 = divide(add(self.input['temp'].values, 5), 68)
        pet = multiply(tmp1, tmp2)
        self.input['et_mcguiness'] = pet
        return pet


    def Makkink(self, a_s=0.23, b_s=0.5):
        """
        using formulation of Makkink
        """
        if 'solar_rad' not in self.input.columns:
            if 'sunshine_hrs' in self.input.columns:
                rs = self.sol_rad_from_sun_hours(a_s=a_s, b_s=b_s)
                if self.verbose:
                    print("Sunshine hour data is used for calculating incoming solar radiation")
            else:
                rs = self._sol_rad_from_t()
                if self.verbose:
                    print("solar radiation is calculated from temperature")
            self.input['solar_rad'] = rs
        else:
            rs = self.input['solar_rad']

        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const

        et = subtract(multiply(multiply(0.61, divide(delta, add(delta, gamma))), divide(rs, 2.45)), 0.12)
        self.input['ET_Makkink'] = et
        return et


    def Linacre(self):
        """
         usingformulation of Linacre 1977 [1] who simplified Penman method.

         [1] Linacre, E. T. (1977). A simple formula for estimating evaporation rates in various climates,
             using temperature data alone. Agricultural meteorology, 18(6), 409-424.
         """
        tm = add(self.input['temp'].values, multiply( 0.006, self.altitude))
        tmp1 = multiply(500, divide(tm, 100-self.lat))
        tmp2 = multiply(15,subtract(self.input['temp'].values, self.input['tdew'].values))
        upar = add(tmp1, tmp2)

        pet = divide(upar, subtract(80, self.input['temp'].values))
        self.input['ET_Linacre'] = pet
        return pet



    def Hargreaves(self):
        """
        estimates daily ETo using Hargreaves method [1]. Equation taken from [2].

        [1] Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature.
            Applied engineering in agriculture, 1(2), 96-99.
        [2] Hargreaves, G. H., & Allen, R. G. (2003). History and evaluation of Hargreaves evapotranspiration equation.
            Journal of Irrigation and Drainage Engineering, 129(1), 53-63.
        """
        tmp1 = multiply(0.0023, add(self.input['temp'], 17.8))
        tmp2 = power(subtract(self.input['tmax'].values, self.input['tmin'].values), 0.5)
        tmp3 = multiply(0.408, self._et_rad())
        return multiply(multiply(tmp1, tmp2), tmp3)


    def Penman_Monteith(self, lm=None):
        """calculates reference evapotrnaspiration according to Penman-Monteith (Allen et al 1998) equation which is
        also recommended by FAO. The etp is calculated at the time step determined by the step size of input data.
        For hourly or sub-hourly calculation, equation 53 is used while for daily time step equation 6 is used.

        # Arguments
        :param
        :return et, a numpy Pandas dataframe consisting of calculated potential etp values.

        http://www.fao.org/3/X0490E/x0490e08.htm#chapter%204%20%20%20determination%20of%20eto
        """
        pet = -9999

        if self.input_freq == 'hourly':
            if lm is None:
                raise ValueError('provide input value of lm')

        wind_2m = self._wind_2m  # wind speed at 2 m height
        D = self.slope_sat_vp(self.input['temp'].values)
        g = self.psy_const

        if self.input_freq=='daily':
            es = self.mean_sat_vp_fao56()
        elif self.input_freq == 'hourly':
            es = self.sat_vp_fao56(self.input['temp'].values)

        ea = self.avp_from_rel_hum()
        #self.input['avp'] = ea
        vp_d = subtract(es, ea)   # vapor pressure deficit


        if 'solar_rad' not in self.input.columns:
            self.input['solar_rad'] = self.sol_rad_from_sun_hours()

        rns = self.net_in_sol_rad()
        rnl = self.net_out_lw_rad()
        rn = subtract(rns, rnl)
        G = self.soil_heat_flux(rn)

        t1 = multiply(0.408 , subtract(rn, G))
        nechay = add(D, multiply(g, add(1.0, multiply(0.34, wind_2m))))

        if self.input_freq=='daily':
            t3 = divide(D, nechay)
            t4 = multiply(t1, t3)
            t5 = multiply(vp_d, divide(g, nechay))
            t6 = divide(multiply(900, wind_2m), add(self.input['temp'].values, 273))
            t7 = multiply(t6, t5)
            pet = add(t4, t7)

        if self.input_freq=='hourly':
            t3 = multiply(divide(37, self.input['temp']+273), g)
            t4 = multiply(t3, vp_d)
            upar = add(t1, t4)
            pet = divide(upar, nechay)

        return pet


    def Thornthwait(self):
        """calculates reference evapotrnaspiration according to empirical temperature based Thornthwaite
        (Thornthwaite 1948) method. The method actualy calculates both ETP and evaporation. It requires only temperature
        and day length as input. Suitable for monthly values.

        # Arguments
        :param
        :return et, a numpy Pandas dataframe consisting of calculated potential etp values.

        # Thornthwaite CW. 1948. An Approach toward a Rational Classification of Climate. Geographical Review 38 (1): 55,
         DOI: 10.2307/210739

        """
        if 'daylight_hrs' not in self.input.columns:
            day_hrs = self.daylight_fao56()
        else:
            day_hrs = self.input['daylight_hrs']

        if 'temp' not in self.input.columns:
            raise ValueError('insufficient input data')

        self.input['adj_t'] = where(self.input['temp'].values<0.0, 0.0, self.input['temp'].values)
        I = self.input['adj_t'].resample('A').apply(custom_resampler)  # heat index (I)
        a = (6.75e-07 * I ** 3) - (7.71e-05 * I ** 2) + (1.792e-02 * I) + 0.49239
        self.input['a'] = a
        a_mon = self.input['a']    # monthly values filled with NaN
        a_mon = pd.DataFrame(a_mon)
        a_ann = pd.DataFrame(a)
        a_monthly = a_mon.merge(a_ann, left_index=True, right_index=True, how='left').fillna(method='bfill')
        self.input['I'] = I
        I_mon = self.input['I']  # monthly values filled with NaN
        I_mon = pd.DataFrame(I_mon)
        I_ann = pd.DataFrame(I)
        I_monthly = I_mon.merge(I_ann, left_index=True, right_index=True, how='left').fillna(method='bfill')

        tmp1 = multiply(1.6, divide(day_hrs, 12.0))
        tmp2 = divide(self.input.index.daysinmonth, 30.0)
        tmp3 = multiply(power(multiply(10.0, divide(self.input['temp'].values, I_monthly['I'].values)), a_monthly['a'].values ), 10.0)
        pet = multiply(tmp1, multiply(tmp2, tmp3))

        self.input['thornwait_mon'] = pet
        self.input['thornwait_daily'] = divide(self.input['thornwait_mon'].values, self.input.index.days_in_month)
        return pet



    @property
    def dec_angle(self):
        """finds solar declination angle"""
        if self.input_freq == 'monthly':
            return  array(0.409 * sin(2*3.14 * self.daily_index.dayofyear/365 - 1.39))
        else:
            return 0.409 * sin(2*3.14 * self.input['jday'].values/365 - 1.39)       # eq 24, declination angle


    def sunset_angle(self):
        """calculates sunset hour angle in radians given by Equation 25  in Fao56 (1)

        1): http://www.fao.org/3/X0490E/x0490e07.htm"""

        j = (3.14/180.0)*self.lat           # eq 22
        d = self.dec_angle       # eq 24, declination angle
        angle = np.arccos(-tan(j)*tan(d))      # eq 25
        return angle

    def daylight_fao56(self):
        """get number of maximum hours of sunlight for a given latitude using equation 34 in Fao56.
        Annual variation of sunlight hours on earth are plotted in figre 14 in ref 1.

        1) http://www.fao.org/3/X0490E/x0490e07.htm"""
        ws = self.sunset_angle()
        hrs = (24/3.14) * ws
        if self.input_freq == 'monthly':
            df = pd.DataFrame(hrs, index=self.daily_index)
            hrs = df.resample('M').mean().values.reshape(-1,)
        return hrs

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


    def Hamon(self, cts=0.0055):
        """calculates evapotranspiration in mm using Hamon 1963 method as given in Lu et al 2005. It uses daily mean
         temperature which can also be calculated
        from daily max and min temperatures. It also requires `daylight_hrs` which is hours of day light, which if not
        provided as input, will be calculated from latitutde. This means if `daylight_hrs` timeseries is not provided as
        input, then argument `lat` must be provided.

        pet = cts * n * n * vdsat
        vdsat = (216.7 * vpsat) / (tavc + 273.3)
        vpsat = 6.108 * exp((17.26939 * tavc)/(tavc + 237.3))

        :param cts: float, or array of 12 values for each month of year or a time series of equal length as input data.
                     if it is float, then that value will be considered for whole year. Default value of 0.0055 was used
                     by Hamon 1961, although he later used different value but I am using same value as it is used by
                     WDMUtil. It should be also noted that 0.0055 is to be used when pet is in inches. So I am dividing
                     the whole pet by 24.5 in order to convert from inches to mm while still using 0.0055.

        Hamon,  W.R.,  1963.  Computation  of  Direct  Runoff  Amounts  FromStorm Rainfall.
            Int. Assoc. Sci. Hydrol. Pub. 63:52-62.
        Lu et al. (2005).  A comparison of six potential evaportranspiration methods for regional use in the southeastern
            United States.  Journal of the American Water Resources Association, 41, 621-633.
         """

        if 'daylight_hrs' not in self.input.columns:
            if self.lat is None:
                raise ValueError('number of daylihgt hours are not given as input so latitude must be provided')
            else:
                print('Calculating daylight hours indirectly from latitude provided.')
                daylight_hrs = divide(self.daylight_fao56(), 12.0)  # shoule be multiple of 12
        else:
            daylight_hrs = self.input['daylight_hrs']

        if 'temp' not in self.input.columns:   # mean temperature is not provided as input
            if 'tmax' not in self.input.columns and 'tmin' not in self.input.columns:
                raise ValueError('tmax and tmin should be provided to calculate mean temperature')
            # calculate mean temperature from tmax and tmin
            else:
                tmean = mean(array([self.input['tmin'].values, self.input['tmax'].values]), axis=0)
        # mean temperature is provided as input
        else:
            tmean = self.input['temp'].values

        vd_sat = self.sat_vpd(tmean)
        other = multiply(cts, power(daylight_hrs, 2.0))
        pet = multiply(other, vd_sat)
        return divide(pet, 24.5)


    def rad_to_evap(self):
        """
         converts solar radiation to equivalent inches of water evaporation

        SRadIn[in/day] = SolRad[Ley/day] / ((597.3-0.57) * temp[centigrade]) * 2.54)    [1]
        or using equation 20 of FAO chapter 3

        from TABLE 3 in FAO chap 3.
        SRadIn[mm/day] = 0.408 * Radiation[MJ m-2 day-1]
        SRadIn[mm/day] = 0.035 * Radiation[Wm-2]
        SRadIn[mm/day] = Radiation[MJ m-2 day-1] / 2.45
        SRadIn[mm/day] = Radiation[J cm-2 day-1] / 245
        SRadIn[mm/day] = Radiation[Wm-2] / 28.4

        [1]  https://github.com/respec/BASINS/blob/4356aa9481eb7217cb2cbc5131a0b80a932907bf/atcMetCmp/modMetCompute.vb#L1251
        https://github.com/DanluGuo/Evapotranspiration/blob/8efa0a2268a3c9fedac56594b28ac4b5197ea3fe/R/Evapotranspiration.R

        """
        # TODO following equation assumes radiations in langleys/day ando output in Inches
        tmp1 = multiply(multiply(597.3-0.57, self.input['temp'].values), 2.54)
        radIn = divide(self.input['solar_rad'].values, tmp1)

        return radIn


    def JensenHaiseR(self, a_s, b_s, ct=0.025, tx=-3):
        """as given (eq 9) in [1] and implemented in [2]

        [1] Xu, C. Y., & Singh, V. P. (2000). Evaluation and generalization of radiation‐based methods for calculating
            evaporation. Hydrological processes, 14(2), 339-349.
        [2] https://github.com/DanluGuo/Evapotranspiration/blob/8efa0a2268a3c9fedac56594b28ac4b5197ea3fe/R/Evapotranspiration.R#L2734
        """

        if 'sunshine_hrs' in self.input.columns:
            rs = self.sol_rad_from_sun_hours(a_s,b_s)
            print('sunshine hour data have been used to calculate incoming solar radiation')
        else:
            rs = self._sol_rad_from_t()
            print('incoming solar radiation is calculated from temperature')
        tmp1 = multiply(multiply(ct, add(self.input['temp'], tx)), rs)
        pet = divide(tmp1, LAMBDA)
        self.input['pet'] = pet
        return


    def Jesnsen(self, cts, ctx):
        """
        This method generates daily pan evaporation (inches) using a coefficient for the month `cts`, , the daily
        average air temperature (F), a coefficient `ctx`, and solar radiation (langleys/day). The computations are
        based on the Jensen and Haise (1963) formula.
                  PET = CTS * (TAVF - CTX) * RIN

            where
                  PET = daily potential evapotranspiration (in)
                  CTS = monthly variable coefficient
                 TAVF = mean daily air temperature (F), computed from max-min
                  CTX = coefficient
                  RIN = daily solar radiation expressed in inches of evaporation

                  RIN = SWRD/(597.3 - (.57 * TAVC)) * 2.54

            where
                 SWRD = daily solar radiation (langleys)
                 TAVC = mean daily air temperature (C)
        :param cts float or array like. Value of monthly coefficient `cts` to be used. If float, then same value is
                assumed for all months. If array like then it must be of length 12.
        :param ctx `float` constant coefficient value of `ctx` to be used in Jensen and Haise formulation.

        [1] Jensen, M. E., & Haise, H. R. (1963). Estimating evapotranspiration from solar radiation. Proceedings of
            the American Society of Civil Engineers, Journal of the Irrigation and Drainage Division, 89, 15-41.
    """
        if not isinstance(cts, float):
            if not isinstance(array(ctx), np.ndarray):
                raise ValueError('cts must be array like')
            else:  # if cts is array like it must be given for 12 months of year, not more not less
                if len(array(cts))>12:
                    raise ValueError('cts must be of length 12')
        else:  # if only one value is given for all moths distribute it as monthly value
            _cts = array([cts for _ in range(12)])

        if not isinstance(ctx, float):
            raise ValueError('ctx must be float')

        # distributing cts values for all dates of input data
        self.input['cts'] = np.nan
        for m,i in zip(self.input.index.month, self.input.index):
            for _m in range(m):
                self.input.at[i, 'cts'] = cts[_m]

        self.input['ctx'] = ctx


        radIn = self.rad_to_evap()
        PanEvp = multiply(multiply(self.input['cts'].values, subtract(self.input['temp'].values, self.input['ctx'].values)), radIn)
        pan_evp = where(PanEvp<0.0, 0.0, PanEvp)
        return pan_evp


    def penman_pan_evap(self, wind_f='pen48', a_s=0.23, b_s=0.5, albedo=0.23):
        """
        calculates pan evaporation from open water using formulation of [1] as mentioned (as eq 12) in [2]. if wind data
        is missing then equation 33 from [4] is used which does not require wind data.

        :param `wind_f` str, if 'pen48 is used then formulation of [1] is used otherwise formulation of [3] requires
                 wind_f to be 2.626.

        [1] Penman, H. L. (1948). Natural evaporation from open water, bare soil and grass. Proceedings of the Royal
            Society of London. Series A. Mathematical and Physical Sciences, 193(1032), 120-145.  http://www.jstor.org/stable/98151
        [2] McMahon, T., Peel, M., Lowe, L., Srikanthan, R. & McVicar, T. 2012. Estimating actual, potential, reference crop
            and pan evaporation using standard meteorological data: a pragmatic synthesis. Hydrology and Earth System
            Sciences Discussions, 9, 11829-11910. https://doi.org/10.5194/hess-17-1331-2013
        [3] Penman, H.L. (1956) Evaporation an Introductory Survey. Netherlands Journal of Agricultural Science, 4, 9-29
        [4] Valiantzas, J. D. (2006). Simplified versions for the Penman evaporation equation using routine weather data.
            Journal of Hydrology, 331(3-4), 690-702. https://doi.org/10.1016/j.jhydrol.2006.06.012
        """
        if wind_f not in ['pen48', 'pen56']:
            raise ValueError('value of given wind_f is not allowed.')

        if wind_f=='pen48':
            _a = 2.626
            _b = 0.09
        else:
            _a = 1.313
            _b = 0.06


        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const

        if 'solar_rad' not in self.input.columns:
            if 'sunshine_hrs' in self.input.columns:
                rs = self.sol_rad_from_sun_hours(a_s=a_s, b_s=b_s)
                if self.verbose:
                    print("Sunshine hour data is used for calculating incoming solar radiation")
            else:
                rs = self._sol_rad_from_t()
                if self.verbose:
                    print("solar radiation is calculated from temperature")
            self.input['solar_rad'] = rs
        else:
            rs = self.input['solar_rad']

        vabar = self.avp_from_rel_hum()  # Vapour pressure
        r_n = self.net_rad(albedo=albedo)  #  net radiation
        vas = self.mean_sat_vp_fao56()

        if 'uz' in self.input.columns:
            if self.verbose:
                print("Wind data have been used for calculating the Penman evaporation.")
            u2 = self._wind_2m
            fau = _a + 1.381 * u2
            Ea = multiply(fau, subtract(vas, vabar))

            tmp1 = divide(delta, add(delta, gamma))
            tmp2 = divide(r_n, LAMBDA)
            tmp3 = multiply(divide(gamma, add(delta, gamma)), Ea)
            pet = add(multiply(tmp1, tmp2), tmp3)
            self.input['pet_PenPan'] = pet
        # if wind data is not available
        else:
            if self.verbose:
                print("Alternative calculation for Penman evaporation without wind data have been performed")

            ra = self._et_rad()
            tmp1 = multiply(multiply(0.047, rs), sqrt(add(self.input['temp'].values, 9.5)))
            tmp2 = multiply(power(divide(rs, ra), 2.0), 2.4)
            tmp3 = multiply(_b, add(self.input['temp'].values, 20))
            tmp4 = subtract(1, divide(self.input['rh_mean'].values, 100))
            tmp5 = multiply(tmp3,tmp4)
            pet = add(subtract(tmp1, tmp2), tmp5)
            self.input['pet_PenPan'] = pet
        return


    def priestley_taylor(self, a_s=0.23, b_s=0.5, alpha_pt=1.26, albedo=0.23):
        """
        following formulation of Priestley & Taylor, 1972 [1].

        :param `alpha_pt` Priestley-Taylor coefficient = 1.26 for Priestley-Taylor model (Priestley and Taylor, 1972)

         [1] Priestley, C. H. B., & Taylor, R. J. (1972). On the assessment of surface heat flux and evaporation using
             large-scale parameters. Monthly weather review, 100(2), 81-92.
         """

        if 'solar_rad' not in self.input.columns:
            if 'sunshine_hrs' in self.input.columns:
                rs = self.sol_rad_from_sun_hours(a_s=a_s, b_s=b_s)
                if self.verbose:
                    print("Sunshine hour data is used for calculating incoming solar radiation")
            else:
                rs = self._sol_rad_from_t()
                if self.verbose:
                    print("solar radiation is calculated from temperature")
            self.input['solar_rad'] = rs
        else:
            rs = self.input['solar_rad']

        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const
        vabar = self.avp_from_rel_hum()
        r_n = self.net_rad(albedo=albedo)  #  net radiation
        vas = self.mean_sat_vp_fao56()
        G = self.soil_heat_flux()

        tmp1 = divide(delta, add(delta, gamma))
        tmp2 = multiply(tmp1, divide(r_n, LAMBDA))
        tmp3 = subtract(tmp2, divide(G, LAMBDA))
        pet = multiply(alpha_pt, tmp3)
        self.input['pet_Priestly_Taylor'] = pet
        return


    @property
    def _wind_2m(self, method='fao56',z_o=0.001):
        """converts wind speed (m/s) measured at height z to 2m using either FAO 56 equation 47 or McMohan eq S4.4.
         u2 = uz [ 4.87/ln(67.8z-5.42) ]         eq 47 in [1]
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


    def net_rad(self, albedo=0.23):
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
        rns = self.net_in_sol_rad(albedo=albedo)
        rnl = self.net_out_lw_rad()

        return subtract(rns, rnl)


    def inv_rel_dist_earth_sun(self):
        """
        Calculate the inverse relative distance between earth and sun from day of the year.
        Based on FAO equation 23 in Allen et al (1998).
        ird = 1.0 + 0.033 * cos( [2pi/365] * j )

        :return: Inverse relative distance between earth and the sun
        :rtype: np array
        """
        inv1 = multiply(2*math.pi/365.0 ,  self.input['jday'].values)
        inv2 = cos(inv1)
        inv3 = multiply(0.033, inv2)
        return add(1.0, inv3)


    def solar_time_cor(self):
        """seasonal correction for solar time by implementation of eqation 32"""
        upar = multiply((2*math.pi), subtract(self.input['jday'].values, 81))
        b =  divide(upar, 364)   # eq 33
        t1 = multiply(0.1645, sin(multiply(2, b)))
        t2 = multiply(0.1255, cos(b))
        t3 = multiply(0.025, sin(b))
        return t1-t2-t3   # eq 32


    def solar_time_angle(self):
        """solar time angle using equation 31"""

        lz = 15.0   #TODO how to calculate this?
        lm = self.long
        t1 = 0.0667*(lz-lm)
        t2 = self.input['half_hr'].values + t1 + self.solar_time_cor()
        t3 = subtract(t2, 12)
        w = multiply((math.pi/12.0) , t3)     # eq 31

        w1 = subtract(w, divide(multiply(math.pi , self.input['t1']).values, 24.0))  # eq 29
        w2 = add(w, divide(multiply(math.pi, self.input['t1']).values, 24.0))   # eq 30
        return w1,w2

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
        if self.input_freq=='hourly':
            j = (3.14/180)*self.lat  # eq 22  phi
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

        elif self.input_freq == 'daily':
            sol_dec = self.dec_angle
            sha = self.sunset_angle()   # sunset hour angle[radians]
            ird = self.inv_rel_dist_earth_sun()
            tmp1 = (24.0 * 60.0) / math.pi
            tmp2 = multiply(sha , multiply(math.sin(self.lat_rad) , sin(sol_dec)))
            tmp3 = multiply(math.cos(self.lat_rad) , multiply(cos(sol_dec) , sin(sha)))
            ra = multiply(tmp1 , multiply(SOLAR_CONSTANT , multiply(ird , add(tmp2 , tmp3)))) # eq 21

        return ra


    def _cs_rad(self):
        """
        Estimate clear sky radiation from altitude and extraterrestrial radiation.

        Based on equation 37 in Allen et al (1998) which is recommended when calibrated Angstrom values are not available.
        et_rad is Extraterrestrial radiation [MJ m-2 day-1]. Can be estimated using ``et_rad()``.

        :return: Clear sky radiation [MJ m-2 day-1]
        :rtype: float
        """
        return (0.00002 * self.altitude + 0.75) * self._et_rad()

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

        et_rad = self._et_rad()
        cs_rad = self._cs_rad()
        sol_rad = multiply(adj , multiply(sqrt(subtract(self.input['tmax'].values , self.input['tmin'].values)) , et_rad))

        # The solar radiation value is constrained by the clear sky radiation
        return np.min( array([sol_rad, cs_rad]), axis=0)


    def dis_sol_pet(self, InTs, DisOpt, Latitude):
        """
        Follows the code from [1] to disaggregate solar radiation and PET from daily to hourly time step.
        :param Latitude `float` latitude in decimal degrees, should be between -66.5 and 66.5
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

        if MetComputeLatitudeMin > Latitude > MetComputeLatitudeMax:
            raise ValueError('Latitude should be between -66.5 and 66.5')

        LatRdn = Latitude * DegreesToRadians

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

def custom_resampler(array_like):
    """calculating heat index using monthly values of temperature."""
    return np.sum(power(divide(array_like, 5.0), 1.514))
