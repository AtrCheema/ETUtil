import pandas as pd
import numpy as np
import math

from convert import Temp, Wind


#: Solar constant [ MJ m-2 min-1]
SOLAR_CONSTANT = 0.0820

# Stefan Boltzmann constant [MJ K-4 m-2 day-1]
STEFAN_BOLTZMANN_CONSTANT = 0.000000004903
"""Stefan Boltzmann constant [MJ K-4 m-2 day-1]"""

class ReferenceET(object):
    """calculates reference evapotranspiration using the `input_method`
    # Arguments
     :param `input_df`: must be a pandas dataframe with some or all following values.
            temp: air temperature
            wind: wind speed
            rel_hum: relative humidity
            solar_rad: solar radiation
            daylight_hrs: number of daylight hours in a day.
            sunshine_hrs: actual sunshine hours
            cloud:   cloud cover
            rh_max: maximum relative humidty
            rh_min: minimum relative humidity

     :param `input_units`: a dictionary containing units for all input time series data.
              it must have one or all of following keys and corresponding values
            temp -- centigrade, fahrenheit, kelvin
            tmin -- centigrade, fahrenheit, kelvin
            tmax -- centigrade, fahrenheit, kelvin
            dewpoint -- centigrade, fahrenheit, kelvin
            wind -- 'MeterPerSecond', 'KilometerPerHour', 'MilesPerHour', 'InchesPerSecond',  'FeetPerSecond'
            rel_hum: relative humidity
            rh_max:
            rh_min:
            solar_rad:
            cloud:
            daylight_hrs: hour
            sunshine_hrs: hour
    :param `lat` float, latitude of measured data in degree decimals. May not be always required. Depends upon the
             method used and input data.
    :param `alatitude` float, Elevation/altitude above sea level [m]
    :param `wind_z` float Height of wind measurement above ground surface [m]
     'method': str, method to be employed to calculated reference evapotranspiration. Must be one of following
        `pm`: penman-monteith fao 56 method
        `thornwait`:
            """

    def __init__(self, input_df, units,  lat = None, altitude=None, wind_z=None):
        self.input = input_df
        self.units = units
        self._check_compatibility()
        self.lat = lat
        self.lat_rad = self.lat * 0.0174533  # degree to radians
        self.altitude = altitude
        self.wind_z = wind_z
        self.input_freq = self.get_in_freq()


    def get_in_freq(self):
        freq = self.input.index.freqstr
        if freq in ['D']:
            return 'daily'
        elif freq in ['H']:
            return 'hourly'
        elif 'min' in freq:
            return 'sub_hourly'
        else:
            raise ValueError('unknown frequency of input data')

    def _check_compatibility(self):
        """units are also converted here."""

        if not isinstance(self.input, pd.DataFrame):
            raise TypeError('input must be a pandas dataframe')

        for col in self.input.columns:
            if col not in self.units.keys():
                raise ValueError('units for input {} are not given'.format(col))

        allowed_units = {'temp': ['centigrade', 'fahrenheit', 'kelvin'],
                         'tmin': ['centigrade', 'fahrenheit', 'kelvin'],
                         'tmax': ['centigrade', 'fahrenheit', 'kelvin'],
                         'dewpoint': ['centigrade', 'fahrenheit', 'kelvin'],
                         'wind':  ['MeterPerSecond', 'KilometerPerHour', 'MilesPerHour', 'InchesPerSecond',
                                   'FeetPerSecond'],
                         'daylight_hrs': ['hour'],
                         'sunshine_hrs': ['hour'],
                         'rel_hum': ['percent'],
                         'rh_min': ['percent'],
                         'rh_max': ['percent'],
                         'solar_rad': [''],
                         'cloud': ['']}

        for _input, _unit in self.units.items():
#            print(_input, _unit, 'here')
            if _unit not in allowed_units[_input]:
                raise ValueError('unit {} of input {} is not allowed. Use any of {}'
                                 .format(_unit, _input, allowed_units[_input]))

        # converting temperature units to celsius
        for val in ['tmin', 'tmax', 'temp']:
            if val in self.input:
                t = Temp(self.input[val].values, self.units[val])
                self.input[val] = t.celsius

        if 'tmean' not in self.input.columns:
            if 'tmin' in self.input.columns and 'tmax' in self.input.columns:
                self.input['tmean'] = np.mean(np.array([self.input['tmin'].values, self.input['tmax'].values]), axis=0)

        if 'wind' in self.input:
            w = Wind(self.input['wind'].values, self.units['wind'])
            self.input['wind'] = w.MeterPerSecond

        # getting julian day
        self.input['jday'] = self.input.index.dayofyear

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
        """ slope of the relationship between saturation vapour pressure and temperature for a given temperature
        according to equation 13.

        :param t: Air temperature [deg C]. Use mean air temperature for use in Penman-Monteith.
        :return: Saturation vapour pressure [kPa degC-1]
        """
        to_exp = np.divide(np.multiply(17.27, t), np.add(t, 237.3))
        tmp = np.multiply(4098 , np.multiply(0.6108 , np.exp(to_exp)))
        return np.divide(tmp , np.power( np.add(t , 237.3), 2))

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
        return np.multiply(0.000665 , self.atm_pressure)

    def avp_from_rhmin_rhmax(self):
        """
        Estimate actual vapour pressure (*ea*) from saturation vapour pressure and relative humidity.

        Based on FAO equation 17 in Allen et al (1998).

        uses  Saturation vapour pressure at daily minimum temperature [kPa].
              Saturation vapour pressure at daily maximum temperature [kPa].
              Minimum relative humidity [%]
              Maximum relative humidity [%]
        :return: Actual vapour pressure [kPa]
        :rtype: float
        """
        avp = 0.0
        if 'rh_min' in self.input.columns and 'rh_max' in self.input.columns:
            tmp1 = np.multiply(self.sat_vp_fao56(self.input['tmin'].values) , np.divide(self.input['rh_max'].values , 100.0))
            tmp2 = np.multiply(self.sat_vp_fao56(self.input['tmax'].values) , np.divide(self.input['rh_min'].values , 100.0))
            avp = np.divide(np.add(tmp1 , tmp2) , 2.0)
        elif 'rel_hum' in self.input.columns:
            # calculation actual vapor pressure from mean humidity
            # equation 19
            t1 = np.divide(self.input['rel_hum'].values, 100)
            t2 = np.divide(np.add(self.sat_vp_fao56(self.input['tmax'].values), self.sat_vp_fao56(self.input['tmin'].values)), 2.0)
            avp = np.multiply(t1,t2)

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
        return np.multiply( np.add(a_s , np.multiply(np.divide(n , N) , b_s)) , self._et_rad())


    def net_in_sol_rad(self, albedo=0.23):
        """
        Calculate net incoming solar (or shortwave) radiation from gross incoming solar radiation, assuming a grass
         reference crop.

        Net incoming solar radiation is the net shortwave radiation resulting from the balance between incoming and
         reflected solar radiation. The output can be converted to equivalent evaporation [mm day-1] using
        ``energy2evap()``.

        Based on FAO equation 38 in Allen et al (1998).

        uses Gross incoming solar radiation [MJ m-2 day-1]. If necessary this can be estimated using functions whose name
            begins with 'sol_rad_from'.
        :param albedo: Albedo of the crop as the proportion of gross incoming solar
            radiation that is reflected by the surface. Default value is 0.23,
            which is the value used by the FAO for a short grass reference crop.
            Albedo can be as high as 0.95 for freshly fallen snow and as low as
            0.05 for wet bare soil. A green vegetation over has an albedo of
            about 0.20-0.25 (Allen et al, 1998).
        :return: Net incoming solar (or shortwave) radiation [MJ m-2 day-1].
        :rtype: float
        """
        return np.multiply((1 - albedo) , self.input['sol_rad'].values)

    def net_out_lw_rad(self ):
        """
        Estimate net outgoing longwave radiation.

        This is the net longwave energy (net energy flux) leaving the earth's surface. It is proportional to the
        absolute temperature of the surface raised to the fourth power according to the Stefan-Boltzmann law. However,
        water vapour, clouds, carbon dioxide and dust are absorbers and emitters of longwave radiation. This function
        corrects the Stefan- Boltzmann law for humidity (using actual vapor pressure) and cloudiness (using solar
        radiation and clear sky radiation). The concentrations of all other absorbers are assumed to be constant.

        The output can be converted to equivalent evaporation [mm day-1] using  ``energy2evap()``.

        Based on FAO equation 39 in Allen et al (1998).

        uses: Absolute daily minimum temperature [degrees Kelvin]
              Absolute daily maximum temperature [degrees Kelvin]
              Solar radiation [MJ m-2 day-1]. If necessary this can be estimated using ``sol+rad()``.
              Clear sky radiation [MJ m-2 day-1]. Can be estimated using  ``cs_rad()``.
              Actual vapour pressure [kPa]. Can be estimated using functions with names beginning with 'avp_from'.
        :return: Net outgoing longwave radiation [MJ m-2 day-1]
        :rtype: float
        """
        added = np.add(np.power(self.input['tmax'].values+273.16, 4), np.power(self.input['tmin'].values+273.16, 4))
        divided = np.divide(added, 2.0)
        tmp1 = np.multiply(STEFAN_BOLTZMANN_CONSTANT , divided)
        tmp2 = np.subtract(0.34 , np.multiply(0.14 , np.sqrt(self.input['avp'].values)))
        tmp3 = np.subtract(np.multiply(1.35 , np.divide(self.input['sol_rad'].values , self._cs_rad())) , 0.35)
        return np.multiply(tmp1 , np.multiply(tmp2 , tmp3))


    @property
    def soil_heat_flux(self):
        return 0.0

    def Penman_Monteith(self):
        """calculates reference evapotrnaspiration according to Penman-Monteith (Allen et al 1998) equation which is
        also recommended by FAO. The etp is calculated at the time step determined by the step size of input data.
        For hourly or sub-hourly calculation, equation 53 is used while for daily time step equation 6 is used.

        # Arguments
        :param
        :return et, a numpy Pandas dataframe consisting of calculated potential etp values.

        http://www.fao.org/3/X0490E/x0490e08.htm#chapter%204%20%20%20determination%20of%20eto
        """

        wind_2m = self._wind_2m  # wind speed at 2 m height
        D = self.slope_sat_vp(self.input['tmean'].values)
        g = self.psy_const

        es = self.mean_sat_vp_fao56()
        ea = self.avp_from_rhmin_rhmax()
        self.input['avp'] = ea
        vp_d = np.subtract(es, ea)   # vapor pressure deficit
        ra = self._et_rad()

        if 'sol_rad' not in self.input.columns:
            self.input['sol_rad'] = self.sol_rad_from_sun_hours()

        rns = self.net_in_sol_rad()
        rnl = self.net_out_lw_rad()
        rn = np.subtract(rns, rnl)
        G = self.soil_heat_flux

        t1 = np.multiply(0.408 , np.subtract(rn, G))
        t2 = np.add(D, np.multiply(g, np.add(1.0, np.multiply(0.34, wind_2m))))
        t3 = np.divide(D, t2)
        t4 = np.multiply(t1, t3)
        t5 = np.multiply(vp_d, np.divide(g, t2))
        t6 = np.divide(np.multiply(900, wind_2m), np.add(self.input['tmean'].values, 273))
        t7 = np.multiply(t6, t5)
        pet = np.add(t4, t7)
        return pet


    def Thornwait(self):
        """calculates reference evapotrnaspiration according to empirical temperature based Thornthwaite
        (Thornthwaite 1948) method. The method actualy calculates both ETP and evaporation. It requires only temperature
        and day length as input.

        # Arguments
        :param
        :return et, a numpy Pandas dataframe consisting of calculated potential etp values.

        # Thornthwaite CW. 1948. An Approach toward a Rational Classification of Climate. Geographical Review 38 (1): 55,
         DOI: 10.2307/210739
        """

    @property
    def dec_angle(self):
        """finds solar declination angle"""
        return 0.409 * np.sin(2*3.14 * self.input['jday'].values/365 - 1.39)       # eq 24, declination angle

    def sunset_angle(self):
        """calculates sunset hour angle in radians given by Equation 25  in Fao56 (1)

        1): http://www.fao.org/3/X0490E/x0490e07.htm"""

        j = (3.14/180.0)*self.lat           # eq 22
        d = self.dec_angle       # eq 24, declination angle
        angle = np.arccos(-np.tan(j)*np.tan(d))      # eq 25
        return angle

    def daylight_fao56(self):
        """get number of maximum hours of sunlight for a given latitude using equation 34 in Fao56.
        Annual variation of sunlight hours on earth are plotted in figre 14 in ref 1.

        1) http://www.fao.org/3/X0490E/x0490e07.htm"""
        ws = self.sunset_angle()
        hrs = (24/3.14) * ws
        return hrs

    def sat_vp_fao56(self, temp):
        """calculates saturation vapor pressure as given in eq 11 of FAO 56 at a given temp which must be in units of
        centigrade.
        using Tetens equation
        es = 0.6108 * exp((17.26*temp)/(temp+273.3))

        Murray, F. W., On the computation of saturation vapor pressure, J. Appl. Meteorol., 6, 203-204, 1967.
        """
        e_not_t = np.multiply(0.6108, np.exp( np.multiply(17.26939, temp) / np.add(temp , 237.3)))
        return e_not_t


    def mean_sat_vp_fao56(self):
        """ calculates mean saturation vapor pressure for a day, weak or month according to eq 12 of FAO 56 using
        tmin and tmax which must be in centigrade units
        """
        es_tmax = self.sat_vp_fao56(self.input['tmin'].values)
        es_tmin = self.sat_vp_fao56(self.input['tmax'].values)
        es = np.mean(np.array([es_tmin, es_tmax]), axis=0)
        return es

    def sat_vpd(self, temp):
        """calculates saturated vapor density at the given temperature.
        """
        esat = self.sat_vp_fao56(temp)
        # multiplying by 10 as in WDMUtil nad Lu et al, they used 6.108 for calculation of saturation vapor pressura
        # while the real equation for calculation of vapor pressure has '0.6108'. I got correct result for Hamon etp when
        # I calculated sat_vp_fao56 with 6.108. As I have put 0.6108 for sat_vp_fao56 calculation, so multiplying with 10
        # here, although not sure why to multiply with 10.
        return np.multiply(np.divide(np.multiply(216.7, esat), np.add(temp, 273.3)), 10)


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
                daylight_hrs = np.divide(self.daylight_fao56(), 12.0)  # shoule be multiple of 12
        else:
            daylight_hrs = self.input['daylight_hrs']

        if 'tmean' not in self.input.columns:   # tmean is not provided as input
            if 'tmax' not in self.input.columns and 'tmin' not in self.input.columns:
                raise ValueError('tmax and tmin should be provided to calculate tmean')
            # calculate tmean from tmax and tmin
            else:
                tmean = np.mean(np.array([self.input['tmin'].values, self.input['tmax'].values]), axis=0)
        # tmean is provided as input
        else:
            tmean = self.input['tmean'].values

        vd_sat = self.sat_vpd(tmean)
        other = np.multiply(cts, np.power(daylight_hrs, 2.0))
        pet = np.multiply(other, vd_sat)
        return np.divide(pet, 24.5)


    def Jesnsen(self):
        """This procedure generates daily potential evapotranspiration (inches) using a
            coefficient for the month, the daily average air temperature (F), a coefficient,
            and solar radiation (langleys/day). The computations are based on the
            Jensen and Haise (1963) formula.
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
    """

    @property
    def _wind_2m(self):
        """converts wind speed (m/s) measured at height z to 2m using FAO 56 equation 47.

        :return: Wind speed at 2 m above the surface [m s-1]
        http://www.fao.org/3/X0490E/x0490e07.htm
        """
        return np.multiply(self.input['wind'] , (4.87 / math.log((67.8 * self.wind_z) - 5.42)))

    def _net_rat(self, ni_sw_rad, no_lw_rad):
        """
            Calculate daily net radiation at the crop surface, assuming a grass reference crop.

        Net radiation is the difference between the incoming net shortwave (or solar) radiation and the outgoing net
        longwave radiation. Output can be converted to equivalent evaporation [mm day-1] using ``energy2evap()``.

        Based on equation 40 in Allen et al (1998).

        :param ni_sw_rad: Net incoming shortwave radiation [MJ m-2 day-1]. Can be
            estimated using ``net_in_sol_rad()``.
        :param no_lw_rad: Net outgoing longwave radiation [MJ m-2 day-1]. Can be
            estimated using ``net_out_lw_rad()``.
        :return: Daily net radiation [MJ m-2 day-1].
        :rtype: float
        """
        return ni_sw_rad - no_lw_rad

    def inv_rel_dist_earth_sun(self):
        """
        Calculate the inverse relative distance between earth and sun from day of the year.
        Based on FAO equation 23 in Allen et al (1998).
        ird = 1.0 + 0.033 * cos( [2pi/365] * j )

        :return: Inverse relative distance between earth and the sun
        :rtype: np array
        """
        inv1 = np.multiply(2*math.pi/365.0 ,  self.input['jday'].values)
        inv2 = np.cos(inv1)
        inv3 = np.multiply(0.033, inv2)
        return np.add(1.0, inv3)

    def _et_rad(self):
        """
        Estimate daily extraterrestrial radiation (*Ra*, 'top of the atmosphere radiation').

        Based on equation 21 in Allen et al (1998). If monthly mean radiation is required make sure *sol_dec*. *sha*
         and *irl* have been calculated using the day of the year that corresponds to the middle of the month.

        **Note**: From Allen et al (1998): "For the winter months in latitudes greater than 55 degrees (N or S), the equations have limited validity.
        Reference should be made to the Smithsonian Tables to assess possible deviations."

        :return: Daily extraterrestrial radiation [MJ m-2 day-1]
        :rtype: float
        """

        sol_dec = self.dec_angle
        sha = self.sunset_angle()   # sunset hour angle[radians]
        ird = self.inv_rel_dist_earth_sun()
        tmp1 = (24.0 * 60.0) / math.pi
        tmp2 = np.multiply(sha , np.multiply(math.sin(self.lat_rad) , np.sin(sol_dec)))
        tmp3 = np.multiply(math.cos(self.lat_rad) , np.multiply(np.cos(sol_dec) , np.sin(sha)))
        return np.multiply(tmp1 , np.multiply(SOLAR_CONSTANT , np.multiply(ird , np.add(tmp2 , tmp3))))

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
        sol_rad = np.multiply(adj , np.multiply(np.sqrt(np.subtract(self.input['tmax'].values , self.input['tmin'].values)) , et_rad))

        # The solar radiation value is constrained by the clear sky radiation
        return np.min( np.array([sol_rad, cs_rad]), axis=0)