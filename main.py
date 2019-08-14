# https://www.intechopen.com/books/advanced-evapotranspiration-methods-and-applications
# https://www.intechopen.com/books/current-perspective-to-predict-actual-evapotranspiration
# https://rdrr.io/cran/Evapotranspiration/man/
# https://www.ncl.ucar.edu/Document/Functions/index.shtml
import pandas as pd
import numpy as np
from numpy import multiply, divide, add, subtract, power, array, where, mean, sqrt
import math

from utils import Util

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

class ReferenceET(Util):
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


    def __init__(self, input_df, units, constants, verbose=True):

        super(ReferenceET, self).__init__(input_df, units, constants=constants, verbose=verbose)



    def Abtew(self):
        """
        daily etp using equation 3 in [1]. `k` is a dimentionless coefficient.

         uses: , k=0.52, a_s=0.23, b_s=0.5
        :param `k` coefficient, default value taken from [1]
        :param `a_s` fraction of extraterrestrial radiation reaching earth on sunless days
        :param `b_s` difference between fracion of extraterrestrial radiation reaching full-sun days
                 and that on sunless days.

         [1] Abtew, W. (1996). EVAPOTRANSPIRATION MEASUREMENTS AND MODELING FOR THREE WETLAND SYSTEMS IN
             SOUTH FLORIDA 1. JAWRA Journal of the American Water Resources Association, 32(3),
             465-473. https://doi.org/10.1111/j.1752-1688.1996.tb04044.x

         """

        rs = self.rs()
        pet = multiply(self.cons['k'], divide(rs, LAMBDA))
        self.input['pet_abtew'] = pet
        return pet


    def Blaney_Criddle(self):
        """using formulation of Blaney-Criddle for daily reference crop ETP using monthly mean tmin and tmax.
        Inaccurate under extreme climates. underestimates in windy, dry and sunny conditions and overestimates under calm, humid
        and clouded conditions.

        [2] Allen, R. G. and Pruitt, W. O.: Rational use of the FAO Blaney-Criddle Formula, J. Irrig. Drain. E. ASCE,
             112, 139–155, 1986."""
        N = self.daylight_fao56()  # mean daily percentage of annual daytime hours
        u2 = self._wind_2m()
        return multiply(N, add(multiply(0.46, self.input['temp'].values), 8.0))


    def BrutsaertStrickler(self):
        """
        using formulation given by BrutsaertStrickler

        :param `alpha_pt` Priestley-Taylor coefficient = 1.26 for Priestley-Taylor model (Priestley and Taylor, 1972)
        :param `a_s` fraction of extraterrestrial radiation reaching earth on sunless days
        :param `b_s` difference between fracion of extraterrestrial radiation reaching full-sun days
                 and that on sunless days.
        :param `albedo`  Any numeric value between 0 and 1 (dimensionless), albedo of the evaporative surface
                representing the portion of the incident radiation that is reflected back at the surface.
                Default is 0.23 for surface covered with short reference crop.
        :return: et

        [1] Brutsaert, W., & Stricker, H. (1979). An advection‐aridity approach to estimate actual regional
             evapotranspiration. Water resources research, 15(2), 443-450.
        """
        rs = self.rs()
        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const
        vabar = self.avp_from_rel_hum()  # Vapour pressure, *ea*
        vas = self.mean_sat_vp_fao56()
        u2 = self._wind_2m()
        f_u2  = add(2.626 , multiply(1.381 , u2))
        r_ng = self.net_rad(rs, vabar)
        alpha_pt = self.cons['alphaPT']

        et = subtract(multiply(multiply((2*alpha_pt-1), divide(delta, add(delta, gamma))), divide(r_ng, LAMBDA)), multiply(multiply(divide(gamma, add(delta, gamma)), f_u2), subtract(vas,vabar)))
        self.input['ET_BrutsaertStrickler'] = et
        return et


    def GrangerGray(self):
        """
        using formulation of Granger & Gray 1989 which is for non-saturated lands and modified form of penman 1948.

         uses: , wind_f`='pen48', a_s=0.23, b_s=0.5, albedo=0.23
        :param `wind_f` str, if 'pen48 is used then formulation of [1] is used otherwise formulation of [3] requires
                 wind_f to be 2.626.
        :param `a_s fraction of extraterrestrial radiation reaching earth on sunless days
        :param `b_s` difference between fracion of extraterrestrial radiation reaching full-sun days
                 and that on sunless days.
        :param `albedo`  Any numeric value between 0 and 1 (dimensionless), albedo of the evaporative surface
                representing the portion of the incident radiation that is reflected back at the surface.
                Default is 0.23 for surface covered with short reference crop.
        :return:

        Granger, R. J., & Gray, D. M. (1989). Evaporation from natural nonsaturated surfaces.
           Journal of Hydrology, 111(1-4), 21-29.
        """
        if self.cons['wind_f'] not in ['pen48', 'pen56']:
            raise ValueError('value of given wind_f is not allowed.')

        if self.cons['wind_f'] =='pen48':
            _a = 2.626
            _b = 0.09
        else:
            _a = 1.313
            _b = 0.06

        rs = self.rs()
        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const

        vabar = self.avp_from_rel_hum()  # Vapour pressure
        r_n = self.net_rad(rs, vabar)  #  net radiation
        vas = self.mean_sat_vp_fao56()

        u2 = self._wind_2m()
        fau = _a + 1.381 * u2
        Ea = multiply(fau, subtract(vas, vabar))

        G = self.soil_heat_flux()

        # dimensionless relative drying power  eq 7 in Granger, 1998
        dry_pow = divide(Ea, add(Ea, divide(subtract(r_n, G), LAMBDA)))
        # eq 6 in Granger, 1998
        G_g = add(divide(1, add(0.793, multiply(0.20, multiply(math.exp(4.902), dry_pow)))), multiply(0.006, dry_pow))

        tmp1 = divide(multiply(delta, G_g), add(multiply(delta, G_g), gamma))
        tmp2 = divide(subtract(r_n, G), LAMBDA)
        tmp3 = multiply(divide(multiply(gamma, G_g), add(multiply(delta, G_g), gamma)), Ea)
        et = add(multiply(tmp1, tmp2), tmp3)
        self.input['ET_GG'] = et
        return et


    def Chapman_Australia(self):
        """using formulation of [1],

        uses: a_s=0.23, b_s=0.5, ap=2.4, alphaA=0.14, albedo=0.23

        [1] Chapman, T. 2001, Estimation of evaporation in rainfall-runoff models,
            in F. Ghassemi, D. Post, M. SivapalanR. Vertessy (eds), MODSIM2001: Integrating models for Natural
            Resources Management across Disciplines, Issues and Scales, MSSANZ, vol. 1, pp. 293-298.
            http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.539.3517&rep=rep1&type=pdf
        """
        A_p = 0.17 + 0.011 * abs(self.cons['lat'])
        B_p = np.power(10, (0.66 - 0.211 * abs(self.cons['lat'])))  # constants (S13.3)
        rs = self.rs()
        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const
        vabar = self.avp_from_rel_hum()  # Vapour pressure
        vas = self.mean_sat_vp_fao56()
        u2 = self._wind_2m()
        r_nl = self.net_out_lw_rad(rs=rs, ea=vabar)   # net outgoing longwave radiation
        ra = self._et_rad()

        # eq 34 in Thom et al., 1981
        f_pan_u = add(1.201 , np.multiply(1.621, u2))

        # eq 4 and 5 in Rotstayn et al., 2006
        p_rad = add(1.32, add(multiply(4e-4, self.cons['lat']), multiply(8e-5, self.cons['lat']**2)))
        f_dir = add(-0.11, multiply(1.31, divide(rs, ra)))
        rs_pan = multiply(add(add(multiply(f_dir,p_rad), multiply(1.42,subtract(1, f_dir))), multiply(0.42, self.cons['albedo'])), rs)
        rn_pan = subtract(multiply(1-self.cons['alphaA'], rs_pan), r_nl)

        # S6.1 in McMohan et al 2013
        tmp1 = multiply(divide(delta, add(delta, multiply(self.cons['ap'], gamma))), divide(rn_pan, LAMBDA))
        tmp2 = divide(multiply(self.cons['ap'], gamma), add(delta, multiply(self.cons['ap'], gamma)))
        tmp3 = multiply(f_pan_u, subtract(vas, vabar))
        tmp4 = multiply(tmp2, tmp3)
        epan = add(tmp1, tmp4)

        et = add(multiply(A_p, epan), B_p)
        self.input['ET_Chapman'] = et
        return et


    def MattShuttleworth(self):
        """
        using formulation of Matt-Shuttleworth and Wallace, 2009. This is designed for semi-arid and windy areas as an
        alternative to FAO-56 Reference Crop method

        Shuttleworth, W. J., & Wallace, J. S. (2009). Calculating the water requirements of irrigated crops in
            Australia using the matt-shuttleworth approach. Transactions of the ASABE, 52(6), 1895-1906.
        """
        self.check_constants(method='Shuttleworth_Wallace')

        ch = self.cons['CH']  # crop height
        ro_a = self.cons['Roua']
        ca = self.cons['Ca']  # specific heat of the air
        s_r = self.cons['surf_res']  # surface resistance (s m-1) of a well-watered crop equivalent to the FAO crop coefficient

        rs = self.rs()
        vabar = self.avp_from_rel_hum()  # Vapour pressure
        vas = self.mean_sat_vp_fao56()
        r_n = self.net_rad(rs, vabar)  # net radiation
        u2 = self._wind_2m()    # Wind speed
        delta = self.slope_sat_vp(self.input['temp'].values)   # slope of vapour pressure curve
        gamma = self.psy_const    # psychrometric constant

        tmp1 = self.seconds * ro_a * ca
        r_clim = multiply( tmp1, divide(subtract(vas, vabar), multiply(delta, r_n)))
        r_clim = where(r_clim==0, 0.1, r_clim)   # correction for r_clim = 0
        u2 = where(u2==0, 0.1, u2)               # correction for u2 = 0

        #  ratio of vapour pressure deficits at 50m to vapour pressure deficits at 2m heights, eq S5.35
        tmp1 = add(multiply(302, add(delta, gamma)), multiply(70, multiply(gamma, u2)))
        tmp2 = add(multiply(208, add(delta, gamma)), multiply(70, multiply(gamma, u2)))
        tmp3 = divide(tmp1, tmp2)
        tmp4 = divide(208, u2)
        tmp5 = divide(302, u2)
        tmp6 = multiply(divide(1,r_clim), subtract(multiply(tmp3, tmp4), tmp5))
        vpd50_to_vpd2 = add(tmp4, tmp6)

        # aerodynamic coefficient for crop height (s*m^-1) (eq S5.36 in McMohan et al 2013)
        tmp1 =  math.log((50.0 - 0.67*ch)/(0.123*ch))
        tmp2 = math.log((50.0 - 0.67*ch)/(0.0123*ch))
        tmp3 = math.log((2-0.08)/0.0148)/math.log((50-0.08)/0.0148)
        r_c50 = (1/0.41**2) * tmp1 * tmp2 * tmp3

        tmp1 = divide(multiply(multiply(ro_a * ca, u2), subtract(vas,vabar)), 50)
        upar = add(multiply(delta, r_n), multiply(tmp1, vpd50_to_vpd2))
        nechay = add(delta, multiply( gamma, add(1, divide(multiply(s_r, u2), r_c50))))
        et = multiply(divide(1,gamma), divide(upar, nechay))
        self.input['ET_Shuttleworth_Wallace'] = et
        return et


    def MortonCRAE(self):
        """
        for monthly pot. ET and wet-environment areal ET and actual ET by Morton 1983.
        :return:

        Morton, F.I. 1983, Operational estimates of areal evapotranspiration and their significance to the science
            and practice of hydrology. Journal of Hydrology, vol. 66, no. 1-4, pp. 1-76.
            https://doi.org/10.1016/0022-1694(83)90177-4
        """



    def Turc(self):
        """
        using Turc 1961 formulation, originaly developed for southern France and Africa. Implemented as given (as eq 5)
         in [2]

        uses
        :param `k` float or array like, monthly crop coefficient. A single value means same crop coefficient for whole year
        :param `a_s` fraction of extraterrestrial radiation reaching earth on sunless days
        :param `b_s` difference between fracion of extraterrestrial radiation reaching full-sun days
                 and that on sunless days.
        [1] Turc, L. (1961). Estimation of irrigation water requirements, potential evapotranspiration: a simple climatic
            formula evolved up to date. Ann. Agron, 12(1), 13-49.
        [2] Alexandris, S., Stricevic, R.Petkovic, S. 2008, Comparative analysis of reference evapotranspiration from
            the surface of rainfed grass in central Serbia, calculated by six empirical methods against the
            Penman-Monteith formula. European Water, vol. 21, no. 22, pp. 17-28. https://www.ewra.net/ew/pdf/EW_2008_21-22_02.pdf
        """
        rs = self.rs()
        ta = self.input['temp'].values
        et = multiply(multiply(self.cons['k'] , (add(multiply(23.88 , rs) , 50))) , divide(ta , (add(ta , 15))))

        if 'rh_mean' in self.input.columns:
            rh_mean = self.input['rh_mean'].values
            eq1 = multiply(multiply(multiply(self.cons['k'] , (add(multiply(23.88 , rs) , 50))) , divide(ta , (add(ta , 15)))) , (add(1 , divide((subtract(50 , rh_mean)) , 70))))
            eq2 = multiply(multiply(self.cons['k'] , (add(multiply(23.88 , rs) , 50))) , divide(ta , (add(ta , 15))))
            et = np.where(rh_mean<50, eq1, eq2)

        self.input['ET_Turc'] = et
        return et


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


    def Makkink(self):
        """
        uses: a_s, b_s
        using formulation of Makkink
        """
        rs = self.rs()

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
        tm = add(self.input['temp'].values, multiply( 0.006, self.cons['altitude']))
        tmp1 = multiply(500, divide(tm, 100-self.cons['lat']))
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


    def Penman_Monteith(self):
        """calculates reference evapotrnaspiration according to Penman-Monteith (Allen et al 1998) equation which is
        also recommended by FAO. The etp is calculated at the time step determined by the step size of input data.
        For hourly or sub-hourly calculation, equation 53 is used while for daily time step equation 6 is used.

        # Arguments
        # uses: lm=None, a_s=0.25, b_s=0.5, albedo=0.23
        :param
        :return et, a numpy Pandas dataframe consisting of calculated potential etp values.

        http://www.fao.org/3/X0490E/x0490e08.htm#chapter%204%20%20%20determination%20of%20eto
        """
        pet = -9999

        if self.input_freq == 'hourly':
            if self.cons['lm'] is None:
                raise ValueError('provide input value of lm')

        wind_2m = self._wind_2m()  # wind speed at 2 m height
        D = self.slope_sat_vp(self.input['temp'].values)
        g = self.psy_const

        if self.input_freq=='daily':
            es = self.mean_sat_vp_fao56()
        elif self.input_freq == 'hourly':
            es = self.sat_vp_fao56(self.input['temp'].values)

        ea = self.avp_from_rel_hum()
        vp_d = subtract(es, ea)   # vapor pressure deficit


        rs = self.rs()

        rns = self.net_in_sol_rad(rs)
        rnl = self.net_out_lw_rad(rs=rs, ea=ea)
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
            if self.cons['lat'] is None:
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


    def JensenHaiseR(self):
        """
        as given (eq 9) in [1] and implemented in [2]

        uses:  a_s, b_s, ct=0.025, tx=-3

        [1] Xu, C. Y., & Singh, V. P. (2000). Evaluation and generalization of radiation‐based methods for calculating
            evaporation. Hydrological processes, 14(2), 339-349.
        [2] https://github.com/DanluGuo/Evapotranspiration/blob/8efa0a2268a3c9fedac56594b28ac4b5197ea3fe/R/Evapotranspiration.R#L2734
        """

        rs = self.rs()
        tmp1 = multiply(multiply(self.cons['ct'], add(self.input['temp'], self.cons['tx'])), rs)
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


    def penman_pan_evap(self):
        """
        calculates pan evaporation from open water using formulation of [1] as mentioned (as eq 12) in [2]. if wind data
        is missing then equation 33 from [4] is used which does not require wind data.

        :uses , wind_f='pen48', a_s=0.23, b_s=0.5, albedo=0.23
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
        if self.cons['wind_f'] not in ['pen48', 'pen56']:
            raise ValueError('value of given wind_f is not allowed.')

        if self.cons['wind_f']=='pen48':
            _a = 2.626
            _b = 0.09
        else:
            _a = 1.313
            _b = 0.06


        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const

        rs = self.rs()

        vabar = self.avp_from_rel_hum()  # Vapour pressure  *ea*
        r_n = self.net_rad(rs, vabar)  #  net radiation
        vas = self.mean_sat_vp_fao56()

        if 'uz' in self.input.columns:
            if self.verbose:
                print("Wind data have been used for calculating the Penman evaporation.")
            u2 = self._wind_2m()
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


    def priestley_taylor(self):
        """
        following formulation of Priestley & Taylor, 1972 [1].
        uses: , a_s=0.23, b_s=0.5, alpha_pt=1.26, albedo=0.23
        :param `alpha_pt` Priestley-Taylor coefficient = 1.26 for Priestley-Taylor model (Priestley and Taylor, 1972)

         [1] Priestley, C. H. B., & Taylor, R. J. (1972). On the assessment of surface heat flux and evaporation using
             large-scale parameters. Monthly weather review, 100(2), 81-92.
         """

        rs = self.rs()

        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const
        vabar = self.avp_from_rel_hum()    #  *ea*
        r_n = self.net_rad(rs, vabar)  #  net radiation
        # vas = self.mean_sat_vp_fao56()
        G = self.soil_heat_flux()

        tmp1 = divide(delta, add(delta, gamma))
        tmp2 = multiply(tmp1, divide(r_n, LAMBDA))
        tmp3 = subtract(tmp2, divide(G, LAMBDA))
        pet = multiply(self.cons['alpha_pt'], tmp3)
        self.input['pet_Priestly_Taylor'] = pet
        return


def custom_resampler(array_like):
    """calculating heat index using monthly values of temperature."""
    return np.sum(power(divide(array_like, 5.0), 1.514))
