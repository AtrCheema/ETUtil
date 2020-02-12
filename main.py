#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://www.intechopen.com/books/advanced-evapotranspiration-methods-and-applications
# https://www.intechopen.com/books/current-perspective-to-predict-actual-evapotranspiration
# https://rdrr.io/cran/Evapotranspiration/man/
# https://www.ncl.ucar.edu/Document/Functions/index.shtml
import pandas as pd
import numpy as np
from numpy import multiply, divide, add, subtract, power, array, where, mean, sqrt
import math

from .utils import Util
from .convert import Temp

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

     :param `units`: a dictionary containing units for all input time series data.
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

    :attributes
        output a dictionary containing calculated et values at different time steps
            """


    def __init__(self, input_df, units, constants, verbose=True):

        super(ReferenceET, self).__init__(input_df.copy(), units, constants=constants, verbose=verbose)



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
        self.check_constants(method='Abtew')  # check that all constants are present

        rs = self.rs()
        et = multiply(self.cons['abtew_k'], divide(rs, LAMBDA))

        self.check_output_freq('Abtew', et)
        return et


    def BlaneyCriddle(self):
        """using formulation of Blaney-Criddle for daily reference crop ETP using monthly mean tmin and tmax.
        Inaccurate under extreme climates. underestimates in windy, dry and sunny conditions and overestimates under calm, humid
        and clouded conditions.

        [2] Allen, R. G. and Pruitt, W. O.: Rational use of the FAO Blaney-Criddle Formula, J. Irrig. Drain. E. ASCE,
             112, 139–155, 1986."""
        self.check_constants(method='BlaneyCriddle')  # check that all constants are present

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
        self.check_constants(method='BrutsaertStrickler')  # check that all constants are present

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
        self.check_constants(method='GrangerGray')  # check that all constants are present

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


    def ChapmanAustralia(self):
        """using formulation of [1],

        uses: a_s=0.23, b_s=0.5, ap=2.4, alphaA=0.14, albedo=0.23

        [1] Chapman, T. 2001, Estimation of evaporation in rainfall-runoff models,
            in F. Ghassemi, D. Post, M. SivapalanR. Vertessy (eds), MODSIM2001: Integrating models for Natural
            Resources Management across Disciplines, Issues and Scales, MSSANZ, vol. 1, pp. 293-298.
            http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.539.3517&rep=rep1&type=pdf
        """
        self.check_constants(method='ChapmanAustralia')  # check that all constants are present

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

        self.check_output_freq('ChapmanAustralia', et)
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
        self.check_constants(method='CRAE')  # check that all constants are present



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
        self.check_constants(method='Turc')  # check that all constants are present

        rs = self.rs()
        ta = self.input['temp'].values
        et = multiply(multiply(self.cons['turc_k'] , (add(multiply(23.88 , rs) , 50))) , divide(ta , (add(ta , 15))))

        if 'rh_mean' in self.input.columns:
            rh_mean = self.input['rh_mean'].values
            eq1 = multiply(multiply(multiply(self.cons['turc_k'] , (add(multiply(23.88 , rs) , 50))) , divide(ta , (add(ta , 15)))) , (add(1 , divide((subtract(50 , rh_mean)) , 70))))
            eq2 = multiply(multiply(self.cons['turc_k'] , (add(multiply(23.88 , rs) , 50))) , divide(ta , (add(ta , 15))))
            et = np.where(rh_mean<50, eq1, eq2)

        self.check_output_freq('Turc', et)
        return et


    def McGuinnessBordne(self):
        """
        calculates evapotranspiration [mm/day] using Mcguinnes Bordne formulation [1].

        [1] McGuinness, J. L., & Bordne, E. F. (1972). A comparison of lysimeter-derived potential evapotranspiration
            with computed values (No. 1452). US Dept. of Agriculture.
        """
        self.check_constants(method='McGuinnessBordne')  # check that all constants are present

        ra = self._et_rad()
        # latent heat of vaporisation, MJ/Kg
        _lambda = LAMBDA # multiply((2.501 - 2.361e-3), self.input['temp'].values)
        tmp1 = multiply((1/_lambda), ra)
        tmp2 = divide(add(self.input['temp'].values, 5), 68)
        et = multiply(tmp1, tmp2)

        self.check_output_freq('McGuinnessBordne', et)
        return et


    def Makkink(self):
        """
        :uses
          a_s, b_s
          temp
          solar_rad

        using formulation of Makkink
        """
        self.check_constants(method='Makkink')  # check that all constants are present

        rs = self.rs()

        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const

        et = subtract(multiply(multiply(0.61, divide(delta, add(delta, gamma))), divide(rs, 2.45)), 0.12)

        self.check_output_freq('Makkink', et)
        return et


    def Linacre(self):
        """
         using formulation of Linacre 1977 [1] who simplified Penman method.
         :uses
           temp
           tdew/rel_hum

         [1] Linacre, E. T. (1977). A simple formula for estimating evaporation rates in various climates,
             using temperature data alone. Agricultural meteorology, 18(6), 409-424.
         """
        self.check_constants(method='Linacre')  # check that all constants are present

        if 'tdew' not in self.input:
            if 'rel_hum' in self.input:
                self.tdew_from_t_rel_hum()

        tm = add(self.input['temp'].values, multiply( 0.006, self.cons['altitude']))
        tmp1 = multiply(500, divide(tm, 100-self.cons['lat']))
        tmp2 = multiply(15,subtract(self.input['temp'].values, self.input['tdew'].values))
        upar = add(tmp1, tmp2)

        et = divide(upar, subtract(80, self.input['temp'].values))

        self.check_output_freq('Linacre', et)
        return et


    def HargreavesSamani(self, method='1985'):
        """
        estimates daily ETo using Hargreaves method [1].
        :uses
          temp
          tmin
          tmax
        :param method: str, if `1985`, then the method of 1985 [1] is followed as calculated by and mentioned by [2]
        if `2003`, then as formula is used as mentioned in [3]
        Note: Current test passes for 1985 method.


        [1] Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature.
            Applied engineering in agriculture, 1(2), 96-99.
        [2] Hargreaves, G. H., & Allen, R. G. (2003). History and evaluation of Hargreaves evapotranspiration equation.
            Journal of Irrigation and Drainage Engineering, 129(1), 53-63.
        [3] https://rdrr.io/cran/Evapotranspiration/man/ET.HargreavesSamani.html
        """
        # self.check_constants(method='HargreavesSamani')  # check that all constants are present

        if method == '2003':
            tmp1 = multiply(0.0023, add(self.input['temp'], 17.8))
            tmp2 = power(subtract(self.input['tmax'].values, self.input['tmin'].values), 0.5)
            tmp3 = multiply(0.408, self._et_rad())
            et = multiply(multiply(tmp1, tmp2), tmp3)

        else:
            ra_my = self._et_rad()
            tmin = self.input['tmin'].values
            tmax = self.input['tmax'].values
            ta = self.input['temp'].values
            # empirical coefficient by Hargreaves and Samani (1985) (S9.13)
            C_HS = 0.00185 *  np.power((subtract(tmax , tmin)),2) - 0.0433 * (subtract(tmax , tmin)) + 0.4023
            et = 0.0135 * C_HS * ra_my / self.cons['LAMDA'] * np.power((subtract(tmax , tmin)), 0.5) * (add(ta , 17.8))

        self.check_output_freq('HargreavesSamani', et)
        return et


    def PenmanMonteith(self):
        """calculates reference evapotrnaspiration according to Penman-Monteith (Allen et al 1998) equation which is
        also recommended by FAO. The etp is calculated at the time step determined by the step size of input data.
        For hourly or sub-hourly calculation, equation 53 is used while for daily time step equation 6 is used.

        # Arguments
        :uses
         lm=None, a_s=0.25, b_s=0.5, albedo=0.23
         temp
         rel_hum

        :param
        :return et, a numpy Pandas dataframe consisting of calculated potential etp values.

        http://www.fao.org/3/X0490E/x0490e08.htm#chapter%204%20%20%20determination%20of%20eto
        """

        self.check_constants(method='PenmanMonteith')

        pet = -9999

        if self.input_freq == 'Hourly':
            if self.cons['lm'] is None:
                raise ValueError('provide input value of lm')

        wind_2m = self._wind_2m()  # wind speed at 2 m height
        D = self.slope_sat_vp(self.input['temp'].values)
        g = self.psy_const

        if self.input_freq=='Daily':
            es = self.mean_sat_vp_fao56()
        elif self.input_freq == 'Hourly':
            es = self.sat_vp_fao56(self.input['temp'].values)
        elif self.input_freq == 'sub_hourly':   #TODO should sub-hourly be same as hourly?
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

        if self.input_freq=='Daily':
            t3 = divide(D, nechay)
            t4 = multiply(t1, t3)
            t5 = multiply(vp_d, divide(g, nechay))
            t6 = divide(multiply(900, wind_2m), add(self.input['temp'].values, 273))
            t7 = multiply(t6, t5)
            pet = add(t4, t7)

        if self.input_freq in ['Hourly', 'sub_hourly']:  #TODO should sub-hourly be same as hourly?
            t3 = multiply(divide(37, self.input['temp']+273), g)
            t4 = multiply(t3, vp_d)
            upar = add(t1, t4)
            pet = divide(upar, nechay)

        self.check_output_freq('PenmanMonteith', pet)
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
        self.check_constants(method='Thornthwait')

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

        self.check_output_freq('Thornthwait', pet)
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
        # TODO not sure if sunshine_hrs is to be used or daylight_hrs
        self.check_constants(method='Hamon')

        # if 'daylight_hrs' not in self.input.columns:
        #     if self.cons['lat'] is None:
        #         raise ValueError('number of daylihgt hours are not given as input so latitude must be provided')
        #     else:
        #         print('Calculating daylight hours indirectly from latitude provided.')
        #         daylight_hrs = divide(self.daylight_fao56(), 12.0)  # shoule be multiple of 12
        # else:
        #     daylight_hrs = self.input['daylight_hrs']

        if 'sunshine_hrs' not in self.input.columns:
            if 'daylight_hrs' not in self.input.columns:
                daylight_hrs = self.daylight_fao56()
            else:
                daylight_hrs = self.input['daylight_hrus']
            sunshine_hrs = daylight_hrs
            print('Warning, sunshine hours are consiered equal to daylight hours')
        else:
            sunshine_hrs = self.input['sunshine_hrs']

        sunshine_hrs = divide(sunshine_hrs, 12.0)

        # preference should be given to tmin and tmax if provided and if tmin, tmax is not provided then use temp which
        # is mean temperature. This is because in original equations, vd_sat is calculated as average of max vapour
        # pressure and minimum vapour pressue.
        if 'tmax' not in self.input.columns:
            if 'temp' not in self.input.columns:
                raise ValueError('Either tmax and tmin or mean temperature should be provided as input')
            else:
                vd_sat = self.sat_vp_fao56(self.input['temp'])
        else:
            vd_sat = self.mean_sat_vp_fao56()

        # in some literature, the equation is divided by 100 by then the cts value is 0.55 instead of 0.0055
        et =cts * 25.4 * power(sunshine_hrs, 2) * (216.7 * vd_sat * 10 / (np.add(self.input['temp'], 273.3)))

        self.check_output_freq('Hamon', et)
        return et


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
        tmp1 = multiply(np.subtract(597.3, multiply(0.57, self.input['temp'].values)), 2.54)
        radIn = divide(self.input['solar_rad'].values, tmp1)

        return radIn


    def JensenHaise(self):
        """
        as given (eq 9) in [1] and implemented in [2]

        uses:  a_s, b_s, ct=0.025, tx=-3

        [1] Xu, C. Y., & Singh, V. P. (2000). Evaluation and generalization of radiation‐based methods for calculating
            evaporation. Hydrological processes, 14(2), 339-349.
        [2] https://github.com/DanluGuo/Evapotranspiration/blob/8efa0a2268a3c9fedac56594b28ac4b5197ea3fe/R/Evapotranspiration.R#L2734
        """
        self.check_constants(method='JensenHaise')

        rs = self.rs()
        tmp1 = multiply(multiply(self.cons['ct'], add(self.input['temp'], self.cons['tx'])), rs)
        et = divide(tmp1, LAMBDA)

        self.check_output_freq('JensenHaise', et)
        return et


    def JesnsenBASINS(self):
        """
        This method generates daily pan evaporation (inches) using a coefficient for the month `cts`, , the daily
        average air temperature (F), a coefficient `ctx`, and solar radiation (langleys/day) as givn in BASINS program[2].
        The computations are
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
        :uses cts float or array like. Value of monthly coefficient `cts` to be used. If float, then same value is
                assumed for all months. If array like then it must be of length 12.
        :uses ctx `float` constant coefficient value of `ctx` to be used in Jensen and Haise formulation.

        [1] Jensen, M. E., & Haise, H. R. (1963). Estimating evapotranspiration from solar radiation. Proceedings of
            the American Society of Civil Engineers, Journal of the Irrigation and Drainage Division, 89, 15-41.
    """
        self.check_constants(method='JensenHaiseBASINS')
        cts = self.cons['cts_jensen']
        ctx = self.cons['ctx_jensen']
        if not isinstance(cts, float):
            if not isinstance(array(ctx), np.ndarray):
                raise ValueError('cts must be array like')
            else:  # if cts is array like it must be given for 12 months of year, not more not less
                if len(array(cts))>12:
                    raise ValueError('cts must be of length 12')
        else:  # if only one value is given for all moths distribute it as monthly value
            cts = array([cts for _ in range(12)])

        if not isinstance(ctx, float):
            raise ValueError('ctx must be float')

        # distributing cts values for all dates of input data
        self.input['cts'] = np.nan
        for m,i in zip(self.input.index.month, self.input.index):
            for _m in range(m):
                self.input.at[i, 'cts'] = cts[_m]

        cts = self.input['cts']
        taf = Temp(self.input['temp'].values, 'centigrade').fahrenheit
        radIn = self.rad_to_evap()
        PanEvp = multiply(multiply(cts, subtract(taf, ctx)), radIn)
        et = where(PanEvp<0.0, 0.0, PanEvp)

        self.check_output_freq('JensenHaiseBASINS', et)
        return et


    def PenPan(self):
        """
        mplementing the PenPan formulation for Class-A pan evaporation

        Rotstayn, L. D., Roderick, M. L. & Farquhar, G. D. 2006. A simple pan-evaporation model for analysis of
            climate simulations: Evaluation over Australia. Geophysical Research Letters, 33.  https://doi.org/10.1029/2006GL027114
        """
        self.check_constants(method='PenPan')

        lat = self.cons['lat']
        ap = self.cons['pen_ap']
        albedo = self.cons['albedo']
        alphaA = self.cons['alphaA']

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

        p_rad = add(1.32, add(multiply(4e-4, lat), multiply(8e-5, lat**2)))
        f_dir = add(-0.11, multiply(1.31, divide(rs, ra)))
        rs_pan = multiply(add(add(multiply(f_dir,p_rad), multiply(1.42,subtract(1, f_dir))), multiply(0.42, albedo)), rs)
        rn_pan = subtract(multiply(1-alphaA, rs_pan), r_nl)

        tmp1 = multiply(divide(delta, add(delta, multiply(ap, gamma))), divide(rn_pan, LAMBDA))
        tmp2 = divide(multiply(ap, gamma), add(delta, multiply(ap, gamma)))
        tmp3 = multiply(f_pan_u, subtract(vas, vabar))
        tmp4 = multiply(tmp2, tmp3)
        epan = add(tmp1, tmp4)

        et = epan

        if self.cons['pan_over_est']:
            if self.cons['pan_est'] == 'pot_et':
                et = multiply(divide(et, 1.078), self.cons['pan_coef'])
            else:
                et = divide(et, 1.078)

        self.check_output_freq('PenPan', et)
        return et


    def Penman(self):
        """
        calculates pan evaporation from open water using formulation of [1] as mentioned (as eq 12) in [2]. if wind data
        is missing then equation 33 from [4] is used which does not require wind data.

        uses:  wind_f='pen48', a_s=0.23, b_s=0.5, albedo=0.23
               uz
               temp
               rs
               reh_hum

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
        self.check_constants(method='Penman')

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
            evap = add(multiply(tmp1, tmp2), tmp3)
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
            evap = add(subtract(tmp1, tmp2), tmp5)

        self.check_output_freq('Penman', evap)
        return


    def PriestleyTaylor(self):
        """
        following formulation of Priestley & Taylor, 1972 [1].
        uses: , a_s=0.23, b_s=0.5, alpha_pt=1.26, albedo=0.23
        :param `alpha_pt` Priestley-Taylor coefficient = 1.26 for Priestley-Taylor model (Priestley and Taylor, 1972)

         [1] Priestley, C. H. B., & Taylor, R. J. (1972). On the assessment of surface heat flux and evaporation using
             large-scale parameters. Monthly weather review, 100(2), 81-92.
         """
        self.check_constants(method='PriestleyTaylor')

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
        et = multiply(self.cons['alpha_pt'], tmp3)

        self.check_output_freq('PriestleyTaylor', et)
        return


    def Romanenko(self):
        """
        using formulation of Romanenko
        uses:
          temp
          rel_hum
        """
        self.check_constants(method='Romanenko')

        t = self.input['temp'].values
        vas = self.mean_sat_vp_fao56()
        vabar = self.avp_from_rel_hum()  # Vapour pressure  *ea*

        tmp1 = power(add(1, divide(t, 25)), 2)
        tmp2 = subtract(1, divide(vabar, vas))
        et = multiply(multiply(4.5, tmp1), tmp2)

        self.check_output_freq('Romanenko', et)
        return et


    def SzilagyiJozsa(self):
        """
        using formulation of Azilagyi, 2007.
        :return: et
        Szilagyi, J. (2007). On the inherent asymmetric nature of the complementary relationship of evaporation. Geophysical Research Letters, 34(2).
        """
        self.check_constants(method='SzilagyiJozsa')

        if self.cons['wind_f']=='pen48':
            _a = 2.626
            _b = 0.09
        else:
            _a = 1.313
            _b = 0.06
        alpha_pt = self.cons['alpha_pt']  # Priestley Taylor constant

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
            et_penman = add(multiply(tmp1, tmp2), tmp3)
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
            et_penman = add(subtract(tmp1, tmp2), tmp5)

       # find equilibrium temperature T_e
        t_e = None

        delta_te = self.slope_sat_vp(t_e)  #   # slope of vapour pressure curve at T_e
        et_pt_te = multiply(alpha_pt, multiply(divide(delta_te, add(delta_te, gamma)), divide(r_n, LAMBDA)))   # Priestley-Taylor evapotranspiration at T_e
        et = subtract(multiply(2, et_pt_te), et_penman)

        self.check_output_freq('SzilagyiJozsa', et)
        return et


def custom_resampler(array_like):
    """calculating heat index using monthly values of temperature."""
    return np.sum(power(divide(array_like, 5.0), 1.514))
