#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["Time", "Distance", "Speed", "Temp", "Pressure", "SolarRad"]

import numpy as np

"""
Unit conversion functions.

:copyright: (c) 2015 by Mark Richards.
:license: BSD 3-Clause, see LICENSE.txt for more details.
"""

import math


TempUnitConverter = {
"Fahrenheit":{
    "Fahrenheit":  lambda fahrenheit: fahrenheit * 1.0,  # fahrenheit to Centigrade
     "Kelvin":     lambda fahrenheit: (fahrenheit + 459.67) * 5/9,  # fahrenheit to kelvin
     "Centigrade": lambda fahrenheit: (fahrenheit - 32) / 1.8  # fahrenheit to Centigrade
},
"Kelvin":{
    "Fahrenheit":  lambda kelvin: kelvin * 9/5 - 459.67,  # kelvin to fahrenheit
     "Kelvin":     lambda k: k*1.0,     # Kelvin to Kelvin
     "Centigrade": lambda kelvin: kelvin - 273.15  # kelvin to Centigrade}
},
"Centigrade":{
    "Fahrenheit":  lambda Centigrade: Centigrade * 1.8 + 32,  # Centigrade to fahrenheit
     "Kelvin":     lambda Centigrade: Centigrade + 273.15,  # Centigrade to kelvin
     "Centigrade": lambda Centigrade: Centigrade * 1.0}
}

metric_dict = {
    'Exa': 1e18,
    'Peta': 1e15,
    'Tera': 1e12,
    'Giga': 1e9,
    'Mega': 1e6,
    'Kilo': 1e3,
    'Hecto': 1e2,
    'Deca': 1e1,
    None: 1,
    'Deci':  1e-1,
    'Centi': 1e-2,
    'Milli': 1e-3,
    'Micro': 1e-6,
    'Nano': 1e-9,
    'Pico': 1e-12,
    'Femto': 1e-15,
    'Atto': 1e-18
}

time_dict = {
        'Year':   31540000,
        'Month':  2628000,
        'Weak':   604800,
        'Day':    86400,
        'Hour':   3600,
        'Minute': 60,
        'Second': 1,
    }

imperial_dist_dict = {
        'Mile': 63360,
        'Furlong': 7920,
        'Rod': 198,
        'Yard': 36,
        'Foot': 12,
        'Inch': 1
    }

PressureConverter = {
"Pascal":{  # Pascal to
    "Pascal": lambda pascal: pascal,
    "Bar": lambda pascal: pascal * 1e-5,
    "Atm": lambda pascal: pascal / 101325,
    "Torr": lambda pascal: pascal * 0.00750062,
    "Psi": lambda pascal: pascal / 6894.76,
    "Ta": lambda pascal: pascal * 1.01971621298E-5
},
"Bar":{ # Bar to
    "Pascal": lambda bar: bar / 0.00001,
    "Bar": lambda bar: bar,
    "Atm": lambda bar: bar / 1.01325,
    "Torr": lambda bar: bar * 750.062,
    "Psi": lambda bar: bar * 14.503,
    "Ta": lambda bar: bar * 1.01972
},
"Atm": {  # Atm to
    "Pascal": lambda atm: atm * 101325,
    "Bar": lambda atm: atm * 1.01325,
    "Atm": lambda atm: atm,
    "Torr": lambda atm: atm * 760,
    "Psi": lambda atm: atm * 14.6959,
    "At": lambda atm: atm * 1.03322755477
},
"Torr": { # Torr to
    "Pascal": lambda torr: torr / 0.00750062,
    "Bar": lambda torr: torr / 750.062,
    "Atm": lambda torr: torr / 760,
    "Torr": lambda tor: tor,
    "Psi": lambda torr: torr / 51.7149,
    "Ta": lambda torr: torr * 0.00135950982242
},
"Psi":{  # Psi to
    "Pascal": lambda psi: psi * 6894.76,
    "Bar": lambda psi: psi / 14.5038,
    "Atm": lambda psi: psi / 14.6959,
    "Torr": lambda psi: psi * 51.7149,
    "Psi": lambda psi: psi,
    "Ta": lambda psi: psi * 0.0703069578296,
},
"Ta":{   # Ta to
    "Pascal": lambda at: at / 1.01971621298E-5,
    "Bar": lambda at: at / 1.0197,
    "Atm": lambda at: at / 1.03322755477,
    "Torr": lambda at: at / 0.00135950982242,
    "Psi": lambda at: at / 0.0703069578296 ,
    "Ta": lambda ta: ta
}
}

DistanceConverter = {
    "Meter":{
        "Meter": lambda meter: meter,
        "Inch": lambda meter: meter * 39.3701
    },
    "Inch":{
        "Meter": lambda inch: inch * 0.0254,
        "Inch": lambda inch: inch
    }
}


unit_plurals = {
    "Inches": "Inch",
    "Miles": "Mile",
    "Meters": "Meter",
    "Feet": "Foot"
}

def split_speed_units(unit):
    dist = unit.split("Per")[0]
    zeit = unit.split("Per")[1]
    if dist in unit_plurals:
        dist = unit_plurals[dist]
    return dist, zeit

import re
def split_units(unit):
    """splits string `unit` based on capital letters"""
    return re.findall('[A-Z][^A-Z]*', unit)


class WrongUnitError(Exception):
    def __init__(self, u_type, qty, unit, allowed, prefix=None):
        self.u_type = u_type
        self.qty = qty
        self.unit = unit
        self.allowed = allowed
        self.pre = prefix

    def __str__(self):
        if self.pre is None:
            return '''
*
*   {} unit `{}` for {} is wrong. Use either of {}
*
'''.format(self.u_type, self.unit, self.qty, self.allowed)
# prefix {milli} provided for {input} unit of {temperature} is wrong. {input} unit is {millipascal}, allowed are {}}
        else:
            return """
*
* prefix `{}` provided for {} unit of {} is wrong.
* {} unit is: {}. Allowed units are
* {}.
*
""".format(self.pre, self.u_type, self.qty, self.u_type, self.unit, self.allowed)



def check_converter(converter):
    super_keys = converter.keys()

    for k, v in converter.items():
        sub_keys = v.keys()

        if all(x in super_keys for x in sub_keys):
            a = 1
        else:
            a = 0

        if all(x in sub_keys for x in super_keys):
            b = 1
        else:
            b = 0

        assert a == b


def check_plurals(unit):
    if unit in unit_plurals:
        unit = unit_plurals[unit]
    return unit

class Distance(object):
    """
    unit converter for distance or length between different imperial and/or metric units.
    ```python
    t = Distance(np.array([2.0]), "Mile")
    np.testing.assert_array_almost_equal(t.Inch, [126720], 5)
    np.testing.assert_array_almost_equal(t.Meter, [3218.688], 5)
    np.testing.assert_array_almost_equal(t.KiloMeter, [3.218688], 5)
    np.testing.assert_array_almost_equal(t.CentiMeter, [321869], 0)
    np.testing.assert_array_almost_equal(t.Foot, [10560.], 5)

    t = Distance(np.array([5000]), "MilliMeter")
    np.testing.assert_array_almost_equal(t.Inch, [196.85039], 5)
    np.testing.assert_array_almost_equal(t.Meter, [5.0], 5)
    np.testing.assert_array_almost_equal(t.KiloMeter, [0.005], 5)
    np.testing.assert_array_almost_equal(t.CentiMeter, [500.0], 5)
    np.testing.assert_array_almost_equal(t.Foot, [16.404199], 5)
    ```
    """

    def __init__(self, val, input_unit):
        self.val = val
        self.input_unit = input_unit

    @property
    def allowed(self):
        return list(imperial_dist_dict.keys()) + ['Meter']

    @property
    def input_unit(self):
        return self._input_unit

    @input_unit.setter
    def input_unit(self, in_unit):
        self._input_unit = in_unit

    def __getattr__(self, out_unit):
        if out_unit.startswith('_'): #pycharm calls this method for its own working, executing default behaviour at such calls
            return self.__getattribute__(out_unit)
        else:
            act_iu, iu_pf = self._preprocess(self.input_unit, "Input")

            act_ou, ou_pf = self._preprocess(out_unit, "Output")

            act_iu = check_plurals(act_iu)
            act_ou = check_plurals(act_ou)

            if act_iu not in self.allowed:
                raise WrongUnitError("Input", self.__class__.__name__, act_iu, self.allowed)
            if act_ou not in self.allowed:
                raise WrongUnitError("output", self.__class__.__name__, act_ou, self.allowed)

            out_in_meter = self._to_meters(ou_pf, act_ou)  # get number of meters in output unit
            input_in_meter = self.val * iu_pf  # for default case when input unit has Meter in it

            # if input unit is in imperial system, first convert it into inches and then into meters
            if act_iu in imperial_dist_dict:
                input_in_inches = imperial_dist_dict[act_iu] * self.val * iu_pf
                input_in_meter = DistanceConverter['Inch']['Meter'](input_in_inches)

            val = input_in_meter / out_in_meter

            return val

    def _to_meters(self, prefix, actual_unit):
        meters = prefix
        if actual_unit != "Meter":
            inches = imperial_dist_dict[actual_unit] * prefix
            meters = DistanceConverter['Inch']['Meter'](inches)
        return meters

    def _preprocess(self, given_unit, io_type="Input"):
        split_u = split_units(given_unit)
        if len(split_u) < 1:  # Given unit contained no capital letter so list is empty
            raise WrongUnitError(io_type, self.__class__.__name__, given_unit, self.allowed)

        pf, ou_pf = 1.0, 1.0
        act_u = split_u[0]
        if len(split_u) > 1:
            pre_u = split_u[0]  # prefix of input unit
            act_u = split_u[1]  # actual input unit

            if pre_u in metric_dict:
                pf = metric_dict[pre_u]  # input unit prefix factor
            else:
                raise WrongUnitError(io_type, self.__class__.__name__, act_u, self.allowed, pre_u)

        return act_u, pf


class Pressure(object):
    """
    ```python
    p = Pressure(20, "Pascal")
    print(p.MilliBar)    #>> 0.2
    print(p.Bar)         #>> 0.0002
    p = Pressure(np.array([10, 20]), "KiloPascal")
    print(p.MilliBar)    # >> [100, 200]
    p = Pressure(np.array([1000, 2000]), "MilliBar")
    print(p.KiloPascal)  #>> [100, 200]
    print(p.Atm)         # >> [0.98692, 1.9738]
    ```
    """

    def __init__(self, val, input_unit):
        self.val = val
        check_converter(PressureConverter)
        self.input_unit = input_unit

    @property
    def allowed(self):
        return list(PressureConverter.keys())

    @property
    def input_unit(self):
        return self._input_unit

    @input_unit.setter
    def input_unit(self, in_unit):

        self._input_unit = in_unit

    def __getattr__(self, out_unit):
        if out_unit.startswith('_'): #pycharm calls this method for its own working, executing default behaviour at such calls
            return self.__getattribute__(out_unit)
        else:
            act_iu, iu_pf = self._preprocess(self.input_unit, "Input")

            act_ou, ou_pf = self._preprocess(out_unit, "Output")

            if act_iu not in self.allowed:
                raise WrongUnitError("Input", self.__class__.__name__, act_iu, self.allowed)
            if act_ou not in self.allowed:
                raise WrongUnitError("output", self.__class__.__name__, act_ou, self.allowed)

            ou_f = PressureConverter[act_iu][act_ou](self.val)

            val = np.round(np.array((iu_pf * ou_f) / ou_pf), 5)
            return val

    def _preprocess(self, given_unit, io_type="Input"):
        split_u = split_units(given_unit)
        if len(split_u) < 1:  # Given unit contained no capital letter so list is empty
            raise WrongUnitError(io_type, self.__class__.__name__, given_unit, self.allowed)

        pf, ou_pf = 1.0, 1.0
        act_u = split_u[0]
        if len(split_u) > 1:
            pre_u = split_u[0]  # prefix of input unit
            act_u = split_u[1]  # actual input unit

            if pre_u in metric_dict:
                pf = metric_dict[pre_u]  # input unit prefix factor
            else:
                raise WrongUnitError(io_type, self.__class__.__name__, act_u, self.allowed, pre_u)

        return act_u, pf


class Temp(object):
    """
    The idea is to write the conversion functions in a dictionary and then dynamically create attribute it the attribute
    is present in converter as key otherwise raise WongUnitError.
    converts temperature among units [kelvin, centigrade, fahrenheit]
    :param `temp`  a numpy array
    :param `input_unit` str, units of temp, should be "Kelvin", "Centigrade" or "Fahrenheit"

    Example:
    ```python
    temp = np.arange(10)
    T = Temp(temp, 'Centigrade')
    T.Kelvin
    >> array([273 274 275 276 277 278 279 280 281 282])
    T.Fahrenheit
    >> array([32. , 33.8, 35.6, 37.4, 39.2, 41. , 42.8, 44.6, 46.4, 48.2])
    T.Centigrade
    >>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ```
    """
    def __init__(self, val, input_unit):
        self.val = val
        check_converter(TempUnitConverter)
        self.input_unit = input_unit

    def __getattr__(self, out_unit):
        if out_unit.startswith('_'): #pycharm calls this method for its own working, executing default behaviour at such calls
            return self.__getattribute__(out_unit)
        else:
            if out_unit not in TempUnitConverter[self.input_unit]:
                raise WrongUnitError("output", self.__class__.__name__, out_unit, self.allowed)

            val = TempUnitConverter[self.input_unit][str(out_unit)](self.val)
            return val

    @property
    def allowed(self):
        return list(TempUnitConverter.keys())

    @property
    def input_unit(self):
        return self._input_unit

    @input_unit.setter
    def input_unit(self, in_unit):
        if in_unit not in self.allowed:
            raise WrongUnitError("Input", self.__class__.__name__, in_unit, self.allowed)
        self._input_unit = in_unit


class Time(object):

    """
    ```python
    t = Time(np.array([100, 200]), "Hour")
    np.testing.assert_array_almost_equal(t.Day, [4.16666667, 8.33333333], 5)
    t = Time(np.array([48, 24]), "Day")
    np.testing.assert_array_almost_equal(t.Minute, [69120., 34560.], 5)
    ```
    """
    def __init__(self, val, input_unit):
        self.val = val
        self.input_unit = input_unit

    @property
    def allowed(self):
        return list(time_dict.keys())

    @property
    def input_unit(self):
        return self._input_unit

    @input_unit.setter
    def input_unit(self, in_unit):
        self._input_unit = in_unit

    def __getattr__(self, out_unit):
        if out_unit.startswith('_'): #pycharm calls this method for its own working, executing default behaviour at such calls
            return self.__getattribute__(out_unit)
        else:
            act_iu, iu_pf = self._preprocess(self.input_unit, "Input")

            act_ou, ou_pf = self._preprocess(out_unit, "Output")

            if act_iu not in self.allowed:
                raise WrongUnitError("Input", self.__class__.__name__, act_iu, self.allowed)
            if act_ou not in self.allowed:
                raise WrongUnitError("output", self.__class__.__name__, act_ou, self.allowed)

            in_sec = time_dict[act_iu] * self.val * iu_pf
            val = in_sec / (time_dict[act_ou]*ou_pf)
            return val

    def _preprocess(self, given_unit, io_type="Input"):
        split_u = split_units(given_unit)
        if len(split_u) < 1:  # Given unit contained no capital letter so list is empty
            raise WrongUnitError(io_type, self.__class__.__name__, given_unit, self.allowed)

        pf, ou_pf = 1.0, 1.0
        act_u = split_u[0]
        if len(split_u) > 1:
            pre_u = split_u[0]  # prefix of input unit
            act_u = split_u[1]  # actual input unit

            if pre_u in metric_dict:
                pf = metric_dict[pre_u]  # input unit prefix factor
            else:
                raise WrongUnitError(io_type, self.__class__.__name__, act_u, self.allowed, pre_u)

        return act_u, pf



class Speed(object):
    """
    converts between different units using Distance and Time classes which convert
    distance and time units separately. This class both classes separately and
    then does the rest of the work.
    ```python
    s = Speed(np.array([10]), "KiloMeterPerHour")
    np.testing.assert_array_almost_equal(s.MeterPerSecond, [2.77777778], 5)
    np.testing.assert_array_almost_equal(s.MilePerHour, [6.21371192], 5)
    np.testing.assert_array_almost_equal(s.FootPerSecond, [9.11344415], 5)
    s = Speed(np.array([14]), "FootPerSecond")
    np.testing.assert_array_almost_equal(s.MeterPerSecond, [4.2672], 5)
    np.testing.assert_array_almost_equal(s.MilePerHour, [9.54545], 5)
    np.testing.assert_array_almost_equal(s.KiloMeterPerHour, [15.3619], 4)
    s = Speed(np.arange(10), 'KiloMetersPerHour')
    o = np.array([ 0. , 10.936, 21.872, 32.808, 43.744, 54.680, 65.616 , 76.552, 87.489, 98.425])
    np.testing.assert_array_almost_equal(s.InchesPerSecond, o, 2)
    ```
    """

    def __init__(self, val, input_unit):
        self.val = val
        self.input_unit = input_unit

    @property
    def input_unit(self):
        return self._input_unit

    @input_unit.setter
    def input_unit(self, in_unit):
        self._input_unit = in_unit


    def __getattr__(self, out_unit):
        if out_unit.startswith('_'): #pycharm calls this method for its own working, executing default behaviour at such calls
            return self.__getattribute__(out_unit)
        else:
            in_dist, in_zeit = split_speed_units(self.input_unit)
            out_dist, out_zeit = split_speed_units(out_unit)

            d = Distance(np.array([1]), in_dist)
            dist_f = getattr(d, out_dist)  # distance factor
            t = Time(np.array([1]), in_zeit)
            time_f = getattr(t, out_zeit)   # time factor

            out_val = self.val * (dist_f/time_f)
            return out_val



class SolarRad(object):
    """
    Watt is power and it is measure of how fast we generate/consume energy.
    Joule is energy.
    Watt as per time built it int. 1 Watt = 1 Joule per second.
    WattHour (not Watt per hour) is energy. If you leave a 4 W device running for one hour, you generate 4 WHr of
    energy, which is 4 JHr/s x 3600 sec / 1 Hr, which is 14400 Joules
    #https://www.physicsforums.com/threads/is-it-watt-per-second-or-watt-per-hour.512361/

    Watt = Newton Meter / Second, amount of work done per second
    #https://www.physicsforums.com/threads/watts-newton-meters-per-second.308266/

    solar_irradiance: measured in W/m^2, Langley/time
    On a type sunny day, solar irradiance on earth will be around 900 W/m^2. #https://www.e-education.psu.edu/eme812/node/644

    Solar irradiance varies from 0 kW/m^2 to 1 kW/m^2  #https://www.pveducation.org/pvcdrom/properties-of-sunlight/measurement-of-solar-radiation

    #solar irradiance is solar power: rate at which solar energy falls onto surface. Power is measured in Watt but we measure
    solar irradiance in power per unit area i.e. W/m^2. If the sun shines at a constant 1000 W/m² for one hour, we say it has delivered 1 kWh/m² of energy.

    1 Langley = 41868 Joules/square_meter                                                i
    1 Langley = 4.184 Joules/square_centimeter
    1 Cal = 4.1868 Joule                                                                 ii
    1 Watt = 1 Joule/Sec                                                                 iii
    so
    1 Langley/hour = 41868/3600.0 = 11.63  W/m^2
    1 Langley/min = 41868/60.0    = 697.3  W/m^2
    1 Ley/day     = 41868/86400   = 0.4845 W/m^2

    * TIM below means timestep in minutes.

    1 W/m^2 = 60 / 41868                               =  0.001433     Ley/min          from ref1                           (1a)
    1 W/m^2 = 3600 / 41868                             =  0.0858       Ley/hour         from ref1                           (1b)
    1 W/m^2 = 86400 / 41868                            =  2.0636       Ley/day                                              (1c)
    1 W/m^2 = (60*TIM) / 41868                         =               Ley/timestep

    1 W/m^2 = 60 / 1000.0                              = 0.06          kJ/m^2  minute   from ref1                           (1d)
    1 W/m^2 = 3600 / 1000.0                            = 3.6           kJ/m^2  hour     from ref1                           (1e)
    1 W/m^2 = 86400 / 1000.0                           = 86.4          kJ/m^2  day     following sequence from above 2
    1 W/m^2 = (60*TIM) / 1000.0                        =               kJ/m^2  timestep

    1 W/m^2 = ?                                        = 0.024         kWh/m^2          (hour)                              (1f)

    1 W/m^2 = (60) / (10000.0 * 4.1868)                = 0.001433      cal/cm2 min      (min)                               (1g)
    1 W/m^2 = (3600) / (10000.0* 4.1868)               = 0.0858        cal/cm2 hour     (hour)                              (1h)
    1 W/m^2 = (86400) / (10000.0 * 4.1868)             = 2.063         cal/cm2 day      following sequence from above 2
    1 W/m^2 = (60*TIM) / (10000.0 * 4.1868)            =               cal/cm^2 timestep

    1 W/m^2 = (86400) / (10000.0* 4.1868)              = 2.06362       cal/cm2 day      ()                                  (1ha)

    1 W/m^2 = (60) / (10000.0)                         = 0.006         J/cm2 min       from (1g) and ii                     (1i)
    1 W/m^2 = (3600) / (10000.0)                       = 0.36          J/cm2 hour      from (1h) and ii                     (1j)
    1 W/m^2 = (86400) / (10000.0)                      = 8.64          J/cm2 day       following sequence from above 2
    1 W/m^2 = (60*TIM) / 10000.0                       =               J/cm2 timestep

    1 W/m^2 = (60) / (1.0)                             = 60            J/m2 min        from (1i) or (1d)                    (1k)
    1 W/m^2 = (3600) / (1.0)                           = 3600          J/m2 hour       from (1j) or (1e)                    (1L)
    1 W/m^2 = (86400) / (1.0)                          = 86400         J/m2 day        from (1j)                            (1M)
    1 W/m^2 = (60*TIM)                                 =               J/m^2 timestep-1

    1 W/m^2 = 86400 / 1e6                              = 0.0864        MJ/m2 day-1       from ref5
    1 W/m^2 = 3600 / 1e6                               = 0.0036        MJ/m2 hour-1
    1 W/m^2 = 60 / 1e6                                 = 6e-5          MJ/m2 min-1
    1 W/m^2 = (60*TIM) / 1e6                           =               MJ/m2 timestep-1


    1 cal/cm^2 day-1                                   = 4.1868 10-2 MJ m-2 day-1       from ref5


    1 Ley/min  = 41868/60.0                            = 697.3         W/m2       from ref1                          (2a)

    1 Ley/min  = (60)/(60.0)                           = 1.0           Ley/min                                       (2b)
    1 Ley/min  = (3600)/(60.0)                         = 60.0          Ley/hour                                      (2c)
    1 Ley/min  = (86400)/(60.0)                        = 1440.0        Ley/day        combining 2a and 1d            (2d)
    1 Ley/min  = (60*TIM)/60.0                         =               Ley/TIM

    1 Ley/min  = (41868*60.0) / (60.0)                 = 41868.0       J/m2 minute    combining 2a and 1L            (2i)
    1 Ley/min  = (41868*3600.0) / (60.0)               = 2512080.0     J/m2 hour                                     (2j)
    1 Ley/min  = (41868*86400) / (60.0)                = 60289920      J/m2 day    following seq from above 2
    1 Ley/min  = (41868*(60*TIM)) / 60                 =               J/m2 TIM

    1 Ley/min  = (41868*60.0) / (60.0* 1e3)            = 41.868        kJ/m^2  minute                                (2e)
    1 Ley/min  = (41868*3600.0) / (60.0* 1e3)          = 2512.08       kJ/m^2  hour                                  (2f)
    1 Ley/min  = (41868*86400.0) / (60.0* 1e3)         = 60289.92      kJ/m^2  day   following sequence from above 2
    1 Ley/min  = (41868*(60*TIM) / 60.0* 1e3)          =               KJ/m^2  TIM

    1 Ley/min  = (41868*60.0) / (60.0 * 1e6)           =  0.04186      MJ/m2 min    TBV
    1 Ley/min  = (41868*3600) / (60.0 * 1e6)           =  2.512        MJ/m2 hour    TBV
    1 Ley/min  = (41868*86400) / (60.0 * 1e6)          =  60.28        MJ/m2 day    TBV
    1 Ley/min  = (41868*(60*TIM)) / (60.0 * 1e6)       =               MJ/m2 TIM

    1 Ley/min  = (41868*60.0) / (60.0 * 1e4)            = 4.1868       J/cm2 min      combinging 2a and 1i           (2g)
    1 Ley/min  = (41868*3600.0) / (60.0 * 1e4)          = 251.208      J/cm2 hour                                    (2h)
    1 Ley/min  = (41868*86400.0) / (60.0 * 1e4)         = 6028.99      J/cm2 day    following sequence from above 2
    1 Ley/tim  = (41868*(60*TIM) / (60.0 * 1e4)         =              J/cm2 TIM

    1 Ley/min  = (41868*60.0) / (60.0 * 1e4 * 1e3)      = 4.186e-3     KJ/cm2 min     TBV
    1 Ley/min  = (41868*3600) / (60.0 * 1e4 * 1e3)      = 0.2512       KJ/cm2 hour    TBV
    1 Ley/min  = (41868*86400) / (60.0 * 1e4 * 1e3)     = 6.0289       KJ/cm2 day     TBV
    1 Ley/min  = (41868*(60*TIM)) / (60.0 * 1e4 * 1e3)  =              KJ/cm2 TIM     TBV

    1 Ley/min  = (41868*60.0) / (60.0 * 1e4 * 1e6)      = 4.186e-6     MJ/cm2 min     TBV
    1 Ley/min  = (41868*3600) / (60.0 * 1e4 * 1e6)      = 2.512e-4     MJ/cm2 hour    TBV
    1 Ley/min  = (41868*86400) / (60.0 * 1e4 * 1e6)     = 6.0289-3     MJ/cm2 day     TBV
    1 Ley/min  = (41868*(60*TIM)) / (60.0 * 1e4 * 1e6)  =              MJ/cm2 TIM     TBV


    1 Ley/hour = 41868/3600                             = 11.63        W/m2       from ref1                          (3a)

    1 Ley/hour = (41868/3600) * 60 / (1.0 * 1e4)        =  0.0697      J/cm^2 minute
    1 Ley/hour = (41868/3600) * 3600 / (1.0 * 1e4)      =  4.1868      J/cm^2 hour
    1 Ley/hour = (41868/3600) * 86400 / (1.0 * 1e4)     =  100.483     J/cm^2 day
    1 Ley/hour = (41868/3600) * (60*TIM) / (1.0 * 1e4)  =              J/cm^2 TIM

    1 Ley/hour = (41868/3600) * 60 / (1e3 * 1e4)        =  6.97e-5     KJ/cm^2 minute
    1 Ley/hour = (41868/3600) * 3600 / (1e3 * 1e4)      =  0.0041868   KJ/cm^2 hour
    1 Ley/hour = (41868/3600) * 86400 / (1e3 * 1e4)     =  0.100483    KJ/cm^2 day
    1 Ley/hour = (41868/3600) * ((60*TIM)/(1e3 * 1e4)   =              KJ/m^2 TIM

    1 Ley/hour = (41868/3600) * 60 / (1e6 * 1e4)        =  6.978e-8    MJ/cm^2 minute
    1 Ley/hour = (41868/3600) * 3600 / (1e6 * 1e4)      =  4.1868e-6   MJ/cm^2 hour
    1 Ley/hour = (41868/3600) * 86400 / (1e6 * 1e4)     =  1.0048e-4   MJ/cm^2 day
    1 Ley/hour = (41868/3600)*((60*TIM)/(1e6 * 1e4))    =              MJ/cm^2 TIM

    1 Ley/hour = (41868/3600) * (60/1.0)                = 697.80       J/m^2 minute
    1 Ley/hour = (41868/3600) * (3600/1.0)              = 41868.0      J/m^2 hour                                    (3)
    1 Ley/hour = (41868/3600) * (86400/1.0)             = 1004832.0    J/m^2 day                                     (3)
    1 Ley/hour = (41868/3600) * ((60*TIM)/1.0)          =              J/m^2 TIM

    1 Ley/hour = (41868/3600) * (60/ 1e3)               = 0.6978       KJ/m^2 minute                                 (3 )
    1 Ley/hour = (41868/3600) * (3600/ 1e3)             = 41.868       KJ/m^2 hour  combining 3a and 1e              (3b)
    1 Ley/hour = (41868/3600) * (86400/ 1e3)            = 1004.83      KJ/m^2 day                                    (3c)
    1 Ley/hour = (41868/3600) * ((60*TIM)/ 1e3)         =              KJ/m^2 TIM

    1 Ley/hour = (41868/3600) * (60/ 1e6)               = 0.0006978    MJ/m^2 minute                                 (3)
    1 Ley/hour = (41868/3600) * (3600/ 1e6)             = 0.041868     MJ/m^2 hour                                   (3)
    1 Ley/hour = (41868/3600) * (86400/ 1e6)            = 1.00483      MJ/m^2 day                                    (3)
    1 Ley/hour = (41868/3600) * ((60*TIM)/ 1e6)         =              MJ/m^2 TIM

    1 Ley/day =  41868/86400                            = 0.4845       W/m2                                          (4a)  val by ref4

    1 Ley/day = (41868/86400) * (60/1.0)                = 29.075       J/m^2 min    TBV
    1 Ley/day = (41868/86400) * (3600/1.0)              = 1744.5       J/m^2 hour  combining 4a and 1L
    1 Ley/day = (41868/86400) * (86400/1.0)             = 41868.0      J/m^2 day
    1 Ley/day = (41868/86400) * ((60*TIM)/1.0)          =              J/m^2 TIM

    1 Ley/day = (41868/86400) * (60/ 1e3)               = 0.0290       KJ/m^2 min     TBV
    1 Ley/day = (41868/86400) * (3600/ 1e3)             = 1.7445       KJ/m^2 hour
    1 Ley/day = (41868/86400) * (86400/ 1e3)            = 41.868       KJ/m^2 day
    1 Ley/day = (41868/86400) * ((TIM*60)/ 1e3)         =              KJ/m^2 TIM

    1 Ley/day = (41868/86400) * (60/ 1e6)               = 2.907e-05    MJ/m^2 minute
    1 Ley/day = (41868/86400) * (3600/ 1e6)             = 0.0017445    MJ/m^2 hour
    1 Ley/day = (41868/86400) * (86400/ 1e6)            = 0.041868     MJ/m^2 day
    1 Ley/day = (41868/86400) * ((TIM*60)/ 1e6)         =              MJ/m^2 TIM

    1 Ley/day = (41868/86400) * 60 / (1.0 * 1e4)        = 2.9075-3     J/cm^2 min    TBV
    1 Ley/day = (41868/86400) * 3600 / (1.0 * 1e4)      = 0.17445      J/cm^2 hour
    1 Ley/day = (41868/86400) * 86400 / (1.0 * 1e4)     = 4.1868       J/cm^2 day
    1 Ley/day = (41868/86400) * (60*TIM) / (1.0 * 1e4)  =              J/cm^2 TIM

    1 Ley/day = (41868/86400) * 60 / (1e3 * 1e4)        = 2.907e-6     KJ/cm^2 min     TBV
    1 Ley/day = (41868/86400) * 3600 / (1e3 * 1e4)      = 1.7445-4     KJ/cm^2 hour
    1 Ley/day = (41868/86400) * 86400 / (1e3 * 1e4)     = 4.1868e-3    KJ/cm^2 day
    1 Ley/day = (41868/86400) * (TIM*60) / (1e3 * 1e4)  =              KJ/cm^2 TIM

    1 Ley/day = (41868/86400) * 60 / (1e6 * 1e4)        = 2.907e-09    MJ/cm^2 minute
    1 Ley/day = (41868/86400) * 3600 / (1e6 * 1e4)      = 1.7445-7     MJ/cm^2 hour
    1 Ley/day = (41868/86400) * 86400 / (1e6 * 1e4)     = 4.1868e-6    MJ/cm^2 day
    1 Ley/day = (41868/86400) * (TIM*60) / (1e6 * 1e4)  =              MJ/cm^2 TIM

    1 J/cm2 min  = 1e4 / 60.0                           = 166.66       W/m2      from (1i)                          (5)

    1 J/cm2 hour = 1e4 / 3600.0                         =  2.7         W/m2      from (1j)                          (6a)  val by ref3

    1 J/cm2 hour = (1e4 * 60) / (3600*41868)            = 0.00398      Ley/min   from (6a) and (1a)                 (6b)
    1 J/cm2 hour = (1e4 * 3600) / (3600*41868)          = 0.2388       Ley/hour  from (6a) and (1b)                 (6c)
    1 J/cm2 hour = (1e4 * 86400) / (3600*41868)         = 5.7323       Ley/day   from (6a) and (1c)                 (6c)
    1 J/cm2 hour = (1e4 * (TIM*10)) / (3600*41868)      = 5.7323       Ley/TIM

    1 J/cm2 hour =  (1e4 * 60) / (1e6*3600)             = 1.667e-4     MJ/m^2 min-1      TBV
    1 J/cm2 hour =  (1e4 * 3600) / (1e6*3600)           = 0.01         MJ/m^2 hour       http://www.fao.org/3/X0490E/x0490e0i.htm
    1 J/cm2 hour =  (1e4 * 86400) / (1e6*3600)          = 0.24         MJ/m^2 day-1      TBV
    1 J/cm2 hour =  (1e4 * (60*TIM)) / (1e6*3600)       =              MJ/m^2 TIM

    1 J/cm2 day = 1e4 / 86400.0                         =  0.11574    W/m2            from (6a)                   (7a)

    1 J/cm2 day =  (1e4 * 60) / (1e6*86400)             =  6.94e-6    MJ/m2 min
    1 J/cm2 day =  (1e4 * 3600) / (1e6*86400)           =  4.167e-4   MJ/m2 hour     TBV
    1 J/cm2 day =  (1e4 * 86400) / (1e6*86400)          =  0.01       MJ/m2 day      from ref5
    1 J/cm2 day =  (1e4 * (TIM*60)) / (1e6*86400)       =             MJ/m2 TIM

    1 J/m2 min  = 1/60.0                                = 0.01667     W/m2       from (1k)                      (8a)

    1 J/m2 hour = 1/3600.0                              = 0.000277    W/m2      from (1L)                      (9a)

    ref1: https://www.nrel.gov/grid/solar-resource/assets/data/3305.pdf
    ref2: https://www.wcc.nrcs.usda.gov/ftpref/wntsc/H&H/GEM/SolarRadConversion.pdf
    ref3: https://www.physicsforums.com/threads/convert-solar-radiation-from-joule-per-cm-square-to-watt-per-meter-2.970019/
    ref4: https://physics.stackexchange.com/questions/189022/converting-langley-to-watts
    ref5: http://www.fao.org/3/X0490E/x0490e0i.htm

    converts solar radiation among units [JoulePerCentimeterSquarePerHour, WattPerMeterSquare]
    :param `temp`  a numpy array
    :param `input_units` str, units of temp, should be one of following strings

               "WattPerMeterSquare":
               "LangleyPerUnitTime":
               "JoulePerCentimeterSquarePerUnitTime":
               "JoulePerMeterSquarePerUnitTime":
            where 'UnitTime can be either 'Day', 'Hour' or x+minutes where x is any integer. It represents


    Example:
    ```python
    rad = np.arange(10)
    rad = SolarRad(rad, 'JoulePerCentimeterSquarePerHour')
    rad.JoulePerCentimeterSquarePerHour
    >> array([0 1 2 3 4 5 6 7 8 9])
    rad.WattPerMeterSquarePerHour
    >> array([0 1 2 3 4 5 6 7 8 9])
    ```
    """
    def __init__(self, solar_rad, input_units):
        self.solar_rad = solar_rad
        self.get_input_units(input_units)

        self.unit_bank = {'WattPerMeterSquare': {'WattPerMeterSquare': 1.0,
                                                 'LangleyPerUnitTime': self.seconds/41868,
                                                 'kiloJoulePerMeterSquarePerUnitTime':self.seconds/1000.0,
                                                 'CaloriePerCentimeterSquarePerUnitTime': self.seconds/(10000.0*4.1868),
                                                 'JoulePerCentimeterSquarePerUnitTime': self.seconds/10000.0,
                                                 'JoulePerMeterSquarePerUnitTime': self.seconds/1.0},
                          'LangleyPerUnitTime': {'WattPerMeterSquare': 1.0,
                                                 'LangleyPerUnitTime': 1.0,
                                                 'kiloJoulePerMeterSquarePerUnitTime':self.seconds/1000.0,
                                                 'CaloriePerCentimeterSquarePerUnitTime': self.seconds/(10000.0*4.1868),
                                                 'JoulePerCentimeterSquarePerUnitTime': self.seconds/10000.0,
                                                 'JoulePerMeterSquarePerUnitTime': self.seconds/1.0},
                          'JoulePerCentimeterSquarePerUnitTime': {'WattPerMeterSquare': 1.0,
                                                 'LangleyPerUnitTime': self.seconds/41868,
                                                 'kiloJoulePerMeterSquarePerUnitTime':self.seconds/1000.0,
                                                 'CaloriePerCentimeterSquarePerUnitTime': self.seconds/(10000.0*4.1868),
                                                 'JoulePerCentimeterSquarePerUnitTime': 1.0,
                                                 'JoulePerMeterSquarePerUnitTime': self.seconds/1.0},
                          'JoulePerMeterSquarePerUnitTime': {'WattPerMeterSquare': 1.0,
                                                 'LangleyPerUnitTime': self.seconds/41868,
                                                 'kiloJoulePerMeterSquarePerUnitTime':self.seconds/1000.0,
                                                 'CaloriePerCentimeterSquarePerUnitTime': self.seconds/(10000.0*4.1868),
                                                 'JoulePerCentimeterSquarePerUnitTime': self.seconds/10000.0,
                                                 'JoulePerMeterSquarePerUnitTime': 1.0}}

    def get_input_units(self, raw_input_unit):
        if raw_input_unit == 'WattPerMeterSquare':
            self.input_units = raw_input_unit
        else:
            unit_parts = raw_input_unit.split('Per')
            unit_time = unit_parts[-1]
            self._find_seconds(unit_time)
            self.input_units = 'Per'.join(unit_parts[0:-1]) + 'Per' + str(self.seconds)


    def _find_seconds(self, UnitTime):
        if UnitTime == 'Hour':
            self.seconds = 3600
        elif UnitTime == 'Day':
            self.seconds = 86400
        elif 'minutes' in UnitTime:
            minutes = int(UnitTime.split('m')[0])
            self.seconds = 60*minutes
        else:
            raise ValueError


def rad2deg(radians):
    """
    Convert radians to angular degrees

    :param radians: Value in radians to be converted.
    :return: Value in angular degrees
    :rtype: float
    """
    return radians * (180.0 / math.pi)
