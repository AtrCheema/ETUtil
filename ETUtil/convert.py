#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

"""
Unit conversion functions.

:copyright: (c) 2015 by Mark Richards.
:license: BSD 3-Clause, see LICENSE.txt for more details.
"""

import math

WindUnitConverter = {
    'MeterPerSecond': {'MeterPerSecond': 1,
                       "KiloMeterPerHour": 0.277,
                       'MilesPerHour': 0.44704,
                       "InchesPerSecond": 0.0254,
                       "FeetPerSecond": 0.3048},
    "KiloMeterPerHour": {'MeterPerSecond': 3.6,
                         "KiloMeterPerHour": 1.0,
                         'MilesPerHour': 1.60934,
                         "InchesPerSecond": 0.09144,
                         "FeetPerSecond": 1.09728},
    "MilesPerHour": {'MeterPerSecond': 2.236,
                     "KiloMeterPerHour": 0.6213,
                     'MilesPerHour': 1.0,
                     "InchesPerSecond": 0.0568,
                     "FeetPerSecond": 0.6818},
    "InchesPerSecond": {'MeterPerSecond': 39.37,
                        "KiloMeterPerHour": 10.93,
                        'MilesPerHour': 17.6,
                        "InchesPerSecond": 1.0,
                        "FeetPerSecond": 12.0},
    "FeetPerSecond": {'MeterPerSecond': 3.28,
                      "KiloMeterPerHour": 0.9113,
                      'MilesPerHour': 1.4667,
                      "InchesPerSecond": 0.0833,
                      "FeetPerSecond": 1.0},
}


PressureUnitConverter = {
    'Pascal': {'Pascal': 1,
                "KiloPascal": 0.001,
                'MegaPascal': 1e-6,
                "Bar": 1e-5,
                "MilliBar": 0.01,
                "KiloBar": 1,
                "MegaBar": 1,
                "mmHG": 0.00750062,
                "atm":9.86923e-6,
                "psi":0.000145038},
    "KiloPascal": {'Pascal': 1,
                "KiloPascal": 1,
                'MegaPascal': 1e-6,
                "Bar": 0.0254,
                "MilliBar": 0.3048,
                "KiloBar": 1,
                "MegaBar": 1,
                "mmHG": 1,
                "atm":1,
                "psi":1},
    "MegaPascal": {'Pascal': 1,
                "KiloPascal": 0.001,
                'MegaPascal': 1,
                "Bar": 0.0254,
                "MilliBar": 0.3048,
                "KiloBar": 1,
                "MegaBar": 1,
                "mmHG": 1,
                "atm":1,
                "psi":1},
    "Bar": {'Pascal': 1,
                "KiloPascal": 0.001,
                'MegaPascal': 1e-6,
                "Bar": 1,
                "MilliBar": 0.3048,
                "KiloBar": 1,
                "MegaBar": 1,
                "mmHG": 1,
                "atm":1,
                "psi":1},
    "MilliBar": {'Pascal': 1,
                "KiloPascal": 0.277,
                'MegaPascal': 0.44704,
                "Bar": 0.0254,
                "MilliBar": 1,
               "KiloBar": 1,
               "MegaBar": 1,
               "mmHG": 1},
    "KiloBar": {'Pascal': 1,
                "KiloPascal": 0.001,
                'MegaPascal': 1e-6,
                "Bar": 0.0254,
                "MilliBar": 0.3048,
                "KiloBar": 1,
                "MegaBar": 1,
                "mmHG": 1,
                "atm":1,
                "psi":1},
    "MegaBar": {'Pascal': 1,
                "KiloPascal": 0.001,
                'MegaPascal': 1e-6,
                "Bar": 0.0254,
                "MilliBar": 0.3048,
                "KiloBar": 1,
                "MegaBar": 1,
                "mmHG": 1,
                "atm":1,
                "psi":1},
    "mmHG": {'Pascal': 1,
                "KiloPascal": 0.001,
                'MegaPascal': 1e-6,
                "Bar": 0.0254,
                "MilliBar": 0.3048,
                "KiloBar": 1,
                "MegaBar": 1,
                "mmHG": 1,
                "atm":1,
                "psi":1},
    "atm": {'Pascal': 1,
             "KiloPascal": 0.001,
             'MegaPascal': 1e-6,
             "Bar": 0.0254,
             "MilliBar": 0.3048,
             "KiloBar": 1,
             "MegaBar": 1,
             "mmHG": 1,
             "atm": 1,
             "psi": 1},
    "psi": {'Pascal': 6894.76,
             "KiloPascal": 6.89476,
             'MegaPascal': 6.89476e-3,
             "Bar": 0.0689476,
             "MilliBar": 0.3048,
             "KiloBar": 1,
             "MegaBar": 1,
             "mmHG": 51.7149,
             "atm": 0.068046,
             "psi": 1},
}


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


class Temp(object):
    """
    converts temperature among units [kelvin, centigrade, fahrenheit]
    :param `temp`  a numpy array
    :param `input_unit` str, units of temp, should be "kelvin", "centigrade" or "fahrenheit"

    Example:
    ```python
    temp = np.arange(10)
    T = Temp(temp, 'centigrade')
    T.kelvin
    >> array([273 274 275 276 277 278 279 280 281 282])
    T.fahrenheit
    >> array([32. , 33.8, 35.6, 37.4, 39.2, 41. , 42.8, 44.6, 46.4, 48.2])
    T.celsius
    >>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ```
    """

    def __init__(self, temp, input_unit):
        self.temp = temp
        self.input_unit = input_unit

    @property
    def celsius(self):
        if self.input_unit == 'centigrade':
            return self.temp
        elif self.input_unit == 'fahrenheit':
            return np.multiply(self.temp - 32, 5 / 9.0)
        elif self.input_unit == 'kelvin':
            return self.temp - 273

    @property
    def fahrenheit(self):
        if self.input_unit == 'centigrade':
            return np.multiply(self.temp, 1.8) + 32
        elif self.input_unit == 'fahrenheit':
            return self.temp
        elif self.input_unit == 'kelvin':
            return np.multiply(self.temp - 273.15, 1.8) + 32

    @property
    def kelvin(self):
        if self.input_unit == 'centigrade':
            return self.temp + 273
        elif self.input_unit == 'fahrenheit':
            return np.multiply(self.temp - 32, 5 / 9.0) + 273.15
        elif self.input_unit == 'kelvin':
            return self.temp


class Wind(object):
    """
    converts wind among units [MeterPerSecond, KilometerPerHour, MilesPerHour, InchesPerSecond, FeetPerSecond]
    :param `temp`  a numpy array
    :param `input_units` str, units of temp, should be "MeterPerSecond", "KilometerPerHour", "MilesPerHour",
           "InchesPerSecond",  "FeetPerSecond"

    Example:
    ```python
    wind = np.arange(10)
    W = Wind(wind, 'KiloMeterPerHour')
    W.MeterPerSecond
    >> array([0.     0.2777 0.5554 0.8331 1.1108 1.3885 1.6662 1.9439 2.2216 2.4993])
    W.KiloMeterPerHour
    >> array([0 1 2 3 4 5 6 7 8 9])
    W.MilesPerHour
    >>array([0.     0.6213 1.2426 1.8639 2.4852 3.1065 3.7278 4.3491 4.9704 5.5917])
    W.InchesPerSecond
    >>array([ 0.   10.93 21.86 32.79 43.72 54.65 65.58 76.51 87.44 98.37])
    W.FeetPerSecond
    >>array([[0.     0.9113 1.8226 2.7339 3.6452 4.5565 5.4678 6.3791 7.2904 8.2017])
    ```
    """
    def __init__(self, wind, input_units):
        self.wind = wind
        check_converter(WindUnitConverter)
        WindUnits = list(WindUnitConverter.keys())
        if input_units not in WindUnits:
            raise ValueError("unknown units {} for wind. Allowed units are {}".format(input_units, WindUnits))
        self.input_units = input_units

    @property
    def MeterPerSecond(self):
        return self.wind * WindUnitConverter['MeterPerSecond'][self.input_units]

    @property
    def KiloMeterPerHour(self):
        return self.wind * WindUnitConverter['KiloMeterPerHour'][self.input_units]

    @property
    def MilesPerHour(self):
        return self.wind * WindUnitConverter['MilesPerHour'][self.input_units]

    @property
    def InchesPerSecond(self):
        return self.wind * WindUnitConverter['InchesPerSecond'][self.input_units]

    @property
    def FeetPerSecond(self):
        return self.wind * WindUnitConverter['FeetPerSecond'][self.input_units]


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
