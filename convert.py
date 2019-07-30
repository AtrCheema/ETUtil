import numpy as np

"""
Unit conversion functions.

:copyright: (c) 2015 by Mark Richards.
:license: BSD 3-Clause, see LICENSE.txt for more details.
"""

import math


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
    W = Wind(wind, 'KilometerPerHour')
    W.MeterPerSecond
    >> array([0.     0.2777 0.5554 0.8331 1.1108 1.3885 1.6662 1.9439 2.2216 2.4993])
    W.KilometerPerHour
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
        self.input_units = input_units

    @property
    def MeterPerSecond(self):
        if self.input_units == 'MeterPerSecond':
            return self.wind
        if self.input_units == 'KilometerPerHour':
            return np.multiply(self.wind, 0.2777)
        if self.input_units == 'MilesPerHour':
            return np.multiply(self.wind, 0.44704)
        if self.input_units == 'InchesPerSecond':
            return np.multiply(self.wind, 0.0254)
        if self.input_units == 'FeetPerSecond':
            return np.multiply(self.wind, 0.3048)

    @property
    def KilometerPerHour(self):
        if self.input_units == 'MeterPerSecond':
            return np.multiply(self.wind, 3.6)
        if self.input_units == 'KilometerPerHour':
            return self.wind
        if self.input_units == 'MilesPerHour':
            return np.multiply(self.wind, 1.60934)
        if self.input_units == 'InchesPerSecond':
            return np.multiply(self.wind, 0.09144)
        if self.input_units == 'FeetPerSecond':
            return np.multiply(self.wind, 1.09728)

    @property
    def MilesPerHour(self):
        if self.input_units == 'MeterPerSecond':
            return np.multiply(self.wind, 2.236)
        if self.input_units == 'KilometerPerHour':
            return np.multiply(self.wind, 0.6213)
        if self.input_units == 'MilesPerHour':
            return self.wind
        if self.input_units == 'InchesPerSecond':
            return np.multiply(self.wind, 0.0568)
        if self.input_units == 'FeetPerSecond':
            return np.multiply(self.wind, 0.6818)

    @property
    def InchesPerSecond(self):
        if self.input_units == 'MeterPerSecond':
            return np.multiply(self.wind, 39.37)
        if self.input_units == 'KilometerPerHour':
            return np.multiply(self.wind, 10.93)
        if self.input_units == 'MilesPerHour':
            return np.multiply(self.wind, 17.6)
        if self.input_units == 'InchesPerSecond':
            return self.wind
        if self.input_units == 'FeetPerSecond':
            return np.multiply(self.wind, 12.0)

    @property
    def FeetPerSecond(self):
        if self.input_units == 'MeterPerSecond':
            return np.multiply(self.wind, 3.28)
        if self.input_units == 'KilometerPerHour':
            return np.multiply(self.wind, 0.9113)
        if self.input_units == 'MilesPerHour':
            return np.multiply(self.wind, 1.4667)
        if self.input_units == 'InchesPerSecond':
            return np.multiply(self.wind, 0.0833)
        if self.input_units == 'FeetPerSecond':
            return self.wind


class SolarRad(object):
    """
    https://www.nrel.gov/grid/solar-resource/assets/data/3305.pdf  *ref1
    https://www.wcc.nrcs.usda.gov/ftpref/wntsc/H&H/GEM/SolarRadConversion.pdf  *ref2
    https://www.physicsforums.com/threads/convert-solar-radiation-from-joule-per-cm-square-to-watt-per-meter-2.970019/  *ref3
    https://physics.stackexchange.com/questions/189022/converting-langley-to-watts  *ref4

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
    1 Cal = 4.1868 Joule                                                                 ii
    1 Watt = 1 Joule/Sec                                                                 iii
    so
    1 Langley/hour = 41868/3600.0 = 11.63 W/m^2
    1 Langley/min = 41868/60.0 = 697.3 W/m^2

    1 W/m^2 = 60/41868                      =  0.001433 Ley/min          from ref1                           (1a)
    1 W/m^2 = 3600/41868                    =  0.0858   Ley/hour         from ref1                           (1b)
    1 W/m^2 = 86400/41868                   =  2.0636   Ley/day                                              (1c)
    1 W/m^2 = 60/1000.0                     = 0.06      kJ/m^2  minute   from ref1                           (1d)
    1 W/m^2 = 3600/1000.0                   = 3.6       kJ/m^2  hour     from ref1                           (1e)
    1 W/m^2 = ?                             = 0.024     kWh/m^2          (hour)                              (1f)
    1 W/m^2 = (60)/(10000.0* 4.1868)        = 0.001433  cal/cm2 min      (min)                               (1g)
    1 W/m^2 = (3600)/(10000.0* 4.1868)      = 0.0858    cal/cm2 hour     (hour)                              (1h)

    1 W/m^2 = (60)/(10000.0)                = 0.006     J/cm2 min       from (1g) and ii                     (1i)
    1 W/m^2 = (3600)/(10000.0)              = 0.36      J/cm2 hour      from (1h) and ii                     (1j)
    1 W/m^2 = (60)/(1.0)                    = 60        J/m2 min        from (1i) or (1d)                    (1k)
    1 W/m^2 = (3600)/(1.0)                  = 3600      J/m2 hour       from (1j) or (1e)                    (1L)
    1 W/m^2 = (86400)/(1.0)                 = 86400     J/m2 day        from (1j)                            (1M)

    1 Ley/min  = 41868/60.0                        = 697.3     W/m2       from ref1                          (2a)
    1 Ley/min  = (60)/(60.0)                       = 1.0       Ley/min                                      (2b)
    1 Ley/min  = (3600)/(60.0)                     = 60.0      Ley/hour                                      (2c)
    1 Ley/min  = (86400)/(60.0)                    = 1440.0    Ley/day                                      (2d)
    1 Ley/min  = (41868*60.0)/(60.0*1000.0)        = 41.868    kJ/m^2  minute                                      (2d)
    1 Ley/min  = (41868*3600.0)/(60.0*1000.0)      = 2512.08   kJ/m^2  hour                                      (2d)
    1 Ley/min  = (41868*60.0)/(60.0*10000.0)       = 4.1868    J/cm2 min                                      (2d)
    1 Ley/min  = (41868*3600.0)/(60.0*10000.0)     = 251.208   J/cm2 hour                                      (2d)
    1 Ley/min  = (41868*60.0)/(60.0)               = 41868.0   J/m2 hour                                      (2d)
    1 Ley/min  = (41868*3600.0)/(60.0)             = 2512080.0 J/m2 hour                                      (2d)

    1 Ley/hour = 41868/3600                        = 11.63   W/m2       from ref1                          (3)

    1 Ley/day =  41868/86400                       = 0.4845  W/m2                                          (4)  val by ref4

    1 J/cm2 min  = 10000.0/60.0                    = 166.66   W/m2      from (1i)                          (5)

    1 J/cm2 hour = 10000.0/3600.0                  =  2.7     W/m2      from (1j)                          (6a)  val by ref3
    1 J/cm2 hour = (10000.0*60)/(3600*41868)       = 0.00398  Ley/min   from (6a) and (1a)                 (6b)
    1 J/cm2 hour = (10000.0*3600)/(3600*41868)     = 0.2388   Ley/hour  from (6a) and (1b)                 (6c)
    1 J/cm2 hour = (10000.0*86400)/(3600*41868)    = 5.7323   Ley/day   from (6a) and (1c)                 (6c)

    1 J/cm2 day = 10000.0/86400.0                  =  0.11574     W/m2   from (6a)                          (7a)

    1 J/m2 min  = 1/60.0                           = 0.0166       W/m2       from (1k)                      (8a)

    1 J/m2 hour = 1/3600.0                         = 0.000277     W/m2      from (1L)                      (9a)


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
