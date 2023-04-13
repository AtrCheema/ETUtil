
import unittest

import numpy as np

from ETUtil.converter import Temp


class TestConverter(unittest.TestCase):

    def test_temp(self):
        temp = np.arange(10)
        T = Temp(temp, 'Centigrade')
        true = np.array([32., 33.8, 35.6, 37.4, 39.2, 41., 42.8, 44.6, 46.4, 48.2])
        np.testing.assert_array_almost_equal(T.Fahrenheit, true, 2)
        true = np.array([273.15, 274.15, 275.15, 276.15, 277.15,
                         278.15, 279.15, 280.15, 281.15, 282.15])
        np.testing.assert_array_almost_equal(T.Kelvin, true, 2)

        T = Temp(temp, 'Celcius')  # check celcius equal to centigrade
        np.testing.assert_array_almost_equal(T.Kelvin, true, 2)

        T = Temp(temp, 'Fahrenheit')
        true = np.array([-17.77777, -17.22222, -16.666, -16.111, -15.555,
                         -15., -14.444, -13.888, -13.333, -12.777])
        np.testing.assert_array_almost_equal(T.Centigrade, true, 2)
        true = np.array([255.372, 255.927, 256.483, 257.038, 257.594,
                         258.15, 258.705, 259.2611, 259.816, 260.372])
        np.testing.assert_array_almost_equal(T.Kelvin, true, 2)

        # test all small spelling
        T = Temp(temp, 'fahrenheit')
        np.testing.assert_array_almost_equal(T.Kelvin, true, 2)

        T = Temp(temp, 'Kelvin')
        true = np.array([-273.15, -272.15, -271.15, -270.15, -269.15,
                         -268.15, -267.15, -266.15, -265.15, -264.15])
        np.testing.assert_array_almost_equal(T.Centigrade, true, 2)
        true = np.array([-459.67, -457.87, -456.07, -454.27, -452.47,
                         -450.67, -448.87, -447.07, -445.27, -443.47])
        np.testing.assert_array_almost_equal(T.Fahrenheit, true, 2)


if __name__ == "__main__":
    unittest.main()