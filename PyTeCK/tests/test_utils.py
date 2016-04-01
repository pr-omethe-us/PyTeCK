# Python 2 compatibility
from __future__ import print_function
from __future__ import division

import numpy as np
import pytest

from .. import utils

class TestToKelvin:
    """Class of tests for function that converts temperature to Kelvin.
    """
    def test_kelvin_to_kelvin(self):
        temp = utils.to_kelvin(300., 'K')
        np.testing.assert_allclose(temp, 300.)

    def test_celsius_to_kelvin(self):
        temp = utils.to_kelvin(1000., 'C')
        np.testing.assert_allclose(temp, 1273.15)

    def test_farenheit_to_kelvin(self):
        temp = utils.to_kelvin(32., 'F')
        np.testing.assert_allclose(temp, 273.15)

    def test_negative_kelvin(self):
        with pytest.raises(ValueError):
            temp = utils.to_kelvin(-1., 'K')

    def test_units_error(self):
        with pytest.raises(KeyError):
            temp = utils.to_kelvin(300., 'R')


class TestToPascal:
    """Class of tests for function that converts pressure to Pascal.
    """
    def test_pa_to_pa(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_pascal(p, 'pa') for p in pres]
        np.testing.assert_allclose(pres_conv, pres)

    def test_atm_to_pa(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_pascal(p, 'atm') for p in pres]
        np.testing.assert_allclose(pres_conv, pres * 101325.)

    def test_kpa_to_pa(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_pascal(p, 'kpa') for p in pres]
        np.testing.assert_allclose(pres_conv, pres * 1.e3)

    def test_mpa_to_pa(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_pascal(p, 'mpa') for p in pres]
        np.testing.assert_allclose(pres_conv, pres * 1.e6)

    def test_torr_to_pa(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_pascal(p, 'torr') for p in pres]
        np.testing.assert_allclose(pres_conv, pres * 133.3224)

    def test_bar_to_pa(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_pascal(p, 'bar') for p in pres]
        np.testing.assert_allclose(pres_conv, pres * 1.e5)

    def test_psi_to_pa(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_pascal(p, 'psi') for p in pres]
        np.testing.assert_allclose(pres_conv, pres * 6894.757293168)

    def test_units_error(self):
        with pytest.raises(KeyError):
            pres = utils.to_pascal(1., 'mmHg')


class TestToAtm:
    """Class of tests for function that converts pressure to atm.
    """
    def test_atm_to_atm(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_atm(p, 'atm') for p in pres]
        np.testing.assert_allclose(pres_conv, pres)

    def test_pa_to_atm(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_atm(p, 'pa') for p in pres]
        np.testing.assert_allclose(pres_conv, pres / 101325.)

    def test_kpa_to_atm(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_atm(p, 'kpa') for p in pres]
        np.testing.assert_allclose(pres_conv, pres * 1.e3 / 101325.)

    def test_mpa_to_atm(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_atm(p, 'mpa') for p in pres]
        np.testing.assert_allclose(pres_conv, pres * 1.e6 / 101325.)

    def test_torr_to_atm(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_atm(p, 'torr') for p in pres]
        np.testing.assert_allclose(pres_conv, pres / 760.)

    def test_bar_to_atm(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_atm(p, 'bar') for p in pres]
        np.testing.assert_allclose(pres_conv, pres * 1.e5 / 101325.)

    def test_psi_to_atm(self):
        pres = np.random.rand(10)
        pres_conv = [utils.to_atm(p, 'psi') for p in pres]
        np.testing.assert_allclose(pres_conv, pres * 6894.757293168 / 101325.)

    def test_units_error(self):
        with pytest.raises(KeyError):
            pres = utils.to_atm(1., 'mmHg')


class TestToSecond:
    """Class of tests for function that converts time to seconds.
    """
    def test_sec_to_sec(self):
        time = np.random.rand(10)
        time_conv = [utils.to_second(t, 's') for t in time]
        np.testing.assert_allclose(time_conv, time)

    def test_ms_to_sec(self):
        time = np.random.rand(10)
        time_conv = [utils.to_second(t, 'ms') for t in time]
        np.testing.assert_allclose(time_conv, time / 1.0e3)

    def test_us_to_sec(self):
        time = np.random.rand(10)
        time_conv = [utils.to_second(t, 'us') for t in time]
        np.testing.assert_allclose(time_conv, time / 1.0e6)

    def test_ns_to_sec(self):
        time = np.random.rand(10)
        time_conv = [utils.to_second(t, 'ns') for t in time]
        np.testing.assert_allclose(time_conv, time / 1.0e9)

    def test_min_to_sec(self):
        time = np.random.rand(10)
        time_conv = [utils.to_second(t, 'min') for t in time]
        np.testing.assert_allclose(time_conv, time * 60.)

    def test_units_error(self):
        with pytest.raises(KeyError):
            time = utils.to_second(1., 'jiffy')
