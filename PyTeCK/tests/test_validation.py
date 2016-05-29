# Python 2 compatibility
from __future__ import print_function
from __future__ import division

from pint import DimensionalityError
import pytest

from ..validation import validate_geq, validate_gt, validate_leq, validate_num
from ..utils import units

class TestValidate_geq:
    """Class of tests for validation function of value lower limit and type.
    """

    def test_ge_wrong_type(self):
        with pytest.raises(TypeError):
            validate_geq('testval', 'five', 0)

    def test_ge_incompatible_units(self):
        with pytest.raises(DimensionalityError):
            validate_geq('testval', 5 * units.second, 0 * units.meter)

    def test_ge_unitless(self):
        with pytest.raises(DimensionalityError):
            validate_geq('testval', 5, 0 * units.meter)

    def test_ge_units_when_unitless(self):
        with pytest.raises(DimensionalityError):
            validate_geq('testval', 5 * units.meter, 0)

    def test_ge_right_type(self):
        value = 5
        low_lim = 0
        assert validate_geq('testval', value, low_lim) == value

    def test_ge_Quantity_type(self):
        value = 5. * units.second
        low_lim = 0. * units.second
        assert validate_geq('testval', value, low_lim) == value

    def test_ge_too_small(self):
        with pytest.raises(RuntimeError):
            validate_geq('testval', -10, 0)

    def test_ge_both_neg(self):
        value = -1.
        low_lim = -10.
        assert validate_geq('testval', value, low_lim) == value


class TestValidate_gt:
    """Class of tests for validation function of value lower limit and type.
    """

    def test_gt_wrong_type(self):
        with pytest.raises(TypeError):
            validate_gt('testval', 'five', 0)

    def test_gt_incompatible_units(self):
        with pytest.raises(DimensionalityError):
            validate_gt('testval', 5 * units.second, 0 * units.meter)

    def test_gt_unitless(self):
        with pytest.raises(DimensionalityError):
            validate_gt('testval', 5, 0 * units.meter)

    def test_gt_units_when_unitless(self):
        with pytest.raises(DimensionalityError):
            validate_gt('testval', 5 * units.meter, 0)

    def test_gt_right_type(self):
        value = 5
        low_lim = 0
        assert validate_gt('testval', value, low_lim) == value

    def test_gt_Quantity_type(self):
        value = 5. * units.second
        low_lim = 0. * units.second
        assert validate_gt('testval', value, low_lim) == value

    def test_gt_too_small(self):
        with pytest.raises(RuntimeError):
            validate_gt('testval', -10, 0)

    def test_gt_both_neg(self):
        value = -1.
        low_lim = -10.
        assert validate_gt('testval', value, low_lim) == value

    def test_gt_int_equal(self):
        with pytest.raises(RuntimeError):
            validate_gt('testval', 0, 0)

    def test_gt_float_equal(self):
        with pytest.raises(RuntimeError):
            validate_gt('testval', 0.0, 0.0)

    def test_gt_Quantity_equal(self):
        with pytest.raises(RuntimeError):
            validate_gt('testval', 0 * units.second, 0 * units.second)

class TestValidate_leq:
    """Class of tests for validation function of value upper limit and type.
    """

    def test_le_wrong_type(self):
        with pytest.raises(TypeError):
            validate_leq('testval', 'ten', 0)

    def test_le_incompatible_units(self):
        with pytest.raises(DimensionalityError):
            validate_leq('testval', 0 * units.second, 10 * units.meter)

    def test_le_unitless(self):
        with pytest.raises(DimensionalityError):
            validate_leq('testval', 0, 5 * units.meter)

    def test_le_units_when_unitless(self):
        with pytest.raises(DimensionalityError):
            validate_leq('testval', 0 * units.meter, 5)

    def test_le_right_type(self):
        value = 0
        upp_lim = 10
        assert validate_leq('testval', value, upp_lim) == value

    def test_le_Quantity_type(self):
        value = -5 * units.second
        upp_lim = 0 * units.second
        assert validate_leq('testval', value, upp_lim) == value

    def test_le_too_large(self):
        with pytest.raises(RuntimeError):
            validate_leq('testval', 10, 0)

    def test_le_both_neg(self):
        value = -10.
        upp_lim = -1.
        assert validate_leq('testval', value, upp_lim) == value

class TestValidate_num:
    """Class of tests for validation function for value being a number.
    """

    def test_num_wrong_type(self):
        with pytest.raises(TypeError):
            validate_num('testval', 'ten')

    def test_num_int(self):
        val = 10
        assert validate_num('testval', val) == val

    def test_num_float(self):
        val = 10.
        assert validate_num('testval', val) == val

    def test_num_Quantity(self):
        val = 10 * units.second
        assert validate_num('testval', val) == val

    def test_num_incompatible_units(self):
        with pytest.raises(DimensionalityError):
            validate_leq('testval', 0 * units.second, 10 * units.meter)
