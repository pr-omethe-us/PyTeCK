"""Validation functions for values.

Based on validation module of pyrk (https://github.com/pyrk).

.. moduleauthor:: Kyle Niemeyer <kyle.niemeyer@gmail.com>
"""

# Python 2 compatibility
from __future__ import print_function
from __future__ import division
import sys
if sys.version_info > (3,):
    long = int

import pint

# Local imports
from .utils import units


def validate_geq(value_name, value, low_lim):
    """Raise error if value lower than specified lower limit or wrong type.

    Parameters
    ---------
    value_name : str
        Name of value being tested
    value : int, float, numpy.ndarray, pint.Quantity
        Value to be tested
    low_lim : type(value)
        Lowest acceptable limit of ``value``

    Returns
    -------
    value : type(value)
        The original value

    """

    try:
        if validate_num(value_name, value) < low_lim:
            msg = (value_name + ' must be greater than or equal to ' +
                   str(low_lim) + '.\n'
                   'Value provided was: ' + str(value)
                   )
            # RuntimeError used to avoid being caught by Pint comparison error.
            # Pint should really raise TypeError (or something) rather than
            # ValueError.
            raise RuntimeError(msg)
        else:
            return value
    except ValueError:
        if isinstance(value, units.Quantity):
            msg = ('\n' + value_name + ' given with units, when variable '
                   'should be dimensionless.'
                   )
            raise pint.DimensionalityError(value.units, None,
                                           extra_msg=msg
                                           )
        else:
            msg = ('\n' + value_name + ' not given in units. '
                   'Correct units share dimensionality with: ' +
                   str(low_lim.units)
                   )
            raise pint.DimensionalityError(None, low_lim.units,
                                           extra_msg=msg
                                           )
    except pint.DimensionalityError:
        msg = ('\n' + value_name + ' given in incompatible units. '
               'Correct units share dimensionality with: ' +
               str(low_lim.units)
               )
        raise pint.DimensionalityError(value.units, low_lim.units,
                                       extra_msg=msg
                                       )
    except:
        raise


def validate_gt(value_name, value, low_lim):
    """Raise error if value not greater than lower limit or wrong type.

    Parameters
    ---------
    value_name : str
        Name of value being tested
    value : int, float, numpy.ndarray, pint.Quantity
        Value to be tested
    low_lim : type(value)
        ``value`` must be greater than this limit

    Returns
    -------
    value : type(value)
        The original value

    """

    try:
        if not validate_num(value_name, value) > low_lim:
            msg = (value_name + ' must be greater than ' +
                   str(low_lim) + '.\n'
                   'Value provided was: ' + str(value)
                   )
            # RuntimeError used to avoid being caught by Pint comparison error.
            # Pint should really raise TypeError (or something) rather than
            # ValueError.
            raise RuntimeError(msg)
        else:
            return value
    except ValueError:
        if isinstance(value, units.Quantity):
            msg = ('\n' + value_name + ' given with units, when variable '
                   'should be dimensionless.'
                   )
            raise pint.DimensionalityError(value.units, None,
                                           extra_msg=msg
                                           )
        else:
            msg = ('\n' + value_name + ' not given in units. '
                   'Correct units share dimensionality with: ' +
                   str(low_lim.units)
                   )
            raise pint.DimensionalityError(None, low_lim.units,
                                           extra_msg=msg
                                           )
    except pint.DimensionalityError:
        msg = ('\n' + value_name + ' given in incompatible units. '
               'Correct units share dimensionality with: ' +
               str(low_lim.units)
               )
        raise pint.DimensionalityError(value.units, low_lim.units,
                                       extra_msg=msg
                                       )
    except:
        raise


def validate_leq(value_name, value, upp_lim):
    """Raise error if value greater than specified upper limit or wrong type.

    Parameters
    ---------
    value_name : str
        Name of value being tested
    value : int, float, numpy.ndarray, pint.Quantity
        Value to be tested
    upp_lim : type(value)
        Highest acceptable limit of ``value``

    Returns
    -------
    value : type(value)
        The original value

    """

    try:
        if validate_num(value_name, value) > upp_lim:
            msg = (value_name + ' must be less than or equal to ' +
                   str(upp_lim) + '.\n'
                   'Value provided was: ' + str(value)
                   )
            # RuntimeError used to avoid being caught by Pint comparison error.
            # Pint should really raise TypeError (or something) rather than
            # ValueError.
            raise RuntimeError(msg)
        else:
            return value
    except ValueError:
        if isinstance(value, units.Quantity):
            msg = ('\n' + value_name + ' given with units, when variable '
                   'should be dimensionless.'
                   )
            raise pint.DimensionalityError(value.units, None,
                                           extra_msg=msg
                                           )
        else:
            msg = ('\n' + value_name + ' not given in units. '
                   'Correct units share dimensionality with: ' +
                   str(upp_lim.units)
                   )
            raise pint.DimensionalityError(None, upp_lim.units,
                                           extra_msg=msg
                                           )
    except pint.DimensionalityError:
        msg = ('\n' + value_name + ' given in incompatible units. '
               'Correct units share dimensionality with: ' +
               str(upp_lim.units)
               )
        raise pint.DimensionalityError(value.units, upp_lim.units,
                                       extra_msg=msg
                                       )
    except:
        raise


def validate_num(value_name, value):
    """Raise error if value is not a number.

    Parameters
    ---------
    value_name : str
        Name of value being tested
    value : int, float, numpy.ndarray, pint.Quantity
        Value to be tested

    Returns
    -------
    value : type(value)
        The original value

    """
    if isinstance(value, (int, long, float, units.Quantity)):
        return value
    else:
        try:
            if isinstance(value.magnitude, (int, long, float, units.Quantity)):
                return value
        except AttributeError:
            pass
    msg = (value_name + ' must be an integer, long, float, or Quantity. \n'
           'The value provided was of type ' + str(type(value)) + ' and '
           'value ' + str(value)
           )
    raise TypeError(msg)
