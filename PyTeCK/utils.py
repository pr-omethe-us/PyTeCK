"""

.. moduleauthor:: Kyle Niemeyer <kyle.niemeyer@gmail.com>
"""

import cantera as ct

def to_second(time, units):
    """Convert time to units of seconds.

    :param float time: Initial time in `units`
    :param str units: Units of `time`
    :return: Converted time in sec
    :rtype: float
    """
    units = units.lower()
    if units == 's':
        return time
    elif units == 'ms':
        return time / 1.0e3
    elif units == 'us':
        return time / 1.0e6
    elif units == 'ns':
        return time / 1.0e9
    elif units == 'min':
        return time * 60.
    else:
        raise KeyError('Time units not recognized: ' + units)

def to_pascal(pres, units):
    """Convert pressure to units of pascal (Cantera pressure unit).

    :param float pres: Initial pressure in `units`
    :param str units: Units of `pres`
    :return: Converted pressure in Pa
    :rtype: float
    """
    units = units.lower()
    if units == 'pa':
        return pres
    elif units == 'kpa':
        return pres * 1.0e3
    elif units == 'mpa':
        return pres * 1.0e6
    elif units == 'atm':
        return pres * ct.one_atm
    elif units == 'torr':
        return pres * 133.3224
    elif units == 'bar':
        return pres * 1.e5
    elif units == 'psi':
        return pres * 6894.757293168
    else:
        raise KeyError('Pressure units not recognized: ' + units)

def to_atm(pres, units):
    """Convert pressure to units of atm.

    :param float pres: Initial pressure in `units`
    :param str units: Units of `pres`
    :return: Converted pressure in atm
    :rtype: float
    """
    units = units.lower()
    if units == 'atm':
        return pres
    elif units == 'pa':
        return pres / ct.one_atm
    elif units == 'kpa':
        return pres * 1000. / ct.one_atm
    elif units == 'mpa':
        return pres * 1.e6 / ct.one_atm
    elif units == 'torr':
        return pres / 760.
    elif units == 'bar':
        return pres * 1.e5 / ct.one_atm
    elif units == 'psi':
        return pres * (6894.757293168 / ct.one_atm)
    else:
        raise KeyError('Pressure units not recognized: ' + units)


def to_kelvin(temp, units):
    """Convert temperature to units of Kelvin.

    :param float temp: Initial temperature in `units`
    :param str units: Units of `temp`
    :return: Converted temperature in Kelvin
    :rtype: float
    """
    if units == 'K':
        temp = temp
    elif units == 'C':
        temp = (temp + 273.15)
    elif units == 'F':
        temp = ((temp + 459.67) * (5.0 / 9.0))
    else:
        raise KeyError('Temperature units not recognized: ' + units)

    if temp < 0:
        raise ValueError('Temperature in Kelvin < zero: ' + str(temp))
    else:
        return temp
