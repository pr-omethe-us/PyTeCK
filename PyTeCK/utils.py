"""

.. moduleauthor:: Kyle Niemeyer <kyle.niemeyer@gmail.com>
"""
from __future__ import print_function

import pint

units = pint.UnitRegistry()

units.define('cm3 = centimeter**3')

get_temp_unit = {'K': 'kelvin',
                 'C': 'degC',
                 'F': 'degF',
                 'R': 'degR'
                 }


# Unique InChI identifier for species
SPEC_KEY = {'1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3': 'nC7H16',
            '1S/C8H18/c1-7(2)6-8(3,4)5/h7H,6H2,1-5H3': 'iC8H18',
            '1S/C7H8/c1-7-5-3-2-4-6-7/h2-6H,1H3': 'C6H5CH3',
            '1S/C2H6O/c1-2-3/h3H,2H2,1H3': 'C2H5OH',
            '1S/O2/c1-2': 'O2',
            '1S/N2/c1-2': 'N2',
            '1S/Ar': 'Ar',
            '1S/He': 'He',
            '1S/CO2/c2-1-3': 'CO2',
            '1S/H2/h1H': 'H2',
            '1S/H2O/h1H2': 'H2O',
           }

SPEC_KEY_REV = {'nC7H16': '1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3',
                'iC8H18': '1S/C8H18/c1-7(2)6-8(3,4)5/h7H,6H2,1-5H3',
                'C6H5CH3': '1S/C7H8/c1-7-5-3-2-4-6-7/h2-6H,1H3',
                'C2H5OH': '1S/C2H6O/c1-2-3/h3H,2H2,1H3',
                'O2': '1S/O2/c1-2',
                'N2': '1S/N2/c1-2',
                'Ar': '1S/Ar',
                'He': '1S/He',
                'CO2': '1S/CO2/c2-1-3',
                'H2': '1S/H2/h1H',
                'H2O': '1S/H2O/h1H2',
               }

SPEC_NAMES = {'nC7H16': 'n-heptane',
              'iC8H18': 'isooctane',
              'C6H5CH3': 'toluene',
              'C2H5OH': 'ethanol',
              'O2': 'oxygen',
              'N2': 'nitrogen',
              'Ar': 'argon',
              'He': 'helium',
              'CO2': 'carbon dioxide',
              'H2': 'hydrogen',
              'H2O': 'water',
             }


def print_species_names():
    """Print species names, internal short name, and InChI identifiers."""

    len_longest = max([len(sp) for sp in SPEC_NAMES.values()])
    header = '{:<{}s} Short name\tInChI key'.format('Species name', len_longest)

    print(header)
    print('-' * len(header.expandtabs()))
    for spec in SPEC_KEY_REV.items():
        print('{:<{}s} {:10}\t{}'.format(SPEC_NAMES[spec[0]], len_longest,
              spec[0], spec[1])
             )
