"""

.. moduleauthor:: Kyle Niemeyer <kyle.niemeyer@gmail.com>
"""

# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import sys
from os.path import splitext, basename
import numpy as np

try:
    import xml.etree.ElementTree as ET
except ImportError:
    import xml.etree.cElementTree as ET

# Local imports
from .utils import units, spec_key, get_temp_unit
from .exceptions import (KeywordError, UndefinedElementError,
                         MissingElementError, MissingAttributeError,
                         UndefinedKeywordError
                         )
from .simulation import Simulation
from . import validation


def get_experiment_kind(root):
    """Read common properties from root of ReSpecTh XML file.

    :param `Element` root: root of ReSpecTh XML file
    :return: Type of experiment ('ST' or 'RCM')
    :rtype: str
    """
    if root.find('experimentType').text != 'Ignition delay measurement':
        raise KeywordError('experimentType not ignition delay measurement')
    try:
        kind = root.find('apparatus/kind').text
        if kind == 'shock tube':
            return 'ST'
        elif kind == 'rapid compression machine':
            return 'RCM'
        else:
            raise NotImplementedError(kind + ' experiment not supported')
    except:
        raise MissingElementError('apparatus/kind')


def get_common_properties(properties, root):
    """Read common properties from root of ReSpecTh XML file.

    :param dict properties: Dictionary with initial properties
    :param `Element` root: root of ReSpecTh XML file
    :return: Dictionary with common properties added
    :rtype: dict
    """
    for elem in root.iterfind('commonProperties/property'):
        name = elem.attrib['name']
        if name == 'initial composition':
            initial_comp = {}
            composition_units = None
            for child in elem.iter('component'):
                # use InChI for unique species identifier (if possible)
                try:
                    spec_id = child.find('speciesLink').attrib['InChI']
                    spec = spec_key[spec_id]
                except KeyError:
                    spec = child.find('speciesLink').attrib['preferredKey']

                # amount of that species
                #initial_comp.append(spec + ':' + child.find('amount').text)
                initial_comp[spec] = float(child.find('amount').text)

                # check consistency of composition_units
                if not composition_units:
                    composition_units = child.find('amount').attrib['units']
                elif composition_units != child.find('amount').attrib['units']:
                    raise KeywordError('inconsistent initial composition units')

            # Convert initial conditions to mole fraction if other.
            if composition_units != 'mole fraction':
                raise NotImplementedError('Non-molar composition unsupported.')

            properties['composition'] = initial_comp
        elif name == 'temperature':
            # Common initial temperature
            try:
                temp_unit = get_temp_unit[elem.attrib['units']]
            except KeyError:
                print('Temperature units not recognized. Must be one of: ' +
                      str(['{}'.format(k) for k in get_temp_unit.keys()])
                      )
                raise
            properties['temperature'] = (float(elem.find('value').text) *
                                         units(temp_unit)
                                         )
            validation.validate_gt('temperature', properties['temperature'],
                                   0. * units.kelvin
                                   )
        elif name == 'pressure':
            # Common initial pressure
            properties['pressure'] = (float(elem.find('value').text) *
                                      units(elem.attrib['units'].lower())
                                      )
            validation.validate_gt('pressure', properties['pressure'],
                                   0. * units.pascal
                                   )
        elif name == 'pressure rise':
            # Constant pressure rise, given in % of initial pressure
            # per unit of time
            if properties['kind'] == 'RCM':
                raise KeywordError('Pressure rise cannot be defined for RCM.')

            properties['pressure-rise'] = (float(elem.find('value').text) /
                                           units(elem.attrib['units'].lower())
                                           )
            validation.validate_geq('pressure rise',
                                    properties['pressure-rise'],
                                    0. / units.second
                                    )
    return properties


def get_ignition_type(properties, root):
    """Gets ignition type and target.

    :param dict properties: Dictionary with initial properties
    :param `Element` root: root of ReSpecTh XML file
    :return: Dictionary with ignition type/target added
    :rtype: dict
    """
    elem = root.find('ignitionType')

    if elem is None:
        raise MissingElementError('ignitionType')

    try:
        ign_target = elem.attrib['target'].rstrip(';').upper()
    except KeyError:
        raise MissingAttributeError('ignitionType target')
    try:
        ign_type = elem.attrib['type']
    except KeyError:
        raise MissingAttributeError('ignitionType type')

    # ReSpecTh allows multiple ignition targets
    if len(ign_target.split(';')) > 1:
        raise NotImplementedError('Multiple ignition targets not implemented.')

    # Acceptable ignition targets include pressure, temperature, and species
    # concentrations
    if ign_target not in ['P', 'T', 'OH', 'OH*', 'CH*', 'CH']:
        raise UndefinedKeywordError(ign_target)

    if ign_type not in ['max', 'd/dt max',
                        'baseline max intercept from d/dt',
                        'baseline min intercept from d/dt',
                        'concentration', 'relative concentration'
                        ]:
        raise UndefinedKeywordError(ign_type)

    if ign_type in ['baseline max intercept from d/dt',
                    'baseline min intercept from d/dt'
                    ]:
        raise NotImplementedError(ign_type + ' not supported')

    properties['ignition type'] = ign_type
    properties['ignition target'] = ign_target

    amt = None
    amt_units = None
    if ign_type in ['concentration', 'relative concentration']:
        try:
            amt = elem.attrib['amount']
        except KeyError:
            raise MissingAttributeError('ignitionType amount')
        try:
            amt_units = elem.attrib['units']
        except KeyError:
            raise MissingAttributeError('ignitionType units')

        properties['ignition target value'] = amt
        properties['ignition target units'] = amt_units

        raise NotImplementedError('concentration ignition delay type '
                                  'not supported'
                                  )
    else:
        properties['ignition target value'] = None

    return properties


def get_datapoints(properties, root):
    """Parse datapoints with ignition delay from file.

    :param dict properties: Dictionary with experimental properties
    :param `Element` root: root of ReSpecTh XML file
    :return: Dictionary with ignition delay data
    :rtype: dict
    """
    # Shock tube experiment will have one data group, while RCM may have one
    # or two (one for ignition delay, one for volume-history)
    property_id = {}
    for dataGroup in root.findall('dataGroup'):

        # get properties of dataGroup
        num = len(dataGroup.findall('dataPoint'))
        for prop in dataGroup.findall('property'):
            property_id[prop.attrib['id']] = prop.attrib['name']
            if prop.attrib['name'] == 'temperature':
                try:
                    temp_unit = get_temp_unit[prop.attrib['units']]
                except KeyError:
                    print('Temperature units not recognized. Must be one of: ' +
                          str(['{}'.format(k) for k in get_temp_unit.keys()])
                          )
                    raise
                val_unit = units(temp_unit)
            else:
                val_unit = units(prop.attrib['units'].lower())
            vals = np.zeros([num]) * val_unit
            properties[prop.attrib['name']] = vals

        # now get data points
        for idx, dp in enumerate(dataGroup.findall('dataPoint')):
            for val in dp:
                prop = property_id[val.tag]
                properties[prop].magnitude[idx] = float(val.text)

                # Check units for correct dimensionality
                if prop == 'ignition delay':
                    validation.validate_gt('ignition-delay',
                                           properties[prop][idx],
                                           0. * units.second
                                           )
                elif prop == 'temperature':
                    validation.validate_gt('temperature',
                                           properties[prop][idx],
                                           0. * units.kelvin
                                           )
                elif prop == 'pressure':
                    validation.validate_gt('pressure',
                                           properties[prop][idx],
                                           0. * units.pascal
                                           )
                elif prop == 'volume':
                    validation.validate_geq('volume',
                                            properties[prop][idx],
                                            0. * units.meter**3
                                            )
                elif prop == 'time':
                    validation.validate_geq('time',
                                            properties[prop][idx],
                                            0. * units.second
                                            )

    return properties


def read_experiment(filename):
    """Reads experiment data from ReSpecTh XML file.

    :param str filename: XML filename in ReSpecTh format with experimental data
    :return: Dictionary with group of experimental properties
    :rtype: dict
    """

    tree = ET.parse(filename)
    root = tree.getroot()

    properties = {}

    # Save name of original data filename
    properties['id'] = splitext(basename(filename))[0]
    properties['data-file'] = basename(filename)

    # Ensure ignition delay, and get which kind of experiment
    properties['kind'] = get_experiment_kind(root)

    # Get properties shared across the file
    properties = get_common_properties(properties, root)

    # Determine definition of ignition delay
    properties = get_ignition_type(properties, root)

    # Now parse ignition delay datapoints
    properties = get_datapoints(properties, root)

    # Get compression time for RCM, if volume history given
    if 'volume' in properties and 'compression-time' not in properties:
        min_volume_idx = np.argmin(properties['volume'])
        min_volume_time = properties['time'][min_volume_idx]
        properties['compression-time'] = min_volume_time

    # Check for missing required properties or conflicts
    for prop in ['composition', 'temperature', 'pressure', 'ignition-delay']:
        if prop not in properties:
            raise MissingElementError(prop)

    if 'volume' in properties and 'time' not in properties:
        raise KeywordError('Time values needed for volume history')
    if 'volume' in properties and 'pressure-rise' in properties:
        raise KeywordError('Both volume history and pressure rise '
                           'cannot be specified'
                           )

    return properties


def create_simulations(properties):
    """Set up individual simulations for each ignition delay value.

    :param dict properties: Dictionary with group of experimental properties
    :return: List of :class:`Simulation` objects for each simulation
    :rtype: list
    """

    simulations = []
    for idx in range(len(properties['ignition-delay'])):
        sim_properties = {}
        # Common properties
        sim_properties['composition'] = properties['composition']
        sim_properties['data-file'] = properties['data-file']
        sim_properties['id'] = properties['id'] + '_' + str(idx)

        for prop in ['temperature', 'pressure', 'ignition-delay']:
            if hasattr(properties[prop].magnitude, '__len__'):
                sim_properties[prop] = properties[prop][idx]
            else:
                # This is a common property, this a scalar
                sim_properties[prop] = properties[prop]

            # Copy pressure rise or volume history if present
            if 'pressure-rise' in properties:
                sim_properties['pressure-rise'] = properties['pressure-rise']

            if 'volume' in properties:
                sim_properties['volume'] = properties['volume']
                sim_properties['time'] = properties['time']

        simulations.append(Simulation(properties['kind'], sim_properties,
                                      properties['ignition target'],
                                      properties['ignition type'],
                                      properties['ignition target value']
                                      ))
    return simulations
