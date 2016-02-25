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
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# Local imports
from .exceptions import (KeywordError, UndefinedElementError,
                         MissingElementError, MissingAttributeError,
                         UndefinedKeywordError
                         )
from .simulation import Property, Simulation


# Unique InChI identifier for species
spec_key = {'1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3': 'nC7H16',
            '1S/C8H18/c1-7(2)6-8(3,4)5/h7H,6H2,1-5H3': 'iC8H18',
            '1S/C7H8/c1-7-5-3-2-4-6-7/h2-6H,1H3': 'C6H5CH3',
            '1S/C2H6O/c1-2-3/h3H,2H2,1H3': 'C2H5OH',
            '1S/O2/c1-2': 'O2',
            '1S/N2/c1-2': 'N2',
            '1S/Ar': 'Ar',
            '1S/He': 'He',
            '1S/CO2/c2-1-3': 'CO2'
            }


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
            #initial_comp = []
            initial_comp = {}
            units = None
            for child in elem.iter('component'):
                # use InChI for unique species identifier (if possible)
                try:
                    spec_id = child.find('speciesLink').attrib['InChI']
                    spec = spec_key[spec_id]
                except KeyError:
                    spec = child.find('speciesLink').attrib['preferredKey']

                # amount of that species
                #initial_comp.append(spec + ':' + child.find('amount').text)
                initial_comp[spec] = child.find('amount').text

                # check consistency of units
                if not units:
                    units = child.find('amount').attrib['units']
                elif units != child.find('amount').attrib['units']:
                    raise KeywordError('inconsistent initial composition units')

            # Convert initial conditions to mole fraction if other.
            if units != 'mole fraction':
                raise NotImplementedError('Non-molar composition unsupported.')

            properties['composition'] = initial_comp
        elif name == 'temperature':
            # Common initial temperature
            properties['temperature'] = Property(float(elem.find('value').text),
                                                 elem.attrib['units']
                                                 )
        elif name == 'pressure':
            # Common initial pressure
            properties['pressure'] = Property(float(elem.find('value').text),
                                              elem.attrib['units']
                                              )
        elif name == 'pressure rise':
            # Constant pressure rise, given in % of initial pressure
            # per unit of time
            if properties['kind'] == 'RCM':
                raise KeywordError('Pressure rise cannot be defined for RCM.')

            properties['pressure rise'] = Property(float(elem.find('value').text),
                                                   elem.attrib['units']
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

        properties['ignition target value'] = Property(amt, amt_units)

        raise NotImplementedError('concentration ignition delay type '
                                  ' not supported'
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
            properties[prop.attrib['name']] = Property(np.zeros([num]),
                                                       prop.attrib['units']
                                                       )

        # now get data points
        for idx, dp in enumerate(dataGroup.findall('dataPoint')):
            for val in dp:
                prop = property_id[val.tag]
                properties[prop].value[idx] = float(val.text)

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
    properties['data file'] = basename(filename)

    # Ensure ignition delay, and get which kind of experiment
    properties['kind'] = get_experiment_kind(root)

    # Get properties shared across the file
    properties = get_common_properties(properties, root)

    # Determine definition of ignition delay
    properties = get_ignition_type(properties, root)

    # Now parse ignition delay datapoints
    properties = get_datapoints(properties, root)

    # Check for missing required properties or conflicts
    for prop in ['composition', 'temperature', 'pressure', 'ignition delay']:
        if prop not in properties:
            raise MissingElementError(prop)

    if 'volume' in properties and 'time' not in properties:
        raise KeywordError('Time values needed for volume history')
    if 'volume' in properties and 'pressure rise' in properties:
        raise KeywordError('Both volume history and pressure rise '
                           'cannot be specified'
                           )

    # Check units
    if properties['ignition delay'].units not in ['s', 'ms', 'us', 'ns', 'min']:
        raise NotImplementedError('Ignition delay units not recognized: ' +
                                  properties['ignition delay'].units
                                  )
    if properties['temperature'].units not in ['K', 'C', 'F']:
        raise NotImplementedError('Temperature units not recognized: ' +
                                  properties['temperature'].units
                                  )
    if properties['pressure'].units.lower() not in ['atm', 'pa', 'kpa', 'mpa'
                                                    'torr', 'bar', 'psi'
                                                    ]:
        raise NotImplementedError('Pressure units not recognized: ' +
                                  properties['pressure'].units
                                  )
    if ('pressure rise' in properties and
        properties['pressure rise'].units not in ['s', 'ms', 'us', 'ns', 'min']
        ):
        raise NotImplementedError('Pressure rise units not recognized: ' +
                                  properties['pressure rise'].units
                                  )
    if ('time' in properties and
        properties['time'].units not in ['s', 'ms', 'us', 'ns', 'min']
        ):
        raise NotImplementedError('Time units not recognized: ' +
                                  properties['time'].units
                                  )
    return properties


def create_simulations(properties):
    """Set up individual simulations for each ignition delay value.

    :param dict properties: Dictionary with group of experimental properties
    :return: List of :class:`Simulation` objects for each simulation
    :rtype: list
    """

    simulations = []
    for idx in range(len(properties['ignition delay'].value)):
        sim_properties = {}
        # Common properties
        sim_properties['composition'] = properties['composition']
        sim_properties['data file'] = properties['data file']
        sim_properties['id'] = properties['id'] + '_' + str(idx)

        for prop in ['temperature', 'pressure', 'ignition delay']:
            if isinstance(properties[prop].value, np.ndarray):
                sim_properties[prop] = Property(properties[prop].value[idx],
                                                properties[prop].units
                                                )
            else:
                # This is a common property, this a scalar
                sim_properties[prop] = Property(properties[prop].value,
                                                properties[prop].units
                                                )

            # Copy pressure rise or volume history if present
            if 'pressure rise' in properties:
                sim_properties['pressure rise'] = properties['pressure rise']

            if 'volume' in properties:
                sim_properties['volume'] = properties['volume']
                sim_properties['time'] = properties['time']

        simulations.append(Simulation(properties['kind'], sim_properties,
                                      properties['ignition target'],
                                      properties['ignition type'],
                                      properties['ignition target value']
                                      ))
    return simulations
