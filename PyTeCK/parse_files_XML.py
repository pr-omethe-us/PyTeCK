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
    from lxml import etree
except ImportError:
    try:
        import xml.etree.cElementTree as etree
    except ImportError:
        try:
            import xml.etree.ElementTree as etree
        except ImportError:
          print("Failed to import ElementTree from any known place")
          raise

# Local imports
from .utils import units, spec_key, get_temp_unit
from .exceptions import (KeywordError, UndefinedElementError,
                         MissingElementError, MissingAttributeError,
                         UndefinedKeywordError
                         )
from .simulation import Simulation
from . import validation

def get_file_metadata(root):
    """Read and parse ReSpecTh XML file metadata (file author, version, etc.)

    Parameters
    ----------
    root : ``etree.Element``
        root of ReSpecTh XML file

    Returns
    -------
    properties : dict
        Dictionary with file metadata

    """
    properties = {}

    properties['file-author'] = {'name': '', 'ORCID': ''}
    try:
        properties['file-author']['name'] = root.find('fileAuthor').text
    except AttributeError:
        print('Warning: no fileAuthor given')

    # Default version is 1.0
    properties['file-version'] = '(1, 0)'
    elem = root.find('fileVersion')
    if elem is None:
        print('Warning: no fileVersion element')

    try:
        version = (int(elem.find('major').text),
                   int(elem.find('minor').text)
                   )
    except AttributeError:
        print('Warning: missing fileVersion major/minor')
    properties['file-version'] = str(version)

    properties['reference'] = {}
    elem = root.find('bibliographyLink')
    try:
        properties['reference']['citation'] = elem.attrib['preferredKey']
    except KeyError:
        print('Warning: missing preferredKey attribute in bibliographyLink')

    try:
        properties['reference']['doi'] = elem.attrib['doi']
    except KeyError:
        print('Warning: missing doi attribute in bibliographyLink')

    return properties


def get_experiment_kind(root):
    """Read common properties from root of ReSpecTh XML file.

    Parameters
    ----------
    root : ``etree.Element``
        root of ReSpecTh XML file

    Returns
    -------
    kind : str
        Type of experiment ('ST' or 'RCM')

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

    Parameters
    ----------
    properties : dict
        Dictionary with initial properties
    root : ``etree.Element``
        root of ReSpecTh XML file

    Returns
    -------
    properties : dict
        Dictionary with common properties added

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


def get_ignition_type(root):
    """Gets ignition type and target.

    Parameters
    ----------
    root : ``etree.Element``
        root of ReSpecTh XML file

    Returns
    -------
    ignition : dict
        Dictionary with ignition type/target information

    """
    ignition = {}
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

    ignition['type'] = ign_type
    ignition['target'] = ign_target

    ignition['target-value'] = None
    ignition['target-units'] = None
    if ign_type in ['concentration', 'relative concentration']:
        try:
            amt = elem.attrib['amount']
        except KeyError:
            raise MissingAttributeError('ignitionType amount')
        try:
            amt_units = elem.attrib['units']
        except KeyError:
            raise MissingAttributeError('ignitionType units')

        ignition['target-value'] = amt
        ignition['target-units'] = amt_units

        raise NotImplementedError('concentration ignition delay type '
                                  'not supported'
                                  )

    return ignition


def get_datapoints(properties, root):
    """Parse datapoints with ignition delay from file.

    Parameters
    ----------
    properties : dict
        Dictionary with experimental properties
    root : ``etree.Element``
        root of ReSpecTh XML file

    Returns
    -------
    properties : dict
        Dictionary with ignition delay data

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
            if prop.attrib['name'] == 'ignition delay':
                properties['ignition-delay'] = vals
            else:
                properties[prop.attrib['name']] = vals

        # now get data points
        for idx, dp in enumerate(dataGroup.findall('dataPoint')):
            for val in dp:
                prop = property_id[val.tag]
                if prop == 'ignition delay':
                    prop = 'ignition-delay'

                properties[prop].magnitude[idx] = float(val.text)

                # Check units for correct dimensionality
                if prop == 'ignition-delay':
                    validation.validate_gt(prop,
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

    Parameters
    ----------
    filename : str
        XML filename in ReSpecTh format with experimental data

    Returns
    -------
    properties : dict
        Dictionary with group of experimental properties

    """

    try:
        tree = etree.parse(filename)
    except OSError:
        raise OSError('Unable to open file ' + filename)
    root = tree.getroot()

    properties = {}

    # Save name of original data filename
    properties['id'] = splitext(basename(filename))[0]
    properties['data-file'] = basename(filename)

    # get file metadata
    properties.update(get_file_metadata(root))

    # Ensure ignition delay, and get which kind of experiment
    properties['kind'] = get_experiment_kind(root)

    # Get properties shared across the file
    properties = get_common_properties(properties, root)

    # Determine definition of ignition delay
    properties['ignition'] = get_ignition_type(root)

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


def convert_XML_to_YAML(filename_xml):
    """Convert ReSpecTh XML file to ChemKED YAML file.

    Parameters
    ----------
    filename_xml : str
        Name of ReSpecTh XML file to be converted.

    Returns
    -------
    filename_yaml : str
        Name of newly created ChemKED YAML file.

    """

    assert os.path.isfile(filename_xml), "XML file missing"

    # get all information from XML file
    properties = read_experiment(filename_xml)

    


    with open(filename_yaml, 'w') as outfile:
        outfile.write(yaml.dump(properties, default_flow_style=False))
