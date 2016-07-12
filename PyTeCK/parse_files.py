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
    import yaml
except ImportError:
    print('Warning: YAML must be installed to read input file.')

# Local imports
from .utils import units, SPEC_KEY, get_temp_unit
from .exceptions import (KeywordError, UndefinedElementError,
                         MissingElementError, MissingAttributeError,
                         UndefinedKeywordError
                         )
from .simulation import Simulation
from . import validation


def get_experiment_kind(raw_properties):
    """Read experiment properties from file.

    Parameters
    ----------
    raw_properties : dict
        Raw properties dict read from file

    Returns
    -------
    kind : str
        Type of experiment ('ST' or 'RCM')

    """
    if not raw_properties.get('experiment-type', None):
        raise MissingElementError('experiment-type')

    if raw_properties['experiment-type'] != 'Ignition delay':
        raise NotImplementedError('experimentType not '
                                  'ignition delay'
                                  )

    apparatus = raw_properties.get('apparatus', None)
    if not apparatus:
        raise MissingElementError('apparatus')

    kind = apparatus.get('kind', None)
    if not kind:
        raise MissingElementError('apparatus/kind')
    elif kind == 'shock tube':
        return 'ST'
    elif kind == 'rapid compression machine':
        return 'RCM'
    else:
        raise NotImplementedError(kind + ' experiment not supported')


def get_ignition_type(ignition_type):
    """Gets (and validates) ignition type and target.

    Parameters
    ----------
    ignition_type : dict
        Dictionary with ignition type information

    Returns
    -------
    ignition : dict
        Dictionary with ignition type/target added

    """
    ignition = {}
    # only supports one ignition target
    if isinstance(ignition_type, list):
        raise NotImplementedError('Multiple ignition targets '
                                  'not implemented.'
                                  )

    ign_target = ignition_type.get('target', None)
    if not ign_target:
        raise MissingAttributeError('target', 'ignition-type')
    ign_type = ignition_type.get('type', None)
    if not ign_type:
        raise MissingAttributeError('type', 'ignition-type')

    # Acceptable ignition targets include pressure, temperature, and species
    # concentrations
    if ign_target not in ['pressure', 'temperature',
                          'OH', 'OH*', 'CH*', 'CH'
                          ]:
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
            ignition['target-value'] = float(ignition_type['amount'])
        except KeyError:
            raise MissingAttributeError('amount', 'ignition-type')
        try:
            ignition['target-units'] = ignition_type['units']
        except KeyError:
            raise MissingAttributeError('units', 'ignition-type')

        # Check value of target
        validation.validate_gt('target-value', ignition['target-value'], 0.0)

        ### THIS ISN'T SUPPORTED NOW
        raise NotImplementedError('concentration ignition delay type '
                                  'not supported'
                                  )

    return ignition


def get_datapoints(properties, raw_properties):
    """Parse datapoints with ignition delay from file.

    Parameters
    ----------
    properties : dict
        Dictionary with experimental properties
    raw_properties : dict
        Dictionary with raw properties read from file

    Returns
    -------
    properties : dict
        Dictionary updated with ignition delay datapoints

    """
    # Each datapoint should have all necessary information associated with it,
    # regardless of experiment type
    if not raw_properties.get('datapoints', None):
        raise MissingElementError('datapoints')

    properties['cases'] = []

    # get properties of datapoints
    for datapoint in raw_properties['datapoints']:
        case = {}

        # Read mandatory initial conditions and properties
        for prop in ['ignition-delay', 'temperature', 'pressure']:
            if not datapoint.get(prop, None):
                raise MissingElementError(prop)
            else:
                temp_units = datapoint[prop].get('units', None)
                if not temp_units:
                    raise MissingAttributeError('units', prop)
                temp_value = datapoint[prop].get('value', None)
                if not temp_value:
                    raise MissingAttributeError('value', prop)

            case[prop] = float(temp_value) * units(temp_units)

        # Check for proper units and values
        validation.validate_gt('ignition delay', case['ignition-delay'],
                               0. * units.second
                               )
        validation.validate_gt('temperature', case['temperature'],
                               0. * units.kelvin
                               )
        validation.validate_gt('pressure', case['pressure'],
                               0. * units.pascal
                               )

        # Initial composition of species
        if not datapoint.get('composition', None):
            raise MissingElementError('composition')
        else:
            initial_comp = {}
            for component in datapoint['composition']:
                # Try to identify specied based on InChI id, otherwise
                # fall back on given name
                spec_id = component.get('InChI', None)
                if not spec_id:
                    spec_name = component.get('species', None)
                    if not spec_name:
                        raise MissingAttributeError('species',
                                                    'composition'
                                                    )
                else:
                    spec_name = SPEC_KEY[spec_id]

                spec_amount = component.get('mole-fraction', None)
                if not spec_amount:
                    raise MissingAttributeError('mole-fraction',
                                                'composition'
                                                )
                initial_comp[spec_name] = spec_amount

            case['composition'] = initial_comp

        # Check ignition type
        if not datapoint.get('ignition-type', None):
            raise MissingElementError('ignition-type')
        else:
            case['ignition'] = get_ignition_type(datapoint['ignition-type'])

        # optional properties

        if datapoint.get('pressure-rise', None):
            temp_units = datapoint['pressure-rise'].get('units', None)
            if not temp_units:
                raise MissingAttributeError('units', 'pressure-rise')
            temp_value = datapoint['pressure-rise'].get('value', None)
            if not temp_value:
                raise MissingAttributeError('value', 'pressure-rise')

            case['pressure-rise'] = float(temp_value) * units(temp_units)
            validation.validate_geq('pressure rise', case['pressure-rise'],
                                    0. / units.second
                                    )

        if datapoint.get('compression-time', None):
            temp_units = datapoint['compression-time'].get('units', None)
            if not temp_units:
                raise MissingAttributeError('units', 'compression-time')
            temp_value = datapoint['compression-time'].get('value', None)
            if not temp_value:
                raise MissingAttributeError('value', 'compression-time')

            case['compression-time'] = float(temp_value) * units(temp_units)
            validation.validate_geq('compression time',
                                    case['compression-time'],
                                    0. * units.second
                                    )

        if datapoint.get('volume-history', None):
            # get information about time and volume
            time_info = datapoint['volume-history'].get('time', None)
            if not time_info:
                raise MissingAttributeError('time', 'volume-history')
            time_units = time_info.get('units', None)
            if not time_units:
                raise MissingAttributeError('units', 'time')
            time_col = time_info.get('column', 0)

            volume_info = datapoint['volume-history'].get('volume', None)
            if not volume_info:
                raise MissingAttributeError('volume', 'volume-history')
            volume_units = volume_info.get('units', None)
            if not volume_units:
                raise MissingAttributeError('units', 'volume')
            volume_col = volume_info.get('column', 1)

            if time_col == volume_col:
                raise KeywordError('time and volume columns are the same in '
                                   'volume-history property.'
                                   )
            if not datapoint['volume-history'].get('values', None):
                raise MissingAttributeError('values', 'volume-history')
            else:
                # convert to NumPy array, ensure appropriate type
                values = np.array(datapoint['volume-history']['values'],
                                  dtype='float64'
                                  )

            case['time'] = values[:, time_col] * units(time_units)
            case['volume'] = values[:, volume_col] * units(volume_units)

            # Check units
            for val in case['time']:
                validation.validate_geq('time', val, 0. * units.second)
            for val in case['volume']:
                validation.validate_geq('volume', val, 0. * units.meter**3)

        properties['cases'].append(case)

    return properties


def read_experiment(filename):
    """Reads experiment data from YAML file.

    Parameters
    ----------
    filename : str
        Name of YAML file in specified format with experimental data

    Returns
    -------
    properties : dict
        Dictionary with group of experimental properties

    """

    with open(filename, 'r') as f:
        raw_properties = yaml.load(f)

    properties = {}

    # Save name of original data filename
    properties['id'] = splitext(basename(filename))[0]
    properties['data-file'] = basename(filename)

    # Ensure ignition delay, and get which kind of experiment
    properties['kind'] = get_experiment_kind(raw_properties)

    # Now parse ignition delay datapoints
    properties = get_datapoints(properties, raw_properties)

    # Get compression time for RCM, if volume history given
    if 'volume' in properties and 'compression-time' not in properties:
        min_volume_idx = np.argmin(properties['volume'])
        min_volume_time = properties['time'][min_volume_idx]
        properties['compression-time'] = min_volume_time

    # Check for missing required properties or conflicts in each case
    for case in properties['cases']:
        for prop in ['composition', 'temperature',
                     'pressure', 'ignition-delay'
                     ]:
            if prop not in case:
                raise MissingElementError(prop)

        if 'volume' in case and 'time' not in case:
            raise KeywordError('Time values needed for volume history')
        if 'volume' in case and 'pressure-rise' in case:
            raise KeywordError('Both volume history and pressure rise '
                               'cannot be specified'
                               )

        # Check that incorrect elements aren't present
        if properties['kind'] == 'ST':
            if 'volume' in case or 'time' in case:
                raise KeywordError('Volume and/or time history not compatible '
                                   'with shock tube experiment.')
            if 'compression-time' in case:
                raise KeywordError('Compression time not compatible '
                                   'with shock tube experiment.')
        elif properties['kind'] == 'RCM':
            if 'pressure-rise' in case:
                raise KeywordError('Pressure rise not compatible '
                                   'with shock tube experiment.')

    return properties


def create_simulations(properties):
    """Set up individual simulations for each ignition delay value.

    Parameters
    ----------
    properties : dict
        Dictionary with group of experimental properties

    Returns
    -------
    simulations : list
        List of :class:`Simulation` objects for each simulation

    """

    simulations = []
    for idx, case in enumerate(properties['cases']):
        sim_properties = {}
        # Common properties
        sim_properties['data-file'] = properties['data-file']
        sim_properties['id'] = properties['id'] + '_' + str(idx)

        for prop in ['temperature', 'pressure', 'ignition-delay',
                     'composition'
                     ]:
            sim_properties[prop] = case[prop]

        # Copy pressure rise or volume history if present
        if 'pressure-rise' in case:
            sim_properties['pressure-rise'] = case['pressure-rise']

        if 'volume' in case:
            sim_properties['volume'] = case['volume']
            sim_properties['time'] = case['time']

        simulations.append(Simulation(properties['kind'], sim_properties,
                                      case['ignition']['target'],
                                      case['ignition']['type'],
                                      case['ignition']['target-value'],
                                      case['ignition']['target-units']
                                      )
                           )
    return simulations
