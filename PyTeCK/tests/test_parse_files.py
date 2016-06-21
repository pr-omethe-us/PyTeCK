# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import os
import pkg_resources

import numpy
import yaml
import pytest

# Local imports
from .. import parse_files
from ..simulation import Simulation
from ..utils import units
from ..exceptions import UndefinedKeywordError

class TestExperimentType:
    """
    """
    def test_shock_tube_experiment(self):
        """Ensure shock tube experiment can be detected.
        """
        file_path = os.path.join('testfile_st.yaml')
        filename = pkg_resources.resource_filename(__name__, file_path)

        with open(filename, 'r') as f:
            raw_properties = yaml.load(f)

        kind = parse_files.get_experiment_kind(raw_properties)
        assert kind == 'ST'

    def test_RCM_experiment(self):
        """Ensure rapid compression machine experiment can be detected.
        """
        file_path = os.path.join('testfile_rcm.yaml')
        filename = pkg_resources.resource_filename(__name__, file_path)

        with open(filename, 'r') as f:
            raw_properties = yaml.load(f)

        kind = parse_files.get_experiment_kind(raw_properties)
        assert kind == 'RCM'


class TestIgnitionType:
    """
    """
    def test_ignition_type_pressure(self):
        """Ensure basic ignition type correctly determined.
        """
        ignition = parse_files.get_ignition_type({'target': 'pressure',
                                                  'type': 'd/dt max'
                                                  })
        assert ignition['target'] == 'pressure'
        assert ignition['type'] == 'd/dt max'

    def test_ignition_type_wrong_target(self):
        """Ensure exception raised for wrong target.
        """
        with pytest.raises(UndefinedKeywordError):
            ignition = parse_files.get_ignition_type({'target': 'volume',
                                                      'type': 'max'
                                                      })

    def test_ignition_type_wrong_type(self):
        """Ensure exception raised for wrong type.
        """
        with pytest.raises(UndefinedKeywordError):
            ignition = parse_files.get_ignition_type({'target': 'pressure',
                                                      'type': 'min'
                                                      })

    def test_pressure_species_target_OH(self):
        """Test species max value as target.
        """
        file_path = os.path.join('testfile_st2.yaml')
        filename = pkg_resources.resource_filename(__name__, file_path)

        with open(filename, 'r') as f:
            properties = yaml.load(f)

        ignition_type = properties['datapoints'][0]['ignition-type']

        ignition = parse_files.get_ignition_type(ignition_type)

        assert ignition['target'] == 'OH'
        assert ignition['type'] == 'max'


class TestDatapoints:
    """
    """
    def test_shock_tube_data_points(self):
        """Test parsing of ignition delay data points for shock tube file.
        """
        file_path = os.path.join('testfile_st.yaml')
        filename = pkg_resources.resource_filename(__name__, file_path)

        with open(filename, 'r') as f:
            raw_properties = yaml.load(f)

        properties = {}
        properties = parse_files.get_datapoints(properties, raw_properties)

        assert len(properties['cases']) == 5

        pressures = numpy.ones(5) * 220. * units.kilopascal
        temperatures = [1164.48, 1164.97, 1264.2,
                              1332.57, 1519.18
                              ] * units.kelvin
        ignition_delays = [471.54, 448.03, 291.57,
                                 205.93, 88.11
                                 ] * units.us

        for idx, case in enumerate(properties['cases']):
            # Ensure correct pressure, temperature, and
            # ignition delay values/units
            assert case['pressure'] == pressures[idx]
            assert case['temperature'] == temperatures[idx]
            assert case['ignition-delay'] == ignition_delays[idx]

            # Check initial composition
            assert case['composition']['H2'] == 0.00444
            assert case['composition']['O2'] == 0.00566
            assert case['composition']['Ar'] == 0.9899

            # Check pressure rise, volume, time
            assert 'pressure-rise' not in case
            assert 'time' not in case
            assert 'volume' not in case

    def test_shock_tube_data_points_pressure_rise(self):
        """Test parsing of ignition delay data points for shock tube file.
        """
        file_path = os.path.join('testfile_st2.yaml')
        filename = pkg_resources.resource_filename(__name__, file_path)

        with open(filename, 'r') as f:
            raw_properties = yaml.load(f)

        properties = {}
        properties = parse_files.get_datapoints(properties, raw_properties)

        case = properties['cases'][0]

        # Ensure correct pressure, temperature, and ignition delay values/units
        assert case['pressure'] == 2.18 * units.atm
        assert case['temperature'] == 1264.2 * units.kelvin
        assert case['ignition-delay'] == 291.57 * units.us

        # Check initial composition
        assert case['composition']['H2'] == 0.00444
        assert case['composition']['O2'] == 0.00566
        assert case['composition']['Ar'] == 0.9899

        # Check pressure rise
        assert case['pressure-rise'] == 0.10 / units.ms

        # Check volume, time
        assert 'time' not in case
        assert 'volume' not in case

    def test_rcm_data_points(self):
        """Test parsing of ignition delay data points for RCM file.
        """
        file_path = os.path.join('testfile_rcm.yaml')
        filename = pkg_resources.resource_filename(__name__, file_path)

        with open(filename, 'r') as f:
            raw_properties = yaml.load(f)

        properties = {}
        properties = parse_files.get_datapoints(properties, raw_properties)

        case = properties['cases'][0]

        # Ensure correct temperature, pressure, and ignition delay values and units
        assert case['temperature'] == 297.4 * units.kelvin
        assert case['pressure'] == 958. * units.torr
        assert case['ignition-delay'] == 1. * units.ms

        assert case['composition']['H2'] == 0.12500
        assert case['composition']['O2'] == 0.06250
        assert case['composition']['N2'] == 0.18125
        assert case['composition']['Ar'] == 0.63125

        # Check other data group with volume history
        numpy.testing.assert_allclose(case['time'],
                                      numpy.arange(0, 9.7e-2, 1.e-3) *
                                      units.second
                                      )

        volumes = numpy.array([
            5.47669375000E+002, 5.46608789894E+002, 5.43427034574E+002,
            5.38124109043E+002, 5.30700013298E+002, 5.21154747340E+002,
            5.09488311170E+002, 4.95700704787E+002, 4.79791928191E+002,
            4.61761981383E+002, 4.41610864362E+002, 4.20399162234E+002,
            3.99187460106E+002, 3.77975757979E+002, 3.56764055851E+002,
            3.35552353723E+002, 3.14340651596E+002, 2.93128949468E+002,
            2.71917247340E+002, 2.50705545213E+002, 2.29493843085E+002,
            2.08282140957E+002, 1.87070438830E+002, 1.65858736702E+002,
            1.44647034574E+002, 1.23435332447E+002, 1.02223630319E+002,
            8.10119281915E+001, 6.33355097518E+001, 5.27296586879E+001,
            4.91943750000E+001, 4.97137623933E+001, 5.02063762048E+001,
            5.06454851923E+001, 5.10218564529E+001, 5.13374097598E+001,
            5.16004693977E+001, 5.18223244382E+001, 5.20148449242E+001,
            5.21889350372E+001, 5.23536351113E+001, 5.25157124459E+001,
            5.26796063730E+001, 5.28476160610E+001, 5.30202402028E+001,
            5.31965961563E+001, 5.33748623839E+001, 5.35527022996E+001,
            5.37276399831E+001, 5.38973687732E+001, 5.40599826225E+001,
            5.42141273988E+001, 5.43590751578E+001, 5.44947289126E+001,
            5.46215686913E+001, 5.47405518236E+001, 5.48529815402E+001,
            5.49603582190E+001, 5.50642270863E+001, 5.51660349836E+001,
            5.52670070646E+001, 5.53680520985E+001, 5.54697025392E+001,
            5.55720927915E+001, 5.56749762728E+001, 5.57777790517E+001,
            5.58796851466E+001, 5.59797461155E+001, 5.60770054561E+001,
            5.61706266985E+001, 5.62600130036E+001, 5.63449057053E+001,
            5.64254496625E+001, 5.65022146282E+001, 5.65761642150E+001,
            5.66485675508E+001, 5.67208534842E+001, 5.67944133373E+001,
            5.68703658198E+001, 5.69493069272E+001, 5.70310785669E+001,
            5.71146023893E+001, 5.71978399741E+001, 5.72779572372E+001,
            5.73517897984E+001, 5.74167271960E+001, 5.74721573687E+001,
            5.75216388520E+001, 5.75759967785E+001, 5.76575701358E+001,
            5.78058719368E+001, 5.80849611077E+001, 5.85928651155E+001,
            5.94734357453E+001, 6.09310671165E+001, 6.32487551103E+001,
            6.68100309742E+001
            ]) * units.cm3
        numpy.testing.assert_allclose(case['volume'], volumes)


class TestCreateSimulations:
    """
    """
    def test_create_st_simulations(self):
        """Ensure appropriate simulations created from shock tube file.
        """
        # Rely on previously tested functions to parse file
        file_path = os.path.join('testfile_st.yaml')
        filename = pkg_resources.resource_filename(__name__, file_path)

        properties = parse_files.read_experiment(filename)

        # Now create list of Simulation objects
        simulations = parse_files.create_simulations(properties)

        comp = {'H2': 0.00444, 'O2': 0.00566, 'Ar': 0.9899}

        # Ensure correct number of simulations
        assert len(simulations) == 5

        pressures = numpy.ones(5) * 220. * units.kilopascal
        temperatures = [1164.48, 1164.97, 1264.2,
                        1332.57, 1519.18
                        ] * units.kelvin
        ignition_delays = [471.54, 448.03, 291.57,
                           205.93, 88.11
                           ] * units.us

        for idx, sim in enumerate(simulations):
            # Ensure correct information
            assert sim.properties['id'] == 'testfile_st_{}'.format(idx)
            assert sim.properties['data-file'] == 'testfile_st.yaml'
            assert sim.kind == 'ST'
            assert sim.properties['temperature'] == temperatures[idx]
            assert sim.properties['pressure'] == pressures[idx]
            assert sim.properties['ignition-delay'] == ignition_delays[idx]
            assert sim.properties['composition'] == comp
            assert sim.ignition_target == 'pressure'
            assert sim.ignition_type == 'd/dt max'
            assert sim.ignition_target_value == None
            assert sim.ignition_target_unit == None

    def test_create_st_simulations_pressure_rise(self):
        """Ensure appropriate simulations created from shock tube file.
        """
        # Rely on previously tested functions to parse file
        file_path = os.path.join('testfile_st2.yaml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        properties = parse_files.read_experiment(filename)

        # Now create list of Simulation objects
        simulations = parse_files.create_simulations(properties)

        # Ensure correct number of simulations
        assert len(simulations) == 1

        sim = simulations[0]
        assert sim.properties['id'] == 'testfile_st2_0'
        assert sim.properties['data-file'] == 'testfile_st2.yaml'
        assert sim.kind == 'ST'
        assert sim.properties['temperature'] == 1264.2 * units.kelvin
        assert sim.properties['pressure'] == 2.18 * units.atm
        assert sim.properties['ignition-delay'] == 291.57 * units.us
        assert sim.properties['pressure-rise'] == 0.10 / units.ms
        comp = {'H2': 0.00444, 'O2': 0.00566, 'Ar': 0.9899}
        assert sim.properties['composition'] == comp
        assert sim.ignition_target == 'OH'
        assert sim.ignition_type == 'max'
        assert sim.ignition_target_value == None
        assert sim.ignition_target_unit == None

    def test_create_rcm_simulations(self):
        """Ensure appropriate simulations created from RCM file.
        """
        # Rely on previously tested functions to parse file
        file_path = os.path.join('testfile_rcm.yaml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        properties = parse_files.read_experiment(filename)

        # Now create list of Simulation objects
        simulations = parse_files.create_simulations(properties)

        # Ensure correct number of simulations
        assert len(simulations) == 1

        # Ensure correct information
        sim1 = simulations[0]
        assert sim1.properties['id'] == 'testfile_rcm_0'
        assert sim1.properties['data-file'] == 'testfile_rcm.yaml'
        assert sim1.kind == 'RCM'
        assert sim1.properties['temperature'] == 297.4 * units.kelvin
        assert sim1.properties['pressure'] == 958. * units.torr
        assert sim1.properties['ignition-delay'] == 1. * units.ms
        comp = {'H2': 0.12500, 'O2': 0.06250, 'N2': 0.18125, 'Ar': 0.63125}
        assert sim1.properties['composition'] == comp
        assert sim1.ignition_target == 'pressure'
        assert sim1.ignition_type == 'd/dt max'
        assert sim1.ignition_target_value == None
        assert sim1.ignition_target_unit == None

        numpy.testing.assert_allclose(sim1.properties['time'],
                                   numpy.arange(0, 9.7e-2, 1.e-3) * units.second
                                   )

        volumes = numpy.array([
            5.47669375000E+002, 5.46608789894E+002, 5.43427034574E+002,
            5.38124109043E+002, 5.30700013298E+002, 5.21154747340E+002,
            5.09488311170E+002, 4.95700704787E+002, 4.79791928191E+002,
            4.61761981383E+002, 4.41610864362E+002, 4.20399162234E+002,
            3.99187460106E+002, 3.77975757979E+002, 3.56764055851E+002,
            3.35552353723E+002, 3.14340651596E+002, 2.93128949468E+002,
            2.71917247340E+002, 2.50705545213E+002, 2.29493843085E+002,
            2.08282140957E+002, 1.87070438830E+002, 1.65858736702E+002,
            1.44647034574E+002, 1.23435332447E+002, 1.02223630319E+002,
            8.10119281915E+001, 6.33355097518E+001, 5.27296586879E+001,
            4.91943750000E+001, 4.97137623933E+001, 5.02063762048E+001,
            5.06454851923E+001, 5.10218564529E+001, 5.13374097598E+001,
            5.16004693977E+001, 5.18223244382E+001, 5.20148449242E+001,
            5.21889350372E+001, 5.23536351113E+001, 5.25157124459E+001,
            5.26796063730E+001, 5.28476160610E+001, 5.30202402028E+001,
            5.31965961563E+001, 5.33748623839E+001, 5.35527022996E+001,
            5.37276399831E+001, 5.38973687732E+001, 5.40599826225E+001,
            5.42141273988E+001, 5.43590751578E+001, 5.44947289126E+001,
            5.46215686913E+001, 5.47405518236E+001, 5.48529815402E+001,
            5.49603582190E+001, 5.50642270863E+001, 5.51660349836E+001,
            5.52670070646E+001, 5.53680520985E+001, 5.54697025392E+001,
            5.55720927915E+001, 5.56749762728E+001, 5.57777790517E+001,
            5.58796851466E+001, 5.59797461155E+001, 5.60770054561E+001,
            5.61706266985E+001, 5.62600130036E+001, 5.63449057053E+001,
            5.64254496625E+001, 5.65022146282E+001, 5.65761642150E+001,
            5.66485675508E+001, 5.67208534842E+001, 5.67944133373E+001,
            5.68703658198E+001, 5.69493069272E+001, 5.70310785669E+001,
            5.71146023893E+001, 5.71978399741E+001, 5.72779572372E+001,
            5.73517897984E+001, 5.74167271960E+001, 5.74721573687E+001,
            5.75216388520E+001, 5.75759967785E+001, 5.76575701358E+001,
            5.78058719368E+001, 5.80849611077E+001, 5.85928651155E+001,
            5.94734357453E+001, 6.09310671165E+001, 6.32487551103E+001,
            6.68100309742E+001
            ]) * units.cm3
        numpy.testing.assert_allclose(sim1.properties['volume'], volumes)
