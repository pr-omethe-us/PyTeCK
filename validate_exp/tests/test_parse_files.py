# Python 2 compatibility
from __future__ import print_function
from __future__ import division

from .. import parse_files
from ..simulation import Property, Simulation

import os
import pkg_resources
import numpy as np

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class TestExperimentType:
    """
    """
    def test_shock_tube_experiment(self):
        """Ensure shock tube experiment can be detected.
        """
        file_path = os.path.join('testfile_st.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        kind = parse_files.get_experiment_kind(root)
        assert kind == 'ST'

    def test_RCM_experiment(self):
        """Ensure rapid compression machine experiment can be detected.
        """
        file_path = os.path.join('testfile_rcm.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        kind = parse_files.get_experiment_kind(root)
        assert kind == 'RCM'


class TestCommonProperties:
    """
    """
    def test_shock_tube_common_properties(self):
        """Ensure basic common properties parsed for shock tube.
        """
        file_path = os.path.join('testfile_st.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        properties = {}
        properties['kind'] = parse_files.get_experiment_kind(root)
        properties = parse_files.get_common_properties(properties, root)

        # Check pressure
        assert properties['pressure'].value == 2.18
        assert properties['pressure'].units == 'atm'

        # Check initial composition
        assert properties['composition']['H2'] == '0.00444'
        assert properties['composition']['O2'] == '0.00566'
        assert properties['composition']['Ar'] == '0.99'

        # Check pressure rise
        assert 'pressure rise' not in properties

        # Make sure no other properties present
        assert (set(properties.keys()) ==
                set(['kind', 'pressure', 'composition'])
                )

    def test_shock_tube_common_properties_pressure_rise(self):
        """Ensure basic common properties parsed for shock tube.
        """
        file_path = os.path.join('testfile_st2.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        properties = {}
        properties['kind'] = parse_files.get_experiment_kind(root)
        properties = parse_files.get_common_properties(properties, root)

        # Check pressure
        assert properties['pressure'].value == 2.18
        assert properties['pressure'].units == 'atm'

        # Check initial composition
        assert properties['composition']['H2'] == '0.00444'
        assert properties['composition']['O2'] == '0.00566'
        assert properties['composition']['Ar'] == '0.99'

        # Check pressure rise
        assert properties['pressure rise'].value == 0.10
        assert properties['pressure rise'].units == 'ms'

        # Make sure no other properties present
        assert (set(properties.keys()) ==
                set(['kind', 'pressure', 'pressure rise', 'composition'])
                )

    def test_rcm_common_properties(self):
        """Ensure basic common properties parsed for RCM.
        """
        file_path = os.path.join('testfile_rcm.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        properties = {}
        properties['kind'] = parse_files.get_experiment_kind(root)
        properties = parse_files.get_common_properties(properties, root)

        # Check initial composition
        assert properties['composition']['N2'] == '0.5'
        assert properties['composition']['O2'] == '0.5'

        assert set(properties.keys()) == set(['kind', 'composition'])


class TestIgnitionType:
    """
    """
    def test_pressure_ignition_target(self):
        """Test pressure max derivative as target for RCM.
        """
        file_path = os.path.join('testfile_rcm.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        properties = {}
        properties = parse_files.get_ignition_type(properties, root)

        assert properties['ignition target'] == 'P'
        assert properties['ignition type'] == 'd/dt max'

    def test_pressure_species_target(self):
        """Test pressure max derivative as target for shock tube.
        """
        file_path = os.path.join('testfile_st.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        properties = {}
        properties = parse_files.get_ignition_type(properties, root)

        assert properties['ignition target'] == 'P'
        assert properties['ignition type'] == 'd/dt max'

    def test_pressure_species_target_OH(self):
        """Test species max value as target.
        """
        file_path = os.path.join('testfile_st2.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        properties = {}
        properties = parse_files.get_ignition_type(properties, root)

        assert properties['ignition target'] == 'OH'
        assert properties['ignition type'] == 'max'


class TestDataGroups:
    """
    """
    def test_shock_tube_data_points(self):
        """Test parsing of ignition delay data points for shock tube file.
        """
        file_path = os.path.join('testfile_st.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        properties = {}
        properties = parse_files.get_datapoints(properties, root)

        # Ensure correct temperature and ignition delay values and units
        np.testing.assert_array_equal(properties['temperature'].value,
                                      [1164.48, 1164.97]
                                      )
        assert properties['temperature'].units == 'K'
        np.testing.assert_array_equal(properties['ignition delay'].value,
                                      [471.54, 448.03]
                                      )
        assert properties['ignition delay'].units == 'us'

    def test_shock_tube_data_points_pressure_rise(self):
        """Test parsing of ignition delay data points for shock tube file.
        """
        file_path = os.path.join('testfile_st2.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        properties = {}
        properties = parse_files.get_datapoints(properties, root)

        # Ensure correct temperature and ignition delay values and units
        np.testing.assert_allclose(properties['temperature'].value, 1264.2)
        assert properties['temperature'].units == 'K'
        np.testing.assert_allclose(properties['ignition delay'].value, 291.57)
        assert properties['ignition delay'].units == 'us'

    def test_rcm_data_points(self):
        """Test parsing of ignition delay data points for RCM file.
        """
        file_path = os.path.join('testfile_rcm.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = ET.parse(filename)
        root = tree.getroot()

        properties = {}
        properties = parse_files.get_datapoints(properties, root)

        # Ensure correct temperature, pressure, and ignition delay values and units
        assert properties['temperature'].value == 1000.
        assert properties['temperature'].units == 'K'
        assert properties['pressure'].value == 40.0
        assert properties['pressure'].units == 'atm'
        assert properties['ignition delay'].value == 1.0
        assert properties['ignition delay'].units == 'ms'

        # Check other data group with volume history
        np.testing.assert_allclose(properties['time'].value,
                                   np.arange(0, 1, 0.1)
                                   )
        assert properties['time'].units == 's'
        np.testing.assert_allclose(properties['volume'].value,
                                   np.arange(5.e2, 6.e2, 10.)
                                   )
        assert properties['volume'].units == 'cm3'


class TestCreateSimulations:
    """
    """
    def test_create_st_simulations(self):
        """Ensure appropriate simulations created from shock tube file.
        """
        # Rely on previously tested functions to parse file
        file_path = os.path.join('testfile_st.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        properties = parse_files.read_experiment(filename)

        # Now create list of Simulation objects
        simulations = parse_files.create_simulations(properties)

        comp = {'H2': '0.00444', 'O2': '0.00566', 'Ar': '0.99'}

        # Ensure correct number of simulations
        assert len(simulations) == 2

        # Ensure correct information
        sim1 = simulations[0]
        assert sim1.properties['id'] == 'testfile_st_0'
        assert sim1.properties['data file'] == 'testfile_st.xml'
        assert sim1.kind == 'ST'
        assert sim1.properties['temperature'] == Property(1164.48, 'K')
        assert sim1.properties['pressure'] == Property(2.18, 'atm')
        assert sim1.properties['ignition delay'] == Property(471.54, 'us')
        assert sim1.properties['composition'] == comp
        assert sim1.ignition_target == 'P'
        assert sim1.ignition_type == 'd/dt max'
        assert sim1.ignition_target_value == None

        sim2 = simulations[1]
        assert sim2.properties['id'] == 'testfile_st_1'
        assert sim2.properties['data file'] == 'testfile_st.xml'
        assert sim2.kind == 'ST'
        assert sim2.properties['temperature'] == Property(1164.97, 'K')
        assert sim1.properties['pressure'] == Property(2.18, 'atm')
        assert sim2.properties['ignition delay'] == Property(448.03, 'us')
        assert sim2.properties['composition'] == comp
        assert sim1.ignition_target == 'P'
        assert sim1.ignition_type == 'd/dt max'
        assert sim2.ignition_target_value == None

    def test_create_st_simulations_pressure_rise(self):
        """Ensure appropriate simulations created from shock tube file.
        """
        # Rely on previously tested functions to parse file
        file_path = os.path.join('testfile_st2.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        properties = parse_files.read_experiment(filename)

        # Now create list of Simulation objects
        simulations = parse_files.create_simulations(properties)

        # Ensure correct number of simulations
        assert len(simulations) == 1

        sim = simulations[0]
        assert sim.properties['id'] == 'testfile_st2_0'
        assert sim.properties['data file'] == 'testfile_st2.xml'
        assert sim.kind == 'ST'
        assert sim.properties['temperature'] == Property(1264.2, 'K')
        assert sim.properties['pressure'] == Property(2.18, 'atm')
        assert sim.properties['ignition delay'] == Property(291.57, 'us')
        assert sim.properties['pressure rise'] == Property(0.10, 'ms')
        comp = {'H2': '0.00444', 'O2': '0.00566', 'Ar': '0.99'}
        assert sim.properties['composition'] == comp
        assert sim.ignition_target == 'OH'
        assert sim.ignition_type == 'max'
        assert sim.ignition_target_value == None

    def test_create_rcm_simulations(self):
        """Ensure appropriate simulations created from RCM file.
        """
        # Rely on previously tested functions to parse file
        file_path = os.path.join('testfile_rcm.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        properties = parse_files.read_experiment(filename)

        # Now create list of Simulation objects
        simulations = parse_files.create_simulations(properties)

        # Ensure correct number of simulations
        assert len(simulations) == 1

        # Ensure correct information
        sim1 = simulations[0]
        assert sim1.properties['id'] == 'testfile_rcm_0'
        assert sim1.properties['data file'] == 'testfile_rcm.xml'
        assert sim1.kind == 'RCM'
        assert sim1.properties['temperature'] == Property(1000., 'K')
        assert sim1.properties['pressure'] == Property(40., 'atm')
        assert sim1.properties['ignition delay'] == Property(1., 'ms')
        assert sim1.properties['composition'] == {'N2': '0.5', 'O2': '0.5'}
        assert sim1.ignition_target == 'P'
        assert sim1.ignition_type == 'd/dt max'
        assert sim1.ignition_target_value == None

        np.testing.assert_allclose(sim1.properties['time'].value,
                                   np.arange(0, 1, 0.1)
                                   )
        assert sim1.properties['time'].units == 's'
        np.testing.assert_allclose(sim1.properties['volume'].value,
                                   np.arange(5.e2, 6.e2, 10.)
                                   )
        assert sim1.properties['volume'].units == 'cm3'
