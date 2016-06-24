# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import os
import pkg_resources

import numpy as np
import pytest

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
from .. import parse_files_XML
from ..simulation import Simulation
from ..utils import units

#pytestmark = pytest.mark.skip(reason="XML converter not completed")

class TestExperimentType:
    """
    """
    def test_shock_tube_experiment(self):
        """Ensure shock tube experiment can be detected.
        """
        file_path = os.path.join('testfile_st.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        kind = parse_files_XML.get_experiment_kind(root)
        assert kind == 'ST'

    def test_RCM_experiment(self):
        """Ensure rapid compression machine experiment can be detected.
        """
        file_path = os.path.join('testfile_rcm.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        kind = parse_files_XML.get_experiment_kind(root)
        assert kind == 'RCM'


class TestCommonProperties:
    """
    """
    def test_shock_tube_common_properties(self):
        """Ensure basic common properties parsed for shock tube.
        """
        file_path = os.path.join('testfile_st.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        properties = {}
        properties['kind'] = parse_files_XML.get_experiment_kind(root)
        properties = parse_files_XML.get_common_properties(properties, root)

        # Check pressure
        assert properties['pressure'] == 2.18 * units.atm

        # Check initial composition
        assert properties['composition']['H2'] == 0.00444
        assert properties['composition']['O2'] == 0.00566
        assert properties['composition']['Ar'] == 0.9899

        # Check pressure rise
        assert 'pressure-rise' not in properties

        # Make sure no other properties present
        assert (set(properties.keys()) ==
                set(['kind', 'pressure', 'composition'])
                )

    def test_shock_tube_common_properties_pressure_rise(self):
        """Ensure basic common properties parsed for shock tube.
        """
        file_path = os.path.join('testfile_st2.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        properties = {}
        properties['kind'] = parse_files_XML.get_experiment_kind(root)
        properties = parse_files_XML.get_common_properties(properties, root)

        # Check pressure
        assert properties['pressure'] == 2.18 * units.atm

        # Check initial composition
        assert properties['composition']['H2'] == 0.00444
        assert properties['composition']['O2'] == 0.00566
        assert properties['composition']['Ar'] == 0.9899

        # Check pressure rise
        assert properties['pressure-rise'] == 0.10 / units.ms

        # Make sure no other properties present
        assert (set(properties.keys()) ==
                set(['kind', 'pressure', 'pressure-rise', 'composition'])
                )

    def test_rcm_common_properties(self):
        """Ensure basic common properties parsed for RCM.
        """
        file_path = os.path.join('testfile_rcm.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        properties = {}
        properties['kind'] = parse_files_XML.get_experiment_kind(root)
        properties = parse_files_XML.get_common_properties(properties, root)

        # Check initial composition
        assert properties['composition']['H2'] == 0.12500
        assert properties['composition']['O2'] == 0.06250
        assert properties['composition']['N2'] == 0.18125
        assert properties['composition']['Ar'] == 0.63125

        assert set(properties.keys()) == set(['kind', 'composition'])


class TestIgnitionType:
    """
    """
    def test_pressure_ignition_target(self):
        """Test pressure max derivative as target for RCM.
        """
        file_path = os.path.join('testfile_rcm.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        ignition = parse_files_XML.get_ignition_type(root)

        assert ignition['target'] == 'P'
        assert ignition['type'] == 'd/dt max'

    def test_pressure_species_target(self):
        """Test pressure max derivative as target for shock tube.
        """
        file_path = os.path.join('testfile_st.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        ignition = parse_files_XML.get_ignition_type(root)

        assert ignition['target'] == 'P'
        assert ignition['type'] == 'd/dt max'

    def test_pressure_species_target_OH(self):
        """Test species max value as targetree.
        """
        file_path = os.path.join('testfile_st2.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        ignition = parse_files_XML.get_ignition_type(root)

        assert ignition['target'] == 'OH'
        assert ignition['type'] == 'max'


class TestDataGroups:
    """
    """
    def test_shock_tube_data_points(self):
        """Test parsing of ignition delay data points for shock tube file.
        """
        file_path = os.path.join('testfile_st.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        properties = {}
        properties = parse_files_XML.get_datapoints(properties, root)

        # Ensure correct temperature and ignition delay values and units
        np.testing.assert_array_equal(properties['temperature'],
                                      [1164.48, 1164.97] * units.kelvin
                                      )
        np.testing.assert_array_equal(properties['ignition-delay'],
                                      [471.54, 448.03] * units.us
                                      )

    def test_shock_tube_data_points_pressure_rise(self):
        """Test parsing of ignition delay data points for shock tube file.
        """
        file_path = os.path.join('testfile_st2.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        properties = {}
        properties = parse_files_XML.get_datapoints(properties, root)

        # Ensure correct temperature and ignition delay values and units
        np.testing.assert_allclose(properties['temperature'],
                                   1264.2 * units.kelvin
                                   )
        np.testing.assert_allclose(properties['ignition-delay'],
                                   291.57 * units.us
                                   )

    def test_rcm_data_points(self):
        """Test parsing of ignition delay data points for RCM file.
        """
        file_path = os.path.join('testfile_rcm.xml')
        filename = pkg_resources.resource_filename(__name__, file_path)
        tree = etree.parse(filename)
        root = tree.getroot()

        properties = {}
        properties = parse_files_XML.get_datapoints(properties, root)

        # Ensure correct temperature, pressure, and ignition delay values and units
        assert properties['temperature'] == 297.4 * units.kelvin
        assert properties['pressure'] == 958. * units.torr
        assert properties['ignition-delay'] == 1. * units.ms

        # Check other data group with volume history
        np.testing.assert_allclose(properties['time'],
                                   np.arange(0, 9.7e-2, 1.e-3) * units.second
                                   )

        volumes = np.array([
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
        np.testing.assert_allclose(properties['volume'], volumes)
