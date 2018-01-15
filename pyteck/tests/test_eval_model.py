# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import os
import pkg_resources

# Third-party libraries
import numpy
import pytest
from pyked.chemked import ChemKED, DataPoint

# Taken from http://stackoverflow.com/a/22726782/1569494
try:
    from tempfile import TemporaryDirectory
except ImportError:
    from contextlib import contextmanager
    import shutil
    import tempfile
    import errno

    @contextmanager
    def TemporaryDirectory():
        name = tempfile.mkdtemp()
        try:
            yield name
        finally:
            try:
                shutil.rmtree(name)
            except OSError as e:
                # Reraise unless ENOENT: No such file or directory
                # (ok if directory has already been deleted)
                if e.errno != errno.ENOENT:
                    raise

# Local imports
from .. import eval_model
from ..simulation import Simulation
from ..utils import units
from ..exceptions import UndefinedKeywordError


class TestEstimateStandardDeviation:
    """
    """
    def test_single_point(self):
        """Check return for single data point.
        """
        changing_variable = numpy.random.rand(1)
        dependent_variable = numpy.random.rand(1)

        standard_dev = eval_model.estimate_std_dev(changing_variable,
                                                   dependent_variable
                                                   )
        assert standard_dev == eval_model.min_deviation

    def test_two_points(self):
        """Check return for two data points.
        """
        changing_variable = numpy.random.rand(2)
        dependent_variable = numpy.random.rand(2)

        standard_dev = eval_model.estimate_std_dev(changing_variable,
                                                   dependent_variable
                                                   )
        assert standard_dev == eval_model.min_deviation

    def test_three_points(self):
        """Check return for perfect, linear three data points.
        """
        changing_variable = numpy.arange(1, 4)
        dependent_variable = numpy.arange(1, 4)

        standard_dev = eval_model.estimate_std_dev(changing_variable,
                                                   dependent_variable
                                                   )
        assert standard_dev == eval_model.min_deviation

    def test_normal_dist_noise(self):
        """Check expected standard deviation for normally distributed noise.
        """
        num = 1000000
        changing_variable = numpy.arange(1, num + 1)
        dependent_variable = numpy.arange(1, num + 1)
        # add normally distributed noise, standard deviation of 1.0
        noise = numpy.random.normal(0.0, 1.0, num)

        standard_dev = eval_model.estimate_std_dev(changing_variable,
                                                   dependent_variable + noise
                                                   )
        assert numpy.isclose(1.0, standard_dev, rtol=1.e-2)

    def test_repeated_points(self):
        """Check that function correctly handles repeated points with no error.
        """
        changing_variable = numpy.arange(1, 10)
        dependent_variable = numpy.arange(1, 10)
        changing_variable[1] = changing_variable[0]

        standard_dev = eval_model.estimate_std_dev(changing_variable,
                                                   dependent_variable
                                                   )
        assert standard_dev == eval_model.min_deviation


class TestGetChangingVariable:
    """
    """
    def test_single_point(self):
        """Check normal behavior for single point.
        """
        cases = [DataPoint({'pressure': [numpy.random.rand(1) * units('atm')],
                            'temperature': [numpy.random.rand(1) * units('K')],
                            'composition':
                                {'kind': 'mole fraction',
                                 'species': [{'species-name': 'O2', 'amount': [1.0]}]
                                 },
                            'ignition-type': None
                            })
                 ]
        variable = eval_model.get_changing_variable(cases)

        assert len(variable) == 1
        assert variable[0] == cases[0].temperature.magnitude

    def test_temperature_changing(self):
        """Check normal behavior for multiple points with temperature changing.
        """
        num = 10
        pressure = numpy.random.rand(1) * units('atm')
        temperatures = numpy.random.rand(num) * units('K')
        cases = []
        for temp in temperatures:
            dp = DataPoint({'pressure': [str(pressure[0])],
                            'temperature': [str(temp)],
                            'composition':
                                {'kind': 'mole fraction',
                                 'species': [{'species-name': 'O2', 'amount': [1.0]}]
                                 },
                            'ignition-type': None
                            })
            cases.append(dp)

        variable = eval_model.get_changing_variable(cases)

        assert len(variable) == num
        assert numpy.allclose(variable, [c.temperature.magnitude for c in cases])

    def test_pressure_changing(self):
        """Check normal behavior for multiple points with pressure changing.
        """
        num = 10
        pressures = numpy.random.rand(num) * units('atm')
        temperature = numpy.random.rand(1) * units('K')
        cases = []
        for pres in pressures:
            dp = DataPoint({'pressure': [str(pres)],
                            'temperature': [str(temperature[0])],
                            'composition':
                                {'kind': 'mole fraction',
                                 'species': [{'species-name': 'O2', 'amount': [1.0]}]
                                 },
                            'ignition-type': None
                            })
            cases.append(dp)

        variable = eval_model.get_changing_variable(cases)

        assert len(variable) == num
        assert numpy.allclose(variable, [c.pressure.magnitude for c in cases])

    def test_both_changing(self):
        """Check fallback behavior for both properties varying.
        """
        num = 10
        pressures = numpy.random.rand(num) * units('atm')
        temperatures = numpy.random.rand(num) * units('K')
        cases = []
        for pres, temp in zip(pressures, temperatures):
            dp = DataPoint({'pressure': [str(pres)],
                            'temperature': [str(temp)],
                            'composition':
                                {'kind': 'mole fraction',
                                 'species': [{'species-name': 'O2', 'amount': [1.0]}]
                                 },
                            'ignition-type': None
                            })
            cases.append(dp)

        variable = eval_model.get_changing_variable(cases)

        assert len(variable) == num
        assert numpy.allclose(variable, [c.temperature.magnitude for c in cases])

class TestEvalModel:
    """
    """
    def relative_location(self, file):
        file_path = os.path.join(file)
        return pkg_resources.resource_filename(__name__, file_path)

    def test(self):
        """
        """

        with TemporaryDirectory() as temp_dir:
            output = eval_model.evaluate_model(
                                      'h2o2.cti',
                                      self.relative_location('spec_keys.yaml'),
                                      self.relative_location('dataset_file.txt'),
                                      data_path=self.relative_location(''),
                                      model_path='',
                                      results_path=temp_dir,
                                      num_threads=1
                                      )
            assert numpy.isclose(output['average error function'], 58.78211242028232, rtol=1.e-3)
            assert numpy.isclose(output['error function standard deviation'], 0.0, rtol=1.e-3)
            assert numpy.isclose(output['average deviation function'], 7.635983785416241, rtol=1.e-3)
