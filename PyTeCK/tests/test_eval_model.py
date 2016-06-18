# Python 2 compatibility
from __future__ import print_function
from __future__ import division

from .. import eval_model
from ..simulation import Simulation
from ..utils import units
from ..exceptions import UndefinedKeywordError

import os
import pkg_resources
import numpy
import pytest

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
        num = 100000
        changing_variable = numpy.arange(1, num + 1)
        dependent_variable = numpy.arange(1, num + 1)
        # add normally distributed noise, standard deviation of 1.0
        noise = numpy.random.normal(0.0, 1.0, num)

        standard_dev = eval_model.estimate_std_dev(changing_variable,
                                                   dependent_variable + noise
                                                   )
        numpy.testing.assert_allclose(1.0, standard_dev, rtol=1.e-3)
