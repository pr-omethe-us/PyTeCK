# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Local imports
from eval_kinetic_models.eval_model import evaluate_model

evaluate_model('Tsurushima-2009.cti', 'nheptane_data.txt')
