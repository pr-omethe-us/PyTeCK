# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Local imports
from PyTeCK.eval_model import evaluate_model

evaluate_model('Tsurushima-2009.cti', 'model_species_keys.json',
               'nheptane_data.txt', 'data', 'models', 'results',
               'model_variant.json')
