# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Local imports
from pyteck.eval_model import evaluate_model

evaluate_model(model_name='Tsurushima-2009.cti',
               spec_keys_file='model_species_keys.json',
               dataset_file='nheptane_data.txt',
               data_path='data',
               model_path='models',
               results_path='results',
               model_variant_file='model_variant.json')
