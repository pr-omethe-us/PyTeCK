from argparse import ArgumentParser
import multiprocessing

from .eval_model import evaluate_model

parser = ArgumentParser(description='PyTeCK: Evaluate '
                                    'performance of kinetic models using '
                                    'experimental ignition delay data.'
                        )
parser.add_argument('-m', '--model',
                    type=str,
                    required=True,
                    help='Input model filename (e.g., mech.cti).'
                    )
parser.add_argument('-k', '--model-keys',
                    type=str,
                    dest='model_keys_file',
                    required=True,
                    help='JSON file with keys for species in models.'
                    )
parser.add_argument('-d', '--dataset',
                    type=str,
                    required=True,
                    help='Filename for list of datasets.'
                    )
parser.add_argument('-dp', '--data-path',
                    type=str,
                    dest='data_path',
                    required=False,
                    default='data',
                    help='Local directory holding dataset files.'
                    )
parser.add_argument('-mp', '--model-path',
                    type=str,
                    dest='model_path',
                    required=False,
                    default='models',
                    help='Local directory holding model files.'
                    )
parser.add_argument('-rp', '--results-path',
                    type=str,
                    dest='results_path',
                    required=False,
                    default='results',
                    help='Local directory holding result HDF5 files.'
                    )
parser.add_argument('-v', '--model-variant',
                    type=str,
                    dest='model_variant_file',
                    required=False,
                    help='JSON with variants for models for, e.g., bath '
                         'gases and pressures.'
                    )
parser.add_argument('-nt', '--num-threads',
                    type=int,
                    dest='num_threads',
                    default=multiprocessing.cpu_count()-1 or 1,
                    required=False,
                    help='The number of threads to use to run simulations in '
                         'parallel.'
                    )
parser.add_argument('-p', '--print',
                    dest='print_results',
                    action='store_true',
                    default=False,
                    help='Print model evaluation results to screen.'
                    )
parser.add_argument('--restart',
                    dest='restart',
                    action='store_true',
                    default=False,
                    help='Reuse prior results files, and only calculate new ones.'
                    )
parser.add_argument('--skip-validation',
                    dest='skip_validation',
                    action='store_true',
                    default=False,
                    help='Skips ChemKED file validation.'
                    )
args = parser.parse_args()

evaluate_model(args.model, args.model_keys_file, args.dataset,
               args.data_path, args.model_path, args.results_path,
               args.model_variant_file, args.num_threads, args.print_results,
               args.restart, args.skip_validation,
               )
