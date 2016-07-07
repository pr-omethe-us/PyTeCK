# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import os
from os.path import splitext, basename
import multiprocessing
from argparse import ArgumentParser

import numpy
from scipy.interpolate import UnivariateSpline

import cantera as ct

try:
    import yaml
except ImportError:
    print('Warning: YAML must be installed to read input file.')

# Local imports
from .utils import units
from . import parse_files
from .simulation import Simulation

min_deviation = 0.10
"""float: minimum allowable standard deviation for experimental data"""

def simulation_worker(sim_tuple):
    """Worker for multiprocessing of simulation cases.

    Parameters
    ----------
    sim_tuple : tuple
        Contains Simulation object and other parameters needed to setup
        and run case.

    Returns
    -------
    sim : ``Simulation``
        Simulation case with calculated ignition delay.

    """
    sim, idx, model_file, model_spec_key, path = sim_tuple

    sim.setup_case(model_file, model_spec_key)
    sim.run_case(idx, path)

    sim = Simulation(sim.kind, sim.properties, sim.ignition_target,
                     sim.ignition_type, sim.ignition_target_value
                     )
    return sim


def estimate_std_dev(indep_variable, dep_variable):
    """

    Parameters
    ----------
    indep_variable : ndarray, list(float)
        Independent variable (e.g., temperature, pressure)
    dep_variable : ndarray, list(float)
        Dependent variable (e.g., ignition delay)

    Returns
    -------
    standard_dev : float
        Standard deviation of difference between data and best-fit line

    """

    assert len(indep_variable) == len(dep_variable), \
        'independent and dependent variables not the same length'

    # spline fit of the data
    if len(indep_variable) == 1 or len(indep_variable) == 2:
        # Fit of data will be perfect
        return min_deviation
    elif len(indep_variable) == 3:
        spline = UnivariateSpline(indep_variable, dep_variable, k=2)
    else:
        spline = UnivariateSpline(indep_variable, dep_variable)

    standard_dev = numpy.std(dep_variable - spline(indep_variable))

    if standard_dev < min_deviation:
        print('Standard deviation of {:.2f} too low, '
              'using {:.2f}'.format(standard_dev, min_deviation))
        standard_dev = min_deviation

    return standard_dev


def get_changing_variable(cases):
    """Identify variable changing across multiple cases.

    Parameters
    ----------
    cases : list(dict)
        List of dictionaries with experimental case data.

    Returns
    -------
    variable : list(float)
        List of floats representing changing experimental variable.

    """
    changing_var = None

    for var_name in ['temperature', 'pressure']:
        var = [case[var_name] for case in cases]
        if not all([x == var[0] for x in var]):
            if not changing_var:
                changing_var = var_name
            else:
                print('Warning: multiple changing variables. '
                      'Using temperature.'
                      )
                changing_var = 'temperature'
                break

    # Temperature is default
    if changing_var is None:
        changing_var = 'temperature'

    variable = [case[changing_var].magnitude for case in cases]
    return variable


def evaluate_model(model_name, spec_keys_file, dataset_file,
                   data_path='data', model_path='models',
                   results_path='results', model_variant_file=None,
                   num_threads=None, print_results=False
                   ):
    """Evaluates the ignition delay error of a model for a given dataset.

    Parameters
    ----------
    model_name : str
        Chemical kinetic model filename
    spec_keys_file : str
        Name of YAML file identifying important species
    dataset_file : str
        Name of file with list of data files
    data_path : str
        Local path for data files. Optional; default = 'data'
    model_path : str
        Local path for model file. Optional; default = 'models'
    results_path : str
        Local path for creating results files. Optional; default = 'results'
    model_variant_file : str
        Name of YAML file identifying ranges of conditions for variants of the
        kinetic model. Optional; default = ``None``
    num_threads : int
        Number of CPU threads to use for performing simulations in parallel.
        Optional; default = ``None``, in which case the available number of
        cores minus one is used.
    print_results : bool
        If ``True``, print results of the model evaluation to screen.

    Returns
    -------
    output : dict
        Dictionary with all information about model evaluation results.

    """

    # Dict to translate species names into those used by models
    with open(spec_keys_file, 'r') as f:
        model_spec_key = yaml.load(f)

    # Keys for models with variants depending on pressure or bath gas
    model_variant = None
    if model_variant_file:
        with open(model_variant_file, 'r') as f:
            model_variant = yaml.load(f)

    # Read dataset list
    with open(dataset_file, 'r') as f:
        dataset_list = f.read().splitlines()

    error_func_sets = numpy.zeros(len(dataset_list))
    dev_func_sets = numpy.zeros(len(dataset_list))

    # Dictionary with all output data
    output = {'model': model_name, 'datasets': []}

    # If number of threads not specified, use either max number of available
    # cores minus 1, or use 1 if multiple cores not available.
    if not num_threads:
        num_threads = multiprocessing.cpu_count()-1 or 1

    # Loop through all datasets
    for idx_set, dataset in enumerate(dataset_list):

        dataset_meta = {'dataset': dataset, 'dataset_id': idx_set}

        # Create individual simulation cases for each datapoint in this dataset
        properties = parse_files.read_experiment(os.path.join(data_path, dataset))
        simulations = parse_files.create_simulations(properties)

        ignition_delays_exp = numpy.zeros(len(simulations))
        ignition_delays_sim = numpy.zeros(len(simulations))

        #############################################
        # Determine standard deviation of the dataset
        #############################################
        ign_delay = [case['ignition-delay'].to('second').magnitude
                     for case in properties['cases']
                     ]

        # get variable that is changing across datapoints
        variable = get_changing_variable(properties['cases'])
        # for ignition delay, use logarithm of values
        standard_dev = estimate_std_dev(variable, numpy.log(ign_delay))
        dataset_meta['standard deviation'] = standard_dev

        #########################################
        # Need to check if Ar or He in reactants,
        # and if so skip this dataset (for now).
        #########################################
        if ((any(['Ar' in case['composition'] for case in properties['cases']])
            and 'Ar' not in model_spec_key[model_name]
            ) or
            (any(['He' in case['composition'] for case in properties['cases']])
             and 'He' not in model_spec_key[model_name]
             )
            ):
            print('Warning: Ar or He in dataset, but not in model. Skipping.')
            error_func_sets[idx_set] = numpy.nan
            continue

        # Use available number of processors minus one,
        # or one process if single core.
        pool = multiprocessing.Pool(processes=num_threads)

        # setup all cases
        jobs = []
        for idx, sim in enumerate(simulations):
            # special treatment based on pressure for Princeton model

            if model_variant and model_name in model_variant:
                model_mod = ''
                if 'bath gases' in model_variant[model_name]:
                    # find any bath gases requiring special treatment
                    bath_gases = set(model_variant[model_name]['bath gases'])
                    gases = bath_gases.intersection(set(sim.properties['composition']))

                    # If only one bath gas present, use that. If multiple, use the
                    # predominant species. If none of the designated bath gases
                    # are present, just use the first one (shouldn't matter.)
                    if len(gases) > 1:
                        max_mole = 0.
                        sp = ''
                        for g in gases:
                            if float(sim.properties['composition'][g]) > max_mole:
                                sp = g
                    elif len(gases) == 1:
                        sp = gases.pop()
                    else:
                        # If no designated bath gas present, use any.
                        sp = bath_gases.pop()
                    model_mod += model_variant[model_name]['bath gases'][sp]

                if 'pressures' in model_variant[model_name]:
                    # pressure to atm
                    pres = sim.properties['pressure'].to('atm').magnitude

                    # choose closest pressure
                    # better way to do this?
                    i = numpy.argmin(numpy.abs(numpy.array(
                        [float(n)
                         for n in list(model_variant[model_name]['pressures'])
                         ]
                        ) - pres))
                    pres = list(model_variant[model_name]['pressures'])[i]
                    model_mod += model_variant[model_name]['pressures'][pres]

                model_file = os.path.join(model_path, model_name + model_mod)
            else:
                model_file = os.path.join(model_path, model_name)

            jobs.append([sim, idx, model_file,
                         model_spec_key[model_name], results_path
                         ])

        # run all cases
        jobs = tuple(jobs)
        results = pool.map(simulation_worker, jobs)

        # not adding more proceses, and ensure all finished
        pool.close()
        pool.join()

        dataset_meta['datapoints'] = []

        for idx, sim in enumerate(results):
            sim.process_results()

            ignition_delays_exp[idx] = sim.properties['ignition-delay'].magnitude
            ignition_delays_sim[idx] = sim.properties['simulated-ignition-delay'].magnitude

            temp = sim.properties['temperature'].to('kelvin').magnitude
            pres = sim.properties['pressure'].to('atm').magnitude

            dataset_meta['datapoints'].append(
                {'experimental ignition delay': ignition_delays_exp[idx],
                 'simulated ignition delay': ignition_delays_sim[idx],
                 'temperature': temp, 'pressure': pres,
                 'composition': sim.properties['composition']
                 })

        # calculate error function for this dataset
        error_func = numpy.power((numpy.log(ignition_delays_sim) -
                               numpy.log(ignition_delays_exp)) / standard_dev, 2
                              )
        error_func = numpy.nanmean(error_func)
        error_func_sets[idx_set] = error_func
        dataset_meta['error function'] = error_func

        dev_func = (numpy.log(ignition_delays_sim) -
                    numpy.log(ignition_delays_exp)
                    ) / standard_dev
        dev_func = numpy.nanmean(dev_func)
        dev_func_sets[idx_set] = dev_func
        dataset_meta['absolute deviation'] = dev_func

        output['datasets'].append(dataset_meta)

    # Overall error function
    error_func = numpy.nanmean(error_func_sets)
    if print_results:
        print('overall error function: ' + repr(error_func))
        print('error standard deviation: ' + repr(numpy.nanstd(error_func_sets)))

    # Absolute deviation function
    abs_dev_func = numpy.nanmean(dev_func_sets)
    if print_results:
        print('absolute deviation function: ' + repr(abs_dev_func))

    output['average error function'] = error_func
    output['error function standard deviation'] = numpy.nanstd(error_func_sets)
    output['average deviation function'] = abs_dev_func

    # Write data to YAML file
    with open(splitext(basename(model_name))[0] + '-results.yaml', 'w') as f:
        yaml.dump(output, f)

    return output
