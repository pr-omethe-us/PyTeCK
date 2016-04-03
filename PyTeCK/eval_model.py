# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import os
from os.path import splitext, basename
import json
import multiprocessing
from argparse import ArgumentParser

import numpy as np
from scipy.interpolate import UnivariateSpline

import cantera as ct

# Local imports
from .utils import to_pascal, to_kelvin, to_atm
from . import parse_files
from .simulation import Property, Simulation


def simulation_worker(sim_tuple):
    """Worker for multiprocessing of simulation cases.

    :param tuple sim_tuple: Contains Simulation object and other parameters needed
    to setup and run case.
    :return: Simulation case with calculated ignition delay.
    :rtype: Simulation object
    """
    sim, idx, model_file, model_spec_key, path = sim_tuple

    sim.setup_case(model_file, model_spec_key)
    sim.run_case(idx, path)

    sim = Simulation(sim.kind, sim.properties, sim.ignition_target,
                     sim.ignition_type, sim.ignition_target_value
                     )
    return sim


def evaluate_model(model_name, spec_keys_file, dataset_file,
                   data_path='data', model_path='models',
                   results_path='results', model_variant_file=None,
                   num_threads=None
                   ):
    """Evaluates the ignition delay error of a model for a given dataset.
    """

    # Dict to translate species names into those used by models
    with open(spec_keys_file, 'r') as f:
        model_spec_key = json.load(f)

    # Keys for models with variants depending on pressure or bath gas
    if model_variant_file:
        with open(model_variant_file, 'r') as f:
            model_variant = json.load(f)

    # Read dataset list
    with open(dataset_file, 'r') as f:
        dataset_list = f.read().splitlines()

    error_func_sets = np.zeros(len(dataset_list))
    dev_func_sets = np.zeros(len(dataset_list))

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

        ignition_delays_exp = np.zeros(len(simulations))
        ignition_delays_sim = np.zeros(len(simulations))

        # Determine standard deviation of the dataset
        if len(simulations) == 1:
            standard_dev = 0.10
        else:
            ign_delay = properties['ignition delay'].value

            # get variable that is changing across datapoints
            var = [var for var in ['temperature', 'pressure', 'composition']
                   if isinstance(properties[var], Property) and
                   isinstance(properties[var].value, np.ndarray)
                   ]
            if len(var) > 1:
                print('Warning: multiple changing variables')
                print('Using ' + var[0])
            var = var[0]
            variable = properties[var].value

            # spline fit of the data
            if len(variable) == 3:
                sp1 = UnivariateSpline(variable, np.log(ign_delay), k=2)
            elif len(variable) == 2:
                sp1 = UnivariateSpline(variable, np.log(ign_delay), k=1)
            else:
                sp1 = UnivariateSpline(variable, np.log(ign_delay))
            diff = np.log(ign_delay) - sp1(variable)
            standard_dev = np.std(diff)

            if standard_dev < 0.10:
                print('Standard deviation too low, using 0.10')
                standard_dev = 0.10

        dataset_meta['standard deviation'] = standard_dev

        # Need to check if Ar or He in reactants,
        # and if so skip this dataset (for now).
        if ('Ar' in properties['composition'] and
            'Ar' not in model_spec_key[model_name]
            ) or ('He' in properties['composition'] and
                  'He' not in model_spec_key[model_name]
                  ):
            print('Warning: Ar or He in dataset, but not in model. Skipping.')
            error_func_sets[idx_set] = np.nan
            continue

        # Use available number of processors minus one,
        # or one process if single core.
        pool = multiprocessing.Pool(processes=num_threads)

        # setup all cases
        jobs = []
        for idx, sim in enumerate(simulations):
            # special treatment based on pressure for Princeton model

            if model_name in model_variant:
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
                    pres = to_atm(sim.properties['pressure'].value,
                                  sim.properties['pressure'].units
                                  )

                    # choose closest pressure
                    # better way to do this?
                    i = np.argmin(np.abs(np.array(
                        [float(n)
                        for n in list(model_variant[model_name]['pressures'])]
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

            ignition_delays_exp[idx] = sim.properties['ignition delay']
            ignition_delays_sim[idx] = sim.properties['simulated ignition delay']

            temp = to_kelvin(sim.properties['temperature'].value,
                             sim.properties['temperature'].units
                             )
            pres = to_atm(sim.properties['pressure'].value,
                          sim.properties['pressure'].units
                          )

            dataset_meta['datapoints'].append(
                {'experimental ignition delay': ignition_delays_exp[idx],
                 'simulated ignition delay': ignition_delays_sim[idx],
                 'temperature': temp, 'pressure': pres,
                 'composition': properties['composition']
                 })

        # calculate error function for this dataset
        error_func = np.power((np.log(ignition_delays_sim) -
                               np.log(ignition_delays_exp)) / standard_dev, 2
                              )
        error_func = np.nanmean(error_func)
        error_func_sets[idx_set] = error_func
        dataset_meta['error function'] = error_func

        dev_func = (np.log(ignition_delays_sim) -
                    np.log(ignition_delays_exp)
                    ) / standard_dev
        dev_func = np.nanmean(dev_func)
        dev_func_sets[idx_set] = dev_func
        dataset_meta['absolute deviation'] = dev_func

        output['datasets'].append(dataset_meta)

    # Overall error function
    error_func = np.nanmean(error_func_sets)
    print('overall error function: ' + repr(error_func))
    print('error standard deviation: ' + repr(np.nanstd(error_func_sets)))

    # Absolute deviation function
    abs_dev_func = np.nanmean(dev_func_sets)
    print('absolute deviation function: ' + repr(abs_dev_func))

    output['average error function'] = error_func
    output['error function standard deviation'] = np.nanstd(error_func_sets)
    output['average deviation function'] = abs_dev_func

    # Write data to JSON file
    with open(splitext(basename(model_name))[0] + '-results.json', 'w') as f:
        json.dump(output, f)
