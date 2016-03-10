# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import os
from os.path import splitext, basename
import json
import multiprocessing
import numpy as np
from scipy.interpolate import UnivariateSpline

import cantera as ct

# Local imports
from test_kinetic_models import parse_files
from test_kinetic_models.simulation import Property, Simulation

def simulation_worker(sim_tuple):
    """Worker for multiprocessing of simulation cases.
    """
    sim, idx, model_file, model_spec_key, path = sim_tuple

    sim.setup_case(model_file, model_spec_key)
    sim.run_case(idx, path)

    sim = Simulation(sim.kind, sim.properties, sim.ignition_target,
                     sim.ignition_type, sim.ignition_target_value
                     )
    return sim

mech_filename = 'Tsurushima-2009.cti'

data_path = 'data'
model_path = 'models'

# Dict to translate species names into those used by models
model_spec_key = {
    'Tsurushima-2009.cti': {'O2': 'O2', 'N2': 'N2', 'nC7H16': 'C7H16', 'CO2': 'CO2'},
    'ERC-2013.cti': {'O2': 'o2', 'N2': 'n2', 'nC7H16': 'nc7h16', 'CO2': 'co2'},
    'Ogura-2007.cti': {'O2': 'O2', 'N2': 'N2', 'nC7H16': 'nC7H16', 'Ar': 'AR', 'He': 'He', 'CO2': 'CO2'},
    'Saisirirat-2011.cti': {'O2': 'o2', 'N2': 'n2', 'nC7H16': 'nc7h16', 'CO2': 'co2'},
    'CNRS-2009.cti': {'O2': 'O2', 'N2': 'N2', 'nC7H16': 'C7H16-1', 'Ar': 'AR', 'He': 'HE', 'CO2': 'CO2'},
    'Sakai-2009.cti': {'O2': 'O2', 'N2': 'N2', 'nC7H16': 'NC7H16', 'Ar': 'AR', 'He': 'HE', 'CO2': 'CO2'},
    'Dalian-2013.cti': {'O2': 'O2', 'N2': 'N2', 'nC7H16': 'C7H16', 'CO2': 'CO2'},
    'Andrae-2013.cti': {'O2': 'O2', 'N2': 'N2', 'nC7H16': 'C7H16', 'Ar': 'AR', 'CO2': 'CO2'},
    'LLNL-2012.cti': {'O2': 'O2', 'N2': 'N2', 'nC7H16': 'NC7H16', 'Ar': 'AR', 'He': 'HE', 'CO2': 'CO2'},
    'Princeton-2009': {'O2': 'o2', 'N2': 'n2', 'nC7H16': 'nc7h16', 'Ar': 'ar', 'He': 'he', 'CO2': 'co2'},
    'Cancino-2011.cti': {'O2': 'O2', 'N2': 'N2', 'nC7H16': 'NC7H16', 'Ar': 'AR', 'CO2': 'CO2'},
    'Tsinghua-2014.cti': {'O2': 'O2', 'N2': 'N2', 'nC7H16': 'NC7H16', 'Ar': 'AR', 'CO2': 'CO2'},
    'CRECK-2014.cti': {'O2': 'O2', 'N2': 'N2', 'nC7H16': 'NC7H16', 'Ar': 'AR', 'He': 'HE', 'CO2': 'CO2'},
    }

# Keys for models with variants depending on pressure or bath gas
model_variant = {
    'Princeton-2009': {
        'bath gases': {
            'N2': {
                '1': '_N2_1atm.cti', '3': '_N2_3atm.cti', '6': '_N2_6atm.cti',
                '9': '_N2_9atm.cti', '12': '_N2_12atm.cti',
                '15': '_N2_15atm.cti', '50': '_N2_50atm.cti'
                },
            'Ar': {
                '1': '_Ar_1atm.cti', '3': '_Ar_3atm.cti', '6': '_Ar_6atm.cti',
                '9': '_Ar_9atm.cti', '12': '_Ar_12atm.cti',
                '15': '_Ar_15atm.cti', '50': '_Ar_50atm.cti'
                },
            }
        }
    }

dataset_list = [
    'st_vermeer_1972.xml',
    'st_burcat_1981-1.xml',
    'st_burcat_1981-2.xml',
    'st_burcat_1981-3.xml',
    'st_burcat_1981-4.xml',
    'st_burcat_1981-5.xml',
    'st_ciezki_1993-1.xml',
    'st_ciezki_1993-2.xml',
    'st_ciezki_1993-3.xml',
    'st_ciezki_1993-4.xml',
    'st_ciezki_1993-5.xml',
    'st_ciezki_1993-6.xml',
    'st_ciezki_1993-7.xml',
    'st_ciezki_1993-8.xml',
    'st_ciezki_1993-9.xml',
    'st_fieweger_1997.xml',
    'st_colket_2001-1.xml',
    'st_colket_2001-2.xml',
    'st_horning_2002-1.xml',
    'st_horning_2002-10.xml',
    'st_horning_2002-11.xml',
    'st_horning_2002-12.xml',
    'st_horning_2002-13.xml',
    'st_horning_2002-14.xml',
    'st_horning_2002-2.xml',
    'st_horning_2002-3.xml',
    'st_horning_2002-4.xml',
    'st_horning_2002-5.xml',
    'st_horning_2002-6.xml',
    'st_horning_2002-7.xml',
    'st_horning_2002-8.xml',
    'st_horning_2002-9.xml',
    'st_gauthier_2004-1.xml',
    'st_gauthier_2004-2.xml',
    'st_gauthier_2004-3.xml',
    'st_gauthier_2004-4.xml',
    'st_smith_2005-1.xml',
    'st_smith_2005-2.xml',
    'st_smith_2005-3.xml',
    'st_smith_2005-4.xml',
    'st_smith_2005-5.xml',
    'st_herzler_2005-1.xml',
    'st_herzler_2005-2.xml',
    'st_herzler_2005-3.xml',
    'st_herzler_2005-4.xml',
    'st_sakai_2007.xml',
    'st_shen_2009-1.xml',
    'st_shen_2009-2.xml',
    'st_shen_2009-3.xml',
    'st_shen_2009-4.xml',
    'st_shen_2009-5.xml',
    'st_shen_2009-6.xml',
    'st_hartmann_2011-1.xml',
    'st_hartmann_2011-2.xml',
    'st_vandersickel_2012-1.xml',
    'st_vandersickel_2012-2.xml',
    'st_vandersickel_2012-3.xml',
    'st_vandersickel_2012-4.xml',
    'st_vandersickel_2012-5.xml',
    'st_vandersickel_2012-6.xml',
    'st_vandersickel_2012-7.xml',
    'rcm_karwat_2013-1.xml',
    'rcm_karwat_2013-2.xml',
    'rcm_karwat_2013-3.xml'
    ]

num_points_per_set = np.zeros(len(dataset_list))
error_func_sets = np.zeros(len(dataset_list))
dev_func_sets = np.zeros(len(dataset_list))
standard_dev_sets = np.zeros(len(dataset_list))

output = {'model': mech_filename, 'datasets': []}

# Loop through all datasets
for idx_set, dataset in enumerate(dataset_list):

    dataset_meta = {'dataset': dataset, 'dataset_id': idx_set}

    # Create individual simulation cases for each datapoint in this dataset
    properties = parse_files.read_experiment(os.path.join(data_path, dataset))
    simulations = parse_files.create_simulations(properties)

    ignition_delays_exp = np.zeros(len(simulations))
    ignition_delays_sim = np.zeros(len(simulations))

    num_points_per_set[idx_set] = len(ignition_delays_exp)

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

    standard_dev_sets[idx_set] = standard_dev
    dataset_meta['standard deviation'] = standard_dev

    # Need to check if Ar or He in reactants, and if so skip this dataset (for now)
    if ('Ar' in properties['composition'] and
        'Ar' not in model_spec_key[mech_filename]
        ) or ('N2' in properties['composition'] and
              'N2' not in model_spec_key[mech_filename]
              ):
        print('Warning: Ar or N2 in dataset, but not in model. Skipping.')
        error_func_sets[idx_set] = np.nan
        continue

    # use available number of processors minus one, or one process if single core
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1 or 1)

    # setup all cases
    jobs = []
    for idx, sim in enumerate(simulations):
        # special treatment based on pressure for Princeton model
        if mech_filename == 'Princeton-2009':
            from test_kinetic_models.simulation import to_pascal
            pres = np.mean(sim.properties['pressure'].value)
            # to Pascal
            pres *= to_pascal[sim.properties['pressure'].units]
            # to atm
            pres /= ct.one_atm
            # choose closest pressure
            nums = np.array([1., 3., 6., 9., 12., 15., 50.])
            mechs = ['_1atm.cti', '_3atm.cti', '_6atm.cti', '_9atm.cti',
                     '_12atm.cti', '_15atm.cti', '_50atm.cti'
                     ]
            i = np.argmin(np.abs(nums - pres))

            model_file = os.path.join(model_path, mech_filename + mechs[i])
        else:
            model_file = os.path.join(model_path, mech_filename)

        jobs.append([sim, idx, model_file,
                     model_spec_key[mech_filename], 'results'
                     ])

    # run all cases
    jobs = tuple(jobs)
    results = pool.map(simulation_worker, jobs)

    # not adding more proceses, and ensure all finished
    pool.close()
    pool.join()

    #for idx, sim in enumerate(simulations):
    #    sim.run_case(idx, path='results')

    # process results to get ignition delays
    #for idx, sim in enumerate(simulations):
    #    sim.process_results()

    #    ignition_delays_exp[idx] = sim.properties['ignition delay']
    #    ignition_delays_sim[idx] = sim.properties['simulated ignition delay']

    dataset_meta['datapoints'] = []

    for idx, sim in enumerate(results):
        sim.process_results()

        ignition_delays_exp[idx] = sim.properties['ignition delay']
        ignition_delays_sim[idx] = sim.properties['simulated ignition delay']

        dataset_meta['datapoints'].append(
            {'experimental ignition delay': ignition_delays_exp[idx],
             'simulated ignition delay': ignition_delays_sim[idx],
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

print('standard deviation: ' + repr(np.nanstd(error_func_sets)))

# Absolute deviation function
abs_dev_func = np.nanmean(dev_func_sets)
print('absolute deviation function: ' + repr(abs_dev_func))

output['average error function'] = error_func
output['error function standard deviation'] = np.nanstd(error_func_sets)
output['average deviation function'] = abs_dev_func

# Write data to JSON file
with open(splitext(basename(mech_filename))[0] + '-results.json', 'w') as f:
    json.dump(output, f)
