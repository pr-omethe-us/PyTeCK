# run case work: needs to include all temperatures in file (or more?)
# initalize 3 parameters in set-up??
# run the actual simulation
# store the data in hdf table??




# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import os
from collections import namedtuple
import warnings
import numpy

# Related modules
try:
    import cantera as ct
    ct.suppress_thermo_warnings()
except ImportError:
    print("Error: Cantera must be installed.")
    raise
try:
    import tables
except ImportError:
    print('PyTables must be installed')
    raise

# Local imports
from .utils import units

class JSR_Simulation(object):
    """Class for jet-stirred reactor simulations."""

    def __init__(self, kind, apparatus, meta, properties):
        """Initialize simulation case.

        :param kind: Kind of experiment (e.g., 'species profile')
        :type kind: str
        :param apparatus: Type of apparatus ('jet-stirred reactor')
        :type apparatus: str
        :param meta: some metadata for this case
        :type meta: dict
        :param properties: set of properties for this case
        :type properties: pyked.chemked.DataPoint
        """
        self.kind = kind
        self.apparatus = apparatus
        self.meta = meta
        self.properties = properties

    def setup_case(self, model_file, species_key, path=''):
        """Sets up the simulation case to be run.

        :param str model_file: Filename for Cantera-format model
        :param dict species_key: Dictionary with species names for `model_file`
        :param str path: Path for data file
        """
        # Establishes the model
        self.gas = ct.Solution(model_file)

        # Set max simulation time, pressure valve coefficient, and max pressure rise
        # These could be set to something in ChemKED file, but haven't seen these specified at all....
        self.maxsimulationtime = 50
        self.pressurevalcof = 0.01
        self.maxpressurerise = 0.01
        
        # Reactor volume needed in m^3 for Cantera
        self.apparatus.volume.ito('m^3')
        
        # Residence time needed in s for Cantera
        self.apparatus.restime.ito('s')

        # Initial temperature needed in Kelvin for Cantera
        self.properties.temperature.ito('kelvin')

        # Initial pressure needed in Pa for Cantera
        self.properties.pressure.ito('pascal')

        # convert reactant names to those needed for model
        reactants = [species_key[self.properties.composition[spec].species_name] + ':' +
                     str(self.properties.composition[spec].amount.magnitude)
                     for spec in self.properties.composition
                     ]
        reactants = ','.join(reactants)

        # need to extract values from quantity or measurement object
        if hasattr(self.properties.temperature, 'value'):
            temp = self.properties.temperature.value.magnitude
        elif hasattr(self.properties.temperature, 'nominal_value'):
            temp = self.properties.temperature.nominal_value
        else:
            temp = self.properties.temperature.magnitude
        if hasattr(self.properties.pressure, 'value'):
            pres = self.properties.pressure.value.magnitude
        elif hasattr(self.properties.pressure, 'nominal_value'):
            pres = self.properties.pressure.nominal_value
        else:
            pres = self.properties.pressure.magnitude

        # Reactants given in format for Cantera
        if self.properties.composition_type in ['mole fraction', 'mole percent']:
            self.gas.TPX = temp, pres, reactants
        elif self.properties.composition_type == 'mass fraction':
            self.gas.TPY = temp, pres, reactants
        else:
            raise(BaseException('error: not supported'))
            return
        
        # Upstream and exhaust
        self.fuelairmix = ct.Reservoir(self.gas)
        self.exhaust = ct.Reservoir(self.gas)

        # Ideal gas reactor 
        self.reactor = ct.IdealGasReactor(self.gas, energy='off', volume=self.volume)
        self.massflowcontrol = ct.MassFlowController(upstream=self.fuelairmix,downstream=self.reactor,mdot=self.reactor.mass/self.restime)
        self.pressureregulator = ct.Valve(upstream=self.reactor,downstream=self.exhaust,K=self.pressurevalcof)

        # Create reactor newtork
        self.reactor_net = ct.ReactorNet([self.reactor])

        # Set file for later data file
        file_path = os.path.join(path, self.meta['id'] + '.h5')
        self.meta['save-file'] = file_path

    def run_case(self, restart=False):
        """Run simulation case set up ``setup_case``.

        :param bool restart: If ``True``, skip if results file exists.
        """

        if restart and os.path.isfile(self.meta['save-file']):
            print('Skipped existing case ', self.meta['id'])
            return

        # Save simulation results in hdf5 table format.
        table_def = {'time': tables.Float64Col(pos=0),
                     'temperature': tables.Float64Col(pos=1),
                     'pressure': tables.Float64Col(pos=2),
                     'volume': tables.Float64Col(pos=3),
                     'mole_fractions': tables.Float64Col(
                          shape=(self.reactor.thermo.n_species), pos=4
                          ),
                     }

        with tables.open_file(self.meta['save-file'], mode='w',
                              title=self.meta['id']
                              ) as h5file:

            table = h5file.create_table(where=h5file.root,
                                        name='simulation',
                                        description=table_def
                                        )
            # Row instance to save timestep information to
            timestep = table.row
            # Save initial conditions
            timestep['time'] = self.reactor_net.time
            timestep['temperature'] = self.reactor.T
            timestep['pressure'] = self.reactor.thermo.P
            timestep['volume'] = self.reactor.volume
            timestep['mole_fractions'] = self.reactor.X
            # Add ``timestep`` to table
            timestep.append()

            # Main time integration loop; continue integration while time of
            # the ``ReactorNet`` is less than specified end time.
            while self.reac_net.time < self.maxsimulationtime:
                self.reactor_net.step()

                # Save new timestep information
                timestep['time'] = self.reactor_net.time
                timestep['temperature'] = self.reactor.T
                timestep['pressure'] = self.reactor.thermo.P
                timestep['volume'] = self.reactor.volume
                timestep['mass_fractions'] = self.reactor.X

                # Add ``timestep`` to table
                timestep.append()

            # Write ``table`` to disk
            table.flush()

        print('Done with case ', self.meta['id'])

    def process_results(self):
        """Process integration results to obtain ignition delay.
        """

        # Load saved integration results
        with tables.open_file(self.meta['save-file'], 'r') as h5file:
            # Load Table with Group name simulation
            table = h5file.root.simulation

            time = table.col('time')
            if self.properties.ignition_target == 'pressure':
                target = table.col('pressure')
            elif self.properties.ignition_target == 'temperature':
                target = table.col('temperature')
            else:
                target = table.col('mass_fractions')[:, self.properties.ignition_target]

        # add units to time
        time = time * units.second

        # Analysis for ignition depends on type specified
        if self.properties.ignition_type in ['max', 'd/dt max']:
            if self.properties.ignition_type == 'd/dt max':
                # Evaluate derivative
                target = first_derivative(time.magnitude, target)

            # Get indices of peaks
            ind = detect_peaks(target)

            # Fall back on derivative if max value doesn't work.
            if len(ind) == 0 and self.properties.ignition_type == 'max':
                target = first_derivative(time.magnitude, target)
                ind = detect_peaks(target)

            # something has gone wrong if there is still no peak
            if len(ind) == 0:
                filename = 'target-data-' + self.meta['id'] + '.out'
                warnings.warn('No peak found, dumping target data to ' +
                              filename + ' and continuing',
                              RuntimeWarning
                              )
                numpy.savetxt(filename, numpy.c_[time.magnitude, target],
                              header=('time, target ('+self.properties.ignition_target+')')
                              )
                self.meta['simulated-ignition-delay'] = 0.0 * units.second
                return


            # Get index of largest peak (overall ignition delay)
            max_ind = ind[numpy.argmax(target[ind])]

            # Will need to subtract compression time for RCM
            time_comp = 0.0
            if hasattr(self.properties.rcm_data, 'compression_time'):
                if hasattr(self.properties.rcm_data.compression_time, 'value'):
                    time_comp = self.properties.rcm_data.compression_time.value
                else:
                    time_comp = self.properties.rcm_data.compression_time


            ign_delays = time[ind[numpy.where((time[ind[ind <= max_ind]] - time_comp)
                                              > 0. * units.second
                                             )]] - time_comp
        elif self.properties.ignition_type == '1/2 max':
            # maximum value, and associated index
            max_val = numpy.max(target)
            ind = detect_peaks(target)
            max_ind = ind[numpy.argmax(target[ind])]

            # TODO: interpolate for actual half-max value
            # Find index associated with the 1/2 max value, but only consider
            # points before the peak
            half_idx = (numpy.abs(target[0:max_ind] - 0.5 * max_val)).argmin()
            ign_delays = [time[half_idx]]

            # TODO: detect two-stage ignition when 1/2 max type?

        # Overall ignition delay
        if len(ign_delays) > 0:
            self.meta['simulated-ignition-delay'] = ign_delays[-1]
        else:
            self.meta['simulated-ignition-delay'] = 0.0 * units.second

        # First-stage ignition delay
        if len(ign_delays) > 1:
            self.meta['simulated-first-stage-delay'] = ign_delays[0]
        else:
            self.meta['simulated-first-stage-delay'] = numpy.nan * units.second
