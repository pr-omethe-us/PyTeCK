"""

.. moduleauthor:: Kyle Niemeyer <kyle.niemeyer@gmail.com>
"""

# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import os
from collections import namedtuple
import numpy as np

# Related modules
try:
    import cantera as ct
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
from .detect_peaks import detect_peaks

def first_derivative(x, y):
    """Evaluates first derivative using second-order finite differences.

    Uses (second-order) centeral difference in interior and second-order
    one-sided difference at boundaries.

    :param x: Independent variable array
    :type x: numpy.ndarray
    :param y: Dependent variable array
    :type y: numpy.ndarray
    :return: First derivative, :math:`dy/dx`
    :rtype: numpy.ndarray
    """
    return (np.gradient(y, np.gradient(x), edge_order=2))


def sample_rising_pressure(time_end, init_pres, freq, pressure_rise_rate):
    """Samples pressure for particular frequency assuming linear rise.

    :param float time_end: End time of simulation in s
    :param float init_pres: Initial pressure
    :param float freq: Frequency of sampling, in Hz
    :param float pressure_rise_rate: Pressure rise rate, in s^-1
    :return: List of times and pressures
    :rtype: list of np.ndarray
    """
    times = np.arange(0.0, time_end + (1.0 / freq), (1.0 / freq))
    pressures = init_pres * (pressure_rise_rate * times + 1.0)
    return [times, pressures]


def create_volume_history(mech, temp, pres, reactants, pres_rise, time_end):
    """Constructs a volume profile based on intiial conditions and pressure rise.

    :param str mech: Cantera-format mechanism file
    :param float temp: Initial temperature in K
    :param float pres: Initial pressure in Pa
    :param str reactants: Reactants composition in mole fraction
    :param float pres_rise: Pressure rise rate, in s^-1
    :param float time_end: End time of simulation in s
    :return: List of times and volumes
    :rtype: list of np.ndarray
    """
    gas = ct.Solution(mech)
    gas.TPX = temp, pres, reactants
    initial_entropy = gas.entropy_mass
    initial_density = gas.density

    # Sample pressure at 20 kHz
    freq = 2.0e4
    [times, pressures] = sample_rising_pressure(time_end, pres, freq, pres_rise)

    # Calculate volume profile based on pressure
    volumes = np.zeros((len(pressures)))
    for i, p in enumerate(pressures):
        gas.SP = initial_entropy, p
        volumes[i] = initial_density / gas.density

    return [times, volumes]


class VolumeProfile(object):
    """Set the velocity of reactor moving wall via specified volume profile.

    The initialization and calling of this class are handled by the
    `Func1
    <http://cantera.github.io/docs/sphinx/html/cython/zerodim.html#cantera.Func1>`_
    interface of Cantera.

    Based on ``VolumeProfile`` implemented in Bryan W. Weber's
    `CanSen <http://bryanwweber.github.io/CanSen/>`
    """

    def __init__(self, properties):
        """Set the initial values of the arrays from the input keywords.

        The time and volume are read from the input file and stored in
        the ``properties`` dictionary. The velocity is calculated by
        assuming a unit area and using central differences. This function is
        only called once when the class is initialized at the beginning of a
        problem so it is efficient.

        :param dict properties: Dictionary of properties read from input file
        """

        # The time and volume are each stored as a ``numpy.array`` in the
        # properties dictionary. The volume is normalized by the first volume
        # element so that a unit area can be used to calculate the velocity.
        self.times = properties['time'].magnitude
        volumes = (properties['volume'].magnitude /
                   properties['volume'].magnitude[0]
                   )

        # The velocity is calculated by the second-order central differences.
        self.velocity = first_derivative(self.times, volumes)

    def __call__(self, time):
        """Return (interpolated) velocity when called during a time step.

        :param float time: Current simulation time in seconds
        :return: Velocity in meters per second
        :rtype: float
        """
        return np.interp(time, self.times, self.velocity, left=0., right=0.)


class PressureRiseProfile(VolumeProfile):
    r"""Set the velocity of reactor moving wall via specified pressure rise.

    The initialization and calling of this class are handled by the
    `Func1 <http://cantera.github.io/docs/sphinx/html/cython/zerodim.html#cantera.Func1>`_
    interface of Cantera.

    The approach used here is based on that discussed by Chaos and Dryer,
    "Chemical-kinetic modeling of ignition delay: Considerations in
    interpreting shock tube data", *Int J Chem Kinet* 2010 42:143-150,
    `doi:10.1002/kin.20471 <http://dx.doi.org/10.1002/kin.20471`.
    A time-dependent polytropic state change is emulated by determining volume
    as a function of time, via a constant linear pressure rise :math:`A`
    (given as a percentage of the initial pressure):

    .. math::
       \frac{dv}{dt} &= -\frac{1}{\gamma} \frac{v(t)}{P(t)} \frac{dP}{dt} \\
       v(t) &= \frac{1}{\rho} \left[ \frac{P(t)}{P_0} \right]^{-1 / \gamma}

       \frac{dP}{dt} &= A P_0 \\
       \therefore P(t) &= P_0 (A t + 1)

       \frac{dv}{dt} = -A \frac{1}{\rho \gamma} (A t + 1)^{-1 / \gamma}

    The expression for :math:`\frac{dv}{dt}` can then be used directly for
    the ``Wall`` velocity.
    """

    def __init__(self, mech_filename, initial_temp, initial_pres,
                 reactants, pressure_rise, time_end
                 ):
        """Set the initial values of properties needed for velocity.

        :param str mech_filename: Cantera-format mechanism
        :param float initial_temp: Initial temperature in K
        :param float initial_pres: Initial pressure in Pa
        :param str reactants: Reactants composition in mole fraction
        :param float pres_rise: Pressure rise rate in s^-1
        :param float time_end: End time of simulation in s
        """

        [self.times, volumes] = create_volume_history(
                    mech_filename, initial_temp, initial_pres,
                    reactants, pressure_rise, time_end
                    )

        # Calculate velocity by second-order finite difference
        self.velocity = first_derivative(self.times, volumes)


class Simulation(object):
    """Class for ignition delay simulations."""

    def __init__(self, kind, properties, ign_target, ign_type,
                 ign_target_val=None, ign_target_unit=None):
        """Initialize simulation case.

        :param kind: Kind of experiment
        :type kind: str
        :param properties: set of physical properties for experiment
        :type properties: dict
        :param ign_target: physical property measured to detect ignition
        :type ign_target: str
        :param ign_type: feature of measured physical property for ignition
        :type ign_type: str
        :param pint.Quantity ign_target_val: Ignition target
        """
        self.kind = kind
        self.properties = properties
        self.ignition_target = ign_target
        self.ignition_type = ign_type
        self.ignition_target_value = ign_target_val
        self.ignition_target_unit = ign_target_unit

    def setup_case(self, mechanism_filename, species_key):
        """Sets up the simulation case to be run.

        :param str mechanism_filename: Cantera-format mechanism
        :param dict species_key: Dictionary with species names for `mechanism_filename`
        """

        self.gas = ct.Solution(mechanism_filename)

        # Convert ignition delay to seconds
        self.properties['ignition-delay'] = self.properties['ignition-delay'].to('second')

        # Set end time of simulation to 100 times the experimental ignition delay
        self.time_end = 100 * self.properties['ignition-delay'].magnitude

        # Initial temperature needed in Kelvin for Cantera
        initial_temp = self.properties['temperature'].to('kelvin').magnitude

        # Initial pressure needed in Pa for Cantera
        initial_pres = self.properties['pressure'].to('pascal').magnitude

        # Initial composition stored in ``properties`` dictionary as dictionary
        # with internal species names as keys and amounts as values.
        # Need to convert to mechanism-specific species name, then format into
        # string with `spec:val` items joined by commas for Cantera
        #reactants = ','.join(self.properties['composition'])
        reactants = [species_key[k] + ':' + str(v)
                     for k, v in self.properties['composition'].items()
                     ]
        reactants = ','.join(reactants)
        self.gas.TPX = initial_temp, initial_pres, reactants

        # Create non-interacting ``Reservoir`` on other side of ``Wall``
        env = ct.Reservoir(ct.Solution('air.xml'))

        # All reactors are ``IdealGasReactor`` objects
        self.reac = ct.IdealGasReactor(self.gas)
        if self.kind == 'ST' and 'pressure-rise' not in self.properties:
            # Shock tube modeled by constant UV
            self.wall = ct.Wall(self.reac, env, A=1.0, velocity=0)

        elif self.kind == 'ST' and 'pressure-rise' in self.properties:
            # Shock tube modeled by constant UV with isentropic compression

            # Need to convert pressure rise units to seconds
            pres_rise = self.properties['pressure-rise']
            self.properties['pressure-rise'] = pres_rise.to('1 / second')

            self.wall = ct.Wall(self.reac, env, A=1.0,
                                velocity=PressureRiseProfile(
                                    mechanism_filename,
                                    initial_temp,
                                    initial_pres,
                                    reactants,
                                    self.properties['pressure-rise'].magnitude,
                                    self.time_end
                                    )
                                )

        elif self.kind == 'RCM' and 'volume' not in self.properties:
            # Rapid compression machine modeled by constant UV
            self.wall = ct.Wall(self.reac, env, A=1.0, velocity=0)

        elif self.kind == 'RCM' and 'volume' in self.properties:
            # Rapid compression machine modeled with volume-time history

            # First convert time units if necessary
            self.properties['time'] = self.properties['time'].to('second')

            self.wall = ct.Wall(self.reac, env, A=1.0,
                                velocity=VolumeProfile(self.properties)
                                )

        # Number of solution variables is number of species + mass,
        # volume, temperature
        self.n_vars = self.reac.kinetics.n_species + 3

        # Create ``ReactorNet`` newtork
        self.reac_net = ct.ReactorNet([self.reac])

        # Set maximum time step based on volume-time history, if present
        if 'time' in self.properties:
            # Minimum difference between volume profile times
            min_time = np.min(np.diff(self.properties['time'].magnitude))
            self.reac_net.set_max_time_step(min_time)

        # Check if species ignition target, that species is present.
        if self.ignition_target not in ['pressure', 'temperature']:
            # Other targets are species
            spec = self.ignition_target

            # Try finding species in upper- and lower-case
            try_list = [spec, spec.lower()]

            # If excited radical, may need to fall back to nonexcited species
            if spec[-1] == '*':
                try_list += [spec[:-1], spec[:-1].lower()]

            ind = None
            for sp in try_list:
                try:
                    ind = self.gas.species_index(sp)
                    break
                except ValueError:
                    pass

            if ind:
                self.ignition_target = ind
            else:
                print('Warning: ' + spec + ' not found in model; '
                      'falling back on pressure.'
                      )
                self.ignition_target = 'pressure'
                self.ignition_type = 'd/dt max'

    def run_case(self, idx, path=None):
        """Run simulation case set up ``setup_case``.

        :param int idx: Simulation case identifier
        :param str path: Path for data file
        """

        # Save simulation results in hdf5 table format.
        table_def = {'time': tables.Float64Col(pos=0),
                     'temperature': tables.Float64Col(pos=1),
                     'pressure': tables.Float64Col(pos=2),
                     'volume': tables.Float64Col(pos=3),
                     'mass-fractions': tables.Float64Col(
                          shape=(self.reac.thermo.n_species), pos=4
                          ),
                     }

        file_path = os.path.join(path, self.properties['id'] + '.h5')
        self.properties['save-file'] = file_path

        with tables.open_file(self.properties['save-file'], mode='w',
                              title=self.properties['id']
                              ) as h5file:

            table = h5file.create_table(where=h5file.root,
                                        name='simulation',
                                        description=table_def
                                        )
            # Row instance to save timestep information to
            timestep = table.row
            # Save initial conditions
            timestep['time'] = self.reac_net.time
            timestep['temperature'] = self.reac.T
            timestep['pressure'] = self.reac.thermo.P
            timestep['volume'] = self.reac.volume
            timestep['mass-fractions'] = self.reac.Y
            # Add ``timestep`` to table
            timestep.append()

            # Main time integration loop; continue integration while time of
            # the ``ReactorNet`` is less than specified end time.
            while self.reac_net.time < self.time_end:
                self.reac_net.step(self.time_end)

                # Interpolate to end time if step took us beyond that point
                if self.reac_net.time > self.time_end:
                    xp = [prev_time, self.reac_net.time]

                    timestep['time'] = self.time_end
                    fp = [prev_temp, self.reac.T]
                    timestep['temperature'] = np.interp(self.time_end, xp, fp)
                    fp = [prev_pres, self.reac.thermo.P]
                    timestep['pressure'] = np.interp(self.time_end, xp, fp)
                    fp = [prev_vol, self.reac.volume]
                    timestep['volume'] = np.interp(self.time_end, xp, fp)
                    mass_fracs = np.zeros(self.reac.Y.size)
                    for i in range(mass_fracs.size):
                        fp = [prev_mass_frac[i], self.reac.Y[i]]
                        mass_fracs[i] = np.interp(self.time_end, xp, fp)
                    timestep['mass-fractions'] = mass_fracs
                else:
                    # Save new timestep information
                    timestep['time'] = self.reac_net.time
                    timestep['temperature'] = self.reac.T
                    timestep['pressure'] = self.reac.thermo.P
                    timestep['volume'] = self.reac.volume
                    timestep['mass-fractions'] = self.reac.Y

                # Add ``timestep`` to table
                timestep.append()

                # Save values for next step in case of interpolation needed
                prev_time = self.reac_net.time
                prev_temp = self.reac.T
                prev_pres = self.reac.thermo.P
                prev_vol = self.reac.volume
                prev_mass_frac = self.reac.Y

            # Write ``table`` to disk
            table.flush()

        print('Done with case', idx)

    def process_results(self):
        """Process integration results to obtain ignition delay.
        """

        # Load saved integration results
        with tables.open_file(self.properties['save-file'], 'r') as h5file:
            # Load Table with Group name simulation
            table = h5file.root.simulation

            time = table.col('time')
            if self.ignition_target == 'pressure':
                target = table.col('pressure')
            elif self.ignition_target == 'temperature':
                target = table.col('temperature')
            else:
                target = table.col('mass-fractions')[:, self.ignition_target]

        # Analysis for ignition depends on type specified
        if self.ignition_type == 'd/dt max':
            # Evaluate derivative
            target = first_derivative(time, target)

        # Get indices of peaks
        ind = detect_peaks(target)

        # Fall back on derivative if max value doesn't work.
        if len(ind) == 0 and self.ignition_type == 'max':
            target = first_derivative(time, target)
            ind = detect_peaks(target)

        # Get index of largest peak (overall ignition delay)
        max_ind = ind[np.argmax(target[ind])]

        # add units to time
        time *= units.second

        # Will need to subtract compression time for RCM
        time_comp = 0.0
        if 'compression-time' in self.properties:
            time_comp = self.properties['compression-time']

        ign_delays = time[ind[np.where((time[ind[ind <= max_ind]] -
                                       time_comp) > 0
                                       )]] - time_comp
        # Overall ignition delay
        if len(ign_delays) > 0:
            self.properties['simulated ignition delay'] = ign_delays[-1]
        else:
            self.properties['simulated ignition delay'] = 0.0 * units.second

        # First-stage ignition delay
        if len(ign_delays) > 1:
            self.properties['simulated first-stage delay'] = ign_delays[0]
        else:
            self.properties['simulated first-stage delay'] = np.nan * units.second
