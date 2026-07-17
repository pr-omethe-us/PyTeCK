import os
from pathlib import Path
from tempfile import TemporaryDirectory

import cantera as ct
import numpy as np
import pytest
import tables
from pyked.chemked import ChemKED, TimeHistory
from scipy.special import erf

from pyteck import simulation
from pyteck.eval_model import create_simulations
from pyteck.simulation import HomogeneousReactorSimulation
from pyteck.utils import units

HERE = Path(__file__).parent


def assert_physically_valid_results(mechanism_filename, time_end, table):
    """Check that saved integration results are physically valid.

    Rather than asserting exact final-state values (which depend on the Cantera
    and mechanism versions and on the exact integrator stepping), verify
    physical invariants that any correct simulation must satisfy.
    """
    # Ensure exact columns present
    assert set(["time", "temperature", "pressure", "volume", "mass_fractions"]) == set(
        table.colnames
    )

    time = table.col("time")
    temperature = table.col("temperature")
    pressure = table.col("pressure")
    mass_fractions = table.col("mass_fractions")

    # Time increases monotonically, and the integration reached the specified
    # end time (a single internal step of overshoot is expected and harmless).
    assert np.all(np.diff(time) > 0.0)
    assert time_end <= time[-1] <= 1.02 * time_end

    # Heat release occurred: the final temperature is above the initial value.
    assert temperature[-1] > temperature[0]

    # Final mass fractions are physical: they sum to one, with no meaningful
    # negative values (tiny negatives are integrator round-off).
    final_mass_fractions = mass_fractions[-1]
    assert np.isclose(np.sum(final_mass_fractions), 1.0, atol=1e-8)
    assert np.all(final_mass_fractions >= -1e-10)

    gas = ct.Solution(mechanism_filename)

    # Elements are conserved between the initial and final states.
    gas.Y = mass_fractions[0]
    initial_elements = [gas.elemental_mass_fraction(e) for e in gas.element_names]
    gas.Y = final_mass_fractions
    final_elements = [gas.elemental_mass_fraction(e) for e in gas.element_names]
    assert np.allclose(initial_elements, final_elements, atol=1e-10)

    # The reactor ran long enough to reach a fully-reacted, steady state: the
    # final composition is the chemical-equilibrium composition for its own
    # final temperature and pressure.
    gas.TPY = temperature[-1], pressure[-1], final_mass_fractions
    final_composition = gas.Y.copy()
    gas.equilibrate("TP")
    assert np.allclose(gas.Y, final_composition, atol=1e-4)


class TestSampleRisingPressure:
    """ """

    def test_sample_pressure_no_rise(self):
        """Test that pressure sampled correctly with no rise."""
        time_end = 10.0
        pres = 1.0
        pres_rise = 0.0
        freq = 2.0e4
        times, pressures = HomogeneousReactorSimulation.sample_rising_pressure(
            time_end, pres, freq, pres_rise
        )
        # Check time array
        assert len(times) == int(freq * time_end + 1)
        assert times[-1] == time_end

        # Ensure pressure all equal to initial pressure
        assert np.allclose(pressures, pres)

    def test_sample_pressure_rise(self):
        """Test that pressure sampled correctly with rise."""
        time_end = 10.0
        pres = 1.0
        pres_rise = 0.05
        freq = 2.0e4
        times, pressures = HomogeneousReactorSimulation.sample_rising_pressure(
            time_end, pres, freq, pres_rise
        )
        # Check time array
        assert len(times) == int(freq * time_end + 1)
        assert times[-1] == time_end

        # Ensure final pressure correct, and check constant derivative
        assert np.allclose(pressures[-1], pres * (pres_rise * time_end + 1))
        dpdt = np.gradient(pressures, times, edge_order=2)
        assert np.allclose(dpdt, pres * pres_rise)


class TestCreateVolumeHistory:
    """ """

    def test_volume_profile_no_pressure_rise(self):
        """Ensure constant volume history if zero pressure rise."""
        times, volume = HomogeneousReactorSimulation.create_volume_history(
            "air.yaml", 300.0, ct.one_atm, "N2:1.0", 0.0, 1.0
        )
        # check that end time is correct and volume unchanged
        assert np.isclose(times[-1], 1.0)
        assert np.allclose(volume, 1.0)

    def test_artificial_volume_profile_nitrogen(self):
        """Check correct volume profile for nitrogen mixture."""
        initial_pres = 1.0 * ct.one_atm
        pres_rise = 0.05
        end_time = 1.0
        initial_temp = 300.0
        times, volumes = HomogeneousReactorSimulation.create_volume_history(
            "air.yaml", initial_temp, initial_pres, "N2:1.0", pres_rise, end_time
        )
        # pressure at end time
        end_pres = initial_pres * (pres_rise * end_time + 1.0)

        gas = ct.Solution("air.yaml")
        gas.TPX = initial_temp, initial_pres, "N2:1.0"

        # assume specific heat ratio roughly constant
        gamma = gas.cp / gas.cv
        volume = (end_pres / initial_pres) ** (-1.0 / gamma)

        # check that end time is correct and volume matches expected
        assert np.allclose(times[-1], 1.0)
        assert np.allclose(volume, volumes[-1], rtol=1e-5)


class TestVolumeProfile:
    """ """

    def test_zero_velocity_after_end(self):
        """Ensure volume profile returns zero velocity after end of time series."""
        tmax = 10.0
        times = np.arange(0, tmax, 0.001)
        volumes = np.cos(times)

        volume_history = TimeHistory(
            time=times * units.second, quantity=volumes * units.cm3, type="volume"
        )
        volume_profile = simulation.VolumeProfile(volume_history)

        assert volume_profile(tmax + 1.0) == 0.0

    def test_interpolated_velocity(self):
        """Ensure volume profile returns correct interpolated velocity."""
        tmax = 10.0
        times = np.arange(0, tmax, 0.001)
        volumes = np.cos(times)

        volume_history = TimeHistory(
            time=times * units.second, quantity=volumes * units.cm3, type="volume"
        )
        velocity_profile = simulation.VolumeProfile(volume_history)

        assert np.allclose(velocity_profile(np.pi), -np.sin(np.pi), rtol=1e-7, atol=1e-10)

    def test_real_interpreted_velocity(self):
        """Ensure correct interpreted velocity from RCM example volume history."""
        filename = str(HERE / "testfile_rcm.yaml")
        properties = ChemKED(filename, skip_validation=True)

        velocity_profile = simulation.VolumeProfile(properties.datapoints[0].volume_history)

        times = np.arange(0, 9.7e-2, 1.0e-3)
        volumes = np.array(
            [
                5.47669375000e002,
                5.46608789894e002,
                5.43427034574e002,
                5.38124109043e002,
                5.30700013298e002,
                5.21154747340e002,
                5.09488311170e002,
                4.95700704787e002,
                4.79791928191e002,
                4.61761981383e002,
                4.41610864362e002,
                4.20399162234e002,
                3.99187460106e002,
                3.77975757979e002,
                3.56764055851e002,
                3.35552353723e002,
                3.14340651596e002,
                2.93128949468e002,
                2.71917247340e002,
                2.50705545213e002,
                2.29493843085e002,
                2.08282140957e002,
                1.87070438830e002,
                1.65858736702e002,
                1.44647034574e002,
                1.23435332447e002,
                1.02223630319e002,
                8.10119281915e001,
                6.33355097518e001,
                5.27296586879e001,
                4.91943750000e001,
                4.97137623933e001,
                5.02063762048e001,
                5.06454851923e001,
                5.10218564529e001,
                5.13374097598e001,
                5.16004693977e001,
                5.18223244382e001,
                5.20148449242e001,
                5.21889350372e001,
                5.23536351113e001,
                5.25157124459e001,
                5.26796063730e001,
                5.28476160610e001,
                5.30202402028e001,
                5.31965961563e001,
                5.33748623839e001,
                5.35527022996e001,
                5.37276399831e001,
                5.38973687732e001,
                5.40599826225e001,
                5.42141273988e001,
                5.43590751578e001,
                5.44947289126e001,
                5.46215686913e001,
                5.47405518236e001,
                5.48529815402e001,
                5.49603582190e001,
                5.50642270863e001,
                5.51660349836e001,
                5.52670070646e001,
                5.53680520985e001,
                5.54697025392e001,
                5.55720927915e001,
                5.56749762728e001,
                5.57777790517e001,
                5.58796851466e001,
                5.59797461155e001,
                5.60770054561e001,
                5.61706266985e001,
                5.62600130036e001,
                5.63449057053e001,
                5.64254496625e001,
                5.65022146282e001,
                5.65761642150e001,
                5.66485675508e001,
                5.67208534842e001,
                5.67944133373e001,
                5.68703658198e001,
                5.69493069272e001,
                5.70310785669e001,
                5.71146023893e001,
                5.71978399741e001,
                5.72779572372e001,
                5.73517897984e001,
                5.74167271960e001,
                5.74721573687e001,
                5.75216388520e001,
                5.75759967785e001,
                5.76575701358e001,
                5.78058719368e001,
                5.80849611077e001,
                5.85928651155e001,
                5.94734357453e001,
                6.09310671165e001,
                6.32487551103e001,
                6.68100309742e001,
            ]
        )
        volumes = volumes / volumes[0]
        dVdt = np.gradient(volumes, times, edge_order=2)
        velocities = np.zeros(times.size)
        for i, time in enumerate(times):
            velocities[i] = velocity_profile(time)
        assert np.allclose(dVdt, velocities)


class TestPressureRiseProfile(object):
    """ """

    def test_artificial_volume_profile(self):
        """ """
        init_temp = 300.0
        init_pressure = 1.0 * ct.one_atm
        pressure_rise = 0.05
        end_time = 10.0

        velocity_profile = simulation.PressureRiseProfile(
            "air.yaml", init_temp, init_pressure, "N2:1.0", pressure_rise, end_time
        )

        # Sample pressure
        times, pressures = HomogeneousReactorSimulation.sample_rising_pressure(
            end_time, init_pressure, 2.0e3, pressure_rise
        )

        # Check velocity profile against "theoretical" volume derivative
        gas = ct.Solution("air.yaml")
        gas.TPX = init_temp, init_pressure, "N2:1.0"
        init_entropy = gas.entropy_mass
        velocities = np.zeros(pressures.size)
        dvolumes = np.zeros(pressures.size)
        for i in range(pressures.size):
            gas.SP = init_entropy, pressures[i]
            gamma = gas.cp / gas.cv
            velocities[i] = velocity_profile(times[i])
            dvolumes[i] = (
                (-1.0 / gamma)
                * pressure_rise
                * (pressures[i] / init_pressure) ** ((-1.0 / gamma) - 1.0)
            )

        assert np.allclose(velocities, dvolumes, rtol=1e-3)


class TestGetIgnitionDelay(object):
    """Tests for `get_ignition_delay` function, using fitted curves.

    Artificial profiles generated by fitting a Gaussian curve to temperature derivative profile from
    a Cantera-based autoignition simulation.
    """

    def test_max_species(self):
        """Test using max value for ignition delay."""
        a, b, c = [5.13293528e04, 3.16147043e-01, 1.05018205e-02]
        times = np.linspace(0, 1, 10000)
        mass_fraction = a * np.exp(-(((times - b) / c) ** 2))
        # max value of this occurs when x == b

        ignition_delays = HomogeneousReactorSimulation.get_ignition_delay(
            times, mass_fraction, "species", "max"
        )

        assert np.allclose(ignition_delays[0], b, rtol=1e-4)

    def test_max_derivative(self):
        """Test using maximum derivative of temperature for ignition delay."""
        a, b, c = [5.13293528e04, 3.16147043e-01, 1.05018205e-02]
        d = 1000 + 0.5 * np.sqrt(np.pi) * a * c * erf(b / c)
        times = np.linspace(0, 1, 10000)
        temperature = -0.5 * np.sqrt(np.pi) * a * c * erf((b - times) / c) + d
        # max derivative of this occurs when x == b

        ignition_delays = HomogeneousReactorSimulation.get_ignition_delay(
            times, temperature, "temperature", "d/dt max"
        )

        assert np.allclose(ignition_delays[0], b, rtol=1e-4)

    def test_max_derivative_species(self):
        """Test using max derivative of a species-looking profile."""
        a, b, c = [5.13293528e04, 3.16147043e-01, 1.05018205e-02]
        times = np.linspace(0, 1, 10000)
        mass_fraction = a * np.exp(-(((times - b) / c) ** 2))
        # first inflection point of Gaussian occurs at b - sqrt(1/2)*c
        # so this is where the maximum derivative occurs

        ignition_delays = HomogeneousReactorSimulation.get_ignition_delay(
            times, mass_fraction, "species", "d/dt max"
        )
        assert np.allclose(ignition_delays[0], b - np.sqrt(1 / 2) * c, rtol=1e-4)

    def test_max_selects_largest_peak_not_first(self):
        """'max' returns the largest peak's time, not the earliest peak (issue #22)."""
        times = np.linspace(0, 1, 10000)
        # a small earlier peak at t=0.2 and a larger (max) peak at t=0.6
        target = 1.0 * np.exp(-(((times - 0.2) / 0.02) ** 2)) + 5.0 * np.exp(
            -(((times - 0.6) / 0.02) ** 2)
        )
        ignition_delays = HomogeneousReactorSimulation.get_ignition_delay(
            times, target, "species", "max"
        )
        assert np.allclose(ignition_delays[0], 0.6, atol=1e-3)

    def test_d_dt_max_selects_largest_derivative_not_first(self):
        """'d/dt max' returns the steepest rise's time, not the earliest (issue #22)."""
        times = np.linspace(0, 1, 10000)
        # a gentle earlier rise (small derivative peak near t=0.2) and a steep
        # later rise (largest derivative peak near t=0.6)
        target = 0.2 * (erf((times - 0.2) / 0.05) + 1.0) + 3.0 * (erf((times - 0.6) / 0.01) + 1.0)
        ignition_delays = HomogeneousReactorSimulation.get_ignition_delay(
            times, target, "temperature", "d/dt max"
        )
        assert np.allclose(ignition_delays[0], 0.6, atol=2e-3)

    def test_half_max(self):
        """Test using half maximum value for ignition delay."""
        a, b, c = [5.13293528e04, 3.16147043e-01, 1.05018205e-02]
        times = np.linspace(0, 1, 10000)
        mass_fraction = a * np.exp(-(((times - b) / c) ** 2))
        # value of peak is `a`, so half max is `a/2`
        # `mass_fraction = a/2` at `b - c*np.sqrt(np.log(2))`
        # (peak minus half of the full width and half max, FWHM)

        ignition_delays = HomogeneousReactorSimulation.get_ignition_delay(
            times, mass_fraction, "species", "1/2 max"
        )

        assert np.allclose(ignition_delays[0], b - c * np.sqrt(np.log(2)), rtol=1e-4)

    def test_derivative_max_extrapolated(self):
        """Test using d/dt max extrapolated value for ignition delay."""
        a, b, c = [5.13293528e04, 3.16147043e-01, 1.05018205e-02]
        times = np.linspace(0, 1, 10000)
        mass_fraction = a * np.exp(-(((times - b) / c) ** 2))
        # first inflection point of Gaussian occurs at b - sqrt(1/2)*c
        # so this is where the maximum derivative occurs
        # derivative:
        # df_dt = (-2*a/c**2) * (times - b) * np.exp(-(b - times)**2 / c**2)

        time_max_dfdt = b - np.sqrt(1 / 2) * c
        dfdt_max = (
            (-2 * a / c**2) * (time_max_dfdt - b) * np.exp(-((b - time_max_dfdt) ** 2) / c**2)
        )

        ignition_delays = HomogeneousReactorSimulation.get_ignition_delay(
            times, mass_fraction, "species", "d/dt max extrapolated"
        )
        assert np.allclose(
            ignition_delays[0],
            time_max_dfdt - a * np.exp(-(((time_max_dfdt - b) / c) ** 2)) / dfdt_max,
            rtol=1e-4,
        )

    def test_derivative_max_extrapolated_nonpositive_derivative_returns_zero(self, monkeypatch):
        """Do not extrapolate when the derivative maximum is not positive."""
        times = np.linspace(0, 1, 3)
        target = np.ones_like(times)
        derivative = np.array([-1.0, 0.0, -1.0])

        monkeypatch.setattr(np, "gradient", lambda target, time, edge_order: derivative)

        ignition_delays = HomogeneousReactorSimulation.get_ignition_delay(
            times, target, "species", "d/dt max extrapolated"
        )

        assert ignition_delays[0] == 0.0

    @pytest.mark.parametrize(
        "ignition_type",
        [
            "max",
            "d/dt max",
            "1/2 max",
            "d/dt max extrapolated",
        ],
    )
    def test_flat_signal_returns_zero_without_dumping_target_data(self, ignition_type):
        """Flat signals are expected non-ignitions, not debug-dump cases."""
        times = np.linspace(0, 1, 100)
        target = np.zeros_like(times)

        cwd = Path.cwd()
        with TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                ignition_delays = HomogeneousReactorSimulation.get_ignition_delay(
                    times, target, "species", ignition_type
                )
                dumped_files = list(Path(temp_dir).glob("target-data-*.out"))
            finally:
                os.chdir(cwd)

        assert ignition_delays[0] == 0.0
        assert dumped_files == []

    def test_not_supported_type(self):
        """Test that a non-supported type raises a warning and returns zero."""
        with pytest.warns(
            RuntimeWarning,
            match="Unable to process ignition type min, setting result to 0 and continuing",
        ):
            ignition_delays = HomogeneousReactorSimulation.get_ignition_delay(
                0.0, 0.0, "emission", "min"
            )

        assert ignition_delays[0] == 0.0


class TestSimulation:
    """Group of tests on `Simulation` class."""

    def test_shock_tube_setup_case(self):
        """Test that shock tube cases are set up properly."""
        filename = str(HERE / "testfile_st.yaml")
        properties = ChemKED(filename, skip_validation=True)

        # Now create list of Simulation objects
        simulations = create_simulations(filename, properties)

        assert len(simulations) == 5

        mechanism_filename = "gri30.yaml"
        SPEC_KEY = {"H2": "H2", "O2": "O2", "N2": "N2", "Ar": "AR"}

        gas = ct.Solution(mechanism_filename)

        sim = simulations[0]
        sim.setup_case(mechanism_filename, SPEC_KEY)

        init_pressure = 220.0 * units.kilopascal

        assert sim.apparatus == "shock tube"
        assert np.allclose(sim.time_end, 4.7154e-2)
        assert np.allclose(sim.gas.T, 1164.48)
        assert np.allclose(sim.gas.P, init_pressure.to("pascal").magnitude)
        mass_fracs = np.zeros(sim.gas.n_species)
        mass_fracs[sim.gas.species_index(SPEC_KEY["H2"])] = 0.00444
        mass_fracs[sim.gas.species_index(SPEC_KEY["O2"])] = 0.00556
        mass_fracs[sim.gas.species_index(SPEC_KEY["Ar"])] = 0.99
        assert np.allclose(sim.gas.X, mass_fracs)
        # no wall velocity
        assert 0.0 == pytest.approx(sim.reac.walls[0].velocity)
        assert sim.n_vars == gas.n_species + 3

        assert sim.properties.ignition_target == "pressure"
        assert sim.properties.ignition_type == "d/dt max"

        sim = simulations[1]
        sim.setup_case(mechanism_filename, SPEC_KEY)

        assert sim.apparatus == "shock tube"
        assert np.allclose(sim.time_end, 4.4803e-2)
        assert np.allclose(sim.gas.T, 1164.97)
        assert np.allclose(sim.gas.P, init_pressure.to("pascal").magnitude)
        mass_fracs = np.zeros(sim.gas.n_species)
        mass_fracs[sim.gas.species_index(SPEC_KEY["H2"])] = 0.00444
        mass_fracs[sim.gas.species_index(SPEC_KEY["O2"])] = 0.00556
        mass_fracs[sim.gas.species_index(SPEC_KEY["Ar"])] = 0.99
        assert np.allclose(sim.gas.X, mass_fracs)
        # no wall velocity
        assert 0.0 == pytest.approx(sim.reac.walls[0].velocity)
        assert sim.n_vars == gas.n_species + 3

        assert sim.properties.ignition_target == "pressure"
        assert sim.properties.ignition_type == "d/dt max"

    def test_shock_tube_temperature_target_setup_case(self):
        """Test that shock tube case with temperature target set up properly."""
        filename = str(HERE / "testfile_st.yaml")
        properties = ChemKED(filename, skip_validation=True)

        properties.datapoints[0].ignition_type["target"] = "temperature"
        properties.datapoints[1].ignition_type["target"] = "temperature"

        # Now create list of Simulation objects
        simulations = create_simulations(filename, properties)

        mechanism_filename = "gri30.yaml"
        SPEC_KEY = {"H2": "H2", "O2": "O2", "N2": "N2", "Ar": "AR"}

        sim = simulations[0]
        sim.setup_case(mechanism_filename, SPEC_KEY)

        # Only thing different from last test: ignition target is temperature
        assert sim.properties.ignition_target == "temperature"

        sim = simulations[1]
        sim.setup_case(mechanism_filename, SPEC_KEY)

        # Only thing different from last test: ignition target is temperature
        assert sim.properties.ignition_target == "temperature"

    def test_shock_tube_pressure_rise_setup_case(self):
        """Test that shock tube case with pressure rise is set up properly."""
        filename = str(HERE / "testfile_st2.yaml")
        properties = ChemKED(filename, skip_validation=True)

        # Now create list of Simulation objects
        simulations = create_simulations(filename, properties)

        assert len(simulations) == 1

        mechanism_filename = "gri30.yaml"
        SPEC_KEY = {"H2": "H2", "O2": "O2", "N2": "N2", "Ar": "AR"}

        init_temp = 1264.2
        init_pres = 2.18 * ct.one_atm

        gas = ct.Solution(mechanism_filename)

        sim = simulations[0]
        sim.setup_case(mechanism_filename, SPEC_KEY)

        assert sim.apparatus == "shock tube"
        assert np.allclose(sim.time_end, 2.9157e-2)
        assert np.allclose(sim.gas.T, init_temp)
        assert np.allclose(sim.gas.P, init_pres)
        mass_fracs = np.zeros(sim.gas.n_species)
        mass_fracs[sim.gas.species_index(SPEC_KEY["H2"])] = 0.00444
        mass_fracs[sim.gas.species_index(SPEC_KEY["O2"])] = 0.00556
        mass_fracs[sim.gas.species_index(SPEC_KEY["Ar"])] = 0.99
        assert np.allclose(sim.gas.X, mass_fracs)
        assert sim.n_vars == gas.n_species + 3

        # Check constructed velocity profile
        times, volumes = HomogeneousReactorSimulation.create_volume_history(
            mechanism_filename,
            init_temp,
            init_pres,
            "H2:0.00444,O2:0.00566,AR:0.9899",
            0.10 * 1000.0,
            sim.time_end,
        )
        volumes = volumes / volumes[0]
        dVdt = np.gradient(volumes, times, edge_order=2)
        # check initial velocity only
        assert dVdt[0] == pytest.approx(sim.reac.walls[0].velocity, rel=1e-3)

    def test_rcm_setup_case(self):
        """Test that RCM case is set up properly."""
        filename = str(HERE / "testfile_rcm.yaml")
        properties = ChemKED(filename, skip_validation=True)

        # Now create list of Simulation objects
        simulations = create_simulations(filename, properties)

        assert len(simulations) == 1

        mechanism_filename = "gri30.yaml"
        SPEC_KEY = {"H2": "H2", "O2": "O2", "N2": "N2", "Ar": "AR"}

        gas = ct.Solution(mechanism_filename)

        sim = simulations[0]
        sim.setup_case(mechanism_filename, SPEC_KEY)

        assert sim.apparatus == "rapid compression machine"
        assert np.allclose(sim.time_end, 0.1)
        assert np.allclose(sim.gas.T, 297.4)
        assert np.allclose(sim.gas.P, 127722.83)
        mass_fracs = np.zeros(sim.gas.n_species)
        mass_fracs[sim.gas.species_index(SPEC_KEY["H2"])] = 0.12500
        mass_fracs[sim.gas.species_index(SPEC_KEY["O2"])] = 0.06250
        mass_fracs[sim.gas.species_index(SPEC_KEY["N2"])] = 0.18125
        mass_fracs[sim.gas.species_index(SPEC_KEY["Ar"])] = 0.63125
        assert np.allclose(sim.gas.X, mass_fracs)

        times = np.arange(0, 9.7e-2, 1.0e-3)
        volumes = np.array(
            [
                5.47669375000e002,
                5.46608789894e002,
                5.43427034574e002,
                5.38124109043e002,
                5.30700013298e002,
                5.21154747340e002,
                5.09488311170e002,
                4.95700704787e002,
                4.79791928191e002,
                4.61761981383e002,
                4.41610864362e002,
                4.20399162234e002,
                3.99187460106e002,
                3.77975757979e002,
                3.56764055851e002,
                3.35552353723e002,
                3.14340651596e002,
                2.93128949468e002,
                2.71917247340e002,
                2.50705545213e002,
                2.29493843085e002,
                2.08282140957e002,
                1.87070438830e002,
                1.65858736702e002,
                1.44647034574e002,
                1.23435332447e002,
                1.02223630319e002,
                8.10119281915e001,
                6.33355097518e001,
                5.27296586879e001,
                4.91943750000e001,
                4.97137623933e001,
                5.02063762048e001,
                5.06454851923e001,
                5.10218564529e001,
                5.13374097598e001,
                5.16004693977e001,
                5.18223244382e001,
                5.20148449242e001,
                5.21889350372e001,
                5.23536351113e001,
                5.25157124459e001,
                5.26796063730e001,
                5.28476160610e001,
                5.30202402028e001,
                5.31965961563e001,
                5.33748623839e001,
                5.35527022996e001,
                5.37276399831e001,
                5.38973687732e001,
                5.40599826225e001,
                5.42141273988e001,
                5.43590751578e001,
                5.44947289126e001,
                5.46215686913e001,
                5.47405518236e001,
                5.48529815402e001,
                5.49603582190e001,
                5.50642270863e001,
                5.51660349836e001,
                5.52670070646e001,
                5.53680520985e001,
                5.54697025392e001,
                5.55720927915e001,
                5.56749762728e001,
                5.57777790517e001,
                5.58796851466e001,
                5.59797461155e001,
                5.60770054561e001,
                5.61706266985e001,
                5.62600130036e001,
                5.63449057053e001,
                5.64254496625e001,
                5.65022146282e001,
                5.65761642150e001,
                5.66485675508e001,
                5.67208534842e001,
                5.67944133373e001,
                5.68703658198e001,
                5.69493069272e001,
                5.70310785669e001,
                5.71146023893e001,
                5.71978399741e001,
                5.72779572372e001,
                5.73517897984e001,
                5.74167271960e001,
                5.74721573687e001,
                5.75216388520e001,
                5.75759967785e001,
                5.76575701358e001,
                5.78058719368e001,
                5.80849611077e001,
                5.85928651155e001,
                5.94734357453e001,
                6.09310671165e001,
                6.32487551103e001,
                6.68100309742e001,
            ]
        )
        volumes = volumes / volumes[0]
        dVdt = np.gradient(volumes, times, edge_order=2)
        # check initial velocity only
        assert dVdt[0] == pytest.approx(sim.reac.walls[0].velocity)

        assert sim.n_vars == gas.n_species + 3

    def test_shock_tube_run_cases(self):
        """Test that shock tube cases run correctly."""
        # Read experiment file
        filename = str(HERE / "testfile_st.yaml")
        properties = ChemKED(filename, skip_validation=True)

        # Now create list of Simulation objects
        simulations = create_simulations(filename, properties)

        mechanism_filename = "gri30.yaml"
        SPEC_KEY = {"H2": "H2", "O2": "O2", "N2": "N2", "Ar": "AR"}

        # Setup and run the first two simulations
        with TemporaryDirectory() as temp_dir:
            for sim in simulations[:2]:
                sim.setup_case(mechanism_filename, SPEC_KEY, path=temp_dir)
                sim.run_case()

                # check for presence of data file
                assert sim.meta["save-file"].exists()
                with tables.open_file(sim.meta["save-file"], "r") as h5file:
                    table = h5file.root.simulation

                    assert_physically_valid_results(mechanism_filename, sim.time_end, table)

                    # This is a constant-volume, adiabatic shock tube, so the
                    # final state is the constant internal-energy/volume
                    # equilibrium of the initial mixture.
                    gas = ct.Solution(mechanism_filename)
                    gas.TPY = (
                        table.col("temperature")[0],
                        table.col("pressure")[0],
                        table.col("mass_fractions")[0],
                    )
                    gas.equilibrate("UV")
                    assert np.isclose(table.col("temperature")[-1], gas.T, rtol=1e-3)
                    assert np.isclose(table.col("pressure")[-1], gas.P, rtol=1e-3)

    def test_shock_tube_pressure_rise_run_cases(self):
        """Test that shock tube cases with pressure rise run correctly."""
        # Read experiment file
        filename = str(HERE / "testfile_st2.yaml")
        properties = ChemKED(filename, skip_validation=True)

        # Now create list of Simulation objects
        simulations = create_simulations(filename, properties)

        mechanism_filename = "gri30.yaml"
        SPEC_KEY = {"H2": "H2", "O2": "O2", "N2": "N2", "Ar": "AR"}

        # Setup and run each simulation
        with TemporaryDirectory() as temp_dir:
            sim = simulations[0]
            sim.setup_case(mechanism_filename, SPEC_KEY, path=temp_dir)
            sim.run_case()

            # check for presence of data file
            assert sim.meta["save-file"].exists()
            with tables.open_file(sim.meta["save-file"], "r") as h5file:
                table = h5file.root.simulation
                assert_physically_valid_results(mechanism_filename, sim.time_end, table)

    def test_rcm_run_cases(self):
        """Test that RCM case runs correctly."""
        # Read experiment file
        filename = str(HERE / "testfile_rcm.yaml")
        properties = ChemKED(filename, skip_validation=True)

        # Now create list of Simulation objects
        simulations = create_simulations(filename, properties)

        mechanism_filename = "gri30.yaml"
        SPEC_KEY = {"H2": "H2", "O2": "O2", "N2": "N2", "Ar": "AR"}

        # Setup and run each simulation
        with TemporaryDirectory() as temp_dir:
            sim = simulations[0]
            sim.setup_case(mechanism_filename, SPEC_KEY, path=temp_dir)
            sim.run_case()

            # check for presence of data file
            assert sim.meta["save-file"].exists()
            with tables.open_file(sim.meta["save-file"], "r") as h5file:
                table = h5file.root.simulation
                assert_physically_valid_results(mechanism_filename, sim.time_end, table)

    # TODO: add test for restart option

    def test_capitalization_species_target(self):
        """Test that species targets with capitalization not matching model works."""
        filename = str(HERE / "testfile_st2.yaml")
        properties = ChemKED(filename, skip_validation=True)

        # ignition target is OH

        # Now create list of Simulation objects
        simulations = create_simulations(filename, properties)

        mechanism_filename = str(HERE / "h2o2-lowercase.yaml")
        SPEC_KEY = {"H2": "h2", "O2": "o2", "N2": "n2", "Ar": "ar"}

        sim = simulations[0]
        sim.setup_case(mechanism_filename, SPEC_KEY)

        # oh is species index 4
        assert sim.properties.ignition_target == 4

        # now try for uppercase in model and lowercase in file.
        properties = ChemKED(filename, skip_validation=True)
        properties.datapoints[0].ignition_type["target"] = "oh"
        SPEC_KEY = {"H2": "H2", "O2": "O2", "N2": "N2", "Ar": "AR"}
        simulations = create_simulations(filename, properties)
        sim = simulations[0]
        sim.setup_case("h2o2.yaml", SPEC_KEY)

        # oh is species index 4
        assert sim.properties.ignition_target == 4
