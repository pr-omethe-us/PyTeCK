import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy
import pytest
from pyked.chemked import DataPoint

# Local imports
from pyteck import eval_model
from pyteck.utils import units

HERE = Path(__file__).parent


class TestReadDatasetList:
    """Tests for reading the dataset-list file (see issue #12)."""

    def test_skips_blank_lines(self, tmp_path):
        """Blank and whitespace-only lines are dropped, not treated as files."""
        dataset_file = tmp_path / "datasets.txt"
        dataset_file.write_text("first.yaml\n\n   \n\t\nsecond.yaml\n\n")
        assert eval_model.read_dataset_list(dataset_file) == ["first.yaml", "second.yaml"]

    def test_strips_surrounding_whitespace(self, tmp_path):
        """Surrounding whitespace on real entries is stripped."""
        dataset_file = tmp_path / "datasets.txt"
        dataset_file.write_text("  first.yaml  \n\tsecond.yaml\t\n")
        assert eval_model.read_dataset_list(dataset_file) == ["first.yaml", "second.yaml"]


def _make_datapoint(species_amounts, pressure="1.0 atm"):
    """Build a DataPoint with the given species mole fractions and pressure."""
    return DataPoint(
        {
            "temperature": ["1000.0 kelvin"],
            "pressure": [pressure],
            "composition": {
                "kind": "mole fraction",
                "species": [
                    {"species-name": name, "amount": [amount]}
                    for name, amount in species_amounts.items()
                ],
            },
            "ignition-type": None,
        }
    )


class TestSelectVariantSuffix:
    """Tests for model-variant selection from a case (see issue #9).

    ``properties.composition`` is a dict keyed by species name in current PyKED,
    which previously broke the bath-gas selection logic.
    """

    def test_single_bath_gas_present(self):
        """The one designated bath gas present selects its suffix."""
        variant = {"bath gases": {"Ar": "-ar", "He": "-he"}}
        props = _make_datapoint({"H2": 0.1, "O2": 0.05, "Ar": 0.85})
        assert eval_model.select_variant_suffix(variant, props) == "-ar"

    def test_multiple_bath_gases_use_predominant(self):
        """With several bath gases, the most abundant one is chosen."""
        variant = {"bath gases": {"Ar": "-ar", "He": "-he"}}
        props = _make_datapoint({"H2": 0.1, "Ar": 0.2, "He": 0.7})
        assert eval_model.select_variant_suffix(variant, props) == "-he"

    def test_no_designated_bath_gas_falls_back(self):
        """If no designated bath gas is present, any valid suffix is returned."""
        variant = {"bath gases": {"Ar": "-ar", "He": "-he"}}
        props = _make_datapoint({"H2": 0.5, "O2": 0.5})
        assert eval_model.select_variant_suffix(variant, props) in {"-ar", "-he"}

    def test_pressure_selects_closest(self):
        """The pressure variant closest to the case pressure is chosen."""
        variant = {"pressures": {"1": "-1atm", "10": "-10atm", "50": "-50atm"}}
        props = _make_datapoint({"H2": 1.0}, pressure="12.0 atm")
        assert eval_model.select_variant_suffix(variant, props) == "-10atm"

    def test_bath_gas_and_pressure_combined(self):
        """Bath-gas and pressure suffixes are concatenated."""
        variant = {
            "bath gases": {"Ar": "-ar"},
            "pressures": {"1": "-1atm", "10": "-10atm"},
        }
        props = _make_datapoint({"H2": 0.1, "Ar": 0.9}, pressure="9.0 atm")
        assert eval_model.select_variant_suffix(variant, props) == "-ar-10atm"

    def test_empty_variant_is_no_suffix(self):
        """A variant with no bath-gas or pressure maps yields an empty suffix."""
        props = _make_datapoint({"H2": 1.0})
        assert eval_model.select_variant_suffix({}, props) == ""


class TestModelKeyValidation:
    """A missing model entry in the species-keys file is reported clearly (issue #10)."""

    def test_missing_model_key_raises_clear_error(self, tmp_path):
        """An unknown model name fails fast with a helpful message, not a bare KeyError."""
        spec_keys = tmp_path / "spec_keys.yaml"
        spec_keys.write_text("othermodel.yaml:\n    H2: H2\n")
        dataset_file = tmp_path / "datasets.txt"
        dataset_file.write_text("some_data.yaml\n")

        with pytest.raises(KeyError, match="not found in species-keys file"):
            eval_model.evaluate_model(
                model_name="h2o2.yaml",
                spec_keys_file=str(spec_keys),
                dataset_file=str(dataset_file),
                data_path=str(tmp_path),
                model_path="",
                results_path=str(tmp_path / "results"),
                skip_validation=True,
            )


class TestEstimateStandardDeviation:
    """ """

    def test_single_point(self):
        """Check return for single data point."""
        rng = numpy.random.default_rng()
        changing_variable = rng.uniform(size=1)
        dependent_variable = rng.uniform(size=1)

        standard_dev = eval_model.estimate_std_dev(changing_variable, dependent_variable)
        assert standard_dev == eval_model.min_deviation

    def test_two_points(self):
        """Check return for two data points."""
        rng = numpy.random.default_rng()
        changing_variable = rng.uniform(size=2)
        dependent_variable = rng.uniform(size=2)

        standard_dev = eval_model.estimate_std_dev(changing_variable, dependent_variable)
        assert standard_dev == eval_model.min_deviation

    def test_three_points(self):
        """Check return for perfect, linear three data points."""
        changing_variable = numpy.arange(1, 4)
        dependent_variable = numpy.arange(1, 4)

        standard_dev = eval_model.estimate_std_dev(changing_variable, dependent_variable)
        assert standard_dev == eval_model.min_deviation

    def test_normal_dist_noise(self):
        """Check expected standard deviation for normally distributed noise."""
        num = 1000000
        changing_variable = numpy.arange(1, num + 1)
        dependent_variable = numpy.arange(1, num + 1)
        # add normally distributed noise, standard deviation of 1.0
        rng = numpy.random.default_rng()
        noise = rng.normal(0.0, 1.0, num)

        standard_dev = eval_model.estimate_std_dev(changing_variable, dependent_variable + noise)
        assert numpy.isclose(1.0, standard_dev, rtol=1.0e-2)

    def test_repeated_points(self):
        """Check that function correctly handles repeated points with no error."""
        changing_variable = numpy.arange(1, 10)
        dependent_variable = numpy.arange(1, 10)
        changing_variable[1] = changing_variable[0]

        standard_dev = eval_model.estimate_std_dev(changing_variable, dependent_variable)
        assert standard_dev == eval_model.min_deviation


class TestGetChangingVariable:
    """ """

    def test_single_point(self):
        """Check normal behavior for single point."""
        rng = numpy.random.default_rng()
        cases = [
            DataPoint(
                {
                    "pressure": [rng.uniform(size=1) * units("atm")],
                    "temperature": [rng.uniform(size=1) * units("K")],
                    "composition": {
                        "kind": "mole fraction",
                        "species": [{"species-name": "O2", "amount": [1.0]}],
                    },
                    "ignition-type": None,
                }
            )
        ]
        variable = eval_model.get_changing_variable(cases)

        assert len(variable) == 1
        assert variable[0] == cases[0].temperature.magnitude

    def test_temperature_changing(self):
        """Check normal behavior for multiple points with temperature changing."""
        num = 10
        rng = numpy.random.default_rng()
        pressure = rng.uniform(size=1) * units("atm")
        temperatures = rng.uniform(size=num) * units("K")
        cases = []
        for temp in temperatures:
            dp = DataPoint(
                {
                    "pressure": [str(pressure[0])],
                    "temperature": [str(temp)],
                    "composition": {
                        "kind": "mole fraction",
                        "species": [{"species-name": "O2", "amount": [1.0]}],
                    },
                    "ignition-type": None,
                }
            )
            cases.append(dp)

        variable = eval_model.get_changing_variable(cases)

        assert len(variable) == num
        assert numpy.allclose(variable, [c.temperature.magnitude for c in cases])

    def test_pressure_changing(self):
        """Check normal behavior for multiple points with pressure changing."""
        num = 10
        rng = numpy.random.default_rng()
        pressures = rng.uniform(size=num) * units("atm")
        temperature = rng.uniform(size=1) * units("K")
        cases = []
        for pres in pressures:
            dp = DataPoint(
                {
                    "pressure": [str(pres)],
                    "temperature": [str(temperature[0])],
                    "composition": {
                        "kind": "mole fraction",
                        "species": [{"species-name": "O2", "amount": [1.0]}],
                    },
                    "ignition-type": None,
                }
            )
            cases.append(dp)

        variable = eval_model.get_changing_variable(cases)

        assert len(variable) == num
        assert numpy.allclose(variable, [c.pressure.magnitude for c in cases])

    def test_both_changing(self):
        """Check fallback behavior for both properties varying."""
        num = 10
        rng = numpy.random.default_rng()
        pressures = rng.uniform(size=num) * units("atm")
        temperatures = rng.uniform(size=num) * units("K")
        cases = []
        for pres, temp in zip(pressures, temperatures):
            dp = DataPoint(
                {
                    "pressure": [str(pres)],
                    "temperature": [str(temp)],
                    "composition": {
                        "kind": "mole fraction",
                        "species": [{"species-name": "O2", "amount": [1.0]}],
                    },
                    "ignition-type": None,
                }
            )
            cases.append(dp)

        with pytest.warns(
            RuntimeWarning, match="Warning: multiple changing variables. Using temperature."
        ):
            variable = eval_model.get_changing_variable(cases)

        assert len(variable) == num
        assert numpy.allclose(variable, [c.temperature.magnitude for c in cases])


class TestEvalModel:
    """ """

    def test(self):
        """Test overall evaluation of model."""

        cwd = Path.cwd()
        with TemporaryDirectory() as temp_dir:
            # Run from within the temporary directory so any files produced
            # (e.g. the summary results YAML) are contained and cleaned up.
            os.chdir(temp_dir)
            try:
                output = eval_model.evaluate_model(
                    model_name="h2o2.yaml",
                    spec_keys_file=str(HERE / "spec_keys.yaml"),
                    dataset_file=str(HERE / "dataset_file.txt"),
                    data_path=str(HERE),
                    model_path="",
                    results_path=temp_dir,
                    num_threads=1,
                    skip_validation=True,
                )
            finally:
                os.chdir(cwd)

            assert numpy.isclose(output["average error function"], 58.78211242028232, rtol=1.0e-3)
            assert numpy.isclose(output["error function standard deviation"], 0.0, rtol=1.0e-3)
            assert numpy.isclose(
                output["average deviation function"], 7.635983785416241, rtol=1.0e-3
            )
