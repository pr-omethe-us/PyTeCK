"""PyTeCK: validation of chemical kinetic models using ChemKED experimental data."""

from ._version import __version__  # noqa: F401
from .eval_model import evaluate_model # noqa: F401

from cantera import suppress_thermo_warnings

# Avoid long warnings from Cantera about thermodynamic polynomials
suppress_thermo_warnings()
