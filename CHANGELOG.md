# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]
### Added
- GitHub Actions for continuous integration (multi-OS, Python 3.10+) and trusted PyPI publishing, replacing Travis CI and AppVeyor
- `pre-commit` configuration with `ruff` linting and formatting
- Expanded documentation: landing page, installation guide (virtual-environment and developer instructions), and a runnable example notebook comparing experimental and simulated ignition delays
- Regression tests for the bug fixes below
- Added handling of zero peaks detected
- Added handling of "d/dt max extrapolated" ignition type
- Added some initial examples in `examples` directory
- Added tests for ignition delay detection

### Changed
- Modern packaging with `pyproject.toml` (hatchling backend), replacing `setup.py`/`setup.cfg`
- Requires Python 3.10 or newer, with updated dependencies (NumPy 2, Cantera 3, PyKED 0.5)
- Compatible with Cantera 3.x (YAML kinetic models)
- Refactored the simulation module into a `BaseSimulation` base class with concrete subclasses and a factory, to allow additional simulation types in the future
- Switched to `pathlib` throughout and standardized docstrings to NumPy style
- Using warnings module for messages

### Fixed
- Report a clear error when the model is missing from the species-keys file instead of a cryptic `KeyError` (#10)
- Skip blank lines in the dataset-list file (#12)
- Correct bath-gas selection for model variants under the current PyKED composition format (#9)
- Exclude non-igniting cases (zero simulated ignition delay) from the dataset error function, so it no longer becomes infinite (#1, #18)
- Select the largest peak for the `max` and `d/dt max` ignition-delay types (#22)
- Fixed handling of uppercase/lowercase species target
- Fixed bug setting pressure when setting simulation up (had `temp` instead of `pres`)


### Removed
- Removed the ReSpecTh/XML conversion module (`parse_files_XML`); PyKED handles this conversion now
- Removed the orphaned `validation` and `exceptions` modules (used only by the removed XML parser)

## [0.2.4] - 2018-05-31
### Fixed
- Fixed ability to handle ChemKED files with uncertainty for various properties.
- Updated handling of RCM volume history and compression time to PyKED v0.4.1
- Fixed searching through composition dict for Ar and He
- Fixed file-author -> file-authors in test files

### Changed
- Removed interpolation from end of integration (was never really necessary).


## [0.2.3] - 2018-02-07
### Fixed
- Standard deviation calculator now averages any duplicates to avoid an error.

## [0.2.2] - 2017-09-02
### Added
- Adds DOI badge to README and CITATION
- Adds AppVeyor build status badge to README
- Adds restart option to skip existing results files.
- Updates PyKED version requirement and adds optional validation skipping

### Fixed
- Fixes ignition delay detection for 1/2 max type (only one value possible, rather than list)
- Fixes test for detecting peaks with min distance
- Ensure time has units when 1/2 max target
- Fixed handling of models with variants

### Changed
- Simulation input parameters now change units in place
- Simulation input composition uses ChemKED Cantera functions

## [0.2.1] - 2017-04-14
### Added
- Adds AppVeyor build for Windows conda packages
- Adds CONTRIBUTING guide

## [0.2.0] - 2017-04-13
### Added
- Adds initial web documentation using Sphinx/Doctr
- Deploys conda and PyPI packages with tagged releases

### Changed
- Major changes for compatibility with PyKED package and newer ChemKED format

## [0.1.0] - 2016-07-12
### Added
- First published version of PyTeCK.
- Supports validation using both shock tube and RCM experimental data in ChemKED format, but RCM not fully functional.

 [Unreleased]: https://github.com/kyleniemeyer/PyTeCK/compare/v0.2.4...HEAD
 [0.2.4]: https://github.com/kyleniemeyer/PyTeCK/compare/v0.2.3...0.2.4
 [0.2.3]: https://github.com/kyleniemeyer/PyTeCK/compare/v0.2.2...0.2.3
 [0.2.2]: https://github.com/kyleniemeyer/PyTeCK/compare/v0.2.1...0.2.2
 [0.2.1]: https://github.com/kyleniemeyer/PyTeCK/compare/v0.2.0...0.2.1
 [0.2.0]: https://github.com/kyleniemeyer/PyTeCK/compare/v0.1...0.2.0
 [0.1.0]: https://github.com/kyleniemeyer/PyTeCK/compare/e99f757b7ea644065a0ee65ce86dbfb8f404be60...v0.1
