# PyTeCK

<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/pr-omethe-us/PyTeCK/main/logo/pyteck-model-data-square-dark.png">
  <img src="https://raw.githubusercontent.com/pr-omethe-us/PyTeCK/main/logo/pyteck-model-data-square.png"
       align="right" width="170" alt="PyTeCK logo: a snake climbing through experimental data points" />
</picture>

[![DOI](https://zenodo.org/badge/53542212.svg)](https://zenodo.org/badge/latestdoi/53542212)
![CI](https://github.com/pr-omethe-us/PyTeCK/actions/workflows/ci.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/pr-omethe-us/PyTeCK/badge.svg?branch=main)](https://coveralls.io/github/pr-omethe-us/PyTeCK)
[![GitHub Release](https://img.shields.io/github/release/pr-omethe-us/PyTeCK.svg)](https://github.com/pr-omethe-us/PyTeCK/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This software package automatically evaluates the performance of a chemical kinetic
model using experimental data given in a specified YAML format.

## Installation

[![PyPI](https://img.shields.io/pypi/v/pyteck)](https://pypi.org/project/pyteck/)
[![Anaconda](https://anaconda.org/pr-omethe-us/pyteck/badges/version.svg)](https://anaconda.org/pr-omethe-us/pyteck)

The easiest way to install PyTeCK is via `conda`. You can install to your environment with

    > conda install -c pr-omethe-us pyteck

which will also handle all the dependencies. Alternatively, you can install from
PyPI with

    > pip install pyteck

If you prefer to install manually, or want a particular version outside of the
tagged releases distributed to those services, you can download the source files
from this repository, navigate to the directory, and install using `pip`:

    > pip install .

## Usage

Once installed, the full list of options can be seen using `pyteck -h` or `pyteck --help`.

## Code of Conduct

In order to have a more open and welcoming community, PyTeCK adheres to a code of
conduct adapted from the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

Please adhere to this code of conduct in any interactions you have in the PyTeCK community.
It is strictly enforced on all official PyTeCK repositories, websites, and resources.
If you encounter someone violating these terms, please let
[@kyleniemeyer](https://github.com/kyleniemeyer) know via email at <kyle.niemeyer@gmail.com>
and we will address it as soon as possible.

## Citation

If you use this package as part of a scholarly publication, please refer to
[CITATION.md](https://github.com/pr-omethe-us/PyTeCK/blob/master/CITATION.md)
for instructions on how to cite this resource directly.

## License

PyTeCK is released under the MIT license; see
[LICENSE](https://github.com/pr-omethe-us/PyTeCK/blob/master/LICENSE) for details.
