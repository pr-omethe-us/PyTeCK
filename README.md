# PyTeCK

[![DOI](https://zenodo.org/badge/53542212.svg)](https://zenodo.org/badge/latestdoi/53542212)
[![Build Status](https://travis-ci.org/kyleniemeyer/PyTeCK.svg?branch=master)](https://travis-ci.org/kyleniemeyer/PyTeCK)
[![Build Status](https://ci.appveyor.com/api/projects/status/a7a3prqgvfg8rr5f?svg=true)](https://ci.appveyor.com/project/kyleniemeyer/pyteck)
[![codecov](https://codecov.io/gh/kyleniemeyer/PyTeCK/branch/master/graph/badge.svg)](https://codecov.io/gh/kyleniemeyer/PyTeCK)
[![Dependency Status](https://dependencyci.com/github/kyleniemeyer/PyTeCK/badge)](https://dependencyci.com/github/kyleniemeyer/PyTeCK)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-contributor%20covenant-green.svg)](http://contributor-covenant.org/version/1/4/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Anaconda](https://anaconda.org/kyleniemeyer/pyteck/badges/version.svg)](https://anaconda.org/kyleniemeyer/pyteck)

This software package automatically evaluates the performance of a chemical kinetic
model using experimental data given in a specified YAML format.

## Installation

The easiest way to install PyTeCK is via `conda`. You can install to your environment with

    > conda install -c kyleniemeyer pyteck

which will also handle all the dependencies. Alternatively, you can install from
PyPI with

    > pip install pyteck

If you prefer to install manually, or want a particular version outside of the
tagged releases distributed to those services, you can download the source files
from this repository, navigate to the directory, and install using either `setuptools`

    > python setup.py install

or `pip`

    > pip install .

`pip` is recommended due to its easy uninstall option (`pip uninstall pyteck`).

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
[CITATION.md](https://github.com/kyleniemeyer/PyTeCK/blob/master/CITATION.md)
for instructions on how to cite this resource directly.

## License

PyTeCK is released under the MIT license; see
[LICENSE](https://github.com/kyleniemeyer/PyTeCK/blob/master/LICENSE) for details.
