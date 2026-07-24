Installation
============

Requirements
------------

PyTeCK requires **Python 3.10 or newer** and the following packages, which are
installed automatically with any of the methods below:

- `NumPy <https://numpy.org>`_ (>= 2.0)
- `SciPy <https://scipy.org>`_
- `PyYAML <https://pyyaml.org>`_
- `Pint <https://pint.readthedocs.io>`_
- `PyTables <https://www.pytables.org>`_
- `Cantera <https://cantera.org>`_ (>= 3.0)
- `PyKED <https://pr-omethe-us.github.io/PyKED/>`_ (>= 0.5)

Installing a released version
-----------------------------

The easiest way to install PyTeCK and all of its dependencies is with ``conda``
from the ``conda-forge`` and ``pr-omethe-us`` channels:

.. code-block:: console

   $ conda install -c conda-forge -c pr-omethe-us pyteck

Alternatively, install from `PyPI <https://pypi.org/project/PyTeCK/>`_ with
``pip``:

.. code-block:: console

   $ pip install pyteck

.. note::

   Cantera and PyTables include compiled extensions. ``conda`` ships prebuilt
   binaries for every platform, so it is the most reliable option if a ``pip``
   install fails to build these dependencies.

Installing from source
----------------------

To install the latest development version, clone the repository and install it
with ``pip``:

.. code-block:: console

   $ git clone https://github.com/pr-omethe-us/PyTeCK.git
   $ cd PyTeCK
   $ pip install .

.. _virtual-environments:

Use a virtual environment
-------------------------

We strongly recommend installing PyTeCK into an isolated **virtual environment**
rather than your system Python. This keeps PyTeCK's dependencies (including
specific versions of Cantera and NumPy) from conflicting with other projects and
makes installations reproducible.

Using the standard-library ``venv``:

.. code-block:: console

   $ python -m venv .venv
   $ source .venv/bin/activate        # Windows: .venv\Scripts\activate
   $ pip install pyteck

Using ``conda`` (recommended when installing the compiled dependencies):

.. code-block:: console

   $ conda create -n pyteck python=3.12
   $ conda activate pyteck
   $ conda install -c conda-forge -c pr-omethe-us pyteck

Deactivate the environment at any time with ``deactivate`` (``venv``) or
``conda deactivate`` (``conda``).

.. _developer-installation:

Developer installation
----------------------

If you want to modify PyTeCK, run its tests, or build this documentation,
install it in **editable** mode inside a virtual environment (see
:ref:`virtual-environments`) so that changes to the source take effect
immediately:

.. code-block:: console

   $ git clone https://github.com/pr-omethe-us/PyTeCK.git
   $ cd PyTeCK
   $ python -m venv .venv
   $ source .venv/bin/activate
   $ pip install -e .

Optional development tools are declared as `dependency groups
<https://packaging.python.org/en/latest/specifications/dependency-groups/>`_ in
``pyproject.toml``:

===========  ================================================================
Group        Contents
===========  ================================================================
``test``     ``pytest``, ``pytest-cov`` — run the test suite
``lint``     ``pre-commit``, ``ruff`` — format and lint the code
``docs``     ``sphinx``, ``nbsphinx``, ``ipython`` — build this documentation
``all``      everything above
===========  ================================================================

Install a group with ``pip`` (25.1 or newer)...

.. code-block:: console

   $ pip install -e . --group all

...or with `uv <https://docs.astral.sh/uv/>`_, which installs the default
groups automatically:

.. code-block:: console

   $ uv sync

Running the tests
~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ pytest

To measure coverage of the ``pyteck`` package:

.. code-block:: console

   $ pytest --cov=pyteck

Linting and formatting
~~~~~~~~~~~~~~~~~~~~~~~

PyTeCK uses `ruff <https://docs.astral.sh/ruff/>`_ for linting and formatting,
wired up through `pre-commit <https://pre-commit.com>`_:

.. code-block:: console

   $ pre-commit install       # run the hooks automatically on every commit
   $ ruff check .             # lint
   $ ruff format .            # format

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ cd docs
   $ make html

The rendered HTML is written to ``docs/_build/html``; open ``index.html`` in a
browser to view it.
