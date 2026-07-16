PyTeCK
======

**PyTeCK** (Python-based Testing of Chemical Kinetics) automatically evaluates
the performance of chemical kinetic models against experimental data. It reads
experimental ignition-delay measurements in the `ChemKED
<https://pr-omethe-us.github.io/PyKED/>`_ YAML format, simulates the
corresponding cases with `Cantera <https://cantera.org>`_, and reports how well
a chemical kinetic model reproduces the measurements.

PyTeCK is released under the `MIT license
<https://github.com/pr-omethe-us/PyTeCK/blob/main/LICENSE>`_ and developed openly
on `GitHub <https://github.com/pr-omethe-us/PyTeCK>`_.

Features
--------

- Evaluate a kinetic model against whole datasets of experimental ignition-delay
  data described with `ChemKED <https://pr-omethe-us.github.io/PyKED/>`_.
- Support for shock tube and rapid compression machine (RCM) experiments,
  including non-ideal effects such as pressure rise and volume-time histories.
- Automatic, per-case parallel execution of simulations.
- A quantitative error function summarizing model performance across a dataset.

Quick start
-----------

Install from PyPI (see :doc:`installation` for conda and development installs):

.. code-block:: console

   $ pip install pyteck

Run an evaluation from the command line:

.. code-block:: console

   $ pyteck --model mech.yaml --model-keys spec_keys.yaml --dataset dataset_file.txt

or from Python:

.. code-block:: python

   from pyteck.eval_model import evaluate_model

   output = evaluate_model(
       model_name="mech.yaml",
       spec_keys_file="spec_keys.yaml",
       dataset_file="dataset_file.txt",
       data_path="data",
       model_path="models",
       results_path="results",
   )
   print(output["average error function"])

The :doc:`example` page walks through a complete, runnable hydrogen shock-tube
case. The full list of command-line options is available with ``pyteck --help``.


User guide
----------

.. toctree::
   :maxdepth: 2

   installation
   example

API reference
-------------

.. toctree::
   :maxdepth: 2

   eval_model
   simulation
   detect_peaks
   utils

Citation
--------

If you use PyTeCK in a scholarly publication, please cite it as described in
`CITATION.md
<https://github.com/pr-omethe-us/PyTeCK/blob/main/CITATION.md>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
