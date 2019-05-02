"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup
from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'pyteck', '_version.py')) as version_file:
    exec(version_file.read())

with open(path.join(here, 'README.md')) as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'CHANGELOG.md')) as changelog_file:
    changelog = changelog_file.read()

with open(path.join(here, 'CITATION.md')) as citation_file:
    citation = citation_file.read()

long_description = readme + '\n\n' + changelog + '\n\n' + citation

install_requires = [
    'pyyaml>=3.12,<4.0',
    'pint>=0.7.2,<0.9',
    'numpy>=1.13.0,<2.0',
    'tables',
    'pyked>=0.4.1',
    'scipy>=1.0.0',
    'cantera>=2.3.0'
]

tests_require = [
    'pytest>=3.2.0',
    'pytest-cov',
]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []

setup(
    name='PyTeCK',
    version=__version__,

    description='Evaluation of chemical kinetic models with experimental data',
    long_description=long_description,
    url='https://github.com/pr-omethe-us/PyTeCK',

    author='Kyle E. Niemeyer',
    author_email='kyle.niemeyer@gmail.com',

    packages=['pyteck', 'pyteck.tests'],
    package_dir={'pyteck': 'pyteck'},
    include_package_data=True,
    package_data={'pyteck': ['tests/*.xml', 'tests/*.yaml', 'tests/dataset_file.txt', 'tests/*.cti']},
    install_requires=install_requires,
    zip_safe=False,

    license='MIT License',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    keywords='chemical_kinetics',

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'pyteck=pyteck.__main__:main',
        ],
    },

    tests_require=tests_require,
    setup_requires=setup_requires,
    #extras_require=extras_require,
)
