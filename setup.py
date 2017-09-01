"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
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

desc = readme + '\n\n' + changelog + '\n\n' + citation
try:
    import pypandoc
    long_description = pypandoc.convert_text(desc, 'rst', format='md')
    with open(path.join(here, 'README.rst'), 'w') as rst_readme:
        rst_readme.write(long_description)
except (ImportError, OSError, IOError):
    long_description = desc

install_requires = [
    'pyyaml>=3.12,<4.0',
    'pint>=0.7.2',
    'numpy>=1.13.0',
    'tables',
    'pyked>=0.2.1',
    'scipy>=0.19.0',
]

tests_require = [
    'pytest>=3.0.1',
    'pytest-cov',
]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []

setup(
    name='PyTeCK',
    version=__version__,

    description='Evaluation of chemical kinetic models with experimental data',
    long_description=long_description,
    url='https://github.com/kyleniemeyer/PyTeCK',

    author='Kyle E. Niemeyer',
    author_email='kyle.niemeyer@gmail.com',

    packages=['pyteck', 'pyteck.tests'],
    package_dir={'pyteck': 'pyteck'},
    include_package_data=True,
    package_data={'pyteck': ['tests/*.xml', 'tests/*.yaml', 'tests/dataset_file.txt']},
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
