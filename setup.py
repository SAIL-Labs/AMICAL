#!/usr/bin/env python

from setuptools import find_packages, setup

project_name = "amical"

setup(
    name=project_name,
    version=1.0,
    packages=find_packages(),
    author='Anthony Soulain',
    author_email='anthony.soulain@sydney.edu.au.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Professional Astronomers',
        'Topic :: High Angular Resolution Astronomy :: Interferometry',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=["matplotlib", "munch", "numpy", "emcee",
                      "astropy", "scipy", "termcolor", "tqdm",
                      "uncertainties", "astroquery",
                      "corner", "h5py", "pytest"],

)
