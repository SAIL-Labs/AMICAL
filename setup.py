#!/usr/bin/env python

from setuptools import setup

project_name = "miamis"

setup(
    name=project_name,
    version=0.1,
    packages=['miamis'],
    author='Anthony Soulain',
    author_email='anthony.soulain@sydney.edu.au.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Professional Astronomers',
        'Topic :: High Angular Resolution Astronomy :: Interferometry',
        'Programming Language :: Python :: 3.7'
    ],
    # package_data={'previs': ['data/eso_limits_matisse.json',
    #                         'data/vizier_catalog_references.json']},
)
