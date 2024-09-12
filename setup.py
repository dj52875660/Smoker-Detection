#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

setup(
    name='smoker_detection',
    packages=find_packages(
        include=['smoker_detection', 'smoker_detection.*']
    ),
    test_suite='tests',
    version="0.1.0",
)
