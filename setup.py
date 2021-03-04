#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Utility inference from choice data in a sequential decision environment
Author: Addison Bohannon
"""

from setuptools import setup

setup(
    name='utility-infer-sde',
    version='0.0.0',
    packages=['utility-infer', 'experiments', 'analysis'],
    url='https://github.com/addisonbohannon/utility-infer.git',
    license=['GNU General Public License v3.0'],
    author='Addison Bohannon',
    author_email='addison.bohannon@gmail.com',
    description='Utility inference from choice data in a sequential decision environment',
    install_requires=['numpy', 'scipy'],
    scripts=[]
)
