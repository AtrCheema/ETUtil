# -*- coding: utf-8 -*-
# Copyright (C) 2025  Ather Abbas
from setuptools import setup

import os
fpath = os.path.join(os.getcwd(), "README.rst")
if os.path.exists(fpath):
    with open(fpath, "r") as fd:
        long_desc = fd.read()

setup(
    name='ETUtil',

    version="1.2",

    description='calculate evapotranspiration from monthly to sub-hourly time step',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/ETUtil',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    classifiers=[
        "Development Status :: 5 - Production/Stable",

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',

        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Utilities",
    ],

    packages=['ETUtil'],

    install_requires=[
        'numpy<=2.0.1, >=1.17',
        'pandas<=2.2, >=0.23',
        'matplotlib',
    ],
    extras_require={
        'all': ["numpy<=2.0.1, >=1.17",
                "scipy<=1.13, >=1.4",
                "openpyxl"],
    }
)
