#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This is for setting up InterpML

"""InterpML

Interpolation Models and Error Bounds for Verifiable Scientific Machine Learning

Python scripts accompanying the paper:

Leveraging Interpolation Models and Error Bounds for Verifiable Scientific Machine Learning
Tyler Chang, Andrew Gillette, Romit Maulik
2024

"""

DOCLINES = (__doc__ or '').split("\n")

from setuptools import setup

exec(open("interpml/version.py").read())

# This command performs the setup
setup(
    name="interpml",
    version=__version__,
    description="Interpolation Models and Error Bounds for Verifiable Scientific Machine Learning",
    long_description="\n".join(DOCLINES[2:]),
    author="Tyler Chang, Andrew Gillette, and Romit Maulik",
    license="BSD 3-clause",

    packages=["interpml"],

    install_requires=["cvxpy",
                      "networkx",
                      "numpy",
                      "pandas",
                      "scipy",
                      "scikit-learn"],

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules"],
)
