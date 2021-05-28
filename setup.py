#!/usr/bin/env python3

import os

# Always prefer setuptools over distutils
import sys

from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="hear2021-evalkit",
    version="0.0.1",
    description="Holistic Evaluation of Audio Representations (HEAR) 2021 -- Evaluation Kit",
    author="",
    author_email="",
    url="https://github.com/neuralaudio/hear2021-eval-kit",
    download_url="https://github.com/neuralaudio/hear2021-eval-kit",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/neuralaudio/hear2021-eval-kit/issues",
        "Source Code": "https://github.com/neuralaudio/hear2021-eval-kit",
    },
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "torch>=1.8",
        "pytorch-lightning",
        # pypi release (only master) doesn't support OrderedDict typing
        # "typing-extensions",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pygments>=2.7.4",  # not directly required, pinned by Snyk to avoid a vulnerability
            "pytest-env",
        ],
        "dev": [
            "pre-commit",
            "nbstripout==0.3.9",  # Used in precommit hooks
            "black==20.8b1",  # Used in precommit hooks
            "jupytext==v1.10.3",  # Used in precommit hooks
            "pytest",
            "pytest-cov",
            "ipython",
            "librosa",
            "scikit-learn>=0.24.2",  # not directly required, pinned by Snyk to avoid a vulnerability
            "matplotlib",
            "numba>=0.49.0",  # not directly required, pinned by Snyk to avoid a vulnerability
            "pygments>=2.7.4",  # not directly required, pinned by Snyk to avoid a vulnerability
            "pytest-env",
            "sphinx>=3.0.4",  # not directly required, pinned by Snyk to avoid a vulnerability
            "unofficial-pt-lightning-sphinx-theme",
            # Temporarily disabled so we can push to pypi
            # "pt-lightning-sphinx-theme @ https://github.com/PyTorchLightning/lightning_sphinx_theme/tarball/master#egg=pt-lightning-sphinx-theme",
            "sphinxcontrib-napoleon",
            "sphinx-autodoc-typehints",
            "mock",
            "sphinx_rtd_theme",
            "myst_parser",
            "linkify-it-py",
        ],
    },
    classifiers=[],
)
