#!/usr/bin/env python3

import os

# Always prefer setuptools over distutils
import sys

from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="heareval",
    version="2021.0.1",
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
    python_requires=">=3.6",
    entry_points={
        "console_scripts": ["hear2021_tasks_coughvid=heareval.tasks.coughvid:main"]
    },
    install_requires=[
        "boto3",
        "luigi",
        "numpy",
        "pandas",
        "python-slugify",
        "requests",
        "soundfile",
        "tqdm",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
        "dev": [
            "pre-commit",
            "black",  # Used in pre-commit hooks
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
    },
    classifiers=[],
)
