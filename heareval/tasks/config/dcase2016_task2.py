"""
Configuration for the nsynth pitch detection task
"""

from .dataset_config import *  # noqa: F403, F401

# HEAR 2021 variation of DCASE 2016 Task 2
# Namely, we ignore the monophonic training data and use the dev
# data for train.
# We also allow training data outside this task.
TASKNAME = "dcase2016-task2-hear2021"

TRAIN_DEV_DOWNLOAD_URL = "https://archive.org/download/dcase2016_task2_train_dev/dcase2016_task2_train_dev.zip"  # pylint: disable=E501
TEST_DOWNLOAD_URL = "https://archive.org/download/dcase2016_task2_test_public/dcase2016_task2_test_public.zip"  # pylint: disable=E501

# SAMPLE_LENGTH_SECONDS = 4.0

# Number files to include
MAX_TRAIN_FILES = None
MAX_TEST_FILES = None
