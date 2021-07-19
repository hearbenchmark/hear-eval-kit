"""
Configuration for the google speech commands task
"""

from .dataset_config import *

TASKNAME = "speech_commands-v0.0.2"
DOWNLOAD_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
TEST_DOWNLOAD_URL = (
    "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
)

SAMPLE_LENGTH_SECONDS = 1.0

# Number files to include -- this is the full dataset
MAX_TRAIN_FILES = 85511
MAX_VAL_FILES = 10102
MAX_TEST_FILES = 4890
