"""
Configuration for the google speech commands task
"""

from .dataset_config import *  # noqa: F403, F401

TASKNAME = "speech_commands-v0.0.2"
DOWNLOAD_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
TEST_DOWNLOAD_URL = (
    "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
)

SAMPLE_LENGTH_SECONDS = 1.0

# Include the entire dataset, 85511 train files, 10102 validation files, and
# 4890 testing samples
MAX_TRAIN_FILES = None
MAX_VAL_FILES = None
MAX_TEST_FILES = None
