"""
Configuration for the nsynth pitch detection task
"""

from .dataset_config import *  # noqa: F403, F401

TASKNAME = "nsynth-pitch-v2.3.3"

TRAIN_DOWNLOAD_URL = (
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
)
VALIDATION_DOWNLOAD_URL = (
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz"
)
TEST_DOWNLOAD_URL = (
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
)

SAMPLE_LENGTH_SECONDS = 4.0

# Number files to include
MAX_TRAIN_FILES = 8000
MAX_VAL_FILES = 1000
MAX_TEST_FILES = 1000
