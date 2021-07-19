"""
Configuration for the google speech commands task
"""

# TODO: move some of these to a global config and import that here
# See: https://github.com/neuralaudio/hear2021-eval-kit/issues/10

TASKNAME = "speech_commands-v0.0.2"
DOWNLOAD_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
TEST_DOWNLOAD_URL = (
    "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
)
SEED = 43

# Number of CPU workers for Luigi jobs
NUM_WORKERS = 4

# If you only use one sample rate, you should have an array with
# one sample rate in it.
# However, if you are evaluating multiple embeddings, you might
# want them all.
SAMPLE_RATES = [48000, 44100, 22050, 16000]

SAMPLE_LENGTH_SECONDS = 1.0

# Number files to include -- this is the full dataset
MAX_TRAIN_FILES = 85511
MAX_VAL_FILES = 10102
MAX_TEST_FILES = 4890
