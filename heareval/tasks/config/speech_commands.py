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


MAX_HOURS = 10

MAX_FILES_PER_CORPUS = int(MAX_HOURS * 60.0 * 60.0 / SAMPLE_LENGTH_SECONDS)
