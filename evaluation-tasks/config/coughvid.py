"""
Configuration for the coughvid task
"""

TASKNAME = "coughvid-v2.0.0"

# If set to true will cache results in S3
S3_CACHE = False
# You should pick a unique handle, since this determine the S3 path
# (which must be globally unique across all S3 users).
HANDLE = "hear"
S3_BUCKET = f"hear2021-{HANDLE}"
# If this is None, boto will use whatever is in your
# ~/.aws/config or AWS_DEFAULT_REGION environment variable
S3_REGION_NAME = "eu-central-1"
# Number of CPU workers for Luigi jobs
NUM_WORKERS = 4
# NUM_WORKERS = 1
# If you only use one sample rate, you should have an array with
# one sample rate in it.
# However, if you are evaluating multiple embeddings, you might
# want them all.
SAMPLE_RATES = [48000, 44100, 22050, 16000]
# TODO: Pick the 75th percentile length?
SAMPLE_LENGTH_SECONDS = 8.0
# TODO: Do we want to call this FRAME_RATE or HOP_SIZE
FRAME_RATE = 4
# Set this to None if you want to use ALL the data.
# NOTE: This will be, expected, 225 test files only :\
# NOTE: You can make this smaller during development of this
# preprocessing script, to keep the pipeline fast.
# WARNING: If you change this value, you *must* delete _workdir
# or working dir.
# Most of the tasks iterate over every audio file present,
# except for the one that downsamples the corpus.
# (This is why we should have one working directory per task)
MAX_FRAMES_PER_CORPUS = 20 * 3600

MAX_FILES_PER_CORPUS = int(MAX_FRAMES_PER_CORPUS / FRAME_RATE / SAMPLE_LENGTH_SECONDS)