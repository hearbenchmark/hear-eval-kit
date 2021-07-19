"""
Generic configuration used by all tasks
"""
# For deterministic dataset generation
SEED = 43

# Number of CPU workers for Luigi jobs
NUM_WORKERS = 4

# If you only use one sample rate, you should have an array with
# one sample rate in it.
# However, if you are evaluating multiple embeddings, you might
# want them all.
SAMPLE_RATES = [48000, 44100, 22050, 16000]
