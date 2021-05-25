#!/usr/bin/env python3
"""
WRITEME [outline the steps in the pipeline]

The idea is that each of these tasks should have a separate working
directory, and clean up the previous tasks working directory when
the current task is done.

Perhaps we should move to having all tasks write to
`_checkpoints/{type(self).__name__}/` and the final output is
`_checkpoints/done-{type(self).__name__}/`. That way, intermediate
working directories can be easily removed.

TODO:
* For tasks with large tqdm lists, it might make sense to add
multiprocessing, e.g. as per
https://github.com/neuralaudio/hear2021-eval-kit/pull/3/files#diff-2ac814f07851f8ddbb7cf1b456ab8ff5947ba33f1bf884e279eecc0cfc9b5262R48
* We also want everything to run in separate taskname/ directories
so different pipelines are isolated from each other.
* We need all the files in the README.md created for each dataset
(task.json, train.csv, etc.).
* After downloading from Zenodo, check that the zipfile has the
correct MD5.
"""

import glob
import hashlib
import os
import shutil
import subprocess

import luigi
import numpy as np
import requests
import soundfile as sf
from tqdm.auto import tqdm

# TODO: Put this in a config.py later
# Number of CPU workers for Luigi jobs
NUM_WORKERS = 1
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
# WARNING: If you change this value, you *must* delete _checkpoints
# or working dir.
# Most of the tasks iterate over every audio file present,
# except for the one that downsamples the corpus.
# (This is why we should have one working directory per task)
MAX_FRAMES_PER_CORPUS = 20 * 3600

max_files_per_corpus = int(MAX_FRAMES_PER_CORPUS / FRAME_RATE / SAMPLE_LENGTH_SECONDS)


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def download_file(url, local_filename):
    """
    The downside of this approach versus `wget -c` is that this
    code does not resume.
    The benefit is that we are sure if the download completely
    successfuly, otherwise we should have an exception.
    From: https://stackoverflow.com/a/16696317/82733
    TODO: Would be nice to have a TQDM progress bar here.
    """
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return local_filename


class DownloadCorpus(luigi.Task):
    def output(self):
        return luigi.LocalTarget("_checkpoints/%s" % (type(self).__name__))

    def run(self):
        # TODO: Change the working dir
        download_file(
            "https://zenodo.org/record/4498364/files/public_dataset.zip",
            "_checkpoints/corpus.zip",
        )
        with self.output().open("w") as outfile:
            pass


class ExtractCorpus(luigi.Task):
    def requires(self):
        return [DownloadCorpus()]

    def output(self):
        return luigi.LocalTarget("_checkpoints/%s" % (type(self).__name__))

    def run(self):
        # TODO: Do this in another directory
        os.system("cd _checkpoints && unzip corpus.zip")
        with self.output().open("w") as outfile:
            pass


def filename_to_inthash(filename):
    # Adapted from Google Speech Commands convention.
    hash_name_hashed = hashlib.sha1(filename.encode("utf-8")).hexdigest()
    return int(hash_name_hashed, 16)


class SubsampleCorpus(luigi.Task):
    def requires(self):
        return [DownloadCorpus()]

    def output(self):
        return luigi.LocalTarget("_checkpoints/%s" % (type(self).__name__))

    def run(self):
        # Really coughvid? webm + ogg?
        audiofiles = list(
            glob.glob("_checkpoints/public_dataset/*.webm")
            + glob.glob("_checkpoints/public_dataset/*.ogg")
        )
        # Deterministically randomly sort all files by their hash
        audiofiles.sort(key=lambda filename: filename_to_inthash(filename))
        print(audiofiles[:10])
        if len(audiofiles) > max_files_per_corpus:
            print(
                "%d audio files in corpus, keeping only %d"
                % (len(audiofiles), max_files_per_corpus)
            )
            for audiofile in audiofiles[max_files_per_corpus:]:
                os.remove(audiofile)
        assert len(
            list(
                glob.glob("_checkpoints/public_dataset/*.webm")
                + glob.glob("_checkpoints/public_dataset/*.ogg")
            )
        ) <= len(audiofiles)
        with self.output().open("w") as outfile:
            pass


def which_set(filename, validation_percentage, testing_percentage):
    """
    Code adapted from Google Speech Commands dataset.

    Determines which data partition the file should belong to, based
    upon the filename.

    We want to keep files in the same training, validation, or testing
    sets even if new ones are added over time. This makes it less
    likely that testing samples will accidentally be reused in training
    when long runs are restarted for example. To keep this stability,
    a hash of the filename is taken and used to determine which set
    it should belong to. This determination only depends on the name
    and the set proportions, so it won't change as other files are
    added.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'train', 'val', or 'test'.
    """
    base_name = os.path.basename(filename)
    percentage_hash = filename_to_inthash(filename) % 100
    if percentage_hash < validation_percentage:
        result = "val"
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = "test"
    else:
        result = "train"
    return result


def convert_to_mono_wav(in_file: str, out_file: str):
    devnull = open(os.devnull, "w")
    # If we knew the sample rate, we could also pad/trim the audio file now, e.g.:
    # ffmpeg -i test.webm -filter_complex apad=whole_len=44100,atrim=end_sample=44100 -ac 1 -c:a pcm_f32le ./test.wav
    ret = subprocess.call(
        ["ffmpeg", "-y", "-i", in_file, "-ac", "1", "-c:a", "pcm_f32le", out_file],
        stdout=devnull,
        stderr=devnull,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0


class ToMonoWavCorpus(luigi.Task):
    """
    Convert all audio to WAV files using Sox.
    We convert to mono, and also ensure that all files are the same length.
    """

    def requires(self):
        return [SubsampleCorpus()]

    def output(self):
        return luigi.LocalTarget("_checkpoints/%s" % (type(self).__name__))

    def run(self):
        for audiofile in tqdm(
            list(
                glob.glob("_checkpoints/public_dataset/*.webm")
                + glob.glob("_checkpoints/public_dataset/*.ogg")
            )
        ):
            convert_to_mono_wav(audiofile, os.path.splitext(audiofile)[0] + ".wav")
        with self.output().open("w") as outfile:
            pass


class EnsureLengthCorpus(luigi.Task):
    """
    Ensure all WAV files are a particular length.
    There might be a one-liner in ffmpeg that we can convert to WAV
    and ensure the file length at the same time.
    """

    def requires(self):
        return [ToMonoWavCorpus()]

    def output(self):
        return luigi.LocalTarget("_checkpoints/%s" % (type(self).__name__))

    def run(self):
        for audiofile in tqdm(list(glob.glob("_checkpoints/public_dataset/*.wav"))):
            x, sr = sf.read(audiofile)
            target_length_samples = int(round(sr * SAMPLE_LENGTH_SECONDS))
            # Convert to mono
            if x.ndim == 2:
                x = np.mean(x, axis=1)
            assert x.ndim == 1, "Audio should be mono"
            # Trim if necessary
            x = x[:target_length_samples]
            if len(x) < target_length_samples:
                x = np.hstack([x, np.zeros(target_length_samples - len(x))])
            assert len(x) == target_length_samples
            sf.write(audiofile, x, sr)
        with self.output().open("w") as outfile:
            pass


class TrainTestCorpus(luigi.Task):
    """
    If there is already a train/test split, we use that.
    Otherwise we deterministically
    """

    def requires(self):
        return [EnsureLengthCorpus()]

    def output(self):
        return luigi.LocalTarget("_checkpoints/%s" % (type(self).__name__))

    def run(self):
        for audiofile in tqdm(list(glob.glob("_checkpoints/public_dataset/*.wav"))):
            partition = which_set(
                audiofile, validation_percentage=0.0, testing_percentage=10.0
            )
            partition_dir = f"_checkpoints/{partition}/"
            ensure_dir(partition_dir)
            shutil.copy2(audiofile, partition_dir)
        with self.output().open("w") as outfile:
            pass


def resample_wav(in_file: str, out_file: str, out_sr: int):
    # TODO: Don't do anything if we are already the right sample rate.
    devnull = open(os.devnull, "w")
    ret = subprocess.call(
        [
            "ffmpeg",
            "-i",
            in_file,
            "-af",
            "aresample=resampler=soxr",
            "-ar",
            str(out_sr),
            out_file,
        ],
        stdout=devnull,
        stderr=devnull,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0


class ResampledCorpus(luigi.Task):
    sr = luigi.IntParameter()

    def requires(self):
        return [TrainTestCorpus()]

    def output(self):
        return luigi.LocalTarget("_checkpoints/%s-%d" % (type(self).__name__, self.sr))

    def run(self):
        for partition in ["train", "test", "val"]:
            for audiofile in tqdm(list(glob.glob(f"_checkpoints/{partition}/*.wav"))):
                for target_sr in SAMPLE_RATES:
                    resample_dir = f"_checkpoints/{target_sr}/{partition}/"
                    ensure_dir(resample_dir)
                    resampled_audiofile = os.path.join(
                        resample_dir,
                        os.path.split(audiofile)[1],
                    )
                    resample_wav(audiofile, resampled_audiofile, target_sr)
        with self.output().open("w") as outfile:
            pass


# TODO: Load from S3 + un-tar if available
class FinalizeCorpus(luigi.Task):
    def requires(self):
        return [ResampledCorpus(sr) for sr in SAMPLE_RATES]

    def output(self):
        return luigi.LocalTarget("_checkpoints/%s" % (type(self).__name__))

    def run(self):
        with self.output().open("w") as outfile:
            pass


if __name__ == "__main__":
    print("max_files_per_corpus = %d" % max_files_per_corpus)
    ensure_dir("_checkpoints")
    luigi.build([FinalizeCorpus()], workers=NUM_WORKERS, local_scheduler=True)
