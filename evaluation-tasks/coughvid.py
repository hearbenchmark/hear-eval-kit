#!/usr/bin/env python3
"""
WRITEME [outline the steps in the pipeline]

The idea is that each of these tasks should has a separate working
directory in _workdir. We remove it only when the entire pipeline
is done. This is safer, even tho it uses more disk space.
(The principle here is we don't remove any working dirs during
processing.)

When hacking on this file, consider only enabling one Task at a
time in __main__.

TODO:
* We also want everything to run in separate taskname/ directories
so different pipelines are isolated from each other.
* We need all the files in the README.md created for each dataset
(task.json, train.csv, etc.).
* After downloading from Zenodo, check that the zipfile has the
correct MD5.
* Would be nice to compute the 50% and 75% percentile audio length
to the metadata.
* Upload to S3 at end (or skip if it exists on S3).
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
from slugify import slugify
from tqdm.auto import tqdm

TASKNAME = "coughvid-v2.0.0"

# TODO: Put this in a config.py later
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


class WorkTask(luigi.Task):
    """
    We assume following conventions:
        * Each luigi Task will have a name property:
            {classname}
            or
            {classname}-{task parameters}
            depending upon what your want the name to be.
            (TODO: Since we always use {classname}, just
            make this constant?)
        * The "output" of each task is a touch'ed file,
        indicating that the task is done. Each .run()
        method should end with this command:
            `_workdir/done-{name}`
            * Optionally, working output of each task will go into:
            `_workdir/{name}`
    Downstream dependencies should be cautious of automatically
    removing the working output, unless they are sure they are the
    only downstream dependency of a particular task (i.e. no
    triangular dependencies).
    """

    @property
    def name(self):
        ...
        # return type(self).__name__

    def output(self):
        return luigi.LocalTarget("_workdir/done-%s" % self.name)

    @property
    def workdir(self):
        d = "_workdir/%s/" % self.name
        ensure_dir(d)
        return d


class DownloadCorpus(WorkTask):
    @property
    def name(self):
        return type(self).__name__

    def run(self):
        # TODO: Change the working dir
        download_file(
            "https://zenodo.org/record/4498364/files/public_dataset.zip",
            os.path.join(self.workdir, "corpus.zip"),
        )
        with self.output().open("w") as outfile:
            pass


class ExtractCorpus(WorkTask):
    def requires(self):
        return DownloadCorpus()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        # TODO: unzip --force ?
        cmd = "cd %s && unzip %s" % (
            self.workdir,
            os.path.realpath(os.path.join(self.requires().workdir, "corpus.zip")),
        )
        # TODO: Don't use os.system, we can't trap any errors
        os.system(cmd)
        with self.output().open("w") as outfile:
            pass


def filename_to_inthash(filename):
    # Adapted from Google Speech Commands convention.
    hash_name_hashed = hashlib.sha1(filename.encode("utf-8")).hexdigest()
    return int(hash_name_hashed, 16)


class SubsampleCorpus(WorkTask):
    """
    Subsample the corpus so that we have the appropriate number of
    audio files.

    Additionally, since the upstream data files might be in subfolders,
    we slugify them here so that we just have one flat directory.
    (TODO: Double check this works.)

    A destructive way of implementing this task is that it removes
    extraneous files, rather than copying them to the next task's
    directory.
    """

    def requires(self):
        return ExtractCorpus()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        # Really coughvid? webm + ogg?
        audiofiles = list(
            glob.glob(os.path.join(self.requires().workdir, "public_dataset/*.webm"))
            + glob.glob(os.path.join(self.requires().workdir, "public_dataset/*.ogg"))
        )
        # Deterministically randomly sort all files by their hash
        audiofiles.sort(key=lambda filename: filename_to_inthash(filename))
        if len(audiofiles) > max_files_per_corpus:
            print(
                "%d audio files in corpus, keeping only %d"
                % (len(audiofiles), max_files_per_corpus)
            )
        # Save diskspace using symlinks
        for audiofile in audiofiles[:max_files_per_corpus]:
            # Extract the audio filename, excluding the old working
            # directory, but including all subfolders
            newaudiofile = os.path.join(
                self.workdir,
                os.path.split(
                    slugify(os.path.relpath(audiofile, self.requires().workdir))
                )[0],
                # This is pretty gnarly but we do it to not slugify the filename extension
                os.path.split(audiofile)[1],
            )
            # Make sure we don't have any duplicates
            assert not os.path.exists(newaudiofile)
            os.symlink(os.path.realpath(audiofile), newaudiofile)
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
    # print(" ".join(["ffmpeg", "-y", "-i", in_file, "-ac", "1", "-c:a", "pcm_f32le", out_file]))
    ret = subprocess.call(
        ["ffmpeg", "-y", "-i", in_file, "-ac", "1", "-c:a", "pcm_f32le", out_file],
        stdout=devnull,
        stderr=devnull,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0


def new_basedir(filename, basedir):
    """
    Rewrite .../filename as basedir/filename
    """
    return os.path.join(basedir, os.path.split(filename)[1])


class ToMonoWavCorpus(WorkTask):
    """
    Convert all audio to WAV files using Sox.
    We convert to mono, and also ensure that all files are the same length.
    """

    def requires(self):
        return SubsampleCorpus()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        audiofiles = list(
            glob.glob(os.path.join(self.requires().workdir, "*.webm"))
            + glob.glob(os.path.join(self.requires().workdir, "*.ogg"))
        )
        for audiofile in tqdm(audiofiles):
            newaudiofile = new_basedir(
                os.path.splitext(audiofile)[0] + ".wav", self.workdir
            )
            convert_to_mono_wav(audiofile, newaudiofile)
        with self.output().open("w") as outfile:
            pass


class EnsureLengthCorpus(WorkTask):
    """
    Ensure all WAV files are a particular length.
    There might be a one-liner in ffmpeg that we can convert to WAV
    and ensure the file length at the same time.
    """

    def requires(self):
        return ToMonoWavCorpus()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        for audiofile in tqdm(
            list(glob.glob(os.path.join(self.requires().workdir, "*.wav")))
        ):
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
            newaudiofile = new_basedir(audiofile, self.workdir)
            sf.write(newaudiofile, x, sr)
        with self.output().open("w") as outfile:
            pass


class TrainTestCorpus(WorkTask):
    """
    If there is already a train/test split, we use that.
    Otherwise we deterministically
    """

    def requires(self):
        return EnsureLengthCorpus()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        for audiofile in tqdm(list(glob.glob(f"{self.requires().workdir}/*.wav"))):
            partition = which_set(
                audiofile, validation_percentage=0.0, testing_percentage=10.0
            )
            partition_dir = f"{self.workdir}/{partition}"
            ensure_dir(partition_dir)
            newaudiofile = new_basedir(audiofile, partition_dir)
            os.symlink(os.path.realpath(audiofile), newaudiofile)
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


class ResampledCorpus(WorkTask):
    sr = luigi.IntParameter()
    partition = luigi.Parameter()

    def requires(self):
        return TrainTestCorpus()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        resample_dir = f"{self.workdir}/{self.sr}/{self.partition}/"
        ensure_dir(resample_dir)
        for audiofile in tqdm(
            list(glob.glob(f"{self.requires().workdir}/{self.partition}/*.wav"))
        ):
            resampled_audiofile = new_basedir(audiofile, resample_dir)
            resample_wav(audiofile, resampled_audiofile, self.sr)
        with self.output().open("w") as outfile:
            pass


class FinalizeCorpus(WorkTask):
    def requires(self):
        return [
            ResampledCorpus(sr, partition)
            for sr in SAMPLE_RATES
            for partition in ["train", "test", "val"]
        ]

    @property
    def name(self):
        return type(self).__name__

    # We overwrite workdir here, because we want the output to be
    # the finalized top-level task directory
    @property
    def workdir(self):
        return TASKNAME

    def run(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        shutil.copytree(self.requires()[0].workdir, self.workdir)
        with self.output().open("w") as outfile:
            pass


class TarCorpus(WorkTask):
    def requires(self):
        return FinalizeCorpus()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        devnull = open(os.devnull, "w")
        ret = subprocess.call(
            [
                "tar",
                "zcvf",
                f"{self.workdir}/{TASKNAME}.tar.gz",
                self.requires().workdir,
            ],
            stdout=devnull,
            stderr=devnull,
        )
        # Make sure the return code is 0 and the command was successful.
        assert ret == 0

        with self.output().open("w") as outfile:
            pass


# TODO: Load from S3 + un-tar if available

if __name__ == "__main__":
    print("max_files_per_corpus = %d" % max_files_per_corpus)
    ensure_dir("_workdir")
    # luigi.build([FinalizeCorpus()], workers=NUM_WORKERS, local_scheduler=True)
    luigi.build([TarCorpus()], workers=NUM_WORKERS, local_scheduler=True)
