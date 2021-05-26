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

Also keep in mind that if the S3 caching is enabled, you
will always just retrieve your S3 cache instead of running
the pipeline.

TODO:
* We need all the files in the README.md created for each dataset
(task.json, train.csv, etc.).
* After downloading from Zenodo, check that the zipfile has the
correct MD5.
* Would be nice to compute the 50% and 75% percentile audio length
to the metadata.
"""

import glob
import hashlib
import os
import shutil
import subprocess

import luigi
import numpy as np
import soundfile as sf
from slugify import slugify
from tqdm.auto import tqdm
from luigi.contrib.s3 import S3Client

import config.coughvid as config
from util.luigi import download_file, ensure_dir, WorkTask
import util.s3 as s3util


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
        if len(audiofiles) > config.MAX_FILES_PER_CORPUS:
            print(
                "%d audio files in corpus, keeping only %d"
                % (len(audiofiles), config.MAX_FILES_PER_CORPUS)
            )
        # Save diskspace using symlinks
        for audiofile in audiofiles[: config.MAX_FILES_PER_CORPUS]:
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
            target_length_samples = int(round(sr * config.SAMPLE_LENGTH_SECONDS))
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
    """
    Create a final corpus, no longer in _workdir but in the top-level
    at directory config.TASKNAME.
    """

    def requires(self):
        return [
            ResampledCorpus(sr, partition)
            for sr in config.SAMPLE_RATES
            for partition in ["train", "test", "val"]
        ]

    @property
    def name(self):
        return type(self).__name__

    # We overwrite workdir here, because we want the output to be
    # the finalized top-level task directory
    @property
    def workdir(self):
        return config.TASKNAME

    def run(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        shutil.copytree(self.requires()[0].workdir, self.workdir)
        with self.output().open("w") as outfile:
            pass


class EnsureBucket(WorkTask):
    """
    Ensure the S3 bucket exists and is readable.

    This S3 code is pretty gnarly, but I'm not sure it can be made
    any cleaner.
    """

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        s3util.check_bucket(config.S3_BUCKET, config.S3_REGION_NAME)
        with self.output().open("w") as outfile:
            pass


class CacheTarCorpus(WorkTask):
    """
    If the tar file is cached in S3, we simply retrieve it and untar
    it, and skip the pipeline.

    If the tar file is NOT in S3, we run the pipeline, create the
    tar-file, and upload it to S3.
    """

    def requires(self):
        return EnsureBucket()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        tarfile = f"{config.TASKNAME}.tar.gz"
        pathtarfile = f"{self.workdir}/{tarfile}"
        client = S3Client()
        s3cache = os.path.join(f"s3://{config.S3_BUCKET}/", tarfile)
        if client.exists(s3cache):
            client.get(s3cache, pathtarfile)
        else:
            # If you yield FinalizeCorpus, this task is suspended
            # and FinalizeCorpus is run, and a LocalFileTarget is
            # returned.
            finalize_corpus = yield FinalizeCorpus()

            # Tar the file
            devnull = open(os.devnull, "w")
            ret = subprocess.call(
                [
                    "tar",
                    "zcvf",
                    pathtarfile,
                    # Unfortunately, we have to hardcode
                    # FinalizeCorpus.workdir()
                    # because we don't have a Task
                    # just a Target in finalize_corpus.
                    config.TASKNAME,
                ],
                stdout=devnull,
                stderr=devnull,
            )
            # Make sure the return code is 0 and the command was successful.
            assert ret == 0

            # Cache to S3
            print("Putting file to S3")
            client.put_multipart(pathtarfile, s3cache)

        with self.output().open("w") as outfile:
            pass


if __name__ == "__main__":
    print("max_files_per_corpus = %d" % config.MAX_FILES_PER_CORPUS)
    ensure_dir("_workdir")
    luigi.build([CacheTarCorpus()], workers=config.NUM_WORKERS, local_scheduler=True)
