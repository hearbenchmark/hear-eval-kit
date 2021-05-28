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
import os
import shutil
import subprocess

import luigi
import numpy as np
import soundfile as sf
from slugify import slugify
from tqdm.auto import tqdm

import hearevalkit.tasks.config.coughvid as config
import hearevalkit.tasks.util.audio as audio_util
import hearevalkit.tasks.util.luigi as luigi_util


class DownloadCorpus(luigi_util.WorkTask):
    @property
    def name(self):
        return type(self).__name__

    def run(self):
        # TODO: Change the working dir
        luigi_util.download_file(
            "https://zenodo.org/record/4498364/files/public_dataset.zip",
            os.path.join(self.workdir, "corpus.zip"),
        )
        with self.output().open("w") as outfile:
            pass

    @property
    def stage_number(self) -> int:
        return 0


class ExtractCorpus(luigi_util.WorkTask):
    def requires(self):
        return DownloadCorpus()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        # Location of zip file to extract. Figure this out before changing
        # the working directory.
        corpus_zip = os.path.realpath(
            os.path.join(self.requires().workdir, "corpus.zip")
        )
        subprocess.check_output(["unzip", "-o", corpus_zip, "-d", self.workdir])

        with self.output().open("w") as outfile:
            pass


class SubsampleCorpus(luigi_util.WorkTask):
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

        # Make sure we found audio files to work with
        if len(audiofiles) == 0:
            raise RuntimeError(f"No audio files found in {self.requires().workdir}")

        # Deterministically randomly sort all files by their hash
        audiofiles.sort(key=lambda filename: luigi_util.filename_to_int_hash(filename))
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


class ToMonoWavCorpus(luigi_util.WorkTask):
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
            newaudiofile = luigi_util.new_basedir(
                os.path.splitext(audiofile)[0] + ".wav", self.workdir
            )
            audio_util.convert_to_mono_wav(audiofile, newaudiofile)
        with self.output().open("w") as outfile:
            pass


class EnsureLengthCorpus(luigi_util.WorkTask):
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
            newaudiofile = luigi_util.new_basedir(audiofile, self.workdir)
            sf.write(newaudiofile, x, sr)
        with self.output().open("w") as outfile:
            pass


class SplitTrainTestCorpus(luigi_util.WorkTask):
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
            partition = luigi_util.which_set(
                audiofile, validation_percentage=0.0, testing_percentage=10.0
            )
            partition_dir = f"{self.workdir}/{partition}"
            luigi_util.ensure_dir(partition_dir)
            newaudiofile = luigi_util.new_basedir(audiofile, partition_dir)
            os.symlink(os.path.realpath(audiofile), newaudiofile)
        with self.output().open("w") as outfile:
            pass


class ResampleSubCorpus(luigi_util.WorkTask):
    sr = luigi.IntParameter()
    partition = luigi.Parameter()

    def requires(self):
        return SplitTrainTestCorpus()

    @property
    def name(self):
        return type(self).__name__

    # Since these tasks have parameters but share the same working
    # directory and name, we postpend the parameters to the output
    # filename, so we can track if one ResampleSubCorpus task finished
    # but others didn't.
    def output(self):
        return luigi.LocalTarget(
            "_workdir/%02d-%s-%d-%s.done"
            % (self.stage_number, self.name, self.sr, self.partition)
        )

    def run(self):
        resample_dir = f"{self.workdir}/{self.sr}/{self.partition}/"
        luigi_util.ensure_dir(resample_dir)
        for audiofile in tqdm(
            list(glob.glob(f"{self.requires().workdir}/{self.partition}/*.wav"))
        ):
            resampled_audiofile = luigi_util.new_basedir(audiofile, resample_dir)
            audio_util.resample_wav(audiofile, resampled_audiofile, self.sr)
        with self.output().open("w") as outfile:
            pass


class FinalizeCorpus(luigi_util.WorkTask):
    """
    Create a final corpus, no longer in _workdir but in the top-level
    at directory config.TASKNAME.
    """

    def requires(self):
        return [
            ResampleSubCorpus(sr, partition)
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


def main():
    print("max_files_per_corpus = %d" % config.MAX_FILES_PER_CORPUS)
    luigi_util.ensure_dir("_workdir")
    luigi.build([FinalizeCorpus()], workers=config.NUM_WORKERS, local_scheduler=True)


if __name__ == "__main__":
    main()
