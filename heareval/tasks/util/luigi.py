"""
Common Luigi classes and functions for evaluation tasks
"""

import hashlib
import os
import shutil
from glob import glob
from pathlib import Path

import luigi
import pandas as pd
import requests
from tqdm import tqdm

import heareval.tasks.util.audio as audio_util

PROCESSMETADATACOLS = [
    "relpath",
    "slug",
    "filename_hash",
    "partition",
    "label",
]


class WorkTask(luigi.Config):
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
            `_workdir/{name}.done`
            * Optionally, working output of each task will go into:
            `_workdir/{name}`
    Downstream dependencies should be cautious of automatically
    removing the working output, unless they are sure they are the
    only downstream dependency of a particular task (i.e. no
    triangular dependencies).
    """

    # Class attribute sets the task name for all inheriting luigi tasks
    data_config = luigi.DictParameter(
        visibility=luigi.parameter.ParameterVisibility.PRIVATE
    )

    @property
    def name(self):
        # Make the default name here. If required change in any
        # subtask. This folder is not named with task_id as the name
        # can be set anytime
        # Should we make it with task_id?
        return type(self).__name__

    def output(self):
        # Replace the name with task_id as it is unique at task and parameter level
        f = os.path.join(
            self.task_subdir, f"{self.stage_number:02d}-{self.task_id}.done"
        )
        return luigi.LocalTarget(f)

    def mark_complete(self):
        # Touches the output file, marking this task as complete
        self.output().open("w").close()

    @property
    def workdir(self):
        d = os.path.join(self.task_subdir, f"{self.stage_number:02d}-{self.name}")
        # In parallel task called from one parent task, ensure_dir might run from
        # multiple calls and it might fail due the condition which is being checked.
        # The folder might not exist while doing the if statement but meanwhile might
        # come into existance from the parallel call.
        os.makedirs(d, exist_ok=True)
        # ensure_dir(d)
        return d

    @property
    def task_subdir(self):
        """
        Task specific subdirectory
        """
        # You must specify a task name for WorkTask
        d = ["_workdir", str(self.versioned_task_name)]
        return os.path.join(*d)

    @property
    def versioned_task_name(self):
        task_name = f"{self.data_config['task_name']}-{self.data_config['version']}"
        return task_name

    @property
    def stage_number(self) -> int:
        """
        Numerically sort the DAG tasks.
        This stage number will go into the name.
            This should be overridden as 0 by any task that has no
        requirements.
        """
        if isinstance(self.requires(), WorkTask):
            return 1 + self.requires().stage_number
        elif isinstance(self.requires(), list):
            return 1 + max([task.stage_number for task in self.requires()])

        # Add a new dict method here so that dicts can be handled
        # The intermediate tasks can also be a list of task.
        elif isinstance(self.requires(), dict):
            parentasks = []
            for task in list(self.requires().values()):
                if isinstance(task, list):
                    parentasks.extend(task)
                else:
                    parentasks.append(task)
            return 1 + max([task.stage_number for task in parentasks])
        else:
            raise ValueError("Unknown requires: %s" % self.requires())


class DownloadCorpus(WorkTask):

    url = luigi.Parameter()
    outfile = luigi.Parameter()

    def run(self):
        download_file(self.url, os.path.join(self.workdir, self.outfile))
        with self.output().open("w") as _:
            pass

    @property
    def stage_number(self) -> int:
        return 0


class ExtractArchive(WorkTask):

    infile = luigi.Parameter()
    download = luigi.TaskParameter(
        visibility=luigi.parameter.ParameterVisibility.PRIVATE
    )
    outdir = luigi.Parameter(default=None)

    def requires(self):
        return {"download": self.download}

    def run(self):
        corpus_zip = os.path.realpath(
            os.path.join(self.requires()["download"].workdir, self.infile)
        )
        # If there are multiple ExtractArchive tasks then we optionally may want the
        # extracted results to be placed into separate dirs within the main workdir.
        output = self.workdir
        if self.outdir is not None:
            output = os.path.join(output, self.outdir)
        shutil.unpack_archive(corpus_zip, output)

        self.mark_complete()


class SubsampleCorpus(WorkTask):
    max_files = luigi.IntParameter()

    def requires(self):
        raise NotImplementedError("This method requires a meta tasks")

    def get_metadata(self):
        metadata = pd.read_csv(
            os.path.join(
                self.requires()["meta"].workdir, self.requires()["meta"].outfile
            ),
            header=None,
            names=PROCESSMETADATACOLS,
        )[["filename_hash", "slug", "relpath", "partition"]]
        return metadata

    def run(self):

        process_metadata = self.get_metadata()
        # Subsample the files based on max files per corpus.
        # The filename hash is used here
        # This task can also be done in the configprocessmetadata as that will give
        # freedom to stratify the selection on some criterion?
        num_files = len(process_metadata)
        max_files = num_files if self.max_files is None else self.max_files
        if num_files > max_files:
            print(
                f"{len(process_metadata)} audio files in corpus, \
                    keeping only {max_files}"
            )

        # Sort by the filename hash and select the max file per corpus
        # The filename hash is done as part of the processmetadata because
        # the string to be hashed for each file is dependent on the data
        process_metadata = process_metadata.sort_values(by="filename_hash").iloc[
            :max_files
        ]

        # Save file using symlinks
        for _, audio in process_metadata.iterrows():
            audiofile = Path(audio["relpath"])
            newaudiofile = Path(os.path.join(self.workdir, audio["slug"]))
            newaudiofile.unlink(missing_ok=True)
            newaudiofile.symlink_to(audiofile.resolve())

        self.mark_complete()


class SubsamplePartition(SubsampleCorpus):
    """
    Performs subsampling on a specific partition
    """

    partition = luigi.Parameter()

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata = metadata[metadata["partition"] == self.partition]
        return metadata


class MonoWavTrimCorpus(WorkTask):
    def requires(self):
        raise NotImplementedError("This method requires a corpus tasks")

    def run(self):
        # TODO: this should check to see if the audio is already a mono wav at the
        #   correct length and just create a symlink if that is this case.
        for audiofile in tqdm(list(glob(f"{self.requires()['corpus'].workdir}/*"))):

            newaudiofile = new_basedir(
                os.path.splitext(audiofile)[0] + ".wav", self.workdir
            )
            audio_util.mono_wav_and_fix_duration(
                audiofile, newaudiofile, duration=self.data_config["sample_duration"]
            )

        self.mark_complete()


class SplitTrainTestCorpus(WorkTask):
    def requires(self):
        raise NotImplementedError("This method requires a meta and a corpus tasks")

    def run(self):
        # Get the process metadata. This gives the freedom of picking the train test
        # label either from the provide metadata file or any other method.

        # Writing slug and partition makes it explicit that these columns are required
        meta = self.requires()["meta"]
        process_metadata = pd.read_csv(
            os.path.join(meta.workdir, meta.outfile),
            header=None,
            names=PROCESSMETADATACOLS,
        )[["slug", "partition"]]

        # Go over the subsampled folder and pick the audio files. The audio files are
        # saved with their slug names and hence the corresponding label can be picked
        # up from the preprocess config
        for audiofile in tqdm(list(glob(f"{self.requires()['corpus'].workdir}/*.wav"))):
            audio_slug = os.path.basename(audiofile)
            partition = process_metadata.loc[
                process_metadata["slug"] == audio_slug, "partition"
            ].values[0]
            partition_dir = f"{self.workdir}/{partition}"
            ensure_dir(partition_dir)

            newaudiofile = new_basedir(audiofile, partition_dir)
            os.symlink(os.path.realpath(audiofile), newaudiofile)
        with self.output().open("w") as _:
            pass


class SplitTrainTestMetadata(WorkTask):
    def requires(self):
        raise NotImplementedError(
            "This method requires a meta and a traintestcorpus task"
        )

    def get_metadata(self):
        metadata = pd.read_csv(
            os.path.join(
                self.requires()["meta"].workdir, self.requires()["meta"].outfile
            ),
            header=None,
            names=PROCESSMETADATACOLS,
        )[["slug", "label"]]
        return metadata

    def run(self):
        # Get the process metadata and select the required columns slug and label
        labeldf = self.get_metadata()

        # Automatically get the partitions from the traintestcorpus task
        # and then get the files there.
        for partition in os.listdir(self.requires()["traintestcorpus"].workdir):
            audiofiles = list(
                glob(
                    os.path.join(
                        self.requires()["traintestcorpus"].workdir,
                        partition,
                        "*.wav",
                    )
                )
            )
            # Check if we got any file in the partition
            if len(audiofiles) == 0:
                raise RuntimeError(
                    "No audio files found in "
                    f"{self.requires()['traintestcorpus'].workdir}/{partition}"
                )
            # Get the filename which is also the slug of the file and the
            # corresponding label can be picked up from the metadata
            audiodf = pd.DataFrame(
                [os.path.basename(a) for a in audiofiles], columns=["slug"]
            )
            assert len(audiofiles) == len(audiodf.drop_duplicates())

            # Get the label from the metadata with the help of the slug of the filename
            sublabeldf = labeldf.merge(audiodf, on="slug")[["slug", "label"]]
            # Check if all the labels were found from the metadata
            assert len(sublabeldf) == len(audiofiles)
            # Save the slug and the label in as the parition metadata
            sublabeldf.to_csv(
                os.path.join(self.workdir, f"{partition}.csv"),
                columns=["slug", "label"],
                index=False,
                header=False,
            )

        with self.output().open("w") as _:
            pass


class MetadataVocabulary(WorkTask):
    def requires(self):
        raise NotImplementedError("This method requires a traintestmeta task")

    def run(self):
        labelset = set()
        # Iterate over all the files in the traintestmeta and get the partition_metadata
        for partition_metadata in os.listdir(self.requires()["traintestmeta"].workdir):
            labeldf = pd.read_csv(
                os.path.join(
                    self.requires()["traintestmeta"].workdir,
                    partition_metadata,
                ),
                header=None,
                names=["filename", "label"],
            )
            labelset = labelset | set(labeldf["label"].unique().tolist())

        # Build the label idx csv and save it
        labelcsv = pd.DataFrame(
            [(label, idx) for (idx, label) in enumerate(sorted(list(labelset)))],
            columns=["idx", "label"],
        )

        labelcsv.to_csv(
            os.path.join(self.workdir, "labelvocabulary.csv"),
            columns=["idx", "label"],
            index=False,
            header=False,
        )

        with self.output().open("w") as _:
            pass


class ResampleSubCorpus(WorkTask):
    sr = luigi.IntParameter()
    partition = luigi.Parameter()

    def requires(self):
        raise NotImplementedError("This method requires a traintestcorpus task")

    def run(self):
        # Check if the original directory exists and then move ahead to make the
        # directory in the resample folder. This ensures val is not made if val is
        # not there
        original_dir = f"{self.requires()['traintestcorpus'].workdir}/{self.partition}"
        resample_dir = f"{self.workdir}/{self.sr}/{self.partition}/"
        if os.path.isdir(original_dir):
            # The below command can run in parallel and might fail.
            # This is because maybe the other resampling task has the directory
            # while the current resampling task just checked for the directory.
            # luigi_util.ensure_dir(resample_dir)

            # This is more safe
            os.makedirs(resample_dir, exist_ok=True)
            for audiofile in tqdm(list(glob(f"{original_dir}/*.wav"))):
                resampled_audiofile = new_basedir(audiofile, resample_dir)
                audio_util.resample_wav(audiofile, resampled_audiofile, self.sr)

        self.mark_complete()


class FinalizeCorpus(WorkTask):
    """
    Create a final corpus, no longer in _workdir but in the top-level
    at directory config.TASKNAME.
    """

    def requires(self):
        raise NotImplementedError(
            "This method requires a list of resample tasks "
            "and a traintestmeta and vocabmeta task"
        )

    # We overwrite workdir here, because we want the output to be
    # the finalized top-level task directory
    @property
    def workdir(self):
        return os.path.join("tasks", self.versioned_task_name)

    def run(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        # The workdirectory of the resample method is same so this would work
        shutil.copytree(self.requires()["resample"][0].workdir, self.workdir)
        shutil.copytree(
            self.requires()["traintestmeta"].workdir,
            self.workdir,
            dirs_exist_ok=True,
        )
        shutil.copytree(
            self.requires()["vocabmeta"].workdir,
            self.workdir,
            dirs_exist_ok=True,
        )

        with self.output().open("w") as _:
            pass


def download_file(url, local_filename):
    """
    The downside of this approach versus `wget -c` is that this
    code does not resume.
    The benefit is that we are sure if the download completely
    successfuly, otherwise we should have an exception.
    From: https://stackoverflow.com/a/16696317/82733
    """
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get("content-length"))
        with open(local_filename, "wb") as f:
            pbar = tqdm(total=total_length)
            chunk_size = 8192
            for chunk in r.iter_content(chunk_size=chunk_size):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                f.write(chunk)
                pbar.update(chunk_size)
            pbar.close()

    return local_filename


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def filename_to_int_hash(filename):
    # Adapted from Google Speech Commands convention.
    hash_name_hashed = hashlib.sha1(filename.encode("utf-8")).hexdigest()
    return int(hash_name_hashed, 16)


def which_set(filename_hash, validation_percentage, testing_percentage):
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
            NOTE: Should be a relative path.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'train', 'val', or 'test'.
    """
    # Change below line to accept the hash directly as the hash
    # is data dependent on which files do we need to consider as a
    # group. For example we might want to keep audio by one speaker
    # in the test or train rather than distributing them.

    percentage_hash = filename_hash % 100
    if percentage_hash < validation_percentage:
        result = "val"
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = "test"
    else:
        result = "train"
    return result


def new_basedir(filename, basedir):
    """
    Rewrite .../filename as basedir/filename
    """
    return os.path.join(basedir, os.path.split(filename)[1])
