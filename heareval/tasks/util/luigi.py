"""
Common Luigi classes and functions for evaluation tasks
"""

import hashlib
import os
import shutil
from glob import glob
from pathlib import Path
import json

import luigi
import pandas as pd
import requests
from tqdm import tqdm

import heareval.tasks.util.audio as audio_util


class WorkTask(luigi.Task):
    """
    We assume following conventions:
        * Each luigi Task will have a name property
        * The "output" of each task is a touch'ed file,
        indicating that the task is done. Each .run()
        method should end with this command:
            `_workdir/{task_subdir}{task_id}.done`
            task_id unique identifies the task by a combination of name and
            input parameters
            * Optionally, working output of each task will go into:
            `_workdir/{task_subdir}{name}`

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
        return type(self).__name__

    def output(self):
        """
        Outfile to mark a task as complete.
        """
        output_name = f"{self.stage_number:02d}-{self.task_id}.done"
        output_file = self.task_subdir.joinpath(output_name)
        return luigi.LocalTarget(output_file)

    def mark_complete(self):
        """Touches the output file, marking this task as complete"""
        self.output().open("w").close()

    @property
    def workdir(self):
        """Working directory"""
        d = self.task_subdir.joinpath(f"{self.stage_number:02d}-{self.name}")
        d.mkdir(exist_ok=True)
        return d

    @property
    def task_subdir(self):
        """Task specific subdirectory"""
        return Path("_workdir").joinpath(str(self.versioned_task_name))

    @property
    def versioned_task_name(self):
        """
        Versioned Task name contains the provided name in the
        data config and the version
        """
        return f"{self.data_config['task_name']}-{self.data_config['version']}"

    @property
    def stage_number(self):
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
        elif isinstance(self.requires(), dict):
            parentasks = []
            for task in list(self.requires().values()):
                if isinstance(task, list):
                    parentasks.extend(task)
                else:
                    parentasks.append(task)
            return 1 + max([task.stage_number for task in parentasks])
        else:
            raise ValueError(f"Unknown requires: {self.requires()}")


def get_download_and_extract_tasks(config: Dict):

    tasks = {}
    for name, url in config["download_urls"].items():
        filename = os.path.basename(urlparse(url).path)
        task = ExtractArchive(
            download=DownloadCorpus(url=url, outfile=filename, data_config=config),
            infile=filename,
            outdir=name,
            data_config=config,
        )
        tasks[name] = task

    return tasks


class ExtractMetadata(WorkTask):
    """
    This is an abstract class that extracts metadata over the full dataset.

    We create a metadata csv file that will be used by downstream
    luigi tasks to curate the final dataset.

    The metadata columns are:
        * relpath - How you find the file path in the original dataset.
        * slug - This is the filename in our dataset. It should be
        unique, it should be obvious what the original filename
        was, and perhaps it should contain the label for audio scene
        tasks.
        * subsample_key - Hash or a tuple of hash to do subsampling
        * split - Split of this particular audio file.
        * label - Label for the scene or event.
        * start, end - Start and end time in seconds of the event,
        for event_labeling tasks.
    """

    outfile = luigi.Parameter()

    # This should have something like the following:
    # train = luigi.TaskParameter()
    # test = luigi.TaskParameter()

    def requires(self):
        ...
        # This should have something like the following:
        # return { "train": self.train, "test": self.test }

    @staticmethod
    def slugify_file_name(relative_path: str) -> str:
        """
        This is the filename in our dataset.

        It should be unique, it should be obvious what the original
        filename was, and perhaps it should contain the label for
        audio scene tasks.
        You can override this and simplify if the slugified filename
        for this dataset is too long.
        TODO: Remove the workdir, if it's present.
        """
        name, ext = os.path.splitext(os.path.basename(relative_path))
        return f"{slugify(str(name))}"

    @staticmethod
    def get_subsample_key(slug: str):
        """
        Gets the subsample key.
        Subsample key is a unique hash at a file level used for subsampling.

        This hash can be composed of multiple hashes as a tuple or can be a
        single hash
        For example - (hash1, hash2, ...). In this case the subsampling is done by
        considering successive hashes in priority.
        This ensures two things -
            1. Priority wise subsampling.
                The fate of files with same hash1 is decided in a group. Either
                they are subsampled or they are not.
            2. Stable sampling
                Each row, either a tuple or not uniquely identifies the data point.
                This ensures that the sampling is unique everytime the pipeline
                runs

        The base method makes the hash of the slug. This can be overridden
        to return a tuple as well
        """
        # Filename hash is a unique hash at a file level.
        filename_hash = filename_to_int_hash(slug)
        # This way the sampling will be at file level
        # To make some grouped subsampling please consider overriding this
        # and returning a tuple (see speech_command.py for example)
        subsample_key = filename_hash
        return subsample_key

    def get_process_metadata(self) -> pd.DataFrame:
        """
        Return a dataframe containing the task metadata for this
        entire task.

        By default, we do one split at a time and then concat them.
        You might consider overriding this for some datasets (like
        Google Speech Commands) where you cannot process metadata
        on a per-split basis.
        """
        process_metadata = pd.concat(
            [
                self.get_split_metadata(split["name"])
                for split in self.data_config["splits"]
            ]
        )
        return process_metadata

    def run(self):
        process_metadata = self.get_process_metadata()

        if self.data_config["task_type"] == "event_labeling":
            assert set(
                ["relpath", "slug", "subsample_key", "split", "label", "start", "end"]
            ).issubset(set(process_metadata.columns))
        elif self.data_config["task_type"] == "scene_labeling":
            assert set(["relpath", "slug", "subsample_key", "split", "label"]).issubset(
                set(process_metadata.columns)
            )
        else:
            raise ValueError("%s task_type unknown" % self.data_config["task_type"])

        process_metadata.to_csv(
            self.workdir.joinpath(self.outfile),
            index=False,
        )

        self.mark_complete()


class DownloadCorpus(WorkTask):
    """
    Downloads from the url and saveds it in the workdir with name
    outfile
    """

    url = luigi.Parameter()
    outfile = luigi.Parameter()

    def run(self):
        download_file(self.url, self.workdir.joinpath(self.outfile))
        self.mark_complete()

    @property
    def stage_number(self) -> int:
        return 0


class ExtractArchive(WorkTask):
    """
    Extracts the downloaded file in the workdir(optionally in subdir inside
    workdir)

    Parameter
        infile: filename which has to be extracted from the
            download task working directory
        download(DownloadCorpus): task which downloads the corpus to be
            extracted
    Requires:
        download(DownloadCorpus): task which downloads the corpus to be
            extracted
    """

    infile = luigi.Parameter()
    download = luigi.TaskParameter(
        visibility=luigi.parameter.ParameterVisibility.PRIVATE
    )
    # Outdir is the sub dir inside the workdir to extract the file.
    # If set to None the file is extracted in the workdir without any
    # subdir
    outdir = luigi.Parameter(default=None)

    def requires(self):
        return {"download": self.download}

    def run(self):
        archive_path = self.requires()["download"].workdir.joinpath(self.infile)
        archive_path = archive_path.absolute()
        output_path = self.workdir
        if self.outdir is not None:
            output_path = output_path.joinpath(self.outdir)
        shutil.unpack_archive(archive_path, output_path)

        self.mark_complete()


class SubsampleSplit(WorkTask):
    """
    A subsampler that acts on a specific split.
    All instances of this will depend on the combined process metadata csv.

    Parameters:
        split: name of the split for which subsampling has to be done
        max_files: maximum files required from the subsampling
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requirements:
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    """

    split = luigi.Parameter()
    max_files = luigi.IntParameter()
    metadata = luigi.TaskParameter()

    def requires(self):
        # The meta files contain the path of the files in the data
        # so we dont need to pass the extract as a dependency here.
        return {
            "metadata": self.metadata,
        }

    def get_metadata(self):
        metadata = pd.read_csv(
            self.requires()["metadata"].workdir.joinpath(
                self.requires()["metadata"].outfile
            )
        )[["subsample_key", "slug", "relpath", "split"]]

        # Since event detection metadata will have duplicates, we de-dup
        # TODO: We might consider different choices of subset
        metadata = (
            metadata.sort_values(by="subsample_key")
            # Drop duplicates as the subsample key is expected to be unique
            .drop_duplicates(subset="subsample_key", ignore_index=True)
            # Select the split to subsample
            .loc[lambda df: df["split"] == self.split]
        )
        return metadata

    def run(self):

        metadata = self.get_metadata()
        num_files = len(metadata)
        max_files = num_files if self.max_files is None else self.max_files
        if num_files > max_files:
            print(
                f"{len(str(num_files))} audio files in corpus, keeping only {max_files}"
            )

        # Sort by the subsample key and select the max_files number of samples
        metadata = metadata.sort_values(by="subsample_key").iloc[
            :max_files
        ]

        for _, audio in metadata.iterrows():
            audiofile = Path(audio["relpath"])
            newaudiofile = Path(
                self.workdir.joinpath(f"{audio['slug']}{audiofile.suffix}")
            )
            newaudiofile.unlink(missing_ok=True)
            newaudiofile.symlink_to(audiofile.resolve())

        self.mark_complete()


class SubsampleSplits(WorkTask):
    """
    Aggregates subsampling of all the splits into a single task as dependencies.

    Parameter:
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires:
        subsample_splits (list(SubsampleSplit)): task subsamples each split
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        # Perform subsampling on each split independently
        subsample_splits = {
            split["name"]: SubsampleSplit(
                metadata=self.metadata,
                split=split["name"],
                max_files=split["max_files"],
                data_config=self.data_config,
            )
            for split in self.data_config["splits"]
        }
        return subsample_splits

    def run(self):
        workdir = Path(self.workdir)
        workdir.rmdir()
        # We need to link the workdir of the requires, they will all be the same
        # for all the requires so just grab the first one.
        key = list(self.requires().keys())[0]
        workdir.symlink_to(Path(self.requires()[key].workdir).absolute())
        self.mark_complete()


class MonoWavTrimCorpus(WorkTask):
    """
    Converts the file to mono, changes to wav encoding,
    trims and pads the audio to be same length

    Parameters
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires:
        corpus(SubsampleSplits): task which aggregates the subsampling for each split
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        return {
            "corpus": SubsampleSplits(
                metadata=self.metadata, data_config=self.data_config
            )
        }

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

    metadata = luigi.TaskParameter()

    def requires(self):
        # The metadata helps in provide the split type for each
        # audio file
        return {
            "corpus": MonoWavTrimCorpus(
                metadata=self.metadata, data_config=self.data_config
            ),
            "metadata": self.metadata,
        }

    def run(self):
        # Get the process metadata. This gives the freedom of picking the train test
        # label either from the provide metadata file or any other method.

        # Writing slug and split makes it explicit that these columns are required
        meta = self.requires()["metadata"]
        process_metadata = pd.read_csv(
            os.path.join(meta.workdir, meta.outfile),
        )[["slug", "split"]]

        # Go over the subsampled folder and pick the audio files. The audio files are
        # saved with their slug names and hence the corresponding label can be picked
        # up from the preprocess config
        for audiofile in tqdm(self.requires()["corpus"].workdir.glob("*.wav")):
            # The metadata slug doesnot have the extension as monowavtrim has not
            # been run yet so get the name without the extension
            split = process_metadata.loc[
                process_metadata["slug"] == audiofile.stem, "split"
            ].values[0]
            split_dir = self.workdir.joinpath(split)
            split_dir.mkdir(exist_ok=True)
            newaudiofile = new_basedir(audiofile, split_dir)
            os.symlink(os.path.realpath(audiofile), newaudiofile)

        self.mark_complete()


class SplitTrainTestMetadata(WorkTask):
    """
    Parameters
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires
        traintestcorpus(SplitTrainTestCorpus): which produces the split
            level corpus
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        return {
            "traintestcorpus": SplitTrainTestCorpus(
                metadata=self.metadata, data_config=self.data_config
            ),
            "metadata": self.metadata,
        }

    def get_metadata(self):
        metadata = pd.read_csv(
            self.requires()["metadata"].workdir.joinpath(
                self.requires()["metadata"].outfile
            )
        )
        return metadata

    def run(self):
        labeldf = self.get_metadata()

        for split_path in self.requires()["traintestcorpus"].workdir.iterdir():
            audiodf = pd.DataFrame(
                [(a.stem, a.suffix) for a in split_path.glob("*.wav")],
                columns=["slug", "ext"],
            )
            assert len(audiodf) != 0, f"No audio files found in: {split_path}"
            assert (
                not audiodf["slug"].duplicated().any()
            ), "Duplicate files in: {split_path}"

            # Get the label from the metadata with the help of the slug of the filename
            sublabeldf = (
                labeldf.merge(audiodf, on="slug")
                .assign(slug=lambda df: df["slug"] + df["ext"])
                .drop("ext", axis=1)
            )

            if self.data_config["task_type"] == "scene_labeling":
                # Check if all the labels were found from the metadata
                assert len(sublabeldf) == len(audiodf)
            elif self.data_config["task_type"] == "event_labeling":
                # This won't work for sound event detection where there might be
                # zero or more than one event per file
                pass
            else:
                raise ValueError("Invalid task_type in dataset config")

            # Save the slug and the label in as the split metadata
            sublabeldf.to_csv(
                self.workdir.joinpath(f"{split_path.stem}.csv"),
                index=False,
            )

        self.mark_complete()


class MetadataVocabulary(WorkTask):
    """
    Parameters
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires
        traintestmeta(SplitTrainTestMetadata): task which produces the split
            level metadata
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        return {
            "traintestmeta": SplitTrainTestMetadata(
                metadata=self.metadata, data_config=self.data_config
            )
        }

    def run(self):
        labelset = set()
        # Iterate over all the files in the traintestmeta and get the split_metadata
        for split_metadata in self.requires()["traintestmeta"].workdir.iterdir():
            labeldf = pd.read_csv(split_metadata)
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
        )

        self.mark_complete()


class ResampleSubCorpus(WorkTask):
    """
    Resamples the Subsampled corpus in different sampling rate
    Parameters
        split(str): The split for which the resampling has to be done
        sr(int): output sampling rate
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires
        traintestcorpus(SplitTrainTestCorpus): task which produces the split
            level corpus
    """

    sr = luigi.IntParameter()
    split = luigi.Parameter()
    metadata = luigi.TaskParameter()

    def requires(self):
        return {
            "traintestcorpus": SplitTrainTestCorpus(
                metadata=self.metadata, data_config=self.data_config
            )
        }

    def run(self):
        original_dir = self.requires()["traintestcorpus"].workdir.joinpath(
            str(self.split)
        )
        resample_dir = self.workdir.joinpath(str(self.sr)).joinpath(str(self.split))
        resample_dir.mkdir(parents=True, exist_ok=True)
        for audiofile in tqdm(original_dir.glob("*.wav")):
            resampled_audiofile = new_basedir(audiofile, resample_dir)
            audio_util.resample_wav(audiofile, resampled_audiofile, self.sr)

        self.mark_complete()


class FinalizeCorpus(WorkTask):
    """
    Create a final corpus, no longer in _workdir but in the top-level
    at directory config.TASKNAME.
    Parameters:
        sample_rates (list(int)): The list of sampling rates in which the corpus
            is required
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires:
        resample(List(ResampleSubCorpus)): list of task which resamples each split
        traintestmeta(SplitTrainTestMetadata): task which produces the split
            level metadata
    """

    sample_rates = luigi.ListParameter()
    metadata = luigi.TaskParameter()

    def requires(self):
        # Will copy the resampled data and the traintestmeta and the vocabmeta
        splits = [p["name"] for p in self.data_config["splits"]]
        return {
            "resample": [
                ResampleSubCorpus(
                    sr=sr,
                    split=split,
                    metadata=self.metadata,
                    data_config=self.data_config,
                )
                for sr in self.sample_rates
                for split in splits
            ],
            "traintestmeta": SplitTrainTestMetadata(
                metadata=self.metadata, data_config=self.data_config
            ),
            "vocabmeta": MetadataVocabulary(
                metadata=self.metadata, data_config=self.data_config
            ),
        }

    # We overwrite workdir here, because we want the output to be
    # the finalized top-level task directory
    @property
    def workdir(self):
        return Path("tasks").joinpath(self.versioned_task_name)

    def run(self):
        if self.workdir.exists():
            shutil.rmtree(self.workdir)

        # Copy the resampled files
        # The workdirectory of the all the resample task is same
        # so select the first one
        shutil.copytree(self.requires()["resample"][0].workdir, self.workdir)

        # Copy the traintestmetadata
        shutil.copytree(
            self.requires()["traintestmeta"].workdir,
            self.workdir,
            dirs_exist_ok=True,
        )
        # Copy the vocabmetadata
        shutil.copytree(
            self.requires()["vocabmeta"].workdir,
            self.workdir,
            dirs_exist_ok=True,
        )
        # Save the dataset config as a json file
        config_out = self.workdir.joinpath("dataset_metadata.json")
        with open(config_out, "w") as fp:
            json.dump(
                self.data_config, fp, indent=True, cls=luigi.parameter._DictParamEncoder
            )

        self.mark_complete()


def run(task: Union[List[luigi.Task], luigi.Task], num_workers: int):
    """
    Run a task / set of tasks

    Args:
        task: a single or list of luigi tasks
        num_workers: Number of CPU workers to use for this task
    """

    # If this is just a single task then add it to a list
    if isinstance(task, luigi.Task):
        task = [task]

    Path("_workdir").mkdir(exist_ok=True)
    luigi.build(
        task,
        workers=num_workers,
        local_scheduler=True,
        log_level="INFO",
    )


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

    Determines which data split the file should belong to, based
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
        result = "valid"
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
