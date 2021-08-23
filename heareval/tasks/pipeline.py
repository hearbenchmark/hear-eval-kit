"""
Generic pipelines for datasets
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Union
from urllib.parse import urlparse

import luigi
import pandas as pd
from pandas import DataFrame, Series
from slugify import slugify
from tqdm import tqdm

import heareval.tasks.util.audio as audio_util
from heareval.tasks.util.luigi import (
    WorkTask,
    download_file,
    filename_to_int_hash,
    new_basedir,
    subsample_metadata,
    which_set,
)

SPLITS = ["train", "valid", "test"]
# This percentage should not be changed as this decides
# the data in the split and hence is not a part of the data config
VALIDATION_PERCENTAGE = 20
TEST_PERCENTAGE = 20

# We want no more than 5 hours of audio per task.
MAX_TASK_DURATION_BY_SPLIT = {
    "train": 3 * 3600,
    "valid": 1 * 3600,
    "test": 1 * 3600,
}


class DownloadCorpus(WorkTask):
    """
    Downloads from the url and saveds it in the workdir with name
    outfile
    """

    url = luigi.Parameter()
    outfile = luigi.Parameter()
    expected_md5 = luigi.Parameter()

    def run(self):
        download_file(self.url, self.workdir.joinpath(self.outfile), self.expected_md5)
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
        audio_util.audio_dir_stats_wav(
            in_dir=output_path,
            out_file=self.workdir.joinpath(f"{self.outdir}_stats.json"),
        )

        self.mark_complete()


def get_download_and_extract_tasks(task_config: Dict):
    """
    Iterates over the dowload urls and builds download and extract
    tasks for them
    """

    tasks = {}
    for urlobj in task_config["download_urls"]:
        name, url, md5 = urlobj["name"], urlobj["url"], urlobj["md5"]
        filename = os.path.basename(urlparse(url).path)
        task = ExtractArchive(
            download=DownloadCorpus(
                url=url, outfile=filename, expected_md5=md5, task_config=task_config
            ),
            infile=filename,
            outdir=name,
            task_config=task_config,
        )
        tasks[name] = task

    return tasks


class ExtractMetadata(WorkTask):
    """
    This is an abstract class that extracts metadata (including labels)
    over the full dataset.
    If we detect that the original dataset doesn't have a full
    train/valid/test split, we extract 20% validation/test if it
    is missing.

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
    def slugify_file_name(relative_path: str):
        """
        This is the filename in our dataset.

        It should be unique, it should be obvious what the original
        filename was, and perhaps it should contain the label for
        audio scene tasks.
        You can override this and simplify if the slugified filename
        for this dataset is too long.

        The basic version here takes the filename and slugifies it.
        """
        slug_text = str(Path(relative_path).stem)
        slug_text = slug_text.replace('-', '_negative_')
        return f"{slugify(slug_text)}"

    @staticmethod
    def get_stratify_key(df: DataFrame) -> Series:
        """
        Get the stratify key

        Subsampling is stratified based on this key.
        Since hashing is only required for ordering the samples
        for subsampling, the stratify key should not necessarily be a hash,
        as it is only used to group the data points before subsampling.
        The actual subsampling is done by the split key and
        the subsample key.

        By default, the label is used for stratification
        """
        assert "label" in df, "label column not found in the dataframe"
        return df["label"]

    @staticmethod
    def get_split_key(df: DataFrame) -> Series:
        """
        Gets the split key.

        This can be the hash of a data specific split like instrument generating
        the file or person recording the sound.
        For subsampling, the data is subsampled in chunks of split.
        By default this is the hash of the filename, but can be overridden if the
        data needs to split into certain groups while subsampling
        i.e leave out some groups and select others based on this key.

        This key is also used to split the data into test and valid if those splits
        are not made explicitly in the get_process_metadata
        """
        assert "relpath" in df, "relpath column not found in the dataframe"
        file_names = df["relpath"].apply(lambda path: Path(path).name)
        return file_names.apply(filename_to_int_hash)

    @staticmethod
    def get_subsample_key(df: DataFrame) -> Series:
        """
        Gets the subsample key.
        Subsample key is a unique hash at a audio file level used for subsampling.
        This is a hash of the slug. This is not recommended to be
        overridden.

        The data is first split by the split key and the subsample key is
        used to ensure stable sampling for groups which are incompletely
        sampled(the last group to be part of the subsample output)
        """
        assert "slug" in df, "slug column not found in the dataframe"
        return df["slug"].apply(str).apply(filename_to_int_hash)

    def get_process_metadata(self) -> pd.DataFrame:
        """
        Return a dataframe containing the task metadata for this
        entire task.

        By default, we do one split at a time and then concat them.
        This runs only for the splits which have a requires task. All
        other splits are sampled with the split train test val function.
        You might consider overriding this for some datasets (like
        Google Speech Commands) where you cannot process metadata
        on a per-split basis.
        """
        process_metadata = pd.concat(
            [
                self.get_split_metadata(split)
                # The splits should come from the requires and not from the data config
                # splits as the splits in data config might need to be generated from
                # the training split with split train test val function
                for split in list(self.requires().keys())
            ]
        )
        return process_metadata

    def split_train_test_val(self, metadata: pd.DataFrame):
        """
        This functions splits the metadata into test, train and valid from train
        split if any of test or valid split is not found
        Three cases might arise -
        1. Validation split not found - Train will be split into valid and train
        2. Test split not found - Train will be split into test and train
        3. Validation and Test split not found - Train will be split into test, train
            and valid
        If there is any data specific split that will already be done in
        get_process_metadata. This function is for automatic splitting if the splits
        are not found
        This uses the split key to do the split with the which set function.
        """

        splits_present = metadata["split"].unique()

        # The metadata should at least have the train split
        # test and valid if not found in the metadata can be sampled
        # from the train
        assert "train" in splits_present, "Train split not found in metadata"
        splits_to_sample = set(SPLITS).difference(splits_present)
        print(
            "Splits not already present in the dataset, "
            + f"now sampled with split key are: {splits_to_sample}"
        )

        # Depending on whether valid and test are already present, the percentage can
        # either be the predefined percentage or 0
        valid_perc = VALIDATION_PERCENTAGE if "valid" in splits_to_sample else 0
        test_perc = TEST_PERCENTAGE if "test" in splits_to_sample else 0

        metadata[metadata["split"] == "train"] = metadata[
            metadata["split"] == "train"
        ].assign(
            split=lambda df: df["split_key"].apply(
                # Use the which set to split the train into the required splits
                lambda split_key: which_set(split_key, valid_perc, test_perc)
            )
        )
        return metadata

    def run(self):
        process_metadata = self.get_process_metadata()
        # Split the metadata to create valid and test set from train if they are not
        # created explicitly in the get process metadata
        process_metadata = self.split_train_test_val(process_metadata)

        if self.task_config["embedding_type"] == "event":
            assert set(
                [
                    "relpath",
                    "slug",
                    "stratify_key",
                    "split_key",
                    "subsample_key",
                    "split",
                    "label",
                    "start",
                    "end",
                ]
            ).issubset(set(process_metadata.columns))
        elif self.task_config["embedding_type"] == "scene":
            assert set(
                [
                    "relpath",
                    "slug",
                    "stratify_key",
                    "split_key",
                    "subsample_key",
                    "split",
                    "label",
                ]
            ).issubset(set(process_metadata.columns))
            # Multiclass predictions should only have a single label per file
            if self.task_config["prediction_type"] == "multiclass":
                label_count = process_metadata.groupby("slug")["label"].aggregate(
                    lambda group: len(group)
                )
                assert (label_count == 1).all()
        else:
            raise ValueError(
                "%s embedding_type unknown" % self.task_config["embedding_type"]
            )

        # Filter the files which actually exists in the data
        exists = process_metadata["relpath"].apply(
            lambda relpath: Path(relpath).exists()
        )

        # If any of the audio files in the metadata is missing, raise an error for the
        # regular dataset. However, in case of small dataset, this is expected and we
        # need to remove those entries from the metadata
        if sum(exists) < len(process_metadata):
            if self.task_config["version"].split("-")[-1] == "small":
                print(
                    "All files in metadata donot exist in the dataset. This is "
                    "expected behavior when small task is running."
                    f"Removing {len(process_metadata) - sum(exists)} entries in the "
                    "metadata"
                )
                process_metadata = process_metadata.loc[exists]
            else:
                raise FileNotFoundError(
                    "Files in the metadata are missing in the directory"
                )

        process_metadata.to_csv(
            self.workdir.joinpath(self.outfile),
            index=False,
        )

        # Save the label count for each split
        for split, split_df in process_metadata.groupby("split"):
            json.dump(
                split_df["label"].value_counts().to_dict(),
                self.workdir.joinpath(f"labelcount_{split}.json").open("w"),
                indent=True,
            )

        self.mark_complete()


class SubsampleSplit(WorkTask):
    """
    A subsampler that acts on a specific split.
    All instances of this will depend on the combined process metadata csv.

    Parameters:
        split: name of the split for which subsampling has to be done
        max_files: maximum files required from the subsampling
        metadata (ExtractMetadata): task which extracts corpus level metadata
    Requirements:
        metadata (ExtractMetadata): task which extracts corpus level metadata
    """

    split = luigi.Parameter()
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
        )[["split", "stratify_key", "split_key", "subsample_key", "slug", "relpath"]]

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
        # This might round badly for small corpora with long audio :\
        # TODO: Issue to check for this
        sample_duration = self.task_config["sample_duration"]
        max_files = int(MAX_TASK_DURATION_BY_SPLIT[self.split] / sample_duration)
        if num_files > max_files:
            print(
                f"{num_files} audio files in corpus."
                f"Max files to subsample: {max_files}"
            )
            sampled_metadata = subsample_metadata(metadata, max_files)
            print(f"Datapoints in split after resampling: {len(sampled_metadata)}")
            assert subsample_metadata(metadata.sample(frac=1), max_files).equals(
                sampled_metadata
            ), "The subsampling is not stable"
        else:
            sampled_metadata = metadata

        for _, audio in sampled_metadata.iterrows():
            audiofile = Path(audio["relpath"])
            # Add the original extension to the slug
            newaudiofile = Path(
                self.workdir.joinpath(f"{audio['slug']}{audiofile.suffix}")
            )
            # missing_ok is python >= 3.8
            if newaudiofile.exists():
                newaudiofile.unlink()
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
            split: SubsampleSplit(
                metadata=self.metadata,
                split=split,
                task_config=self.task_config,
            )
            for split in SPLITS
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
        corpus (SubsampleSplits): task which aggregates the subsampling for each split
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        return {
            "corpus": SubsampleSplits(
                metadata=self.metadata, task_config=self.task_config
            )
        }

    def run(self):
        # TODO: this should check to see if the audio is already a mono wav at the
        #   correct length and just create a symlink if that is this case.
        for audiofile in tqdm(list(self.requires()["corpus"].workdir.iterdir())):
            newaudiofile = self.workdir.joinpath(f"{audiofile.stem}.wav")
            audio_util.mono_wav_and_fix_duration(
                audiofile, newaudiofile, duration=self.task_config["sample_duration"]
            )

        self.mark_complete()


class SplitData(WorkTask):
    """
    Go over the subsampled folder and pick the audio files. The audio files are
    saved with their slug names and hence the corresponding label can be picked
    up from the preprocess config. (These are symlinks.)

    Parameters
        metadata (ExtractMetadata): task which extracts a corpus level metadata
            the metadata helps to provide the split type of each audio file
    Requires
        corpus(MonoWavTrimCorpus): which processes the audio file and converts
            them to wav format
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        # The metadata helps in provide the split type for each
        # audio file
        return {
            "corpus": MonoWavTrimCorpus(
                metadata=self.metadata, task_config=self.task_config
            ),
            "metadata": self.metadata,
        }

    def run(self):

        meta = self.requires()["metadata"]
        metadata = pd.read_csv(
            os.path.join(meta.workdir, meta.outfile),
        )[["slug", "split"]]

        for audiofile in tqdm(list(self.requires()["corpus"].workdir.glob("*.wav"))):
            # Compare the filename with the slug.
            # Please note that the slug doesnot has the extension of the file
            split = metadata.loc[metadata["slug"] == audiofile.stem, "split"].values[0]
            split_dir = self.workdir.joinpath(split)
            split_dir.mkdir(exist_ok=True)
            newaudiofile = new_basedir(audiofile, split_dir)
            os.symlink(os.path.realpath(audiofile), newaudiofile)

        self.mark_complete()


class SplitMetadata(WorkTask):
    """
    Splits the label dataframe.

    Parameters
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires
        data (SplitData): which produces the split level corpus
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        return {
            "data": SplitData(metadata=self.metadata, task_config=self.task_config),
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

        for split_path in self.requires()["data"].workdir.iterdir():
            audiodf = pd.DataFrame(
                [(a.stem, a.suffix) for a in list(split_path.glob("*.wav"))],
                columns=["slug", "ext"],
            )
            assert len(audiodf) != 0, f"No audio files found in: {split_path}"
            assert (
                not audiodf["slug"].duplicated().any()
            ), "Duplicate files in: {split_path}"

            # Get the label from the metadata with the help of the slug of the filename
            audiolabel_df = (
                labeldf.merge(audiodf, on="slug")
                .assign(slug_path=lambda df: df["slug"] + df["ext"])
                .drop("ext", axis=1)
            )

            if self.task_config["embedding_type"] == "scene":
                # Create a dictionary containing a list of metadata keyed on the slug.
                audiolabel_json = (
                    audiolabel_df[["slug_path", "label"]]
                    .groupby("slug_path")["label"]
                    .apply(list)
                    .to_dict()
                )

            elif self.task_config["embedding_type"] == "event":
                # For event labeling each file will have a list of metadata
                audiolabel_json = (
                    audiolabel_df[["slug_path", "label", "start", "end"]]
                    .set_index("slug_path")
                    .groupby(level=0)
                    .apply(lambda group: group.to_dict(orient="records"))
                    .to_dict()
                )
            else:
                raise ValueError("Invalid embedding_type in dataset config")

            # Save the json used for training purpose
            json.dump(
                audiolabel_json,
                self.workdir.joinpath(f"{split_path.stem}.json").open("w"),
                indent=True,
            )

            # Save the slug and the label in as the split metadata
            audiolabel_df.to_csv(
                self.workdir.joinpath(f"{split_path.stem}.csv"),
                index=False,
            )

        self.mark_complete()


class MetadataVocabulary(WorkTask):
    """
    Creates the vocabulary CSV file for a task.

    Parameters
        metadata (ExtractMetadata): task which extracts a corpus level metadata
    Requires
        splitmeta (SplitMetadata): task which produces the split
            level metadata
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        return {
            "splitmeta": SplitMetadata(
                metadata=self.metadata, task_config=self.task_config
            )
        }

    def run(self):
        labelset = set()
        # Iterate over all the files in the split metadata and get the
        # split_metadata
        for split_metadata in list(self.requires()["splitmeta"].workdir.glob("*.csv")):
            labeldf = pd.read_csv(split_metadata)
            json.dump(
                labeldf["label"].value_counts().to_dict(),
                self.workdir.joinpath(f"labelcount_{split_metadata.stem}.json").open(
                    "w"
                ),
                indent=True,
            )
            labelset = labelset | set(labeldf["label"].unique().tolist())

        # Build the label idx csv and save it
        labelcsv = pd.DataFrame(
            [(idx, label) for (idx, label) in enumerate(sorted(list(labelset)))],
            columns=["idx", "label"],
        )

        labelcsv.to_csv(
            os.path.join(self.workdir, "labelvocabulary.csv"),
            columns=["idx", "label"],
            index=False,
        )

        self.mark_complete()


class ResampleSubcorpus(WorkTask):
    """
    Resamples the Subsampled corpus in different sampling rate
    Parameters
        split(str): The split for which the resampling has to be done
        sr(int): output sampling rate
        metadata (ExtractMetadata): task which extracts corpus level metadata
    Requires
        data (SplitData): task which produces the split
            level corpus
    """

    sr = luigi.IntParameter()
    split = luigi.Parameter()
    metadata = luigi.TaskParameter()

    def requires(self):
        return {"data": SplitData(metadata=self.metadata, task_config=self.task_config)}

    def run(self):
        original_dir = self.requires()["data"].workdir.joinpath(str(self.split))
        resample_dir = self.workdir.joinpath(str(self.sr)).joinpath(str(self.split))
        resample_dir.mkdir(parents=True, exist_ok=True)
        for audiofile in tqdm(list(original_dir.glob("*.wav"))):
            resampled_audiofile = new_basedir(audiofile, resample_dir)
            audio_util.resample_wav(audiofile, resampled_audiofile, self.sr)

        audio_util.audio_dir_stats_wav(
            in_dir=resample_dir,
            out_file=self.workdir.joinpath(str(self.sr)).joinpath(
                f"{self.split}_stats.json"
            ),
        )
        self.mark_complete()


class ResampleSubcorpuses(WorkTask):
    """
    Aggregates resampling of all the splits and sampling rates
    into a single task as dependencies.

    Parameter:
        metadata (ExtractMetadata): task which extracts corpus level metadata
    Requires:
        subsample_splits (list(SubsampleSplit)): task subsamples each split
    """

    sample_rates = luigi.ListParameter()
    metadata = luigi.TaskParameter()

    def requires(self):
        # Perform resampling on each split and sampling rate independently
        resample_splits = [
            ResampleSubcorpus(
                sr=sr,
                split=split,
                metadata=self.metadata,
                task_config=self.task_config,
            )
            for sr in self.sample_rates
            for split in SPLITS
        ]
        return resample_splits

    def run(self):
        workdir = Path(self.workdir)
        workdir.rmdir()
        # We need to link the workdir of the requires, they will all be the same
        # for all the requires so just grab the first one.
        requires_workdir = Path(self.requires()[0].workdir).absolute()
        workdir.symlink_to(requires_workdir)
        self.mark_complete()


class FinalizeCorpus(WorkTask):
    """
    Create a final corpus, no longer in _workdir but in the top-level
    at directory config.TASKNAME.
    Parameters:
        sample_rates (list(int)): The list of sampling rates in which the corpus
            is required
        metadata (ExtractMetadata): task which extracts corpus level metadata
    Requires:
        resample (List(ResampleSubCorpus)): list of task which resamples each split
        splitmeta (SplitMetadata): task which produces the split
            level metadata
    """

    sample_rates = luigi.ListParameter()
    metadata = luigi.TaskParameter()
    tasks_dir = luigi.Parameter()

    def requires(self):
        # Will copy the resampled data and the split metadata and the vocabmeta
        return {
            "resample": ResampleSubcorpuses(
                sample_rates=self.sample_rates,
                metadata=self.metadata,
                task_config=self.task_config,
            ),
            "splitmeta": SplitMetadata(
                metadata=self.metadata, task_config=self.task_config
            ),
            "vocabmeta": MetadataVocabulary(
                metadata=self.metadata, task_config=self.task_config
            ),
        }

    # We overwrite workdir here, because we want the output to be
    # the finalized top-level task directory
    @property
    def workdir(self):
        return Path(self.tasks_dir).joinpath(self.versioned_task_name)

    def run(self):
        if self.workdir.exists():
            shutil.rmtree(self.workdir)

        # Copy the resampled files
        shutil.copytree(self.requires()["resample"].workdir, self.workdir)

        # Copy labelvocabulary.csv
        shutil.copy2(
            self.requires()["vocabmeta"].workdir.joinpath("labelvocabulary.csv"),
            self.workdir.joinpath("labelvocabulary.csv"),
        )
        # Copy the train test metadata jsons
        src = self.requires()["splitmeta"].workdir
        dst = self.workdir
        for item in os.listdir(src):
            if item.endswith(".json"):
                # Based upon https://stackoverflow.com/a/27161799
                assert not dst.joinpath(item).exists()
                assert not src.joinpath(item).is_dir()
                shutil.copy2(src.joinpath(item), dst.joinpath(item))
        # Python >= 3.8 only
        # shutil.copytree(src, dst, dirs_exist_ok=True, \
        #        ignore=shutil.ignore_patterns("*.csv"))
        # Save the dataset config as a json file
        config_out = self.workdir.joinpath("task_metadata.json")
        with open(config_out, "w") as fp:
            json.dump(
                self.task_config, fp, indent=True, cls=luigi.parameter._DictParamEncoder
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

    luigi.build(
        task,
        workers=num_workers,
        local_scheduler=True,
        log_level="INFO",
    )
