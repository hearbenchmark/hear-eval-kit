"""
Generic pipelines for datasets
"""

import os
from pathlib import Path
from typing import Dict, List, Union
from urllib.parse import urlparse
from slugify import slugify

import luigi
import pandas as pd

import heareval.tasks.util.luigi as luigi_util


def get_download_and_extract_tasks(config: Dict):

    tasks = {}
    for name, url in config["download_urls"].items():
        filename = os.path.basename(urlparse(url).path)
        task = luigi_util.ExtractArchive(
            download=luigi_util.DownloadCorpus(
                url=url, outfile=filename, data_config=config
            ),
            infile=filename,
            outdir=name,
            data_config=config,
        )
        tasks[name] = task

    return tasks


class ExtractMetadata(luigi_util.WorkTask):
    """
    This is an abstract class that ... over the full dataset

    Custom metadata pre-processing. Creates a metadata csv
    file that will be used by downstream luigi tasks to curate the final dataset.
    TODO: It would be nice to have a better description of what this pattern is
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
        It should be unique, it should be obvious what the original filename was,
        and perhaps it should contain the label for audio scene tasks.
        You can override this and simplify if the slugified filename for this dataset is too long.
        TODO: Remove the workdir, if it's present.
        """
        return f"{slugify(relative_path)}.wav"

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
            assert set(process_metadata.columns) == set(
                ["relpath", "slug", "filename_hash", "split", "label", "start", "end"]
            )
        elif self.data_config["task_type"] == "scene_labeling":
            assert set(process_metadata.columns) == set(
                ["relpath", "slug", "filename_hash", "split", "label"]
            )
        else:
            raise ValueError("%s task_type unknown" % self.data_config["task_type"])

        process_metadata.to_csv(
            os.path.join(self.workdir, self.outfile),
            index=False,
        )

        self.mark_complete()


class SubsampleSplit(luigi_util.SubsampleSplit):
    """
    A subsampler that acts on a specific split.
    All instances of this will depend on the combined process metadata csv.
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        # The meta files contain the path of the files in the data
        # so we dont need to pass the extract as a dependency here.
        return {
            "meta": self.metadata,
        }


class SubsampleSplits(luigi_util.WorkTask):
    """
    Aggregates subsampling of all the splits into a single task as dependencies.
    All the subsampled files are stored in the requires workdir, so we just link to
    that since there aren't any real outputs associated with this task.
    This is a bit of a hack -- but it allows us to avoid rewriting
    the Subsample task as well as take advantage of Luigi concurrency.
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


class MonoWavTrimCorpus(luigi_util.MonoWavTrimCorpus):

    metadata = luigi.TaskParameter()

    def requires(self):
        return {
            "corpus": SubsampleSplits(
                metadata=self.metadata, data_config=self.data_config
            )
        }


class SplitTrainTestCorpus(luigi_util.SplitTrainTestCorpus):

    metadata = luigi.TaskParameter()

    def requires(self):
        # The metadata helps in provide the split type for each
        # audio file
        return {
            "corpus": MonoWavTrimCorpus(
                metadata=self.metadata, data_config=self.data_config
            ),
            "meta": self.metadata,
        }


class SplitTrainTestMetadata(luigi_util.SplitTrainTestMetadata):

    metadata = luigi.TaskParameter()

    def requires(self):
        # Requires the traintestcorpus and the metadata.
        # The metadata is split into train and test files
        # which are in the traintestcorpus
        return {
            "traintestcorpus": SplitTrainTestCorpus(
                metadata=self.metadata, data_config=self.data_config
            ),
            "meta": self.metadata,
        }


class MetadataVocabulary(luigi_util.MetadataVocabulary):

    metadata = luigi.TaskParameter()

    def requires(self):
        # Depends only on the train test metadata
        return {
            "traintestmeta": SplitTrainTestMetadata(
                metadata=self.metadata, data_config=self.data_config
            )
        }


class ResampleSubCorpus(luigi_util.ResampleSubCorpus):

    metadata = luigi.TaskParameter()

    def requires(self):
        # Requires the train test corpus and will take in
        # parameter for which split and sr the resampling
        # has to be done
        return {
            "traintestcorpus": SplitTrainTestCorpus(
                metadata=self.metadata, data_config=self.data_config
            )
        }


class FinalizeCorpus(luigi_util.FinalizeCorpus):

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

    luigi_util.ensure_dir("_workdir")
    luigi.build(
        task,
        workers=num_workers,
        local_scheduler=True,
        log_level="INFO",
    )
