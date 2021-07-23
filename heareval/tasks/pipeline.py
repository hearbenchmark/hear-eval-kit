"""
Generic pipelines for datasets
"""

import os
from pathlib import Path
from typing import Dict, List, Union
from urllib.parse import urlparse

import luigi

import heareval.tasks.util.luigi as luigi_util
from heareval.tasks.dataset_config import DatasetConfig


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


class SubsamplePartition(luigi_util.SubsamplePartition):
    """
    A subsampler that acts on a specific partition.
    All instances of this will depend on the combined process metadata csv.
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        # The meta files contain the path of the files in the data
        # so we dont need to pass the extract as a dependency here.
        return {
            "meta": self.metadata,
        }


class SubsamplePartitions(luigi_util.WorkTask):
    """
    Aggregates subsampling of all the partitions into a single task as dependencies.
    All the subsampled files are stored in the requires workdir, so we just link to
    that since there aren't any real outputs associated with this task.
    This is a bit of a hack -- but it allows us to avoid rewriting
    the Subsample task as well as take advantage of Luigi concurrency.
    """

    metadata = luigi.TaskParameter()

    def requires(self):
        # Perform subsampling on each partition independently
        subsample_partitions = {
            partition["name"]: SubsamplePartition(
                metadata=self.metadata,
                partition=partition["name"],
                max_files=partition["max_files"],
                data_config=self.data_config,
            )
            for partition in self.data_config["partitions"]
        }
        return subsample_partitions

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
            "corpus": SubsamplePartitions(
                metadata=self.metadata, data_config=self.data_config
            )
        }


class SplitTrainTestCorpus(luigi_util.SplitTrainTestCorpus):

    metadata = luigi.TaskParameter()

    def requires(self):
        # The metadata helps in provide the partition type for each
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
        # parameter for which partition and sr the resampling
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
        partitions = [p["name"] for p in self.data_config["partitions"]]
        return {
            "resample": [
                ResampleSubCorpus(
                    sr=sr,
                    partition=partition,
                    metadata=self.metadata,
                    data_config=self.data_config,
                )
                for sr in self.sample_rates
                for partition in partitions
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
