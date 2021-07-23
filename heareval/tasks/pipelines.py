"""
Generic pipelines for datasets
"""

from pathlib import Path
import os
from urllib.parse import urlparse

import luigi

from heareval.tasks.dataset_config import DatasetConfig
import heareval.tasks.util.luigi as luigi_util


def get_download_and_extract_tasks(config: DatasetConfig):

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
