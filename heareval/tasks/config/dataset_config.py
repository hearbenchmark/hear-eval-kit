"""
Generic configuration used by all tasks
"""

from typing import Dict, List


class DatasetConfig:
    """
    A base class config class for HEAR datasets.

    Args:
        task_name: Unique name for this task
        version: version string for the dataset
        download_urls: A dictionary of URLs to download the dataset files from
        sample_duration: All samples with be padded / trimmed to this length
    """

    def __init__(
        self, task_name: str, version: str, download_urls: Dict, sample_duration: float
    ):
        self.task_name = task_name
        self.version = version
        self.download_urls = download_urls
        self.sample_duration = sample_duration

        # For deterministic dataset generation
        self.seed = 43

        # Number of CPU works for Luigi jobs
        self.num_workers = 4

        # Default sample rates for HEAR evaluation. If you
        # only use one sample rate this can be a list with
        # only a single rate in it.
        self.sample_rates = [48000, 44100, 22050, 16000]

    @property
    def versioned_task_name(self):
        return f"{self.task_name}-{self.version}"


class PartitionConfig:
    """
    A configuration class for creating named partitions in a dataset

    Args:
        name: name of the partition
        max_files: an integer number of samples to cap this partition at,
            defaults to None for no maximum.
    """

    def __init__(self, name: str, max_files: int = None):
        self.name = name
        self.max_files = max_files


class PartitionedDatasetConfig(DatasetConfig):
    """
    A base class config class for HEAR datasets. This config should be used when
    there are pre-defined data partitions.

    Args:
        task_name: Unique name for this task
        version: version string for the dataset
        download_urls: A dictionary of URLs to download the dataset files from
        sample_duration: All samples with be padded / trimmed to this length
        partitions: A list of PartitionConfig objects describing the partitions
    """

    def __init__(
        self,
        task_name: str,
        version: str,
        download_urls: Dict,
        sample_duration: float,
        partitions: List[PartitionConfig],
    ):
        super().__init__(task_name, version, download_urls, sample_duration)
        self.partitions = partitions
