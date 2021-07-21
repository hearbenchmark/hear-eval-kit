"""
Configuration for the nsynth pitch detection task
"""

from .dataset_config import PartitionedDatasetConfig, PartitionConfig


class NSynthPitchConfig(PartitionedDatasetConfig):
    def __init__(self):
        super().__init__(
            task_name="nsynth-pitch",
            version="v2.2.3",
            download_urls={
                "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
                "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
                "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
            },
            # All samples will be trimmed / padded to this length
            sample_duration=4.0,
            # Pre-defined partitions in the dataset. Number of files in each split is
            # train: 85,111; valid: 10,102; test: 4890.
            # To subsample a partition, set the max_files to an integer.
            partitions=[
                PartitionConfig(name="train", max_files=1000),
                PartitionConfig(name="valid", max_files=100),
                PartitionConfig(name="test", max_files=100),
            ],
        )
