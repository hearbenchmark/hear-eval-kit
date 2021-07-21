"""
Configuration for the google speech commands task
"""

from .dataset_config import DatasetConfig, PartitionConfig


class SpeechCommands(DatasetConfig):
    def __init__(self):
        super().__init__(task_name="speech_commands", version="v0.0.2")

        # We can potentially add all of these to the base config class to ensure that
        # they get set -- we could also have another base class type PartitionedDataset
        # or something for datasets where there is a predefined partition like in nsynth
        # and google speech commands.
        self.download_urls = {
            "train": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
            "test": "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz",
        }

        # All samples will be trimmed / padded to this length
        self.sample_duration = 1.0

        # Pre-defined partitions in the dataset. Number of files in each split is
        # train: 85,111; valid: 10,102; test: 4890.
        # To subsample a partition, set the max_files to an integer.
        self.partitions = [
            PartitionConfig(name="train", max_files=1000),
            PartitionConfig(name="valid", max_files=100),
            PartitionConfig(name="test", max_files=100),
        ]
