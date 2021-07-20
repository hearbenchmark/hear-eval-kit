"""
Configuration for the google speech commands task
"""

from .dataset_config import DatasetConfig


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

        # This could also be a list of PartitionConfig classes or something.
        self.partitions = {
            "train": {"max_files": None},
            "valid": {"max_files": None},
            "test": {"max_files": None},
        }
