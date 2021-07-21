"""
Configuration for the google speech commands task
"""

from .dataset_config import PartitionedDatasetConfig, PartitionConfig


# TODO: Instead of having this as a python class -- this could be a json file and we could
#   have a function that loads it and constructs the correct config object with that
#   values from the json file. We could have a default json file that is loaded
#   automatically -- and then if the user wants to adjust these values they could
#   pass in an additional json config as a command line arg that would overwrite any
#   default values that are specified.
class SpeechCommandsConfig(PartitionedDatasetConfig):
    def __init__(self):
        super().__init__(
            task_name="speech_commands",
            version="v0.0.2",
            download_urls={
                "train": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                "test": "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz",
            },
            # All samples will be trimmed / padded to this length
            sample_duration=1.0,
            # Pre-defined partitions in the dataset. Number of files in each split is
            # train: 85,111; valid: 10,102; test: 4890.
            # To subsample a partition, set the max_files to an integer.
            partitions=[
                PartitionConfig(name="train", max_files=1000),
                PartitionConfig(name="valid", max_files=100),
                PartitionConfig(name="test", max_files=100),
            ],
        )
