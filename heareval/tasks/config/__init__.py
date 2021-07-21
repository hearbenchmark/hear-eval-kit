from .dataset_config import DatasetConfig, PartitionedDatasetConfig
from .speech_commands import SpeechCommands

available_configs = {
    "speech_commands": SpeechCommands,
}


def get_config(task: str) -> DatasetConfig:
    """Returns an instantiated dataset config object"""
    if task not in available_configs:
        raise ValueError(f"{task} is not a valid task name.")

    return available_configs[task]()
