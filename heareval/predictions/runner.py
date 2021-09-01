#!/usr/bin/env python3
"""
Downstream training, using embeddings as input features and learning
predictions.

TODO: Add CUDA device.
"""

import json
import os
import random
from importlib import import_module
from pathlib import Path
from typing import Optional

import click
import torch
from slugify import slugify
from tqdm import tqdm

from heareval.predictions.task_predictions import task_predictions


@click.command()
@click.argument("module", type=str)
@click.option(
    "--embeddings-dir",
    default="embeddings",
    help="Location of task embeddings to compute predictions on",
    type=click.Path(exists=True),
)
@click.option(
    "--model",
    default=None,
    help="Location of model weights file",
    type=click.Path(exists=True),
)
@click.option(
    "--task",
    default="all",
    help="Task to run. (Default: all)",
    type=str,
)
@click.option(
    "--gpus",
    default=None if not torch.cuda.is_available() else "[0]",
    help='GPUs to use, as JSON string (default: "[0]" if any '
    "are available, none if not). "
    "See https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#select-gpu-devices",  # noqa
    type=str,
)
@click.option(
    "--model-options", default="{}", help="A JSON dict of kwargs to pass to load_model"
)
def runner(
    module: str,
    embeddings_dir: str = "embeddings",
    model: Optional[str] = None,
    task: str = "all",
    gpus: Optional[str] = None if not torch.cuda.is_available() else "[0]",
    model_options: str = "{}",
) -> None:
    if gpus is not None:
        gpus = json.loads(gpus)
    model_options_dict = json.loads(model_options)
    if isinstance(model_options_dict, dict):
        if model_options_dict:
            options_str = "-" + "-".join(
                [
                    "%s=%s" % (slugify(k), slugify(str(v)))
                    for k, v in model_options_dict.items()
                ]
            )
        else:
            options_str = ""
    else:
        raise ValueError("model_options should be a JSON dict")

    embeddings_dir_path = Path(embeddings_dir).joinpath(module + options_str)
    if not embeddings_dir_path.is_dir():
        raise ValueError(
            "Cannot locate directory containing embeddings. "
            f"Ensure that directory named {embeddings_dir_path} exists."
        )

    # We only load this to get the embedding sizes.
    # This is because there's no simple API for querying that.
    # (And perhaps it is model specific anyway.)
    module_clr = import_module(module)
    # Load the model using the model weights path if they were provided
    if model is not None:
        print(f"Loading {module} using: {model}, {model_options_dict}")
        model_obj = module_clr.load_model(model, **model_options_dict)  # type: ignore
    else:
        print(f"Loading {module} using: {model}, {model_options_dict}")
        model_obj = module_clr.load_model("", **model_options_dict)  # type: ignore
    scene_embedding_size = model_obj.scene_embedding_size
    timestamp_embedding_size = model_obj.timestamp_embedding_size
    # Free model obj
    model_obj = None

    if task == "all":
        tasks = list(embeddings_dir_path.iterdir())
    else:
        tasks = [embeddings_dir_path.joinpath(task)]
        assert os.path.exists(tasks[0]), f"{tasks[0]} does not exist"
    random.shuffle(tasks)
    for task_path in tqdm(tasks):
        print(f"Computing predictions for {task_path.name}")
        task_predictions(
            task_path, scene_embedding_size, timestamp_embedding_size, gpus
        )


if __name__ == "__main__":
    runner()
