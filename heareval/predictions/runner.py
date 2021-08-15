#!/usr/bin/env python3
"""
Downstream training, using embeddings as input features and learning
predictions.

TODO: Add CUDA device.
"""

from importlib import import_module
from pathlib import Path
from typing import Optional

import click
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
def runner(
    module: str, embeddings_dir: str = "embeddings", model: Optional[str] = None
) -> None:
    embeddings_dir_path = Path(embeddings_dir).joinpath(module)
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
        print(f"Loading model using: {model}")
        model_obj = module_clr.load_model(model)  # type: ignore
    else:
        model_obj = module_clr.load_model()  # type: ignore
    scene_embedding_size = model_obj.scene_embedding_size
    timestamp_embedding_size = model_obj.timestamp_embedding_size

    tasks = list(embeddings_dir_path.iterdir())
    for task_path in tqdm(tasks):
        # TODO: Currently has no validation
        if "dcase2016_task2-hear" in str(task_path):
            continue
        print(f"Computing predictions for {task_path.name}")
        task_predictions(task_path, scene_embedding_size, timestamp_embedding_size)


if __name__ == "__main__":
    runner()
