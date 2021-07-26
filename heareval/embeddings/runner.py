#!/usr/bin/env python3
"""
Computes embeddings on a set of tasks
"""

import importlib
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Union

import click
import torch

TORCH = "torch"
TENSORFLOW = "tf"


class Embedding:
    def __init__(self, module_name: str, model_path: str = None):
        print(f"Importing {module_name}")
        self.module = importlib.import_module(module_name)

        # Load the model using the model weights path if they were provided
        if model_path is not None:
            print(f"Loading model using: {model_path}")
            self.model = self.module.load_model(model_path)
        else:
            self.model = self.module.load_model()

        # Check to see what type of model this is: torch or tensorflow
        if isinstance(self.model, torch.nn.Module):
            self.type = TORCH
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        else:
            import tensorflow as tf

            if isinstance(self.model, tf.Module):
                self.type = TENSORFLOW
                raise NotImplementedError("TensorFlow embeddings not supported yet.")
            else:
                raise ValueError(f"Unsupported model type received: {type(self.model)}")

        print(self.model)


@click.command()
@click.argument("module", type=str)
@click.option(
    "--model",
    default=None,
    help="Location of model weights file",
    type=click.Path(exists=True),
)
@click.option(
    "--tasks-dir",
    default=None,
    help="Location of tasks to compute embeddings on",
    type=click.Path(exists=True),
)
def runner(module: str, model: str = None, tasks_dir: str = None) -> None:

    # Check for directory containing the tasks
    tasks_dir = Path("tasks") if tasks_dir is None else Path(tasks_dir)
    if not tasks_dir.is_dir():
        raise ValueError(
            "Cannot locate directory containing tasks. "
            f"Ensure that directory named {tasks_dir} exists."
        )

    metadata = tasks_dir.glob("*/task_metadata.json")
    print(list(metadata))

    embedding = Embedding(module, model)


if __name__ == "__main__":
    runner()
