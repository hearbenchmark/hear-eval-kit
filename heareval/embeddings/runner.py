#!/usr/bin/env python3
"""
Computes embeddings on a set of tasks
"""

import json
import os
import shutil
import time
from pathlib import Path

import click
import tensorflow as tf
import torch
from slugify import slugify
from tqdm import tqdm

import heareval.gpu_max_mem as gpu_max_mem
from heareval.embeddings.task_embeddings import Embedding, task_embeddings

if torch.cuda.is_available() and not tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
):
    raise ValueError("GPUs not available in tensorflow, but found by pytorch")


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
    default="tasks",
    help="Location of tasks to compute embeddings on",
    type=str,
)
@click.option(
    "--task",
    default="all",
    help="Task to run. (Default: all)",
    type=str,
)
@click.option(
    "--embeddings-dir", default="embeddings", help="Location to save task embeddings"
)
@click.option(
    "--model-options", default="{}", help="A JSON dict of kwargs to pass to load_model"
)
def runner(
    module: str,
    model: str = None,
    tasks_dir: str = "tasks",
    task: str = "tasks",
    embeddings_dir: str = "embeddings",
    model_options: str = "{}",
) -> None:
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

    # Check for directory containing the tasks
    tasks_dir_path = Path(tasks_dir)
    embeddings_dir_path = Path(embeddings_dir)
    print(embeddings_dir_path)
    if not tasks_dir_path.is_dir():
        raise ValueError(
            "Cannot locate directory containing tasks. "
            f"Ensure that directory named {tasks_dir_path} exists or specify a folder "
            f"containing HEAR tasks using the argument --tasks-dir"
        )

    # Load the embedding model
    embedding = Embedding(module, model, model_options_dict)

    if task == "all":
        tasks = list(tasks_dir_path.iterdir())
    else:
        tasks = [tasks_dir_path.joinpath(task)]
        assert os.path.exists(tasks[0]), f"{tasks[0]} does not exist"
    for task_path in tqdm(tasks):
        # TODO: Would be good to include the version here
        # https://github.com/neuralaudio/hear2021-eval-kit/issues/37
        embed_dir = embeddings_dir_path.joinpath(embedding.name + options_str)

        task_name = task_path.name
        embed_task_dir = embed_dir.joinpath(task_name)

        done_embeddings = embed_task_dir.joinpath(".done.embeddings")
        if os.path.exists(done_embeddings):
            continue

        if os.path.exists(embed_task_dir):
            shutil.rmtree(embed_task_dir)

        start = time.time()
        gpu_max_mem.reset()

        task_embeddings(embedding, task_path, embed_task_dir)

        time_elapsed = time.time() - start
        gpu_max_mem_used = gpu_max_mem.measure()
        print(
            f"...computed embeddings in {time_elapsed} sec "
            f"(GPU max mem {gpu_max_mem_used}) "
            f"for {task_path.name} using {module} {model_options}"
        )
        open(embed_task_dir.joinpath("profile.embeddings.json"), "wt").write(
            json.dumps(
                {
                    "time_elapsed": time_elapsed,
                    "gpu_max_mem": gpu_max_mem_used,
                    "gpu_device_name": gpu_max_mem.device_name(),
                },
                indent=4,
            )
        )

        # Touch this file to indicate that processing completed successfully
        open(done_embeddings, "wt")


if __name__ == "__main__":
    runner()
