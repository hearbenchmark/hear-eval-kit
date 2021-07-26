#!/usr/bin/env python3
"""
Computes embeddings on a set of tasks
"""

import os
from dataclasses import dataclass
from typing import Any, List, Union

import click


@click.command()
@click.argument("module", type=str)
@click.option(
    "--model",
    default=None,
    help="Location of model weights file.",
    type=click.Path(exists=True),
)
def runner(module: str, model: str) -> None:
    print(module)
    print(type(model))


if __name__ == "__main__":
    runner()
