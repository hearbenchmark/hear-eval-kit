#!/usr/bin/env python3

"""A simple python script template.
"""

import os
import sys
import argparse
import importlib

import torch
import tensorflow as tf


class ModelError(BaseException):
    """Class for errors in models"""

    pass


class ValidateModel:

    ACCEPTABLE_SAMPLE_RATE = [16000, 22050, 44100, 48000]

    def __init__(self, module_name: str, model_file_path: str):
        self.module_name = module_name
        self.model_file_path = model_file_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.module = None
        self.sample_rate = None
        self.model = None

    def __call__(self):
        self.import_model()
        self.check_input_sample_rate()
        self.check_load_model()

    def import_model(self):
        print(f"Importing {self.module_name}")
        self.module = importlib.import_module(self.module_name)

    def check_input_sample_rate(self):
        print("Checking input_sample_rate")
        self.sample_rate = self.module.input_sample_rate()
        if self.sample_rate not in self.ACCEPTABLE_SAMPLE_RATE:
            raise ModelError(
                f"Input sample rate of {self.sample_rate} is invalid. "
                f"Must be one of {self.ACCEPTABLE_SAMPLE_RATE}"
            )

    def check_load_model(self):
        print("Checking load_model")
        self.model = self.module.load_model(self.model_file_path, self.device)

        if not isinstance(self.model, tf.Module):
            raise ModelError(
                f"Model must be either a PyTorch module: "
                f"https://pytorch.org/docs/stable/generated/torch.nn.Module.html "
                f"or a tensorflow module: "
                f"https://www.tensorflow.org/api_docs/python/tf/Module"
            )


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "module", help="Name of the model package to validate", type=str
    )
    parser.add_argument(
        "--model",
        "-m",
        default="",
        type=str,
        help="Load model weights from this location",
    )
    args = parser.parse_args(arguments)

    ValidateModel(args.module, args.model)()

    print("Looks good!")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
