#!/usr/bin/env python3

"""
Command line tool for pre-processing audio files

- Convert to WAV from another format
- TODO: Resampling
"""

import os
import sys
import argparse
import subprocess
import multiprocessing
from typing import List
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map


def convert_to_wav(in_file: str, out_file: str):
    devnull = open(os.devnull, "w")
    subprocess.call(
        ["ffmpeg", "-y", "-i", in_file, "-c:a", "pcm_f32le", out_file],
        stdout=devnull,
        stderr=devnull,
    )


def batch_convert(folder: str, out_folder: str, extensions: List):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    files_to_convert = []
    output_file_names = []
    for item in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, item)):
            filename, file_ext = os.path.splitext(item)
            if file_ext in extensions:
                output = os.path.join(out_folder, f"{filename}.wav")
                files_to_convert.append(item)
                output_file_names.append(output)

    num_workers = min(32, multiprocessing.cpu_count() + 4)
    print(
        f"Processing {len(files_to_convert)} files using max of {num_workers} workers"
    )
    process_map(
        convert_to_wav,
        files_to_convert,
        output_file_names,
        chunksize=1,
        max_workers=num_workers,
    )


def batch_resample(in_folder, out_folder, sr):
    # TODO: in progress
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", help="Input directory to look for audio files in", type=str
    )
    parser.add_argument(
        "output", help="Output directory to save processed audio files", type=str
    )

    args = parser.parse_args(arguments)

    # TODO: make the the extensions to look for a command line argument?
    batch_convert(args.input, args.output, [".ogg", ".webm"])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
