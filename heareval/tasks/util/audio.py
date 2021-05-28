"""
Audio utility functions for evaluation task preparation
"""

import os
import subprocess


def convert_to_mono_wav(in_file: str, out_file: str):
    devnull = open(os.devnull, "w")
    # If we knew the sample rate, we could also pad/trim the audio file now, e.g.:
    # ffmpeg -i test.webm -filter_complex apad=whole_len=44100,atrim=end_sample=44100 -ac 1 -c:a pcm_f32le ./test.wav
    # print(" ".join(["ffmpeg", "-y", "-i", in_file, "-ac", "1", "-c:a", "pcm_f32le", out_file]))
    ret = subprocess.call(
        ["ffmpeg", "-y", "-i", in_file, "-ac", "1", "-c:a", "pcm_f32le", out_file],
        stdout=devnull,
        stderr=devnull,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0


def resample_wav(in_file: str, out_file: str, out_sr: int):
    # TODO: Don't do anything if we are already the right sample rate.
    devnull = open(os.devnull, "w")
    ret = subprocess.call(
        [
            "ffmpeg",
            "-i",
            in_file,
            "-af",
            "aresample=resampler=soxr",
            "-ar",
            str(out_sr),
            out_file,
        ],
        stdout=devnull,
        stderr=devnull,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0
