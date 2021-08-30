"""
Audio utility functions for evaluation task preparation
"""

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import soundfile as sf
from tqdm import tqdm


def mono_wav_and_fix_duration(in_file: str, out_file: str, duration: float):
    """
    Convert to WAV file and trim to be equal to or less than a specific length
    """
    ret = subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(in_file),
            "-filter_complex",
            f"apad=whole_dur={duration},atrim=end={duration}",
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            str(out_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0, f"ret = {ret}"


def convert_to_mono_wav(in_file: str, out_file: str):
    devnull = open(os.devnull, "w")
    # If we knew the sample rate, we could also pad/trim the audio file now, e.g.:
    # ffmpeg -i test.webm -filter_complex \
    #    apad=whole_len=44100,atrim=end_sample=44100 \
    #    -ac 1 -c:a pcm_f32le ./test.wav
    # print(" ".join(["ffmpeg", "-y", "-i", in_file,
    #    "-ac", "1", "-c:a", "pcm_f32le", out_file]))
    ret = subprocess.call(
        ["ffmpeg", "-y", "-i", in_file, "-ac", "1", "-c:a", "pcm_f32le", out_file],
        stdout=devnull,
        stderr=devnull,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0


def resample_wav(in_file: str, out_file: str, out_sr: int):
    """
    Resample a wave file using SoX high quality mode
    """
    ret = subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            in_file,
            "-af",
            "aresample=resampler=soxr",
            "-ar",
            str(out_sr),
            out_file,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Make sure the return code is 0 and the command was successful.
    assert ret == 0


def audio_stats_wav(in_file: Union[str, Path]):
    """Get statistics for a single wav file"""
    audio = sf.SoundFile(str(in_file))
    return {
        "samples": len(audio),
        "sample_rate": audio.samplerate,
        "duration": len(audio) / audio.samplerate,
    }


def audio_dir_stats_wav(
    in_dir: Union[str, Path], out_file: str, exts: Optional[List[str]] = None
):
    """Produce summary by recursively searching a directory for wav files"""
    if exts is None:
        exts = [".wav", ".mp3", ".ogg"]

    # Filter the files in the directory for the required extensions
    audio_paths = list(
        filter(
            lambda audio_path: audio_path.suffix.lower()
            in map(str.lower, exts),  # type: ignore
            Path(in_dir).absolute().rglob("*"),
        )
    )
    audio_dir_stats = list(
        map(
            audio_stats_wav,
            tqdm(audio_paths),
        )
    )

    durations = [stats["duration"] for stats in audio_dir_stats]
    unique_sample_rates = dict(
        Counter([stats["sample_rate"] for stats in audio_dir_stats])
    )

    stats = {
        "audio_count": len(durations),
        "audio_samplerate_count": unique_sample_rates,
        "audio_mean_dur(sec)": np.mean(durations),
        "audio_median_dur(sec)": np.median(durations),
    }
    stats.update(
        {
            f"{str(p)}th percentile dur(sec)": np.percentile(durations, p)
            for p in [10, 25, 75, 90]
        }
    )
    json.dump(stats, open(out_file, "w"), indent=True)
