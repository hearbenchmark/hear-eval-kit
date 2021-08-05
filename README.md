# hear2021-eval-kit

Evaluation kit for HEAR 2021 NeurIPS competition

See [ROADMAP](ROADMAP.md).

## Usage

```
pip install heareval
```

You will need `ffmpeg>=4.2` installed (possibly from conda-forge).
You will need `soxr` support, which might require package
libsox-fmt-ffmpeg or [installing from
source](https://github.com/neuralaudio/hear-eval-kit/issues/156#issuecomment-893151305).

### Evaluation Tasks

These Luigi pipelines are used to preprocess the evaluation tasks
into a common format for downstream evaluation.

To run the preprocessing pipeline for Google Speech Commands:
```
python3 -m heareval.tasks.runner speech_commands
```

For NSynth pitch:
```
python3 -m heareval.tasks.runner nsynth_pitch
```

For DCASE 2016, Task 2 (sound event detection):
```
python3 -m heareval.tasks.runner dcase2016_task2
```

These commands will download and preprocess the entire dataset. An intermediary dir
call `_workdir` will be created, and then a final directory called `tasks` will contain
the completed dataset.

Options:
```
Options:
  --num-workers INTEGER  Number of CPU workers to use when running. If not
                         provided all CPUs are used.

  --sample-rate INTEGER  Perform resampling only to this sample rate. By
                         default we resample to 16000, 22050, 44100, 48000.
```

Additionally, to check the stats of an audio directory:
```
python3 -m heareval.tasks.audio_dir_stats {input folder} {output json file}
```
Stats include: audio_count, audio_samplerate_count, 
mean meadian and certain (10, 25, 75, 90) percentile durations.
This is helpful in getting a quick glance of the audio files in a folder and 
helps in decideing the preprocessing configurations.

### Computing embeddings

Once a set of tasks has been generated, embeddings can be computed using any audio
embedding model that follows the
[HEAR API](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api).

To compute embeddings using the [HEAR baseline](https://github.com/neuralaudio/hear-baseline):

1) Install the hearbaseline and download the model weights:
```
pip install hearbaseline
wget https://github.com/neuralaudio/hear-baseline/raw/main/saved_models/naive_baseline.pt
```

2) Compute the embeddings for all the tasks
```
python3 -m heareval.embeddings.runner hearbaseline --model ./naive_baseline.pt
```

This assumes that your current working directory contains a folder called `tasks`
produced by `heareval.tasks.runner`. If this directory is in a different location or
named something different you can use the option `--tasks-dir`:
```
python3 -m heareval.embeddings.runner hearbaseline --model ./naive_baseline.pt --tasks-dir /path/to/tasks
```

### Downstream Evaluation

```
python3 heareval/task_embeddings.py
```

[TODO: make sure this works with pip3 install]

## Development

Clone repo:
```
git clone https://github.com/neuralaudio/hear2021-eval-kit
cd hear2021-eval-kit
```
Install in development mode:
```
pip install -e ".[dev]"
```

Make sure you have pre-commit hooks installed:
```
pre-commit install
```

Running tests:
```
python3 -m pytest
```
