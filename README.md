# hear2021-eval-kit

Evaluation kit for HEAR 2021 NeurIPS competition

## Installation

```
pip3 install heareval
```

You can use our preprocessed datasets. Otherwise, see "Development > Preprocessing"

## Evaluation

### Computing embeddings

Once a set of tasks has been generated, embeddings can be computed
using any audio embedding model that follows the [HEAR
API](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api).

To compute embeddings using the [HEAR
baseline](https://github.com/neuralaudio/hear-baseline):

1) Install the hearbaseline and download the model weights:
```
pip3 install hearbaseline
wget https://github.com/neuralaudio/hear-baseline/raw/main/saved_models/naive_baseline.pt
```

2) Compute the embeddings for all the tasks
```
python3 -m heareval.embeddings.runner hearbaseline --model ./naive_baseline.pt
```

This assumes that your current working directory contains a folder
called `tasks` produced by `heareval.tasks.runner`. If this directory
is in a different location or named something different you can use
the option `--tasks-dir`:
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
pip3 install -e ".[dev]"
```

Make sure you have pre-commit hooks installed:
```
pre-commit install
```

Running tests:
```
python3 -m pytest
```

### Preprocessing

You probably don't need to do this unless you are implementing the
HEAR challenge.

If you want to run preprocessing yourself:
* You will need `ffmpeg>=4.2` installed (possibly from conda-forge).
* You will need `soxr` support, which might require package
libsox-fmt-ffmpeg or [installing from
source](https://github.com/neuralaudio/hear-eval-kit/issues/156#issuecomment-893151305).

These Luigi pipelines are used to preprocess the evaluation tasks
into a common format for downstream evaluation.

To run the preprocessing pipeline for all open tasks:
```
python3 -m heareval.tasks.runner all
```
You can also just run individual tasks:
python3 -m heareval.tasks.runner [speech_commands|nsynth_pitch|dcase2016_task2]

These commands will download and preprocess the entire dataset. An
intermediary directory defined by the option `luigi-dir`(default
`_workdir`) will be created, and then a final directory defined by
the option `tasks-dir` (default `tasks`) will contain the completed
dataset.

Options:
```
Options:
  --num-workers INTEGER  Number of CPU workers to use when running. If not
                         provided all CPUs are used.

  --sample-rate INTEGER  Perform resampling only to this sample rate. By
                         default we resample to 16000, 22050, 44100, 48000.
  
  --small       FLAG     If passed, the task will run on a small-version of the 
                         data.

  --luigi-dir   STRING   Path to dir to store the intermediate luigi task outputs.
                         By default this is set to _workdir in the module root directory

  --tasks-dir   STRING   Path to dir to store the final task outputs.
                         By default this is set to tasks in the module root directory
```

To check the stats of an audio directory:
```
python3 -m heareval.tasks.audio_dir_stats {input folder} {output json file}
```
Stats include: audio_count, audio_samplerate_count, mean meadian
and certain (10, 25, 75, 90) percentile durations.  This is helpful
in getting a quick glance of the audio files in a folder and helps
in decideing the preprocessing configurations.

The pipeline will also generate some stats of the original and
preprocessed data sets, e.g.:
```
speech_commands-v0.0.2/01-ExtractArchive/test_stats.json
speech_commands-v0.0.2/01-ExtractArchive/train_stats.json
speech_commands-v0.0.2/03-ExtractMetadata/labelcount_test.json
speech_commands-v0.0.2/03-ExtractMetadata/labelcount_train.json
speech_commands-v0.0.2/03-ExtractMetadata/labelcount_valid.json
```

### Faster preprocessing, for development

The small flag runs the preprocessing pipeline on a small version
of each dataset stored at [Downsampled HEAR Open
Tasks](https://github.com/turian/hear2021-open-tasks-downsampled). This
is used for development and continuous integration tests for the
pipeline.

These small versions of the data can be generated
deterministically with the following command:
```
python3 -m heareval.tasks.sampler <taskname>
```

**_NOTE_** : Each task config has `dataset_fraction`. The data in
each split is subsampled by this fraction in the final output. This
is not to be confused with the `--small` flag which is used to run
the task on a small version of the dataset for development. Also
the small version in the config has its own `dataset_fraction` which
can be used to subsample the small dataset when the small flag is
passed.

## DEPRECATED

See [ROADMAP](ROADMAP.md).
