# hear2021-eval-kit

Evaluation kit for HEAR 2021 NeurIPS competition

## Installation

```
pip3 install heareval
```

Tested with Python 3.7 and 3.8. Python 3.9 is not officially supported
because pip3 installs are very finicky, but it might work.

You can use our preprocessed datasets. Otherwise, see "Development > Preprocessing"


## Evaluation

The easiest way to do evaluation is to launch a Spotty GCP instance:
```
spotty start
spotty sh
```

This requires the heareval Docker image, which is pre-built and
published on Dockerhub for your convenience.

Please refer to `README.spotty` for more details.

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

If you want to use your pip HEAR module, substitute `hearbaseline`
and `./naive_baseline.pt` below with your pip module name and model
weight path.

2) Compute the embeddings for all the tasks ("all") or one task:
```
python3 -m heareval.embeddings.runner hearbaseline --model ./naive_baseline.pt
    [--tasks-dir tasks]
    [--task task]
    [--embeddings-dir embeddings]
```

This assumes that your current working directory contains a folder
called `tasks` produced by `heareval.tasks.runner`. If this directory
is in a different location or named something different you can use
the option `--tasks-dir`. 

By default embeddings will be computed in a folder named `embeddings`
in the current working directory. To generate in a different location
use the option `--embeddings-dir`.

### Downstream Evaluation

For evaluation of each task, a shallow model will be trained on the
embeddings followed by task specific evaluations. The names of the
scoring functions used for these task specific evalutions can be
found in the `task_metadata.json` inside every task directory.

1) Train the shallow model and generate the test set predictions
for all tasks or one task:
```
python3 -m heareval.predictions.runner hearbaseline --model ./naive_baseline.pt \
    [--embeddings-dir embeddings]
    [--task task]
    [--gpus INT]
```

2) Evaluate the generated predictions for the test set for one or
all modules and for one or all tasks:
```
python3 -m heareval.evaluation.runner \
    [module]
    [--embeddings-dir embeddings]
    [--task task]
```

By default, both the steps above assume a folder named `embeddings`,
generated in the compute embeddings step. If this directory is
different, the option `--embeddings-dir` can be used.

Running the above will generate `evaluation_results.json` in the
current working directory containing the evalution scores for each
task.

## Development

Clone repo:
```
git clone https://github.com/neuralaudio/hear2021-eval-kit
cd hear2021-eval-kit
```
Add secret task submodule:
```
git submodule init
git submodule update
```
**_NOTE_**: Secret tasks are not available to participants. They should skip the above step. 

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

This will take about 2 user-CPU-hours for the open tasks. 100 GB free
disk space is required while processing. Final output is 11 GB.

These Luigi pipelines are used to preprocess the evaluation tasks
into a common format for downstream evaluation.

To run the preprocessing pipeline for all available tasks:
```
python3 -m heareval.tasks.runner all
```

You can also just run individual tasks:
```
python3 -m heareval.tasks.runner [speech_commands|nsynth_pitch|dcase2016_task2]
```
**_NOTE__**: To run the pipeline on secret tasks please ensure to initialise, update and install the `hearsecrettasks` submodule. This repository is not available for participants. If the submodule is set up :
- Both the aforementioned commands will work for secret tasks as well. 
- Running with the `all` option will trigger all the available set of open and secret tasks. 
- To run individual tasks, please use the corresponding `task` name. The secret task names are are also hidden and listed in the `hearsecrettasks` submodule.


Each pipeline will download and preprocess each dataset according
to the following DAG:
* DownloadCorpus
* ExtractArchive
* ExtractMetadata: Create splits over the entire corpus and find
the label metadata for them.
* SubsampleSplit (subsample each split) => MonoWavTrimCorpus => SplitData (symlinks)
* SplitData => {SplitMetadata, ResampleSubcorpus}
* SplitMetadata => MetadataVocabulary
* FinalizeCorpus

In terms of sampling:
* We create a 60/20/20 split if train/valid/test does not exist.
* We cap each split at 3/1/1/ hours of audio, defined as
* If further small sampling happens, that chooses a particular
number of audio samples per task.

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

  --work-dir    STRING   Temporary directory to save all the
                         intermediate tasks (will not be deleted afterwords).
                         It will require as much disk space as the final output,
                         if not more.
                         By default this is set to _workdir in the
                         module root directory.

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

**_NOTE_** : The `--small` flag which is used to run the task on a
small version of the dataset for development.

### End to end testing

To test the pipeline with small version of each task(currently only running on 
speech_commands.py), please run the following bash scripts
```
bash run-small.sh <path/to/temorary_dir>
```
All the required subfolders will be generated in the `temporary directory` provided above.
## DEPRECATED

See [ROADMAP](ROADMAP.md).
