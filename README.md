# hear-eval-kit

Evaluation kit for the [HEAR Benchmark](https://hearbenchmark.com) using tasks from
[hear-preprocess](https://github.com/hearbenchmark/hear-preprocess)
and audio embedding models that follow the 
[HEAR API](https://hearbenchmark.com/hear-api.html).

Downstream evaluation on each task involves two
steps:
* computing audio embeddings
* learning a shallow fully-connected predictor

The first step's speed depends upon a variety of factors.
The second step's speed is relatively similar between models.

If you have any questions or comments:
* File an [issue](https://github.com/hearbenchmark/hear-eval-kit/issues).
* Email us (deep at neuralaudio dot ai).

## Requirements

Tested with Python 3.7 and 3.8. Python 3.9 is not officially supported
because pip3 installs are very finicky, but it might work.

We officially support Torch 1.9 and Tensorflor 2.6.0, as well as
Tensorflow 2.4.2 using the hack described in the [Dockerfile
README](docker/README.md). We use CUDA 11.2. Other versions are
possible, please contact us.

We test on 16GB GCP GPUs.

## Quickstart

Here is a simple example to evaluate the 
[hearbaseline wav2vec2](https://github.com/hearbenchmark/hear-baseline) audio embedding model on the 
[Mridingam Tonic](https://doi.org/10.5281/zenodo.4068196) task, which is a classification
task using sounds from a pitched percussion instrument called a Mridingam. 

This example shows how to compute embeddings on a pre-processed version of the data, and
then learn a shallow prediction model on the embeddings for evaluation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hearbenchmark/hear-eval-kit/blob/main/heareval_evaluation_example.ipynb)


## Installation

There are 3 ways to run `heareval`:
1) Locally, through pip3 install (or conda)
2) Using Docker
3) On the cloud

You are welcome to contact us if you have any questions or issues.

### Local installation

```
pip3 install heareval
```

### Docker

We have docker images containing the `heareval` environment.
`turian/heareval:stable` contains the latest stable image with all
dependencies bundled in.

### Cloud GPUs

The easiest way to do evaluation is to launch a Spotty GCP instance.
You can easily adapt Spotty also for AWS GPU instances.

Prepare a `spotty.yaml` file with the provided template file:
```
cp spotty.yaml.tmpl spotty.yaml
```
Change the instance name in the copied file. Specifically, change `"USERNAME"` 
suffix in `instances: name` to allow for multiple users in the same project 
to make separate gcp instances and volumes to avoid conflicts within the project.

Run spotty:
```
spotty start
spotty sh
```

This requires the heareval Docker image, which is pre-built and
published on Dockerhub for your convenience.

Please refer to `README.spotty` for more details.

## Downloading Tasks

We've released pre-processed versions of all HEAR 2021 open and secret tasks on Zenodo,
you can access those here: https://doi.org/10.5281/zenodo.5885750

**Note on Sample Rate:** 
All the tasks hosted on Zenodo have been pre-processed to 
48kHz. If the embedding model that you are using requires a different sample rate,
then you will need to resample the audio to that rate before running. 

For other sampling rates (16000, 22050, 32000, 44100), please download 
files (requester pays) from Google Storage 
[gs://hear2021-archive/tasks/](https://console.cloud.google.com/storage/browser/hear2021-archive/tasks)

Alternatively, you
can generate the pre-processed datasets using 
[hear-preprocess](https://github.com/hearbenchmark/hear-preprocess).

## Compute embeddings

```
time python3 -m heareval.embeddings.runner MODULE_NAME --model WEIGHTS_FILE --tasks-dir hear-2021.0.3/tasks/
```
where `MODULE_NAME` is your embedding model name.

This will create directories `embeddings/MODULE_NAME/TASK/` with
your embeddings. If you run the above command multiple times, it
will skip tasks it has already performed embedding on. You can
delete directories if you want to recompute embeddings.

There is an advanced option `--model-options` whereby you can pass
a JSON string of parameters to the model. This is useful for
experimenting with model hyperparameters. These options appear in
the embeddings output directory name, so you can run several different
model variations at once.

## Evaluation over embeddings

You can then run final downstream evaluation on these embeddings as follows:

```
python3 -m heareval.predictions.runner embeddings/{MODULE_NAME}/*
```

This will run on a particular module, over all tasks, with determinism
and the default number of grid points. Embeddings will be loaded
into CPU memory, for speed of training.
Logs will be sent to stdout and concise logs will be in `logs/`.
If you run this multiple times, it should be deterministic, but will
always start from scratch.

Ignore warnings about `Leaking Caffe2 thread-pool after fork`, this
is a known torch bug.

More advanced flags allow different downstream training regimes

Final test scores are logged to stdout and also to
`{EMBEDDINGS_DIR}/{MODULE_NAME}/{TASK_NAME}/test.predicted-scores.json`.

## Note on Speed

Models with larger embeddings scale sub-linearly in training time
(because of GPU optimizations) and linearly in hop-size (for
event-based prediction tasks). The main hyperparameters controlling
downstream training time are the maximum number of epochs and number
of grid points for grid search.

## Development

If you are developing this repo, clone repo:
```
git clone https://github.com/hearbenchmark/hear-eval-kit
cd hear-eval-kit
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


**_NOTE_** : Please make sure the workflows for each of the open task (`./gihub/workflows/task-{task_name}.yml`) is using the correct version of preprocessed tasks from the [Preprocessed Downsampled HEAR Open
Tasks](https://github.com/hearbenchmark/hear2021-open-tasks-downsampled/tree/main/preprocessed) Repo
for Continuous Integration.

Current hearpropress version used for Continuous Integration - `2021.0.6`

Please keep the version in sync with hearpreprocess
