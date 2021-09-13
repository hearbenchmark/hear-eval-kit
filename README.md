![HEAR2021](https://neuralaudio.ai/assets/img/hear-header-sponsor.jpg)
# hear-eval-kit

Evaluation kit for HEAR 2021 NeurIPS competition, using tasks from
[hear-preprocess](https://github.com/neuralaudio/hear-preprocess).

# hear-eval-kit
Downstream evaluation on each task involves two
steps:
* computing audio embeddings
* learning a shallow fully-connected predictor

The first step's speed depends upon a variety of factors.
The second step's speed is relatively similar between models.

If you have any questions or comments:
* File an [issue](github.com/neuralaudio/hear-eval-kit/issues).
* Post on the [discussion board](discuss.neuralaudio.ai/).
* [Email us](mailto:deep at neuralaudio dot ai).

## Requirements

Tested with Python 3.7 and 3.8. Python 3.9 is not officially supported
because pip3 installs are very finicky, but it might work.

We officially support Torch 1.9 and Tensorflor 2.6.0, as well as
Tensorflow 2.4.2 using the hack described in the [Dockerfile
README](docker/README.md). We use CUDA 11.2. Other versions are
possible, please contact us.

We test on 16GB GCP GPUs.

## Quickstart

Here is a simple quickstart to evaluate `hearbaseline` using random
projections and a tiny subset of the open tasks. More detailed
instructions are below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralaudio/hear-eval-kit/blob/master/heareval_quickstart.ipynb)


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

## Download Open Tasks

If you are on GCP cloud, you can freely download open tasks as follows:

```
gsutil -m cp gs://hear2021/open-tasks/hear-2021.0.3-*-{SAMPLE_RATE}.gz . && for f in hear-*.gz; do tar zxf "$f"; done
```
where `SAMPLE_RATE` in `{16000, 20050, 32000, 44100, 48000}`  is
the sample rate your model desires.

If you are downloading from HTTPS, please only download open
tasks once and mirror them internally, because cloud downloads are
expensive for us. We are looking for longer-term hosting options.

Download:
```
https://storage.googleapis.com/hear2021/open-tasks/hear-2021.0.3-{TASK}-{SAMPLE_RATE}.tar.gz
```
for the following tasks:
```
    dcase2016_task2-hear2021-full
    nsynth_pitch-v2.2.3-5h
    nsynth_pitch-v2.2.3-50h
    speech_commands-v0.0.2-5h
    speech_commands-v0.0.2-full
```
where `SAMPLE_RATE` in `{16000, 20050, 32000, 44100, 48000}` is the
sample rate your model desires.

Untar all the files.

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
git clone https://github.com/neuralaudio/hear-eval-kit
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
