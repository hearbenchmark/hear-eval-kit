# ROADMAP

See [README.md](README.md) for developer getting start instructions.

## Minimal complete pipeline

* `heareval/baseline.py` - Simple unlearned embedding algorithm
following the API.

* `heareval/tasks/` - One time preprocessing of tasks into train +
test WAVs and `train/test.csv`.

* `heareval/predict/` - [not implemented] For each task, map from
API-compatible embeddings to `predicted-test.csv`. (This involves
a linear model for most task types.)

* `heareval/evaluate/` - Given task type, `test.csv`, and
`predicted-test.csv` output evaluation scores.
(Note that we can implement this now just by creating random `test.csv`
and `predicted-test.csv` files, Christian is starting this task.)

## Task Types + Prediction

See [heareval/tasks/README.md](heareval/tasks/README.md) for more
details.

### Learning Based Tasks

Entire audio, multiclass:
    * Concat all embeddings for the audio, make a multiclass prediction.

Entire audio, multilabel:
    * Concat all embeddings for the audio, make a multilabel prediction.

Frame-based multilabel (transcription + sound event detection):
    * For each frame, make a multilabel prediction.

### Unlearned tasks

JND (audio1 vs audio2 has a perceptible difference):
    * np.abs(embedding1 - embedding2), where embedding1 is the
    concatenation of all audio1 frames, same for embedding2.
    * Compute AUC

Ranking tasks:
    * This requires more thought. There are two options:
	    1) Instead of each train and test row being a list of
	    files in ranked order, we could convert it to a long
	    set of file triples where the ordering is known.
    	This is a bit uglier and less general but simpler to work
    	with.
        2) We could perform a forced ranking on all the files.
        This makes the prediction code for ranking more gnarly,
        but the `predict-train.csv` is clean and consistent and
        we can just run Spearman correlation for eval.
    * Regardless of the final output of the predictor, we will
    probably need triplets.

## Components

Here are the basic components of the heareval package.

We are open to changing around the directory structure.

### Baseline

* `heareval/baseline.py` - A simple unlearned embedding model based
upon Mel-spectograms and random projection. This has code for the
entire competition API, including simple framing, etc.

### Tasks

These are pipelines that you literally run only once as a preprocessing
step, to put the tasks in a common format, deterministically. We
use the Luigi pipeline library to break down preprocessing into a
series of repeatable steps.

* `heareval/tasks/` - Preprocessing code that, for each task,
downloads it, standardizes the audio length, resamples, partitions
into train/test, and creates CSV files `train.csv` and `test.csv`
with the labels. See [tasks/README.md](tasks/README.md) for more
details about the task format spec.

Once the heareval package has been installed on your system: (i.e.
`pip install -e ".[dev]"`from project root for development), you
can run the pipeline for a particular task using the following
command from any directory:
```python
python3 -m heareval.tasks.<taskname>
```
Example, for coughvid:
```python
python3 -m heareval.tasks.coughvid
```
This will create the following:
* `_workdir` - intermediate temporary work directory used by Luigi to checkpoint and 
  save output from each stage of the pipeline
* `tasks` - directory that holds the finalized preprocessed data for each task

We currently only have COUGHVID implemented. Once we have a second
evaluation task implemented in Luigi, we are hoping to make the
Luigi pipeline super generic and modular, and abstract away all
boilerplate, so that adding the next 20 tasks is really straightforward.

### Evaluation pipeline

* `heareval/task_embeddings.py` - Given a particular pip-module
that follows our API, e.g. `heareval.baseline`, run it over every
task and cache every audio file's frames' embeddings as numpy to
disk. (Currently, pytorch only, but should be ported to TF2.)

* `heareval/embeddings_to_labels.py` - [not implemented yet] Embedding
to label prediction. For each task, given the `train.csv` and
`test.csv`, impute `test-predictions.csv` in the same format as
`test.csv`.
    * This will involve training for the training-based tasks listed above.

* `heareval/evaluation.py` - [not implemented yet] Compute various
task scores using `test.csv` and `predicted-test.csv`.

## Follow-On Work

* Port baseline etc to TF.

* `heareval/learned-baseline.py` - 
    1) For each individual task, train on `train.csv` to learn an
    embedding just for this task. Maybe use a Mel-spec followed by a
    conv-net?
    2) Multi-task training of one embedding over all tasks, i.e. instead
    of learning a different embedding per task, train over all tasks
    at the same time.
