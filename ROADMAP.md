# ROADMAP

See [README.md](README.md) for developer getting start instructions.

## Components

We are open to changing around the directory structure.

### Baseline

* heareval/baseline.py - A simple unlearned embedding model based
upon Mel-spectograms and random projection. This has code for the
entire competition API, including simple framing, etc.

### Tasks

* `heareval/tasks/` - This is preprocessing code that, for each
task, downloads it, standardizes the audio length, resamples, and
creates CSV files with the labels. See [tasks/README.md](tasks/README.md)
for more details about the spec. This also breaks things into train
and test splits.

This should be run from the repository root directory, and will generate:
* `_workdir` - intermediate temporary work
* `tasks` - 

We use the Luigi pipeline to break down preprocessing into a series
of repeatable steps.

We currently only have COUGHVID implemented. Once we have a second
evaluation task, we are hoping to make the Luigi pipeline super
generic and modular, and abstract away all boilerplate, so that
adding the next 20 tasks is really simple.

### Evaluation pipeline

* `heareval/task_embeddings.py` - Given a particular pip-module
that follows our API, e.g. `heareval.baseline`, run it over every
task and cache embeddings as numpy to disk. (Currently, pytorch
only.)




Think of the luigi pipeline as something that you literally only run once as a preprocessing step
The tentative plan is that heareval/task_embeddings.py runs the model on all the tasks and pickles the embeddings to disk.
Next we have a simple linear or single-hidden layer model that trains the mapping from embedding to label (or multilabel or frame-based multilabel).
That model is applied to test.csv, maybe something called: model-predictions.csv is output.
Finally (and where I'm proposing you jump in) is code that takes test.csv and model-predictions.csv and computes scores. First for multiclass and then other eval task types
