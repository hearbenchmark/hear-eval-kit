# ROADMAP

See [README.md](README.md) for developer getting start instructions.

## Components

Here are the basic components of the heareval package.

We are open to changing around the directory structure.

### Baseline

* `heareval/baseline.py` - A simple unlearned embedding model based
upon Mel-spectograms and random projection. This has code for the
entire competition API, including simple framing, etc.

### Tasks

These are pipelines that you literally run only once as a preprocessing
step, to put the tasks in a common format. We use the Luigi pipeline
library to break down preprocessing into a series of repeatable
steps.

* `heareval/tasks/` - Preprocessing code that, for each task,
downloads it, standardizes the audio length, resamples, partitions
into train/test, and creates CSV files `train.csv` and `test.csv`
with the labels. See [tasks/README.md](tasks/README.md) for more
details about the task format spec.

Once the heareval package has been installed on your system: 
(i.e. `pip install -e ".[dev]"`from project root for development), you can run the 
pipeline for a particular task using the following command 
from any directory:
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

* `heareval/embeddings_to_labels.py` - [not implemented yet] For
each task, given the `train.csv` and `test.csv`, impute
`test-predictions.csv` in the same format as `test.csv`.

For ranking tasks, there is no learning necessary

For learning-based tasks (classification, multiclass, multilabel),
this should use the embeddings as feature input, learn a linear or
single hidden layer model, and predict outputs. Some sort of sensible
validation and early stopping should probably be implemented. The
final model should then make predictions over the `test.csv` files.


The tentative plan is that heareval/task_embeddings.py runs the model on all the tasks and pickles the embeddings to disk.
Next we have a simple linear or single-hidden layer model that trains the mapping from embedding to label (or multilabel or frame-based multilabel).
That model is applied to test.csv, maybe something called: model-predictions.csv is output.
Finally (and where I'm proposing you jump in) is code that takes test.csv and model-predictions.csv and computes scores. First for multiclass and then other eval task types


### Lower priority tasks

Port baseline etc to TF.
