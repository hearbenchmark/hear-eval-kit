evaluation-tasks
================

This folder contain Luigi pipelines to download and preprocess
evaluation tasks into a common format. Luigi checkpoints are saved
into directory .checkpoints so preprocessing can be resumed if
interrupted. After preprocessing, tar'ed outputs are saved to your
S3 bucket. This avoids hitting dataset providers repeatedly.

For each evaluation task, the directory structure is:
    taskname/
        task.json
        README.md
        LICENSE
        train.csv
            [filename],...
        test.csv
            [filename],...
        audio/[sr]/train/[filename]

## More details

task.json also specifies the frame_rate that we will use for the
evaluation.

If this is a task involving multiple classes or labels, the max
number of classes/labels will be provided. We might have two versions
of label files, ones with strings and ones converted to ints for
convenience.

## train.csv and test.csv

For classification/multi-classification of the entire sound:
```
filename, string label
```

For tagging (multilabel sound event classification) of the entire sound:
```
filename, list of string labels
```

For frame-based temporal multilabel (e.g. transcription and sound event detection):
```
filename, float timestamp in milliseconds, list of string labels
```

For ranking tasks:
```
list of filenames in ranked order
```

For JND tasks:
```
filename1, filename2, 0/1 indicates whether the audio is perceptually different to human listeners.
```

If the dataset provides a validation.csv, that will be included
too. Otherwise, participants do partition train into train/val
however they like.

## labelvocabulary.csv

A CSV file of the format:

```
string label, non-negative int starting at 0
```
