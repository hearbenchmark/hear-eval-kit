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
        README
        LICENSE
        train.csv
            [filename],...
        test.csv
            [filename],...
        audio/[sr]/train/[filename]

## More details

task.json also specifies the hop_size that we will use for the
evaluation.

If this is a task involving multiple classes or labels, the max
number of classes/labels will be provided. We might have two versions
of label files, ones with strings and ones converted to ints for
convenience.

## train.csv and test.csv

For classification/multi-classification of the entire sound:
```
filename, non-negative integer class
```

For tagging (multilabel sound event classification) of the entire sound:
```
filename, list of string labels
```

For frame-based temporal multilabel (e.g. transcription and sound event detection):
```
filename, float timestamp in seconds, list of string labels
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

## Caching with S3

1. Download and configure the AWS CLI if you haven't done that already:
    * [Intallation](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
    * [Configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html)

2. Update the S3 config file: `config/s3.py`
    * `S3_CACHE = True` enables S3 caching. Set this to False if you want to disable
        caching for all tasks.
    * `HANDLE` is a string that is used to create an S3 bucket for all the evaluation
        tasks. Every S3 bucket must have a unique name, so you should use this to create
        one for yourself. The value of `HANDLE` is appended to `hear2021-`. For example,
        if I set `HANDLE=jordie` then all my tasks will be cached in a bucket named
        `hear2021-jordie`.
    * `S3_REGION_NAME` sets the region for your S3 buckets. You can set this to `None`
        to use the default value set during CLI configuration.
