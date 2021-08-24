"""
Common Luigi classes and functions for evaluation tasks
"""

import hashlib
import os
import os.path
from functools import partial
from pathlib import Path

import luigi
import pandas as pd
import requests
from tqdm.auto import tqdm


class WorkTask(luigi.Task):
    """
    We assume following conventions:
        * Each luigi Task will have a name property
        * The "output" of each task is a touch'ed file,
        indicating that the task is done. Each .run()
        method should end with this command:
            `_workdir/{task_subdir}{task_id}.done`
            task_id unique identifies the task by a combination of name and
            input parameters
            * Optionally, working output of each task will go into:
            `_workdir/{task_subdir}{name}`

    Downstream dependencies should be cautious of automatically
    removing the working output, unless they are sure they are the
    only downstream dependency of a particular task (i.e. no
    triangular dependencies).
    """

    # Class attribute sets the task name for all inheriting luigi tasks
    task_config = luigi.DictParameter(
        visibility=luigi.parameter.ParameterVisibility.PRIVATE
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add versioned task name to the task id, so that each task is unique across
        # different datasets. This helps in tracking task in the pipeline's DAG.
        self.task_id = f"{self.task_id}_{self.versioned_task_name}"

    @property
    def name(self):
        return type(self).__name__

    def output(self):
        """
        Outfile to mark a task as complete.
        """
        output_name = f"{self.stage_number:02d}-{self.task_id}.done"
        output_file = self.task_subdir.joinpath(output_name)
        return luigi.LocalTarget(output_file)

    def mark_complete(self):
        """Touches the output file, marking this task as complete"""
        self.output().open("w").close()

    @property
    def workdir(self):
        """Working directory"""
        d = self.task_subdir.joinpath(f"{self.stage_number:02d}-{self.name}")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def task_subdir(self):
        """Task specific subdirectory"""
        return Path(self.task_config.get("luigi_dir", "_workdir")).joinpath(
            str(self.versioned_task_name)
        )

    @property
    def versioned_task_name(self):
        """
        Versioned Task name contains the provided name in the
        data config and the version
        """
        return f"{self.task_config['task_name']}-{self.task_config['version']}"

    @property
    def stage_number(self):
        """
        Numerically sort the DAG tasks.
        This stage number will go into the name.

        This should be overridden as 0 by any task that has no
        requirements.
        """
        if isinstance(self.requires(), WorkTask):
            return 1 + self.requires().stage_number
        elif isinstance(self.requires(), list):
            return 1 + max([task.stage_number for task in self.requires()])
        elif isinstance(self.requires(), dict):
            parentasks = []
            for task in list(self.requires().values()):
                if isinstance(task, list):
                    parentasks.extend(task)
                else:
                    parentasks.append(task)
            return 1 + max([task.stage_number for task in parentasks])
        else:
            raise ValueError(f"Unknown requires: {self.requires()}")


def download_file(url, local_filename, expected_md5):
    """
    The downside of this approach versus `wget -c` is that this
    code does not resume.

    The benefit is that we are sure if the download completely
    successfuly, otherwise we should have an exception.
    From: https://stackoverflow.com/a/16696317/82733
    """
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get("content-length"))
        with open(local_filename, "wb") as f:
            pbar = tqdm(total=total_length)
            chunk_size = 8192
            for chunk in r.iter_content(chunk_size=chunk_size):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                f.write(chunk)
                pbar.update(chunk_size)
            pbar.close()
    assert (
        md5sum(local_filename) == expected_md5
    ), f"Md5sum for url: {url} is: {md5sum(local_filename)}"
    return local_filename


def filename_to_int_hash(text):
    """
    Returns the sha1 hash of the text passed in.
    """
    hash_name_hashed = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(hash_name_hashed, 16)


def which_set(filename_hash, validation_percentage, testing_percentage):
    """
    Code adapted from Google Speech Commands dataset.

    Determines which data split the file should belong to, based
    upon the filename int hash.

    We want to keep files in the same training, validation, or testing
    sets even if new ones are added over time. This makes it less
    likely that testing samples will accidentally be reused in training
    when long runs are restarted for example. To keep this stability,
    a hash of the filename is taken and used to determine which set
    it should belong to. This determination only depends on the name
    and the set proportions, so it won't change as other files are
    added.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'train', 'valid', or 'test'.
    """

    percentage_hash = filename_hash % 100
    if percentage_hash < validation_percentage:
        result = "valid"
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = "test"
    else:
        result = "train"
    return result


def new_basedir(filename, basedir):
    """
    Rewrite .../filename as basedir/filename
    """
    return os.path.join(basedir, os.path.split(filename)[1])


def md5sum(filename):
    """
    NOTE: Identical hash value as running md5sum from the command-line.
    """
    with open(filename, mode="rb") as f:
        with tqdm(total=os.path.getsize(filename)) as pbar:
            d = hashlib.md5()
            for buf in iter(partial(f.read, 32768), b""):
                d.update(buf)
                pbar.update(32768)
    return d.hexdigest()


def subsample_metadata(metadata: pd.DataFrame, max_files: int):
    """
    Returns the sampled metadata
    Stratification is currently commented out in this function

    Common Step -
        1. Sort the metadata by the split_key and the subsample_key
    If stratification is Done:
        2. Get the max files required for each stratify key by accounting for
            the fraction of the stratify key in the data
        3. For all the files associated with one stratify key, select max allowed files
            (calculated in the previous step)
        4. In case the stratify key has percentage low enough, and not even a single
            file is selected, select at least one file for each group

    If stratification is not Done:
        2. Select max number of files required for subsampling.

    """
    assert set(["split_key", "subsample_key"]).issubset(
        metadata.columns
    ), "All columns not found in input metadata"
    # Sort by the split key and the subsample key
    metadata = metadata.sort_values(
        by=["split_key", "subsample_key"], ascending=[True, True]
    )

    # Below code piece is for stratified subsampling.
    # However there are certain caveats while using it for Event based and Multilabel
    # tasks. For this reason we have disabled stratification across all the tasks.

    # Function - In the below code, stratify-key level max_file is calculated
    # (by accounting for the fraction of that stratify key in the datset) and the
    # corresponding number of audio files for that key is selected(deterministically).

    # Limitation - However, this is not the case for Event and Multilabel tasks, as one
    # file can span across multiple stratify keys(each file can have multiple labels).
    # Inclusion of one file due to one stratify key will effect other stratify keys
    # (the other labels for the file) and the max_files for one key will not be
    # maintained when selecting files for each group.
    """
    # Get group count for each group
    grp_count = metadata["stratify_key"].value_counts() * max_files / len(metadata)
    # Groupby and select the required sample from each group
    sampled_metadata = pd.concat(
        [
            # Ensure at least 1 sample is selected for each group
            stratify_grp.head(max(1, int(grp_count[grp])))
            for grp, stratify_grp in metadata.groupby("stratify_key")
        ]
    )
    # Assertions
    # if all the labels are there in the metadata after subsampling
    assert set(sampled_metadata["stratify_key"].unique()) == set(
        metadata["stratify_key"].unique()
    ), "All stratify groups are not in the sampled metadata."

    # If the subsampled data points are more than the max_subsample +
    # len(grp_count). The length of group count is here since some groups
    # might be too small and we might need to take one sample for the group.
    assert len(sampled_metadata) <= max_files + len(
        grp_count
    ), "Sampled metadata is more than the allowed max files + unique groups"
    """

    # The above code is commented and the below lines are added
    # The lines below donot do any stratification and just select the max_file
    # number of files, from the sorted metadata(by split key and subsample key)
    sampled_metadata = metadata.head(max_files)
    assert (
        len(sampled_metadata) <= max_files
    ), "Sampled metadata is more than the allowed max files"
    return sampled_metadata.sort_values("subsample_key")
