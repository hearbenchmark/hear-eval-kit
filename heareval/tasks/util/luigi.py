"""
Common Luigi classes and functions for evaluation tasks
"""

import hashlib
import os
import luigi
import requests
from tqdm import tqdm


class WorkTask(luigi.Task):
    """
    We assume following conventions:
        * Each luigi Task will have a name property:
            {classname}
            or
            {classname}-{task parameters}
            depending upon what your want the name to be.
            (TODO: Since we always use {classname}, just
            make this constant?)
        * The "output" of each task is a touch'ed file,
        indicating that the task is done. Each .run()
        method should end with this command:
            `_workdir/done-{name}`
            * Optionally, working output of each task will go into:
            `_workdir/{name}`
    Downstream dependencies should be cautious of automatically
    removing the working output, unless they are sure they are the
    only downstream dependency of a particular task (i.e. no
    triangular dependencies).
    """

    @property
    def name(self):
        ...
        # return type(self).__name__

    def output(self):
        return luigi.LocalTarget(
            "_workdir/%02d-%s.done" % (self.stage_number, self.name)
        )

    @property
    def workdir(self):
        d = "_workdir/%02d-%s/" % (self.stage_number, self.name)
        ensure_dir(d)
        return d

    @property
    def stage_number(self) -> int:
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
        else:
            raise ValueError("Unknown requires: %s" % self.requires())


def download_file(url, local_filename):
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
    return local_filename


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def filename_to_int_hash(filename):
    # Adapted from Google Speech Commands convention.
    hash_name_hashed = hashlib.sha1(filename.encode("utf-8")).hexdigest()
    return int(hash_name_hashed, 16)


def which_set(filename, validation_percentage, testing_percentage):
    """
    Code adapted from Google Speech Commands dataset.

    Determines which data partition the file should belong to, based
    upon the filename.

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
      String, one of 'train', 'val', or 'test'.
    """
    base_name = os.path.basename(filename)
    percentage_hash = filename_to_int_hash(filename) % 100
    if percentage_hash < validation_percentage:
        result = "val"
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
