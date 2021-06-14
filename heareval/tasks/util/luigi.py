"""
Common Luigi classes and functions for evaluation tasks
"""

import hashlib
import os
import luigi
import requests
import subprocess
import shutil
# Required for shutil to work for tar.gz
import zipfile

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

    # Class attribute sets the task name for all inheriting luigi tasks
    task_name = None

    @property
    def name(self):
        ...
        # return type(self).__name__

    def output(self):
        #Replace the name with task_id as it is unique at task and parameter level
        f = os.path.join(self.task_subdir, f"{self.stage_number:02d}-{self.task_id}.done")
        return luigi.LocalTarget(f)

    @property
    def workdir(self):
        d = os.path.join(self.task_subdir, f"{self.stage_number:02d}-{self.name}")
        ensure_dir(d)
        return d

    @property
    def task_subdir(self):
        """
        Task specific subdirectory
        """
        # You must specify a task name for WorkTask
        assert self.task_name is not None
        d = ["_workdir", str(self.task_name)]
        return os.path.join(*d)

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


class DownloadCorpus(WorkTask):
    """
    Task for downloading a dataset to a specific filename
    """

    # URL to download dataset from
    url = luigi.Parameter()
    outfile = luigi.Parameter()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        download_file(self.url, os.path.join(self.workdir, self.outfile))
        with self.output().open("w") as _:
            pass

    @property
    def stage_number(self) -> int:
        return 0


class ExtractArchive(WorkTask):
    
    infile = luigi.Parameter()
    #The previous task will be passed in as a parameter.
    #This must have workdir attribute and the zip file to extract should be stored inside it.
    prev_task = luigi.TaskParameter()
    
    @property
    def name(self):
        return type(self).__name__

    def requires(self):
        return self.prev_task

    def run(self):
        assert hasattr(self.requires(), "workdir"), \
        "The Task requires a task with a workdir attribute from which the file to unzip will be picked"

        corpus_zip = os.path.realpath(
            os.path.join(self.requires().workdir, self.infile)
        )
        shutil.unpack_archive(corpus_zip, self.workdir) 
        with self.output().open("w") as _:
            pass

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
            NOTE: Should be a relative path.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'train', 'val', or 'test'.
    """
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
