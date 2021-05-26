"""
Common Luigi classes and functions for evaluation tasks
"""

import os
import luigi
import requests


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
        return luigi.LocalTarget("_workdir/done-%s" % self.name)

    @property
    def workdir(self):
        d = "_workdir/%s/" % self.name
        ensure_dir(d)
        return d


def download_file(url, local_filename):
    """
    The downside of this approach versus `wget -c` is that this
    code does not resume.
    The benefit is that we are sure if the download completely
    successfuly, otherwise we should have an exception.
    From: https://stackoverflow.com/a/16696317/82733
    TODO: Would be nice to have a TQDM progress bar here.
    """
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return local_filename


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
